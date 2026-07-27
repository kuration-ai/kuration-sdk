"""Private runtime for the intersection-anchored chunked SLAM reconstruction prototyped in
`intersection.ipynb`: SALAD-embed frames -> find
cross-clip intersection points -> chunk each clip's timeline around those
intersections (or into equal-size windows if there's nothing to anchor to)
-> run each chunk through a depth model (DA3) in parallel across devices ->
stitch every chunk into one global trajectory (+ optional point cloud) via
per-boundary affine alignment on a shared camera frustum.

``depth_model`` may be passed as an already-constructed model object, or as
the string name of a Depth Anything 3 preset (e.g. ``"da3-base"``), in which
case the vendored ``depth_anything_3`` package loads and moves the pretrained
model onto this runtime's home device internally -- the caller never needs
to import ``depth_anything_3`` directly.
"""
import collections
import copy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

# NOTE: this module deliberately does NOT set KMP_DUPLICATE_LIB_OK. Importing torch next to
# another OpenMP-linked wheel (faiss, sklearn) can abort with a duplicate-runtime error, but
# suppressing that abort is what makes the resulting numerical corruption silent -- and an SDK
# must not make that decision for the whole process. Callers who accept the risk can export
# KMP_DUPLICATE_LIB_OK=TRUE themselves before importing kuration.

import numpy as np
import torch
from tqdm.auto import tqdm

# ----------------------------------------------------------------------
# Inlined from depth_anything_3.utils.geometry / utils.export.glb (vendored
# DA3 repo), verbatim, so this module has no import-time dependency on that
# package being installed or present on disk -- only on `depth_model` (built
# from it and passed in already-initialized) exposing `.inference(...)`.
# ----------------------------------------------------------------------


def _transpose_last_two_axes(arr):
    if arr.ndim < 2:
        return arr
    axes = list(range(arr.ndim))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    return arr.transpose(axes)


def _affine_inverse_np(A: np.ndarray) -> np.ndarray:
    """Inverse of a batch of (..., 4, 4) homogeneous transforms."""
    R = A[..., :3, :3]
    T = A[..., :3, 3:]
    P = A[..., 3:, :]
    return np.concatenate(
        [np.concatenate([_transpose_last_two_axes(R), -_transpose_last_two_axes(R) @ T], axis=-1), P],
        axis=-2,
    )


def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    """Accept (4,4) or (3,4) extrinsics, return (4,4)."""
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        H = np.eye(4, dtype=ext.dtype)
        H[:3, :4] = ext
        return H
    raise ValueError(f"extrinsic must be (4,4) or (3,4), got {ext.shape}")


def _get_conf_thresh(
    prediction,
    sky_mask,
    conf_thresh: float,
    conf_thresh_percentile: float = 10.0,
    ensure_thresh_percentile: float = 90.0,
) -> float:
    if sky_mask is not None and (~sky_mask).sum() > 10:
        conf_pixels = prediction.conf[~sky_mask]
    else:
        conf_pixels = prediction.conf
    lower = np.percentile(conf_pixels, conf_thresh_percentile)
    upper = np.percentile(conf_pixels, ensure_thresh_percentile)
    return min(max(conf_thresh, lower), upper)


def _depths_to_world_points_with_colors(
    depth: np.ndarray,
    K: np.ndarray,
    ext_w2c: np.ndarray,
    images_u8: np.ndarray,
    conf: Optional[np.ndarray],
    conf_thr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each frame, cast rays through K^-1, scale by depth to get camera-
    frame points, then map to world via (w2c)^-1. Colors are read off the
    same pixels."""
    N, H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (H*W,3)

    pts_all, col_all = [], []
    for i in range(N):
        d = depth[i]
        valid = np.isfinite(d) & (d > 0)
        if conf is not None:
            valid &= conf[i] >= conf_thr
        if not np.any(valid):
            continue

        d_flat = d.reshape(-1)
        vidx = np.flatnonzero(valid.reshape(-1))

        K_inv = np.linalg.inv(K[i])
        c2w = np.linalg.inv(_as_homogeneous44(ext_w2c[i]))

        rays = K_inv @ pix[vidx].T
        Xc = rays * d_flat[vidx][None, :]
        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
        Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)

        cols = images_u8[i].reshape(-1, 3)[vidx].astype(np.uint8)

        pts_all.append(Xw)
        col_all.append(cols)

    if len(pts_all) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    return np.concatenate(pts_all, 0), np.concatenate(col_all, 0)


def _estimate_scene_scale(points: np.ndarray, fallback: float = 1.0) -> float:
    if points.shape[0] < 2:
        return fallback
    lo = np.percentile(points, 5, axis=0)
    hi = np.percentile(points, 95, axis=0)
    diag = np.linalg.norm(hi - lo)
    return float(diag if np.isfinite(diag) and diag > 0 else fallback)


@dataclass
class CameraPose:
    """Pose of one camera in its connected component's stitched frame.

    `frustum_points` is `(5,3)`: apex, top-left, top-right, bottom-left,
    bottom-right. `cx`, `cy`, `f` are the camera intrinsics in real pixels
    (principal point + focal length, `f` averaged from fx/fy). `x_axis`,
    `y_axis`, `z_axis` are normalized world-space basis vectors for this
    camera's orientation, using standard OpenCV camera-local axes: `x` right,
    `y` down, `z` forward (through the focal point)."""
    frustum_points: np.ndarray
    cx: float
    cy: float
    f: float
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray


@dataclass
class SLAMOutput:
    """`trajectories`/`poses` are keyed by clip name, ordered by frame
    index, de-duplicated across overlapping chunks (first reconstruction of
    a shared frame wins). `poses[name]` is a list of `CameraPose`, one per
    frame in `trajectories[name]`. Place-recognition-connected clips share a
    coordinate frame; disconnected clips retain independent local frames.
    `points`/`colors` are `None` unless the runtime was built with
    `output_pointcloud=True`."""
    trajectories: Dict[str, np.ndarray]
    poses: Dict[str, List[CameraPose]]
    points: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    provenance: Optional[Mapping[str, Any]] = None


class SLAMOperator:
    """One object driving the full intersection-anchored reconstruction
    runtime. Call it directly (`runtime(data)`) to run every step in
    order, or call `find_intersections`/`build_chunks`/`process_chunks`/
    `stitch_chunks` individually.

    Args:
        depth_model: Required. Either a DA3-style model exposing
            `.inference(image=frames, process_res=..., export_dir=None) ->
            Prediction` (with `.depth`, `.intrinsics`, `.extrinsics`,
            `.conf`, `.processed_images`), already loaded onto its "home"
            device -- or the string name of a Depth Anything 3 preset (e.g.
            `"da3-base"`, `"da3-large"`, or a full `org/repo` Hub id), in
            which case the pretrained model is downloaded, constructed, and
            moved onto this runtime's home device internally.
        embedding_model: Optional. Any object exposing `.encode(frames)` or
            `__call__(frames)` that takes a list of `(H, W, 3)` uint8 frames
            and returns an `(N, D)` array of global descriptors (e.g. a
            SALAD model). All frame preprocessing (resize/normalize/color
            conversion) is the model's own responsibility -- this SDK just
            forwards the raw frames it was given.
        embeddings: Optional, mutually exclusive with `embedding_model`.
            Precomputed embeddings, either a list aligned with `data`'s clip
            order or a dict keyed by clip name, each an `(N_i, D)` array.
        output_pointcloud: Whether `stitch_chunks()` also returns a merged
            point cloud, or just trajectories/poses.
        devices: Device(s) to run the depth model on. `cpu` entries are
            dropped unless the whole list is CPU-only. Defaults to
            autodetecting a single cuda/mps/cpu device.
        window_size, overlap: Frames per single-clip chunk, and shared
            frames between temporally-adjacent chunks.
        bridge_size: Frames strictly before an intersection frame that its
            bridge anchor claims; internally split into
            `bridge_backward = bridge_size` and
            `bridge_forward = window_size - bridge_size` so the two always
            sum to `window_size`, matching the notebook's bridge-anchor
            invariant with a single parameter.
        process_res: Passed straight through to the depth model's
            `process_res` (longest-side resize target).
        conf_thresh: Percentile (0-100) used to adaptively pick the
            confidence threshold for point-cloud filtering (the notebook's
            `CONF_THRESH_PERCENTILE`); the separate absolute floor the
            notebook also uses (`conf_thresh=1.05`) is kept fixed internally.
        num_max_points: Cap on the final *merged* point cloud, applied once
            in `stitch_chunks()` -- not per chunk, which would let an
            N-chunk scene balloon to `N x num_max_points` points.
        top_k, similarity_threshold: FAISS neighbor search width and minimum
            cosine similarity for a candidate cross-clip intersection match.
        seed: Seed for deterministic final point-cloud capping.
        verbose: If True, print a tqdm progress bar while `process_chunks()`
            runs each chunk through the depth model.
    """

    def __init__(
        self,
        depth_model: Any,
        embedding_model: Optional[Any] = None,
        embeddings: Optional[Union[Sequence[np.ndarray], Mapping[str, np.ndarray]]] = None,
        output_pointcloud: bool = True,
        devices: Optional[Sequence[Union[str, "torch.device"]]] = None,
        window_size: int = 48,
        overlap: int = 4,
        bridge_size: int = 24,
        process_res: int = 504,
        conf_thresh: float = 20.0,
        num_max_points: int = 1_000_000,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        seed: int = 0,
        verbose: bool = False,
    ):
        if embedding_model is not None and embeddings is not None:
            raise ValueError("Pass either embedding_model or embeddings, not both.")
        integer_options = {
            "window_size": window_size,
            "overlap": overlap,
            "bridge_size": bridge_size,
            "process_res": process_res,
            "num_max_points": num_max_points,
            "top_k": top_k,
            "seed": seed,
        }
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in integer_options.values()):
            raise ValueError("SLAM integer options must be integers, not booleans")
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        if overlap < 1 or overlap >= window_size:
            raise ValueError("overlap must be in [1, window_size)")
        if bridge_size < 0 or bridge_size > window_size:
            raise ValueError("bridge_size must be in [0, window_size]")
        if process_res < 1 or num_max_points < 1 or top_k < 1:
            raise ValueError("process_res, num_max_points, and top_k must be positive")
        if not np.isfinite(conf_thresh) or not 0 <= conf_thresh <= 100:
            raise ValueError("conf_thresh must be a finite percentile in [0, 100]")
        if not np.isfinite(similarity_threshold) or not -1 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be finite and in [-1, 1]")

        self.embedding_model = embedding_model
        self.embeddings = embeddings
        self.output_pointcloud = output_pointcloud

        self.window_size = window_size
        self.overlap = overlap
        self.bridge_backward = bridge_size
        self.bridge_forward = window_size - bridge_size
        self.process_res = process_res
        self.conf_thresh_percentile = conf_thresh
        self.num_max_points = num_max_points
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.seed = seed
        self.verbose = verbose

        self.devices = self._resolve_devices(devices)
        self.depth_model = (
            self._load_depth_model(depth_model, self.devices[0])
            if isinstance(depth_model, str)
            else depth_model
        )
        self._device_models: Dict[torch.device, Any] = {}
        # Replicas are created under this lock (and, in `process_chunks`, before any worker
        # thread starts) so two per-device threads can never race on `_device_models`, nor
        # deepcopy the source model while another thread is running inference on it.
        self._model_lock = threading.Lock()

        self._clips: Optional[List[Dict[str, Any]]] = None
        self._bridge_pairs: Optional[Dict[Tuple[int, int], Tuple[float, int, int]]] = None
        self._all_chunks: Optional[List[Dict[str, Any]]] = None
        self._per_clip_chunk_order: Optional[Dict[int, List[Tuple[int, int, int]]]] = None
        self._chunk_results: Optional[List[Optional[Dict[str, Any]]]] = None

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_devices(devices) -> List[torch.device]:
        if not devices:
            if torch.cuda.is_available():
                return [torch.device("cuda")]
            if torch.backends.mps.is_available():
                return [torch.device("mps")]
            return [torch.device("cpu")]
        resolved = [torch.device(d) for d in devices]
        non_cpu = [d for d in resolved if d.type != "cpu"]
        return non_cpu if non_cpu else resolved

    @staticmethod
    def _load_depth_model(model_name: str, device: "torch.device") -> Any:
        """Download + construct a pretrained Depth Anything 3 model by preset
        name (e.g. `"da3-base"`) or full Hub id (e.g. `"depth-anything/da3-base"`),
        and move it onto `device` in eval mode. Imported lazily so this module
        has no import-time dependency on `depth_anything_3` for callers who
        already pass a constructed `depth_model` object."""
        from ._vendor.depth_anything_3.api import DepthAnything3

        repo_id = model_name if "/" in model_name else f"depth-anything/{model_name}"
        model = DepthAnything3.from_pretrained(repo_id)
        return model.to(device).eval()

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        module = type(x).__module__
        if module.startswith("torch"):
            return x.detach().cpu().numpy()
        if module.startswith("tensorflow"):
            return x.numpy()
        return np.asarray(x)

    @staticmethod
    def _to_uint8(arr: np.ndarray) -> np.ndarray:
        if arr.dtype == np.uint8:
            return arr
        if np.issubdtype(arr.dtype, np.floating):
            if arr.size and arr.max() <= 1.0 + 1e-6:
                arr = arr * 255.0
            return np.clip(np.round(arr), 0, 255).astype(np.uint8)
        return arr.astype(np.uint8)

    def _normalize_sequence(self, sequence) -> List[np.ndarray]:
        """One clip's frames (numpy/torch/tf array shaped (T,H,W,3), or a
        list of individual (H,W,3) frames) -> a list of (H,W,3) uint8
        numpy arrays, in whatever channel order the depth/embedding models
        expect (this SDK does not reorder channels)."""
        if isinstance(sequence, (list, tuple)):
            return [self._to_uint8(self._to_numpy(f)) for f in sequence]
        arr = self._to_numpy(sequence)
        if arr.ndim != 4:
            raise ValueError(f"Expected a sequence shaped (T, H, W, 3); got shape {arr.shape}")
        arr = self._to_uint8(arr)
        return [arr[t] for t in range(arr.shape[0])]

    def _ingest_data(self, data) -> List[Dict[str, Any]]:
        if isinstance(data, dict):
            names = list(data.keys())
            sequences = [data[n] for n in names]
        else:
            sequences = list(data)
            names = [f"clip_{i}" for i in range(len(sequences))]

        self._clips = [
            {"name": name, "frames": self._normalize_sequence(seq), "embedding": None}
            for name, seq in zip(names, sequences)
        ]
        self._bridge_pairs = None
        self._all_chunks = None
        self._per_clip_chunk_order = None
        self._chunk_results = None
        return self._clips

    def _require_clips(self, data) -> List[Dict[str, Any]]:
        if data is not None:
            return self._ingest_data(data)
        if self._clips is None:
            raise RuntimeError("No data ingested yet -- pass `data` or call the runtime with data first.")
        return self._clips

    # ------------------------------------------------------------------
    # Step 1 + 2: embed frames, find cross-clip intersection points
    # ------------------------------------------------------------------

    def _embed_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        assert self.embedding_model is not None
        if hasattr(self.embedding_model, "encode"):
            emb = self.embedding_model.encode(frames)
        else:
            emb = self.embedding_model(frames)
        return self._to_numpy(emb).astype(np.float32)

    def _assign_precomputed_embeddings(self):
        assert self._clips is not None
        assert self.embeddings is not None
        if isinstance(self.embeddings, dict):
            for clip in self._clips:
                if clip["name"] not in self.embeddings:
                    raise KeyError(f"No precomputed embedding provided for clip '{clip['name']}'.")
                clip["embedding"] = self._to_numpy(self.embeddings[clip["name"]]).astype(np.float32)
        else:
            embeddings_list = list(self.embeddings)
            if len(embeddings_list) != len(self._clips):
                raise ValueError(
                    f"Got {len(embeddings_list)} precomputed embedding array(s) for "
                    f"{len(self._clips)} clip(s)."
                )
            for clip, emb in zip(self._clips, embeddings_list):
                clip["embedding"] = self._to_numpy(emb).astype(np.float32)

        for clip in self._clips:
            if clip["embedding"].shape[0] != len(clip["frames"]):
                raise ValueError(
                    f"Clip '{clip['name']}' has {len(clip['frames'])} frames but "
                    f"{clip['embedding'].shape[0]} precomputed embeddings."
                )

    def _compute_best_matches(self) -> Dict[Tuple[int, int], Tuple[float, int, int]]:
        """One FAISS flat-index nearest-neighbor search over every clip's
        embeddings *together* already surfaces both cross-clip matches (an
        intersection between two different clips) and same-clip matches (a
        loop closure: this clip crossing its own earlier path) in a single
        pass -- we just no longer skip the `ci == cj` case, gated by the
        constraint that a same-clip match must be at least `window_size`
        frames apart in time (otherwise every match would trivially be a
        frame against its own near-identical neighbors). Only the single
        best match is kept per (clip, clip) key -- same simplification the
        runtime already makes for cross-clip pairs, now also covering the
        same clip paired with itself."""
        import faiss

        assert self._clips is not None
        all_embeddings = np.concatenate([c["embedding"] for c in self._clips], axis=0).astype(np.float32)
        owners = np.array([(ci, fi) for ci, c in enumerate(self._clips) for fi in range(len(c["frames"]))])

        norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        all_embeddings = all_embeddings / norms

        index = faiss.IndexFlatIP(all_embeddings.shape[1])
        index.add(all_embeddings)
        sims, neighbors = index.search(all_embeddings, self.top_k + 1)  # +1 since rank 0 is self

        best_match: Dict[Tuple[int, int], Tuple[float, int, int]] = {}
        for i in range(all_embeddings.shape[0]):
            ci, fi = owners[i]
            for rank in range(1, self.top_k + 1):
                j = neighbors[i, rank]
                score = sims[i, rank]
                if j < 0 or score < self.similarity_threshold:
                    continue
                cj, fj = owners[j]

                if ci == cj:
                    if abs(int(fi) - int(fj)) < self.window_size:
                        continue  # too close in time to be a real loop closure
                    cu, cv, fu, fv = ci, ci, min(int(fi), int(fj)), max(int(fi), int(fj))
                elif ci < cj:
                    cu, cv, fu, fv = ci, cj, fi, fj
                else:
                    cu, cv, fu, fv = cj, ci, fj, fi

                pair_key = (int(cu), int(cv))
                if pair_key not in best_match or score > best_match[pair_key][0]:
                    best_match[pair_key] = (float(score), int(fu), int(fv))
        return best_match

    def find_intersections(self, data=None):
        """Only does anything if an `embedding_model` or precomputed
        `embeddings` was provided at init -- otherwise a no-op returning
        `None`, so `build_chunks()` falls back to equal-size windows.
        Finds both cross-clip intersections and same-clip loop closures."""
        if self.embedding_model is None and self.embeddings is None:
            return None

        self._require_clips(data)
        assert self._clips is not None

        if self.embedding_model is not None:
            for clip in self._clips:
                clip["embedding"] = self._embed_frames(clip["frames"])
        else:
            self._assign_precomputed_embeddings()

        self._bridge_pairs = self._compute_best_matches()
        return self._bridge_pairs

    # ------------------------------------------------------------------
    # Step 3: chunk every clip's timeline
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_range(start, end, n_frames):
        length = end - start + 1
        if start < 0:
            start, end = 0, length - 1
        if end > n_frames - 1:
            end = n_frames - 1
            start = max(0, end - length + 1)
        return start, end

    def _build_clip_chunks(self, n_frames, anchors):
        """`anchors`: sorted, non-overlapping (start, end, bridge_id)
        ranges already claimed by intersection points in this clip. With no
        anchors at all, this degrades to plain equal-size window_size/
        overlap chunking across the whole clip -- the fallback used when
        no embedding model/embeddings were given."""
        window_size, overlap = self.window_size, self.overlap
        step = window_size - overlap
        chunks = list(anchors)

        boundaries = [(-1, -1, None)] + list(anchors) + [(n_frames, n_frames, None)]
        for (_, l_end, l_id), (r_start, _, r_id) in zip(boundaries, boundaries[1:]):
            gap_start = l_end + 1
            gap_end = r_start - 1
            has_left = l_id is not None
            has_right = r_id is not None

            if gap_start > gap_end:
                # Anchors already touch/overlap -- nothing to fill, except
                # when two different bridge anchors land in the same clip
                # sharing zero real frames; fall through so the merge-chunk
                # branch below still emits one small bridging chunk sharing
                # real frames with both anchors (needed for stitching to
                # have a shared frame to align on).
                if not (has_left and has_right and (l_end - r_start + 1) < 1):
                    continue

            if has_left and has_right:
                midpoint = (gap_start + gap_end) // 2

                forward_end = l_end
                start = l_end - overlap + 1
                while start + window_size - 1 <= midpoint:
                    end = start + window_size - 1
                    chunks.append((start, end, None))
                    forward_end = end
                    start += step

                backward_start = r_start
                end = r_start + overlap - 1
                while end - window_size + 1 >= midpoint + 1:
                    start2 = end - window_size + 1
                    chunks.append((start2, end, None))
                    backward_start = start2
                    end -= step

                merge_start = forward_end - overlap + 1
                merge_end = backward_start + overlap - 1
                chunks.append((merge_start, merge_end, None))
            elif has_right and not has_left:
                end = r_start + overlap - 1
                while end >= gap_start:
                    start = max(end - window_size + 1, gap_start)
                    chunks.append((start, end, None))
                    if start <= gap_start:
                        break
                    end -= step
            else:
                start = (l_end - overlap + 1) if has_left else gap_start
                while start <= gap_end:
                    end = min(start + window_size - 1, gap_end)
                    chunks.append((start, end, None))
                    if end >= gap_end:
                        break
                    start += step

        chunks.sort(key=lambda c: c[0])
        return chunks

    @staticmethod
    def _local_frame_refs(chunk):
        if chunk["type"] == "single":
            return [(chunk["clip"], f) for f in range(chunk["start"], chunk["end"] + 1)]
        refs = [(chunk["clip_u"], f) for f in range(chunk["range_u"][0], chunk["range_u"][1] + 1)]
        refs += [(chunk["clip_v"], f) for f in range(chunk["range_v"][0], chunk["range_v"][1] + 1)]
        return refs

    def _validate_chunk_adjacency(self):
        # Each entry in `order` is (chunk_idx, start, end) -- the explicit
        # range this clip occupies within that chunk, captured directly at
        # build time. This disambiguates loop-closure bridge chunks that
        # touch the same clip twice (clip_u == clip_v) via two different
        # ranges, which a lookup keyed only on chunk+clip couldn't resolve.
        assert self._per_clip_chunk_order is not None
        assert self._clips is not None
        for ci, order in self._per_clip_chunk_order.items():
            for (a_idx, a_start, a_end), (b_idx, b_start, b_end) in zip(order, order[1:]):
                if a_idx == b_idx:
                    continue
                shared_frame = max(a_start, b_start)
                assert shared_frame <= min(a_end, b_end), (
                    f"chunks {a_idx} and {b_idx} don't share any frame in clip {ci} "
                    f"({self._clips[ci]['name']}) -- chunking layout is broken"
                )

    def build_chunks(self, data=None):
        self._require_clips(data)
        assert self._clips is not None

        bridge_specs: Dict[int, Dict[str, Any]] = {}
        clip_anchors: Dict[int, List[Tuple[int, int, Optional[int]]]] = {
            ci: [] for ci in range(len(self._clips))
        }

        if self._bridge_pairs:
            for bridge_id, ((cu, cv), (_, fu, fv)) in enumerate(self._bridge_pairs.items()):
                n_u = len(self._clips[cu]["frames"])
                n_v = len(self._clips[cv]["frames"])
                range_u = self._clamp_range(fu - self.bridge_backward, fu + self.bridge_forward - 1, n_u)
                range_v = self._clamp_range(fv - self.bridge_backward, fv + self.bridge_forward - 1, n_v)
                bridge_specs[bridge_id] = {"clip_u": cu, "range_u": range_u, "clip_v": cv, "range_v": range_v}
                clip_anchors[cu].append((range_u[0], range_u[1], bridge_id))
                clip_anchors[cv].append((range_v[0], range_v[1], bridge_id))
            for ci in clip_anchors:
                clip_anchors[ci].sort(key=lambda a: a[0])

        all_chunks: List[Dict[str, Any]] = []
        bridge_chunk_index: Dict[int, int] = {}
        per_clip_chunk_order: Dict[int, List[Tuple[int, int, int]]] = {
            ci: [] for ci in range(len(self._clips))
        }

        for ci, clip in enumerate(self._clips):
            n_frames = len(clip["frames"])
            clip_chunks = self._build_clip_chunks(n_frames, clip_anchors[ci])

            for start, end, bridge_id in clip_chunks:
                if bridge_id is None:
                    idx = len(all_chunks)
                    all_chunks.append({"type": "single", "clip": ci, "start": start, "end": end})
                else:
                    if bridge_id not in bridge_chunk_index:
                        spec = bridge_specs[bridge_id]
                        idx = len(all_chunks)
                        all_chunks.append({
                            "type": "bridge", "bridge_id": bridge_id,
                            "clip_u": spec["clip_u"], "range_u": spec["range_u"],
                            "clip_v": spec["clip_v"], "range_v": spec["range_v"],
                        })
                        bridge_chunk_index[bridge_id] = idx
                    idx = bridge_chunk_index[bridge_id]
                per_clip_chunk_order[ci].append((idx, start, end))

        self._all_chunks = all_chunks
        self._per_clip_chunk_order = per_clip_chunk_order
        self._validate_chunk_adjacency()
        return all_chunks

    # ------------------------------------------------------------------
    # Step 4: run every chunk through the depth model, split across devices
    # ------------------------------------------------------------------

    @staticmethod
    def _same_device(a: Optional["torch.device"], b: "torch.device") -> bool:
        """Whether two devices name the same physical device.

        `torch.device("cuda") != torch.device("cuda:0")` even though both mean the first GPU,
        and autodetection (`_resolve_devices`) produces the *unindexed* form while a parameter
        always reports the *indexed* one. Comparing them raw made the single-device path always
        take the deepcopy branch, replicating the model onto the accelerator it already lived on
        (~2x VRAM, OOM on a full-size DA3). Normalize both sides to (type, index-or-0)."""
        if a is None:
            return False
        return (a.type, 0 if a.index is None else a.index) == \
            (b.type, 0 if b.index is None else b.index)

    @staticmethod
    def _clear_cached_device(model: Any) -> None:
        """Invalidate a lazily cached `.device` attribute on a freshly moved replica.

        Depth Anything 3 memoizes its own device in `self.device` on first use (see
        `_vendor/depth_anything_3/api.py::_get_model_device`) and `nn.Module.to()` does not
        reset it, so a deepcopy'd replica would keep moving its inputs to the *source* device --
        a timing-dependent cross-device RuntimeError. A `device` implemented as a property is
        recomputed on every access and has nothing to invalidate."""
        if isinstance(getattr(type(model), "device", None), property):
            return
        if isinstance(getattr(model, "device", None), torch.device):
            model.device = None

    def _get_model_for_device(self, device: torch.device):
        with self._model_lock:
            if device in self._device_models:
                return self._device_models[device]
            try:
                current_device = next(self.depth_model.parameters()).device
            except (StopIteration, AttributeError, TypeError):
                current_device = None
            if self._same_device(current_device, device):
                model = self.depth_model
            else:
                model = copy.deepcopy(self.depth_model).to(device).eval()
                self._clear_cached_device(model)
            self._device_models[device] = model
            return model

    @staticmethod
    def _frustum_points(K, ext_w2c, W, H, scale) -> np.ndarray:
        """(5,3): apex, top-left, top-right, bottom-left, bottom-right --
        same ray-casting math as DA3's `_camera_frustum_lines`, but
        returning the 5 unique points directly (image-space corner order is
        known up front: (0,0)=TL, (W-1,0)=TR, (W-1,H-1)=BR, (0,H-1)=BL)
        instead of 8 duplicated line-segment endpoints that then need a
        vertex-count/up-vector heuristic to re-identify -- the fragile trick
        CLAUDE.md documents as silently picking the wrong vertex under
        float32 rounding once round-tripped through a GLB export. We never
        export to GLB here, so we can just keep the ordering exact."""
        corners_px = np.array(
            [[0, 0, 1.0], [W - 1, 0, 1.0], [W - 1, H - 1, 1.0], [0, H - 1, 1.0]]
        )  # TL, TR, BR, BL
        K_inv = np.linalg.inv(K)
        c2w = np.linalg.inv(_as_homogeneous44(ext_w2c))
        apex = (c2w @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]

        rays = (K_inv @ corners_px.T).T
        z = rays[:, 2:3].copy()
        z[z == 0] = 1.0
        plane_cam = (rays / z) * scale
        plane_w = np.stack(
            [(c2w @ np.array([p[0], p[1], p[2], 1.0]))[:3] for p in plane_cam], axis=0
        )
        tl, tr, br, bl = plane_w
        return np.stack([apex, tl, tr, bl, br], axis=0)

    @staticmethod
    def _camera_basis(K, c2w_rotation):
        """cx, cy, f (real pixels) from the intrinsics, plus x/y/z world-
        space unit basis vectors read straight off the camera-to-world
        rotation's columns (OpenCV camera-local axes: x right, y down, z
        forward, since those are exactly `c2w_rotation @ [1,0,0]` etc.)."""
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        f = float((fx + fy) / 2.0)
        x_axis = c2w_rotation[:, 0]
        y_axis = c2w_rotation[:, 1]
        z_axis = c2w_rotation[:, 2]
        return (
            float(cx), float(cy), f,
            x_axis / np.linalg.norm(x_axis),
            y_axis / np.linalg.norm(y_axis),
            z_axis / np.linalg.norm(z_axis),
        )

    def _run_chunk(self, chunk, model) -> Dict[str, Any]:
        assert self._clips is not None
        refs = self._local_frame_refs(chunk)
        frames = [self._clips[ci]["frames"][fi] for ci, fi in refs]

        prediction = model.inference(image=frames, process_res=self.process_res, export_dir=None)

        c2w = _affine_inverse_np(prediction.extrinsics)
        camera_centers = c2w[:, :3, 3]

        if self.output_pointcloud:
            conf_thr = _get_conf_thresh(
                prediction, None, conf_thresh=1.05, conf_thresh_percentile=self.conf_thresh_percentile,
            )
            points, colors = _depths_to_world_points_with_colors(
                prediction.depth, prediction.intrinsics, prediction.extrinsics,
                prediction.processed_images, prediction.conf, conf_thr,
            )
            scene_scale = _estimate_scene_scale(points, fallback=1.0)
        else:
            points = np.zeros((0, 3), dtype=np.float32)
            colors = np.zeros((0, 3), dtype=np.uint8)
            scene_scale = _estimate_scene_scale(camera_centers, fallback=1.0)

        H, W = prediction.depth.shape[1:]
        frustum_points = [
            self._frustum_points(prediction.intrinsics[i], prediction.extrinsics[i], W, H, scene_scale * 0.01)
            for i in range(len(frames))
        ]
        camera_basis = [
            self._camera_basis(prediction.intrinsics[i], c2w[i, :3, :3])
            for i in range(len(frames))
        ]

        return {
            "refs": refs,
            "points": points,
            "colors": colors,
            "camera_centers": camera_centers,
            "frustum_points": frustum_points,
            "camera_basis": camera_basis,
        }

    def process_chunks(self):
        if self._all_chunks is None:
            raise RuntimeError("No chunks built yet -- call build_chunks() first.")

        n_chunks = len(self._all_chunks)
        self._chunk_results = [None] * n_chunks

        # Partition chunks across devices up front (round-robin) rather than
        # a shared work queue, so each device's model replica is only ever
        # touched by the one thread driving it.
        device_queues: Dict[torch.device, List[int]] = {d: [] for d in self.devices}
        for idx in range(n_chunks):
            device = self.devices[idx % len(self.devices)]
            device_queues[device].append(idx)

        # Materialize every replica up front, on this thread: deepcopying the source model while
        # another device's thread is already running inference on it is not safe.
        device_models = {
            device: self._get_model_for_device(device)
            for device, indices in device_queues.items() if indices
        }

        progress = tqdm(total=n_chunks, desc="Reconstructing chunks", disable=not self.verbose)
        progress_lock = threading.Lock()

        def _run_on_device(device, indices):
            assert self._all_chunks is not None
            model = device_models[device]
            results = {}
            for idx in indices:
                results[idx] = self._run_chunk(self._all_chunks[idx], model)
                with progress_lock:
                    progress.update(1)
            return results

        try:
            with ThreadPoolExecutor(max_workers=len(self.devices)) as pool:
                futures = [pool.submit(_run_on_device, d, idxs) for d, idxs in device_queues.items() if idxs]
                for future in as_completed(futures):
                    for idx, result in future.result().items():
                        self._chunk_results[idx] = result
        finally:
            progress.close()

        assert all(result is not None for result in self._chunk_results)
        return cast(List[Dict[str, Any]], self._chunk_results)

    # ------------------------------------------------------------------
    # Step 5: stitch every chunk into one global scene
    # ------------------------------------------------------------------

    _IDENTITY_AFFINE = np.hstack([np.eye(3), np.zeros((3, 1))])

    @staticmethod
    def _affine_alignment(source, target):
        n = source.shape[0]
        src_h = np.hstack([source, np.ones((n, 1))])
        M, *_ = np.linalg.lstsq(src_h, target, rcond=None)
        return M.T  # (3,4)

    @staticmethod
    def _apply_affine(vertices, M):
        n = vertices.shape[0]
        v_h = np.hstack([vertices, np.ones((n, 1))])
        return (M @ v_h.T).T

    @staticmethod
    def _to_homogeneous(M):
        return np.vstack([M, [0.0, 0.0, 0.0, 1.0]])

    def _build_adjacency(self):
        adjacency = collections.defaultdict(list)
        assert self._per_clip_chunk_order is not None
        for ci, order in self._per_clip_chunk_order.items():
            for (a_idx, a_start, a_end), (b_idx, b_start, b_end) in zip(order, order[1:]):
                if a_idx == b_idx:
                    continue
                adjacency[a_idx].append((b_idx, ci, (a_start, a_end), (b_start, b_end)))
                adjacency[b_idx].append((a_idx, ci, (b_start, b_end), (a_start, a_end)))
        return adjacency

    def _fit_chunk_edge(self, cur_idx, nxt_idx, clip_idx, r_cur, r_nxt):
        """Affine transform mapping chunk `nxt_idx`'s local frustum
        coordinates onto chunk `cur_idx`'s, using one frame both chunks
        independently reconstructed -- 5 correspondence points (apex + 4
        labelled corners) straight from each chunk's own local frame.
        `r_cur`/`r_nxt` are the explicit (start, end) ranges this clip
        occupies within each chunk, passed in directly so loop-closure
        bridge chunks touching the same clip twice aren't ambiguous."""
        shared_frame = max(r_cur[0], r_nxt[0])
        assert shared_frame <= min(r_cur[1], r_nxt[1]), (
            f"chunks {cur_idx} and {nxt_idx} don't share any frame in clip {clip_idx}"
        )

        assert self._chunk_results is not None
        current = self._chunk_results[cur_idx]
        following = self._chunk_results[nxt_idx]
        assert current is not None and following is not None
        cur_refs = current["refs"]
        nxt_refs = following["refs"]
        target = current["frustum_points"][cur_refs.index((clip_idx, shared_frame))]
        source = following["frustum_points"][nxt_refs.index((clip_idx, shared_frame))]
        return self._affine_alignment(source, target)

    def _assemble_output(self, global_transform) -> SLAMOutput:
        trajectories: Dict[str, np.ndarray] = {}
        poses: Dict[str, List[CameraPose]] = {}
        assert self._per_clip_chunk_order is not None
        assert self._clips is not None
        assert self._chunk_results is not None
        assert self._all_chunks is not None

        for ci, order in self._per_clip_chunk_order.items():
            name = self._clips[ci]["name"]
            seen: Dict[int, Tuple[np.ndarray, CameraPose]] = {}
            for chunk_idx, start, end in order:
                result = self._chunk_results[chunk_idx]
                assert result is not None
                M = global_transform[chunk_idx]
                R = M[:, :3]  # linear part only -- direction vectors don't translate
                ref_index = {ref: i for i, ref in enumerate(result["refs"])}
                for f in range(start, end + 1):
                    if f in seen:
                        continue
                    i = ref_index[(ci, f)]
                    center_g = self._apply_affine(result["camera_centers"][i:i + 1], M)[0]
                    frustum_g = self._apply_affine(result["frustum_points"][i], M)
                    cx, cy, foc, x_axis, y_axis, z_axis = result["camera_basis"][i]
                    pose = CameraPose(
                        frustum_points=frustum_g,
                        cx=cx, cy=cy, f=foc,
                        x_axis=self._rotate_and_normalize(x_axis, R),
                        y_axis=self._rotate_and_normalize(y_axis, R),
                        z_axis=self._rotate_and_normalize(z_axis, R),
                    )
                    seen[f] = (center_g, pose)

            ordered_frames = sorted(seen)
            if ordered_frames:
                trajectories[name] = np.stack([seen[f][0] for f in ordered_frames], axis=0)
                poses[name] = [seen[f][1] for f in ordered_frames]
            else:
                trajectories[name] = np.zeros((0, 3), dtype=np.float32)
                poses[name] = []

        points, colors = None, None
        if self.output_pointcloud:
            all_points, all_colors = [], []
            for chunk_idx in range(len(self._all_chunks)):
                result = self._chunk_results[chunk_idx]
                assert result is not None
                if result["points"].shape[0] == 0:
                    continue
                M = global_transform[chunk_idx]
                all_points.append(self._apply_affine(result["points"], M))
                all_colors.append(result["colors"])

            if all_points:
                points = np.concatenate(all_points, axis=0).astype(np.float32)
                colors = np.concatenate(all_colors, axis=0)
                points, colors = self._cap_points(points, colors, self.num_max_points)
            else:
                points = np.zeros((0, 3), dtype=np.float32)
                colors = np.zeros((0, 3), dtype=np.uint8)

        return SLAMOutput(trajectories=trajectories, poses=poses, points=points, colors=colors)

    @staticmethod
    def _rotate_and_normalize(vec, R):
        v = R @ vec
        return v / np.linalg.norm(v)

    def _cap_points(self, points, colors, num_max):
        if points.shape[0] <= num_max:
            return points, colors
        idx = np.random.default_rng(self.seed).choice(points.shape[0], num_max, replace=False)
        return points[idx], colors[idx]

    def stitch_chunks(self) -> SLAMOutput:
        if self._chunk_results is None:
            raise RuntimeError("No chunk results yet -- call process_chunks() first.")
        if self._per_clip_chunk_order is None or self._all_chunks is None:
            raise RuntimeError("No chunk layout yet -- call build_chunks() first.")

        adjacency = self._build_adjacency()
        global_transform = {}
        visited = set()

        # Place-recognition matches connect clips that share one reconstructed scene. Clips with
        # no qualifying match are still valid inputs: reconstruct each disconnected component in
        # its own local coordinate frame instead of failing the entire multi-clip run.
        for root in range(len(self._all_chunks)):
            if root in visited:
                continue
            global_transform[root] = self._IDENTITY_AFFINE.copy()
            visited.add(root)
            queue = collections.deque([root])
            while queue:
                cur = queue.popleft()
                for nxt, ci, r_cur, r_nxt in adjacency[cur]:
                    if nxt in visited:
                        continue
                    M_local = self._fit_chunk_edge(cur, nxt, ci, r_cur, r_nxt)
                    global_transform[nxt] = (
                        self._to_homogeneous(global_transform[cur])
                        @ self._to_homogeneous(M_local)
                    )[:3, :]
                    visited.add(nxt)
                    queue.append(nxt)

        return self._assemble_output(global_transform)

    # ------------------------------------------------------------------

    def __call__(self, data) -> SLAMOutput:
        self._ingest_data(data)
        self.find_intersections()
        self.build_chunks()
        self.process_chunks()
        return self.stitch_chunks()
