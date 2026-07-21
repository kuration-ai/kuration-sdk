"""Optional global SLAM execution before bounded Kuration curation begins."""
from __future__ import annotations

from collections.abc import Callable, Mapping
import importlib
import json
from pathlib import Path

import numpy as np

from .export import EXPORT_FORMAT, EXPORT_VERSION, export_slam_output


def _pipeline_factory(factory: Callable | str | None) -> Callable:
    if factory is None:
        factory = "slam_pipeline:SLAMPipeline"
    if isinstance(factory, str):
        module_name, separator, attribute = factory.partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError("pipeline_factory strings must use 'module:attribute' syntax")
        try:
            module = importlib.import_module(module_name)
            factory = getattr(module, attribute)
        except (ImportError, AttributeError) as error:
            raise ImportError(
                f"cannot load SLAM pipeline factory {module_name}:{attribute}; install or "
                "provide the SLAMPipeline implementation, or pass pipeline_factory= explicitly"
            ) from error
    if not callable(factory):
        raise TypeError("pipeline_factory must be callable or an import string")
    return factory


def _validated_frames(data: Mapping[str, object]) -> dict[str, np.ndarray]:
    if not isinstance(data, Mapping) or not data:
        raise ValueError("raw SLAM input must be a non-empty {clip_name: frames} mapping")
    validated = {}
    for name, frames in data.items():
        if not isinstance(name, str) or not name:
            raise ValueError("raw SLAM clip names must be non-empty strings")
        value = np.asarray(frames)
        if value.ndim != 4 or value.shape[0] < 1 or value.shape[-1] != 3:
            raise ValueError(f"raw SLAM frames for {name!r} must have shape (T, H, W, 3)")
        if value.dtype.kind not in "uif" or not np.isfinite(value).all():
            raise ValueError(f"raw SLAM frames for {name!r} must be a finite numeric array")
        validated[name] = value
    return validated


def decode_video_clips(
    videos: Mapping[str, str | Path],
    *,
    frame_skip: int = 1,
    batch_size: int = 64,
    color_space: str = "BGR",
    provider=None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Decode video paths into the frame mapping and timestamps expected by SLAMPipeline."""
    if not isinstance(videos, Mapping) or not videos:
        raise ValueError("video input must be a non-empty {clip_name: path} mapping")
    if not isinstance(frame_skip, int) or isinstance(frame_skip, bool) or frame_skip < 1:
        raise ValueError("frame_skip must be a positive integer")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("video decode batch_size must be a positive integer")
    if color_space not in {"RGB", "BGR"}:
        raise ValueError("SLAM video color_space must be 'RGB' or 'BGR'")
    if provider is None:
        try:
            from kuration import PyAVMediaProvider

            provider = PyAVMediaProvider()
        except ImportError as error:
            raise ImportError(
                "decoding video for SLAM needs the video extra: "
                "pip install 'kuration-plugin-slam[video]'"
            ) from error
    frames_by_clip = {}
    timestamps = {}
    for name, uri in videos.items():
        if not isinstance(name, str) or not name:
            raise ValueError("video clip names must be non-empty strings")
        if not isinstance(uri, (str, Path)) or not str(uri):
            raise ValueError(f"video path for {name!r} must be a non-empty path or URI")
        refs = list(provider.iter_refs(str(uri), source_id=name))[::frame_skip]
        if not refs:
            raise ValueError(f"video for {name!r} has no frames")
        batches = [
            np.asarray(provider.decode_batch(refs[start:start + batch_size]))
            for start in range(0, len(refs), batch_size)
        ]
        frames = np.concatenate(batches, axis=0)
        if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) != len(refs):
            raise ValueError(
                f"video provider for {name!r} must decode shape (T, H, W, 3), got {frames.shape}"
            )
        # Kuration's video provider declares RGB. The example notebook's cv2 loader supplies
        # BGR, so BGR remains the compatibility default for SLAMPipeline integrations.
        if color_space == "BGR":
            frames = frames[..., ::-1].copy()
        frames_by_clip[name] = frames
        timestamps[name] = np.asarray([ref.timestamp for ref in refs], dtype=np.float64)
    return frames_by_clip, timestamps


def run_slam_pipeline(
    data: Mapping[str, object],
    *,
    depth_model=None,
    embedding_model=None,
    embeddings=None,
    pipeline_factory: Callable | str | None = None,
    **pipeline_options,
):
    """Run the notebook-style ``SLAMPipeline`` over raw in-memory clip frames.

    Models are injected by the caller so the plugin has no mandatory Torch, Depth Anything,
    SALAD, or FAISS dependency. With no explicit factory, the function lazily imports
    ``slam_pipeline.SLAMPipeline`` exactly as the example notebook does.
    """
    frames = _validated_frames(data)
    if depth_model is None:
        raise ValueError("running SLAM needs depth_model; precomputed SLAM input does not")
    if embedding_model is not None and embeddings is not None:
        raise ValueError("pass embedding_model or embeddings, not both")
    options = dict(pipeline_options)
    options["depth_model"] = depth_model
    if embedding_model is not None:
        options["embedding_model"] = embedding_model
    if embeddings is not None:
        options["embeddings"] = embeddings
    pipeline = _pipeline_factory(pipeline_factory)(**options)
    if not callable(pipeline):
        raise TypeError("pipeline_factory must construct a callable SLAM pipeline")
    output = pipeline(frames)
    if not isinstance(getattr(output, "trajectories", None), Mapping) \
            or not isinstance(getattr(output, "poses", None), Mapping):
        raise ValueError("SLAM pipeline output must expose trajectory and pose mappings")
    return output


def _existing_manifest(source) -> Path | None:
    if not isinstance(source, (str, Path)):
        return None
    path = Path(source)
    if path.is_dir():
        path = path / "manifest.json"
    if not path.is_file():
        raise ValueError(f"SLAM input path does not contain a manifest: {path}")
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read SLAM manifest {str(path)!r}") from error
    if document.get("format") != EXPORT_FORMAT or document.get("version") != EXPORT_VERSION:
        raise ValueError(
            f"SLAM manifest must declare format={EXPORT_FORMAT!r}, version={EXPORT_VERSION}"
        )
    return path


def prepare_slam_input(
    source,
    out_dir: str | Path | None = None,
    *,
    run_if_needed: bool = True,
    runner_options: Mapping[str, object] | None = None,
    decoder_options: Mapping[str, object] | None = None,
    export_options: Mapping[str, object] | None = None,
) -> Path:
    """Normalize existing manifests, completed outputs, or raw frames into one manifest.

    - A manifest path is returned unchanged.
    - A completed SLAM output object is exported without constructing a pipeline.
    - A raw ``{clip_name: frames}`` mapping runs SLAM only when ``run_if_needed`` is true.
    """
    manifest = _existing_manifest(source)
    if manifest is not None:
        return manifest
    if out_dir is None:
        raise ValueError("out_dir is required when the input is not an existing SLAM manifest")
    export_kwargs = dict(export_options or {})
    if isinstance(getattr(source, "trajectories", None), Mapping) \
            and isinstance(getattr(source, "poses", None), Mapping):
        return export_slam_output(source, out_dir, **export_kwargs)
    if isinstance(source, Mapping):
        if not run_if_needed:
            raise ValueError("raw frame input requires SLAM, but run_if_needed=False")
        if source and all(isinstance(value, (str, Path)) for value in source.values()):
            frames, timestamps = decode_video_clips(source, **dict(decoder_options or {}))
            export_kwargs.setdefault("timestamps", timestamps)
            export_kwargs.setdefault("videos", {name: str(path) for name, path in source.items()})
        else:
            frames = source
        output = run_slam_pipeline(frames, **dict(runner_options or {}))
        return export_slam_output(output, out_dir, **export_kwargs)
    raise TypeError(
        "SLAM input must be a manifest path, a completed output object, or raw frame mapping"
    )
