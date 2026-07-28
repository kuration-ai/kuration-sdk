"""Portable export of a completed ``SLAMOperator`` reconstruction."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Mapping
from urllib.parse import unquote, urlparse

import numpy as np

from ._paths import local_path
from .pointcloud import POINTCLOUD_FORMAT, POINTCLOUD_VERSION, PointCloudArtifact


EXPORT_FORMAT = "kuration-slam-export"
EXPORT_VERSION = 1


def _json_mapping(value: Mapping[str, object] | None, *, name: str) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON-safe mapping or None")
    try:
        encoded = json.dumps(dict(value), allow_nan=False, sort_keys=True)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a JSON-safe mapping") from error
    if not isinstance(decoded, dict):  # pragma: no cover - guarded by Mapping above
        raise ValueError(f"{name} must be a JSON object")
    return decoded


def _finite_array(value, *, name: str, shape_tail: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array") from error
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        raise ValueError(f"{name} must have shape (T, {', '.join(map(str, shape_tail))})")
    if not len(array):
        raise ValueError(f"{name} must contain at least one pose")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _payload_digest(values: Mapping[str, np.ndarray]) -> str:
    digest = sha256()
    for name, value in sorted(values.items()):
        array = np.ascontiguousarray(value)
        header = json.dumps(
            {"name": name, "dtype": str(array.dtype), "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(header.encode("utf-8"))
        digest.update(b"\0")
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _clip_filename(name: str, payload_digest: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-") or "clip"
    name_digest = sha256(name.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:56]}-{name_digest}-{payload_digest[:12]}.npz"


def _atomic_savez(target: Path, payload: Mapping[str, np.ndarray]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(target: Path, value: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_media_suffix(source: Path) -> str:
    suffix = source.suffix.lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,10}", suffix) else ".media"


def _bundle_video(source: Path, root: Path, *, name: str) -> str:
    """Copy one external local video into the export using an atomic content address."""
    media_dir = root / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    if media_dir.is_symlink():
        raise ValueError(f"{name} cannot be bundled through symlinked directory {media_dir!s}")
    try:
        media_dir.resolve().relative_to(root.resolve(strict=False))
    except ValueError as error:
        raise ValueError(f"{name} media directory escapes the SLAM export root") from error

    descriptor, temporary_name = tempfile.mkstemp(
        dir=media_dir,
        prefix=".video.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    digest = sha256()
    try:
        with source.open("rb") as source_handle, os.fdopen(descriptor, "wb") as target_handle:
            while True:
                chunk = source_handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                target_handle.write(chunk)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        filename = f"video-{digest.hexdigest()}{_safe_media_suffix(source)}"
        os.replace(temporary, media_dir / filename)
    except Exception:
        # ``os.fdopen`` owns the descriptor after it succeeds; close it when opening the
        # source failed first.
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return (Path("media") / filename).as_posix()


def _video_reference(value: str | Path, root: Path, *, name: str) -> str:
    """Return a portable video reference, bundling local files outside the export tree."""
    text = str(value)
    if not text:
        raise ValueError(f"{name} must be a non-empty path or URI")
    parsed = urlparse(text)
    if parsed.scheme and len(parsed.scheme) > 1:
        if parsed.query or parsed.fragment or parsed.params:
            raise ValueError(
                f"{name} remote URI must not contain query parameters, path parameters, "
                "or a fragment; use a stable credential-free URI"
            )
        try:
            has_credentials = parsed.username is not None or parsed.password is not None
        except ValueError as error:
            raise ValueError(f"{name} is not a valid remote URI") from error
        if has_credentials:
            raise ValueError(
                f"{name} remote URI must not contain embedded credentials; "
                "configure credentials through the media provider"
            )
        if parsed.scheme.lower() != "file":
            return text
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError(f"{name} file URI must refer to the local host")
        text = unquote(parsed.path)

    try:
        source = Path(text).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"{name} local video does not exist: {text!r}") from error
    try:
        mode = source.stat().st_mode
    except OSError as error:
        raise ValueError(f"{name} local video cannot be inspected: {text!r}") from error
    if not stat.S_ISREG(mode):
        raise ValueError(f"{name} local video must be a regular file: {text!r}")

    try:
        return source.relative_to(root.resolve(strict=False)).as_posix()
    except ValueError:
        return _bundle_video(source, root, name=name)


def _times_for_clip(
    name: str,
    length: int,
    *,
    timestamps: Mapping[str, object] | None,
    sample_hz: float | None,
) -> np.ndarray:
    if timestamps is not None:
        if name not in timestamps:
            raise ValueError(f"timestamps is missing clip {name!r}")
        values = np.asarray(timestamps[name], dtype=np.float64)
    elif sample_hz is not None:
        values = np.arange(length, dtype=np.float64) / sample_hz
    else:
        values = np.arange(length, dtype=np.float64)
    if values.shape != (length,):
        raise ValueError(f"timestamps[{name!r}] must have shape ({length},)")
    if not np.isfinite(values).all():
        raise ValueError(f"timestamps[{name!r}] must contain only finite values")
    if len(values) > 1 and np.any(np.diff(values) <= 0):
        raise ValueError(f"timestamps[{name!r}] must be strictly increasing")
    return values


def _confidence_for_clip(
    name: str,
    length: int,
    tracking_confidence: Mapping[str, object] | None,
) -> np.ndarray:
    if tracking_confidence is None or name not in tracking_confidence:
        return np.ones(length, dtype=np.float64)
    values = np.asarray(tracking_confidence[name], dtype=np.float64)
    if values.shape != (length,):
        raise ValueError(f"tracking_confidence[{name!r}] must have shape ({length},)")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(
            f"tracking_confidence[{name!r}] must contain finite values in [0, 1]"
        )
    return values


def _pose_arrays(name: str, poses, length: int) -> dict[str, np.ndarray]:
    rows = list(poses)
    if len(rows) != length:
        raise ValueError(
            f"output.poses[{name!r}] has {len(rows)} poses; trajectory has {length} rows"
        )

    def vectors(attribute: str, width: int = 3) -> np.ndarray:
        values = [getattr(pose, attribute) for pose in rows]
        return _finite_array(values, name=f"{name}.{attribute}", shape_tail=(width,))

    frustums = _finite_array(
        [pose.frustum_points for pose in rows],
        name=f"{name}.frustum_points",
        shape_tail=(5, 3),
    )
    intrinsics = _finite_array(
        [[pose.cx, pose.cy, pose.f] for pose in rows],
        name=f"{name}.intrinsics",
        shape_tail=(3,),
    )
    return {
        "camera_x_axis": vectors("x_axis"),
        "camera_y_axis": vectors("y_axis"),
        "camera_z_axis": vectors("z_axis"),
        "camera_intrinsics": intrinsics,
        "camera_frustum": frustums.reshape(length, 15),
    }


def _point_cloud(
    output,
    point_attributes: Mapping[str, object] | None,
) -> tuple[dict, dict[str, np.ndarray]] | None:
    points = getattr(output, "points", None)
    if points is None:
        return None
    artifact = PointCloudArtifact(
        points,
        getattr(output, "colors", None),
        point_attributes or {},
    )
    payload = {
        "_format": np.asarray(POINTCLOUD_FORMAT),
        "_version": np.asarray(POINTCLOUD_VERSION, dtype=np.int64),
        "points": artifact.points,
        **dict(artifact.attributes),
    }
    if artifact.colors is not None:
        payload["colors"] = artifact.colors
    relative = f"scene_pointcloud-{_payload_digest(payload)[:12]}.npz"
    metadata = {
        "pointcloud": relative,
        "format": POINTCLOUD_FORMAT,
        "version": POINTCLOUD_VERSION,
        "count": len(artifact),
        "attributes": sorted(artifact.attributes),
    }
    return metadata, payload


def export_slam_output(
    output,
    out_dir: str | Path,
    *,
    timestamps: Mapping[str, object] | None = None,
    sample_hz: float | None = None,
    videos: Mapping[str, str | Path] | None = None,
    point_attributes: Mapping[str, object] | None = None,
    tracking_confidence: Mapping[str, object] | None = None,
    provenance: Mapping[str, object] | None = None,
) -> Path:
    """Export one completed ``SLAMOperator`` output into a Kuration-readable manifest.

    The function is intentionally duck-typed: importing this package never imports or installs
    the heavyweight SLAM implementation. ``output`` needs the attributes demonstrated by the
    example notebook: ``trajectories``, ``poses``, and optional ``points``/``colors``.
    """
    if sample_hz is not None:
        if isinstance(sample_hz, bool) or not isinstance(sample_hz, (int, float)) \
                or not np.isfinite(sample_hz) or sample_hz <= 0:
            raise ValueError("sample_hz must be a positive finite number or None")
        sample_hz = float(sample_hz)
    trajectories = getattr(output, "trajectories", None)
    poses = getattr(output, "poses", None)
    if not isinstance(trajectories, Mapping) or not trajectories:
        raise ValueError("output.trajectories must be a non-empty mapping")
    if not isinstance(poses, Mapping):
        raise ValueError("output.poses must be a mapping")
    if videos is not None and not isinstance(videos, Mapping):
        raise ValueError("videos must be a mapping or None")

    clip_names = sorted(trajectories, key=str)
    if any(not isinstance(name, str) or not name for name in clip_names):
        raise ValueError("SLAM clip names must be non-empty strings")
    missing_poses = [name for name in clip_names if name not in poses]
    if missing_poses:
        raise ValueError(f"output.poses is missing clip(s) {missing_poses}")
    if timestamps is not None:
        if not isinstance(timestamps, Mapping):
            raise ValueError("timestamps must be a mapping covering every SLAM clip")
        missing = [name for name in clip_names if name not in timestamps]
        extra = [name for name in timestamps if name not in trajectories]
        if missing or extra:
            raise ValueError(
                "timestamps must cover exactly the exported clips; "
                f"missing={missing}, unknown={sorted(extra, key=str)}")

    root = local_path(out_dir, name="SLAM export out_dir")

    recorded_provenance = provenance
    if recorded_provenance is None:
        recorded_provenance = getattr(output, "provenance", None)
    normalized_provenance = _json_mapping(recorded_provenance, name="provenance")

    prepared_clips: dict[str, tuple[str, dict[str, np.ndarray], dict[str, str]]] = {}
    manifest_clips = {}
    for name in clip_names:
        trajectory = _finite_array(
            trajectories[name], name=f"output.trajectories[{name!r}]", shape_tail=(3,)
        )
        times = _times_for_clip(
            name, len(trajectory), timestamps=timestamps, sample_hz=sample_hz
        )
        arrays = {
            "time": times,
            "camera_position": trajectory,
            "tracking_confidence": _confidence_for_clip(
                name, len(trajectory), tracking_confidence
            ),
            **_pose_arrays(name, poses[name], len(trajectory)),
        }
        filename = _clip_filename(name, _payload_digest(arrays))
        relative = str(Path("clips") / filename)
        entry = {"trajectory": relative}
        manifest_clips[name] = entry
        prepared_clips[name] = (filename, arrays, entry)

    point_cloud = _point_cloud(output, point_attributes)
    if videos is not None:
        for name in clip_names:
            if name in videos:
                reference = _video_reference(
                    videos[name], root, name=f"videos[{name!r}]"
                )
                manifest_clips[name]["video"] = reference

    manifest = {
        "format": EXPORT_FORMAT,
        "version": EXPORT_VERSION,
        "timebase": "timestamp" if timestamps is not None or sample_hz is not None
        else "frame_index",
        "sample_hz": sample_hz,
        "clips": manifest_clips,
    }
    if normalized_provenance is not None:
        manifest["provenance"] = normalized_provenance
    if point_cloud is not None:
        manifest["scene"] = point_cloud[0]

    clips_dir = root / "clips"
    for filename, arrays, _entry in prepared_clips.values():
        _atomic_savez(clips_dir / filename, arrays)
    pointcloud_filename = None
    if point_cloud is not None:
        pointcloud_filename = point_cloud[0]["pointcloud"]
        _atomic_savez(root / pointcloud_filename, point_cloud[1])

    manifest_path = root / "manifest.json"
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    # Content-addressed payloads are immutable and may have been staged by another writer.
    # Pruning files not referenced by *this* manifest can race with that writer between its
    # payload and manifest commits, leaving the eventual manifest broken. Superseded payloads
    # are therefore retained; garbage collection must be an explicit target-level operation.
    return manifest_path
