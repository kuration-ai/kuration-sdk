"""Portable export of a completed ``SLAMPipeline`` reconstruction."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np

from .pointcloud import PointCloudArtifact


EXPORT_FORMAT = "kuration-slam-export"
EXPORT_VERSION = 1


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


def _clip_filename(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-") or "clip"
    digest = sha256(name.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:70]}-{digest}.npz"


def _times_for_clip(
    name: str,
    length: int,
    *,
    timestamps: Mapping[str, object] | None,
    sample_hz: float | None,
) -> np.ndarray:
    if timestamps is not None and name in timestamps:
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
    root: Path,
    point_attributes: Mapping[str, object] | None,
) -> dict | None:
    points = getattr(output, "points", None)
    if points is None:
        return None
    artifact = PointCloudArtifact(
        points,
        getattr(output, "colors", None),
        point_attributes or {},
    )
    relative = "scene_pointcloud.npz"
    artifact.save(root / relative)
    return {
        "pointcloud": relative,
        "format": "kuration-pointcloud-npz",
        "version": 1,
        "count": len(artifact),
        "attributes": sorted(artifact.attributes),
    }


def export_slam_output(
    output,
    out_dir: str | Path,
    *,
    timestamps: Mapping[str, object] | None = None,
    sample_hz: float | None = None,
    videos: Mapping[str, str | Path] | None = None,
    point_attributes: Mapping[str, object] | None = None,
    tracking_confidence: Mapping[str, object] | None = None,
) -> Path:
    """Export one completed ``SLAMPipeline`` output into a Kuration-readable manifest.

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

    root = Path(out_dir)
    clips_dir = root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest_clips = {}
    for raw_name in sorted(trajectories, key=str):
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("SLAM clip names must be non-empty strings")
        name = raw_name
        if name not in poses:
            raise ValueError(f"output.poses is missing clip {name!r}")
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
        filename = _clip_filename(name)
        relative = str(Path("clips") / filename)
        np.savez_compressed(clips_dir / filename, **arrays)
        entry = {"trajectory": relative}
        if videos is not None and name in videos:
            video = str(videos[name])
            if not video:
                raise ValueError(f"videos[{name!r}] must be a non-empty path or URI")
            entry["video"] = video
        manifest_clips[name] = entry

    manifest = {
        "format": EXPORT_FORMAT,
        "version": EXPORT_VERSION,
        "timebase": "timestamp" if timestamps is not None or sample_hz is not None
        else "frame_index",
        "sample_hz": sample_hz,
        "clips": manifest_clips,
    }
    point_cloud = _point_cloud(output, root, point_attributes)
    if point_cloud is not None:
        manifest["scene"] = point_cloud
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path
