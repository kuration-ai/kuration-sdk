"""Validated scene point-cloud artifacts and deterministic curation methods."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import operator
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, SupportsIndex, cast

import numpy as np

from ._paths import local_path


POINTCLOUD_FORMAT = "kuration-pointcloud-npz"
POINTCLOUD_VERSION = 1
_RESERVED = {"points", "colors", "_format", "_version"}


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(cast(SupportsIndex, value))
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def _number(value: object, *, name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _points(value) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("point cloud points must be a numeric array") from error
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("point cloud points must have shape (N, 3)")
    if not len(array):
        raise ValueError("point cloud must contain at least one point")
    if not np.isfinite(array).all():
        raise ValueError("point cloud points must contain only finite values")
    return np.array(array, copy=True)


def _colors(value, length: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] != length or array.shape[1] not in (3, 4):
        raise ValueError("point cloud colors must have shape (N, 3) or (N, 4)")
    if array.dtype.kind not in "uif" or not np.isfinite(array).all():
        raise ValueError("point cloud colors must be a finite numeric array")
    return np.array(array, copy=True)


def _attributes(values: Mapping[str, object] | None, length: int) -> Mapping[str, np.ndarray]:
    out = {}
    for name, value in (values or {}).items():
        if not isinstance(name, str) or not name or name in _RESERVED:
            raise ValueError(f"invalid or reserved point-cloud attribute name {name!r}")
        array = np.asarray(value)
        if array.ndim < 1 or array.shape[0] != length:
            raise ValueError(f"point-cloud attribute {name!r} must start with length {length}")
        if array.dtype.kind not in "uifb" or not np.isfinite(array).all():
            raise ValueError(f"point-cloud attribute {name!r} must be finite and numeric")
        out[name] = np.array(array, copy=True)
    return MappingProxyType(out)


@dataclass(frozen=True)
class PointCloudArtifact:
    """A portable point cloud plus row-aligned color and quality attributes."""

    points: np.ndarray
    colors: np.ndarray | None = None
    attributes: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = _points(self.points)
        colors = _colors(self.colors, len(points))
        attributes = _attributes(self.attributes, len(points))
        points.setflags(write=False)
        if colors is not None:
            colors.setflags(write=False)
        for value in attributes.values():
            value.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "colors", colors)
        object.__setattr__(self, "attributes", attributes)

    def __len__(self) -> int:
        return len(self.points)

    def select(self, indices) -> "PointCloudArtifact":
        chosen = np.asarray(indices, dtype=np.int64)
        if chosen.ndim != 1 or np.any(chosen < 0) or np.any(chosen >= len(self)):
            raise ValueError("point-cloud selection indices must be a valid 1-D index array")
        return PointCloudArtifact(
            self.points[chosen],
            None if self.colors is None else self.colors[chosen],
            {name: value[chosen] for name, value in self.attributes.items()},
        )

    def save(self, path: str | Path) -> Path:
        target = local_path(path, name="point-cloud output")
        if target.suffix.lower() != ".npz":
            raise ValueError("point-cloud output currently requires a .npz path")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_format": np.asarray(POINTCLOUD_FORMAT),
            "_version": np.asarray(POINTCLOUD_VERSION, dtype=np.int64),
            "points": self.points,
            **dict(self.attributes),
        }
        if self.colors is not None:
            payload["colors"] = self.colors
        np.savez_compressed(target, **payload)
        return target


def _manifest_pointcloud(path: Path) -> Path:
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"SLAM manifest {str(path)!r} must be a JSON object")
    scene = document.get("scene")
    if scene is not None and not isinstance(scene, dict):
        raise ValueError(f"SLAM manifest {str(path)!r} scene must be an object")
    scene = scene or {}
    value = scene.get("pointcloud")
    if not isinstance(value, str) or not value:
        raise ValueError(f"SLAM manifest {str(path)!r} has no scene point cloud")
    format_value = scene.get("format")
    if format_value != POINTCLOUD_FORMAT:
        raise ValueError(
            f"SLAM manifest {str(path)!r} scene has format {format_value!r}; "
            f"expected {POINTCLOUD_FORMAT!r}"
        )
    version_value = scene.get("version")
    try:
        version = _integer(version_value, name="scene version")
    except ValueError:
        version = None
    if version != POINTCLOUD_VERSION:
        raise ValueError(
            f"SLAM manifest {str(path)!r} scene has unsupported version "
            f"{version_value!r}; expected {POINTCLOUD_VERSION}"
        )
    source = Path(value)
    return source if source.is_absolute() else path.parent / source


def load_point_cloud(source: str | Path | PointCloudArtifact) -> PointCloudArtifact:
    """Load an artifact from an NPZ path, SLAM manifest, or existing artifact."""
    if isinstance(source, PointCloudArtifact):
        return source
    path = local_path(source, name="point-cloud source")
    if path.is_dir():
        path = path / "manifest.json"
    if path.name == "manifest.json" or path.suffix.lower() == ".json":
        path = _manifest_pointcloud(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            points = np.asarray(archive["points"])
            colors = np.asarray(archive["colors"]) if "colors" in names else None
            format_value = np.asarray(archive["_format"]).item() \
                if "_format" in names else None
            version_value = np.asarray(archive["_version"]).item() \
                if "_version" in names else None
            attributes = {
                name: np.asarray(archive[name]) for name in sorted(names - _RESERVED)
            }
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(f"cannot read point cloud {str(path)!r}") from error
    if format_value != POINTCLOUD_FORMAT:
        raise ValueError(
            f"point cloud {str(path)!r} has format {format_value!r}; "
            f"expected {POINTCLOUD_FORMAT!r}")
    try:
        version = _integer(version_value, name="point-cloud version")
    except ValueError:
        version = None
    if version != POINTCLOUD_VERSION:
        raise ValueError(
            f"point cloud {str(path)!r} has unsupported version {version_value!r}; "
            f"expected {POINTCLOUD_VERSION}")
    return PointCloudArtifact(points, colors, attributes)


def _limited(indices: np.ndarray, maximum: int | None) -> np.ndarray:
    if maximum is None or len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)
    return indices[positions]


def curate_point_cloud(
    source: str | Path | PointCloudArtifact,
    *,
    method: str = "voxel",
    max_points: int | None = None,
    voxel_size: float | None = None,
    min_confidence: float | None = None,
    seed: int = 0,
    output: str | Path | None = None,
) -> PointCloudArtifact:
    """Filter and deterministically downsample a point cloud.

    ``voxel`` preserves spatial coverage, ``random`` produces a seeded baseline, and
    ``confidence`` retains the highest ``attributes['confidence']`` rows.
    """
    cloud = load_point_cloud(source)
    if max_points is not None:
        try:
            max_points = _integer(max_points, name="max_points")
        except ValueError as error:
            raise ValueError("max_points must be a positive integer or None") from error
        if max_points < 1:
            raise ValueError("max_points must be a positive integer or None")
    seed = _integer(seed, name="seed")
    indices = np.arange(len(cloud), dtype=np.int64)
    confidence = cloud.attributes.get("confidence")
    if min_confidence is not None:
        if confidence is None or confidence.ndim != 1:
            raise ValueError("min_confidence requires a scalar 'confidence' point attribute")
        confidence_floor = _number(min_confidence, name="min_confidence")
        indices = indices[confidence >= confidence_floor]
        if not len(indices):
            raise ValueError("point-cloud confidence filter removed every point")
    if method == "voxel":
        if voxel_size is None:
            raise ValueError("voxel curation needs a positive finite voxel_size")
        try:
            size = _number(voxel_size, name="voxel_size")
        except ValueError as error:
            raise ValueError("voxel curation needs a positive finite voxel_size") from error
        if size <= 0:
            raise ValueError("voxel curation needs a positive finite voxel_size")
        cells = np.floor(cloud.points[indices] / size).astype(np.int64)
        _, first = np.unique(cells, axis=0, return_index=True)
        indices = indices[np.sort(first)]
        indices = _limited(indices, max_points)
    elif method == "random":
        if max_points is None:
            raise ValueError("random curation needs max_points")
        if len(indices) > max_points:
            indices = np.sort(np.random.default_rng(seed).choice(
                indices, size=max_points, replace=False
            ))
    elif method == "confidence":
        if confidence is None or confidence.ndim != 1:
            raise ValueError("confidence curation needs a scalar 'confidence' point attribute")
        if max_points is None:
            raise ValueError("confidence curation needs max_points")
        order = np.argsort(-confidence[indices], kind="stable")
        indices = indices[order[:max_points]]
    else:
        raise ValueError("point-cloud method must be 'voxel', 'random', or 'confidence'")
    curated = cloud.select(indices)
    if output is not None:
        curated.save(output)
    return curated
