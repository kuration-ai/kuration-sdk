"""Reload a completed SLAM reconstruction from disk and view it in 3D."""
from __future__ import annotations

import colorsys
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from ._paths import local_path
from .adapter import SLAMTrajectoryAdapter
from .export import EXPORT_FORMAT, EXPORT_VERSION
from .pointcloud import load_point_cloud
from .schema import (
    CAMERA_FRUSTUM,
    CAMERA_INTRINSICS,
    CAMERA_POSITION,
    CAMERA_X_AXIS,
    CAMERA_Y_AXIS,
    CAMERA_Z_AXIS,
)

_CV_TO_GLTF = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def load_slam_output(manifest_path: str | Path) -> SimpleNamespace:
    """Rebuild an object shaped like `SLAMOperator`'s live output
    (`.trajectories`/`.poses`/`.points`/`.colors`) from an exported SLAM
    manifest, so code written against that interface -- like
    `build_pointcloud_scene` below -- runs the same whether it's handed a
    freshly-run `SLAMOperator` or a manifest reloaded from disk."""
    manifest_path = local_path(manifest_path, name="SLAM manifest")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("SLAM manifest must be a JSON object")
    if (
        manifest.get("format") != EXPORT_FORMAT
        or manifest.get("version") != EXPORT_VERSION
    ):
        raise ValueError(
            f"SLAM manifest must declare format={EXPORT_FORMAT!r}, "
            f"version={EXPORT_VERSION}"
        )
    if not isinstance(manifest.get("clips"), dict) or not manifest["clips"]:
        raise ValueError("SLAM manifest 'clips' must be a non-empty object")
    scene = manifest.get("scene")
    if scene is not None and not isinstance(scene, dict):
        raise ValueError("SLAM manifest 'scene' must be an object or null")
    adapter = SLAMTrajectoryAdapter(manifest=manifest_path)

    trajectories: dict[str, np.ndarray] = {}
    poses: dict[str, list[SimpleNamespace]] = {}
    for name in adapter.sources():
        path = adapter._path(name)
        try:
            with np.load(path, allow_pickle=False) as archive:
                trajectory = np.asarray(archive[CAMERA_POSITION])
                frustums = np.asarray(archive[CAMERA_FRUSTUM])
                intrinsics = np.asarray(archive[CAMERA_INTRINSICS])
                x_axis, y_axis, z_axis = (
                    np.asarray(archive[CAMERA_X_AXIS]),
                    np.asarray(archive[CAMERA_Y_AXIS]),
                    np.asarray(archive[CAMERA_Z_AXIS]),
                )
        except (OSError, KeyError, ValueError) as error:
            raise ValueError(
                f"cannot read SLAM trajectory for clip {name!r}: {path}"
            ) from error
        length = trajectory.shape[0] if trajectory.ndim else 0
        expected_shapes = {
            CAMERA_POSITION: (length, 3),
            CAMERA_FRUSTUM: (length, 15),
            CAMERA_INTRINSICS: (length, 3),
            CAMERA_X_AXIS: (length, 3),
            CAMERA_Y_AXIS: (length, 3),
            CAMERA_Z_AXIS: (length, 3),
        }
        arrays = {
            CAMERA_POSITION: trajectory,
            CAMERA_FRUSTUM: frustums,
            CAMERA_INTRINSICS: intrinsics,
            CAMERA_X_AXIS: x_axis,
            CAMERA_Y_AXIS: y_axis,
            CAMERA_Z_AXIS: z_axis,
        }
        invalid = {
            field: {"actual": value.shape, "expected": expected_shapes[field]}
            for field, value in arrays.items()
            if value.shape != expected_shapes[field]
        }
        if length < 1 or invalid:
            raise ValueError(
                f"SLAM trajectory for clip {name!r} has invalid pose-array shapes: {invalid}"
            )
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError(
                f"SLAM trajectory for clip {name!r} must contain only finite values"
            )
        frustums = frustums.reshape(-1, 5, 3)
        trajectories[name] = trajectory
        poses[name] = [
            SimpleNamespace(
                frustum_points=frustums[i],
                cx=intrinsics[i, 0],
                cy=intrinsics[i, 1],
                f=intrinsics[i, 2],
                x_axis=x_axis[i],
                y_axis=y_axis[i],
                z_axis=z_axis[i],
            )
            for i in range(length)
        ]

    points = colors = None
    if scene:
        cloud = load_point_cloud(manifest_path)
        points, colors = cloud.points, cloud.colors

    return SimpleNamespace(trajectories=trajectories, poses=poses, points=points, colors=colors)


def _flip_cv_to_gltf(points: np.ndarray) -> np.ndarray:
    return points @ _CV_TO_GLTF.T


def _clip_color(index: int, count: int) -> list[int]:
    hue = index / max(count, 1)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return [int(r * 255), int(g * 255), int(b * 255), 255]


def _path_from_segments(trimesh: Any, segments: np.ndarray, color: list[int]) -> Any:
    """Build a path directly, without trimesh's optional graph backends."""
    vertices = np.asarray(segments).reshape((-1, 3))
    entities = [
        trimesh.path.entities.Line([index, index + 1])
        for index in range(0, len(vertices), 2)
    ]
    return trimesh.path.Path3D(
        entities=entities,
        vertices=vertices,
        colors=np.tile(color, (len(entities), 1)),
        process=False,
    )


def build_pointcloud_scene(output: Any, *, max_display_points: int = int(3e6)) -> Any:
    """Build a `trimesh.Scene` of `output`'s point cloud plus each clip's
    camera trajectory and per-frame frustums, flipped into glTF (Y-up) axes
    -- one call away from `trimesh.viewer.notebook.scene_to_html(...)` for a
    notebook viewer. `output` needs `.trajectories`/`.poses`/`.points`/
    `.colors`, the interface `SLAMOperator` and `load_slam_output` share.

    Needs `trimesh`, which is not part of the `slam` extra:
    `pip install trimesh`.
    """
    if (
        isinstance(max_display_points, bool)
        or not isinstance(max_display_points, (int, np.integer))
        or max_display_points < 1
    ):
        raise ValueError("max_display_points must be a positive integer")
    max_display_points = int(max_display_points)
    try:
        import trimesh
    except ImportError as error:
        raise ImportError(
            "build_pointcloud_scene needs trimesh: pip install trimesh"
        ) from error

    combined = trimesh.Scene()

    if output.points is not None and output.points.shape[0]:
        points_g = _flip_cv_to_gltf(output.points)
        # Colors are optional on both interfaces (`SLAMOperator` with `output_pointcloud` off
        # per-point colors, and a manifest whose point cloud carries no `colors` array), so an
        # uncolored cloud is displayed in neutral grey rather than crashing.
        colors = getattr(output, "colors", None)
        if colors is None:
            rgb = np.full((points_g.shape[0], 3), 170, dtype=np.uint8)
        else:
            rgb = np.asarray(colors)[:, :3].astype(np.uint8)
        if len(points_g) > max_display_points:
            selected = np.linspace(
                0,
                len(points_g) - 1,
                max_display_points,
                dtype=np.int64,
            )
            points_g = points_g[selected]
            rgb = rgb[selected]
        alpha = np.full((rgb.shape[0], 1), 255, dtype=np.uint8)
        rgba = np.concatenate([rgb, alpha], axis=1)
        cloud = trimesh.PointCloud(vertices=points_g, colors=rgba)
        combined.add_geometry(cloud, node_name="points")

    clip_names = list(output.trajectories.keys())
    for clip_index, name in enumerate(clip_names):
        color = _clip_color(clip_index, len(clip_names))

        centers_g = _flip_cv_to_gltf(output.trajectories[name])
        if len(centers_g) > 1:
            segments = centers_g[np.array([[i, i + 1] for i in range(len(centers_g) - 1)])]
            trajectory = _path_from_segments(trimesh, segments, color)
            combined.add_geometry(trajectory, node_name=f"{name}_trajectory")

        for pose_index, pose in enumerate(output.poses[name]):
            apex, top_left, top_right, bottom_left, bottom_right = _flip_cv_to_gltf(
                pose.frustum_points
            )
            segments = np.array([
                [apex, top_left], [apex, top_right], [apex, bottom_left], [apex, bottom_right],
                [top_left, top_right], [top_right, bottom_right],
                [bottom_right, bottom_left], [bottom_left, top_left],
            ])
            frustum = _path_from_segments(trimesh, segments, color)
            combined.add_geometry(frustum, node_name=f"{name}_frustum_{pose_index}")

    return combined
