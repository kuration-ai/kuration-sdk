"""Reload a completed SLAM reconstruction from disk and view it in 3D."""
from __future__ import annotations

import colorsys
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

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
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent

    trajectories: dict[str, np.ndarray] = {}
    poses: dict[str, list[SimpleNamespace]] = {}
    for name, entry in manifest["clips"].items():
        with np.load(root / entry["trajectory"]) as archive:
            trajectories[name] = archive[CAMERA_POSITION]
            frustums = archive[CAMERA_FRUSTUM].reshape(-1, 5, 3)
            intrinsics = archive[CAMERA_INTRINSICS]
            x_axis, y_axis, z_axis = (
                archive[CAMERA_X_AXIS], archive[CAMERA_Y_AXIS], archive[CAMERA_Z_AXIS]
            )
            poses[name] = [
                SimpleNamespace(
                    frustum_points=frustums[i],
                    cx=intrinsics[i, 0], cy=intrinsics[i, 1], f=intrinsics[i, 2],
                    x_axis=x_axis[i], y_axis=y_axis[i], z_axis=z_axis[i],
                )
                for i in range(len(trajectories[name]))
            ]

    points = colors = None
    if manifest.get("scene"):
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
    try:
        import trimesh
    except ImportError as error:
        raise ImportError(
            "build_pointcloud_scene needs trimesh: pip install trimesh"
        ) from error

    combined = trimesh.Scene()

    if output.points is not None and output.points.shape[0]:
        points_g = _flip_cv_to_gltf(output.points)
        rgb = output.colors[:, :3].astype(np.uint8)
        alpha = np.full((rgb.shape[0], 1), 255, dtype=np.uint8)
        rgba = np.concatenate([rgb, alpha], axis=1)
        cloud = trimesh.PointCloud(vertices=points_g, colors=rgba)
        if cloud.vertices.shape[0] > max_display_points:
            skip = max(1, cloud.vertices.shape[0] // max_display_points)
            cloud.vertices = cloud.vertices[::skip].copy()
            cloud.colors = cloud.colors[::skip].copy()
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
