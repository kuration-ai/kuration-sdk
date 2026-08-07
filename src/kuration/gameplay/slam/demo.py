"""Small deterministic reconstruction used by the offline golden-path example."""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np

from .export import export_slam_output


def _poses(positions: np.ndarray) -> list[SimpleNamespace]:
    rows = []
    differences = np.diff(positions, axis=0, prepend=positions[:1])
    for index, (position, difference) in enumerate(zip(positions, differences, strict=True)):
        if not np.linalg.norm(difference):
            difference = np.array([1.0, 0.0, 0.0])
        forward = difference / np.linalg.norm(difference)
        up = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(forward, up))) > 0.95:
            up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(up, forward)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(forward, x_axis)
        frustum = np.stack([
            position,
            position + forward - x_axis - y_axis,
            position + forward + x_axis - y_axis,
            position + forward - x_axis + y_axis,
            position + forward + x_axis + y_axis,
        ])
        rows.append(SimpleNamespace(
            frustum_points=frustum,
            cx=320.0,
            cy=240.0,
            f=500.0,
            x_axis=x_axis,
            y_axis=y_axis,
            z_axis=forward,
            frame_index=index,
        ))
    return rows


def write_demo_reconstruction(
    out_dir: str | Path,
    *,
    frames_per_clip: int = 96,
    sample_hz: float = 12.0,
) -> Path:
    """Generate a tiny, model-free SLAM export suitable for documentation and smoke tests."""
    if not isinstance(frames_per_clip, int) or isinstance(frames_per_clip, bool) \
            or frames_per_clip < 16:
        raise ValueError("frames_per_clip must be an integer of at least 16")
    time = np.arange(frames_per_clip, dtype=np.float64) / sample_hz
    trajectories = {
        "demo-orbit": np.column_stack((np.cos(time), np.sin(time), 0.05 * time)),
        "demo-strafe": np.column_stack((0.2 * time, 0.3 * np.sin(time * 2), 0.1 * time)),
    }
    confidence = {
        "demo-orbit": np.full(frames_per_clip, 0.98),
        "demo-strafe": np.full(frames_per_clip, 0.94),
    }
    # One low-confidence region makes the quality gate visible without invalidating the export.
    confidence["demo-strafe"][frames_per_clip // 3:frames_per_clip // 3 + 4] = 0.2
    points = np.column_stack((
        np.cos(np.linspace(0, 8 * np.pi, 2_000)),
        np.sin(np.linspace(0, 8 * np.pi, 2_000)),
        np.linspace(-1, 1, 2_000),
    ))
    colors = np.clip((points - points.min(axis=0)) / np.ptp(points, axis=0) * 255, 0, 255)
    output = SimpleNamespace(
        trajectories=trajectories,
        poses={name: _poses(value) for name, value in trajectories.items()},
        points=points,
        colors=colors.astype(np.uint8),
    )
    return export_slam_output(
        output,
        out_dir,
        sample_hz=sample_hz,
        tracking_confidence=confidence,
        point_attributes={"confidence": np.linspace(0.5, 1.0, len(points))},
    )
