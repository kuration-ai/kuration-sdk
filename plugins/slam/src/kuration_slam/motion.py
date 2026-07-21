"""Bounded motion features over aligned camera trajectories."""
from __future__ import annotations

import numpy as np

from .schema import CAMERA_POSITION, CAMERA_Z_AXIS


OUTPUT_FEATURES = (
    "slam_path_length",
    "slam_displacement",
    "slam_motion_efficiency",
    "slam_mean_step",
    "slam_max_step",
    "slam_turn_degrees",
    "slam_mean_turn_degrees",
    "slam_max_turn_degrees",
    "slam_spatial_extent",
)


def _vectors(batch, token: str) -> np.ndarray:
    value = np.asarray(batch[token], dtype=np.float64)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{token} must have batch shape (B, T, 3), got {value.shape}")
    if value.shape[1] < 1:
        raise ValueError(f"{token} windows must contain at least one timestep")
    if not np.isfinite(value).all():
        raise ValueError(f"{token} must contain only finite values")
    return value


class SLAMMotion:
    """Compute translation and view-direction motion for every segment in a batch."""

    output_features = OUTPUT_FEATURES

    def transform_batch(self, batch, ctx=None):
        positions = _vectors(batch, f"field:{CAMERA_POSITION}")
        forwards = _vectors(batch, f"field:{CAMERA_Z_AXIS}")
        batch_size, steps, _ = positions.shape
        if steps == 1:
            zeros = np.zeros(batch_size, dtype=np.float64)
            extent = np.linalg.norm(positions.max(axis=1) - positions.min(axis=1), axis=1)
            return {name: (extent if name == "slam_spatial_extent" else zeros.copy())
                    for name in OUTPUT_FEATURES}

        translations = np.linalg.norm(np.diff(positions, axis=1), axis=2)
        path_length = translations.sum(axis=1)
        displacement = np.linalg.norm(positions[:, -1] - positions[:, 0], axis=1)
        efficiency = np.divide(
            displacement,
            path_length,
            out=np.zeros_like(displacement),
            where=path_length > np.finfo(np.float64).eps,
        )
        norms = np.linalg.norm(forwards, axis=2)
        if np.any(norms <= np.finfo(np.float64).eps):
            raise ValueError(f"field:{CAMERA_Z_AXIS} contains a zero-length direction vector")
        unit = forwards / norms[..., None]
        dots = np.sum(unit[:, :-1] * unit[:, 1:], axis=2)
        turns = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
        extent = np.linalg.norm(positions.max(axis=1) - positions.min(axis=1), axis=1)
        return {
            "slam_path_length": path_length,
            "slam_displacement": displacement,
            "slam_motion_efficiency": efficiency,
            "slam_mean_step": translations.mean(axis=1),
            "slam_max_step": translations.max(axis=1),
            "slam_turn_degrees": turns.sum(axis=1),
            "slam_mean_turn_degrees": turns.mean(axis=1),
            "slam_max_turn_degrees": turns.max(axis=1),
            "slam_spatial_extent": extent,
        }


def slam_motion_diversity(
    pipeline,
    *,
    axis: str = "slam_path_length",
    bins: int = 8,
):
    """Add SLAM motion features and organize segments along one diversity axis."""
    return pipeline.transform("aintropy/slam-motion").bin(axis, bins=bins)
