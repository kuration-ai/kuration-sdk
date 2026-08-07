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

POSITION_ROLE = f"field:{CAMERA_POSITION}"
FORWARD_ROLE = f"field:{CAMERA_Z_AXIS}"


def _vectors(batch, token: str) -> np.ndarray:
    value = np.asarray(batch[token], dtype=np.float64)
    if value.ndim == 4:
        if value.shape[2] != 1:
            raise ValueError(
                f"{token} requires exactly one camera entity per timestep, got {value.shape}")
        value = value[:, :, 0, :]
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(
            f"{token} must have batch shape (B, T, 3) or (B, T, 1, 3), got {value.shape}")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{token} windows must contain at least one timestep")
    return value


def _input_token(ctx, role: str) -> str:
    return ctx.input_token(role) if hasattr(ctx, "input_token") else role


def _valid_steps(ctx, *, batch_size: int, padded_steps: int) -> np.ndarray:
    values = getattr(ctx, "valid_steps", None) if ctx is not None else None
    if values is None:
        return np.full(batch_size, padded_steps, dtype=np.int64)
    lengths = np.asarray(values)
    if (
        lengths.shape != (batch_size,)
        or lengths.dtype.kind not in "iu"
        or np.any(lengths < 1)
        or np.any(lengths > padded_steps)
    ):
        raise ValueError(
            "SLAM motion valid_steps must contain one in-range positive integer per batch row"
        )
    return lengths.astype(np.int64, copy=False)


def _motion_row(positions: np.ndarray, forwards: np.ndarray) -> dict[str, float]:
    if not np.isfinite(positions).all() or not np.isfinite(forwards).all():
        raise ValueError("SLAM motion inputs must contain only finite values in valid timesteps")
    if len(positions) == 1:
        return {
            name: (
                float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
                if name == "slam_spatial_extent"
                else 0.0
            )
            for name in OUTPUT_FEATURES
        }

    translations = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    path_length = float(translations.sum())
    displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    efficiency = (
        displacement / path_length
        if path_length > np.finfo(np.float64).eps
        else 0.0
    )
    norms = np.linalg.norm(forwards, axis=1)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError(f"{FORWARD_ROLE} contains a zero-length direction vector")
    unit = forwards / norms[:, None]
    dots = np.sum(unit[:-1] * unit[1:], axis=1)
    turns = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    return {
        "slam_path_length": path_length,
        "slam_displacement": displacement,
        "slam_motion_efficiency": efficiency,
        "slam_mean_step": float(translations.mean()),
        "slam_max_step": float(translations.max()),
        "slam_turn_degrees": float(turns.sum()),
        "slam_mean_turn_degrees": float(turns.mean()),
        "slam_max_turn_degrees": float(turns.max()),
        "slam_spatial_extent": float(
            np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))
        ),
    }


class SLAMMotion:
    """Compute translation and view-direction motion for every segment in a batch."""

    output_features = OUTPUT_FEATURES

    def transform_batch(self, batch, ctx=None):
        positions = _vectors(batch, _input_token(ctx, POSITION_ROLE))
        forwards = _vectors(batch, _input_token(ctx, FORWARD_ROLE))
        if forwards.shape != positions.shape:
            raise ValueError("camera forward axes must match camera-position shape")
        batch_size, steps, _ = positions.shape
        valid_steps = _valid_steps(ctx, batch_size=batch_size, padded_steps=steps)
        rows = [
            _motion_row(positions[index, :length], forwards[index, :length])
            for index, length in enumerate(valid_steps)
        ]
        return {
            name: np.asarray([row[name] for row in rows], dtype=np.float64)
            for name in OUTPUT_FEATURES
        }


def slam_motion_diversity(
    pipeline,
    *,
    axis: str = "slam_path_length",
    bins: int = 8,
):
    """Add SLAM motion features and organize segments along one diversity axis."""
    return pipeline.transform("aintropy/slam-motion").bin(axis, bins=bins)
