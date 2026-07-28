"""Bounded reconstruction-quality signals over aligned camera trajectories."""
from __future__ import annotations

import numpy as np

from .schema import CAMERA_POSITION, CAMERA_Z_AXIS, TRACKING_CONFIDENCE


QUALITY_FEATURES = (
    "slam_tracking_confidence_mean",
    "slam_tracking_gap_fraction",
    "slam_translation_jump_fraction",
    "slam_rotation_jump_fraction",
    "slam_step_scale_drift",
    "slam_quality_score",
)

POSITION_ROLE = f"field:{CAMERA_POSITION}"
FORWARD_ROLE = f"field:{CAMERA_Z_AXIS}"
CONFIDENCE_ROLE = f"field:{TRACKING_CONFIDENCE}"


def _positive(value, name: str, *, maximum: float | None = None) -> float:
    try:
        number = float(value) if not isinstance(value, (bool, str, bytes)) else float("nan")
    except (TypeError, ValueError):
        number = float("nan")
    if not np.isfinite(number) or number <= 0 or (maximum is not None and number > maximum):
        suffix = f" at most {maximum:g}" if maximum is not None else ""
        raise ValueError(f"{name} must be a positive finite number{suffix}")
    return number


def _fraction(value, name: str) -> float:
    try:
        number = float(value) if not isinstance(value, (bool, str, bytes)) else float("nan")
    except (TypeError, ValueError):
        number = float("nan")
    if not np.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return number


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
            "SLAM quality valid_steps must contain one in-range positive integer per batch row"
        )
    return lengths.astype(np.int64, copy=False)


class SLAMQuality:
    """Measure confidence gaps, pose jumps, and a conservative scale-drift proxy.

    Translation jumps are scale-aware: each window derives a robust threshold from its median
    and median absolute deviation. ``max_translation_step`` can add a domain-specific absolute
    ceiling when the reconstruction's world-unit scale is known.
    """

    output_features = QUALITY_FEATURES

    def __init__(
        self,
        *,
        min_tracking_confidence: float = 0.5,
        translation_ratio: float = 10.0,
        translation_mad_multiplier: float = 8.0,
        max_translation_step: float | None = None,
        max_turn_degrees: float = 90.0,
    ) -> None:
        self.min_tracking_confidence = _fraction(
            min_tracking_confidence, "min_tracking_confidence"
        )
        self.translation_ratio = _positive(translation_ratio, "translation_ratio")
        self.translation_mad_multiplier = _positive(
            translation_mad_multiplier, "translation_mad_multiplier"
        )
        self.max_translation_step = (
            None if max_translation_step is None
            else _positive(max_translation_step, "max_translation_step")
        )
        self.max_turn_degrees = _positive(
            max_turn_degrees, "max_turn_degrees", maximum=180.0
        )

    def _quality_row(
        self,
        positions: np.ndarray,
        forwards: np.ndarray,
        confidence: np.ndarray,
    ) -> dict[str, float]:
        if (
            not np.isfinite(positions).all()
            or not np.isfinite(forwards).all()
            or not np.isfinite(confidence).all()
        ):
            raise ValueError(
                "SLAM quality inputs must contain only finite values in valid timesteps"
            )
        if np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("tracking confidence must be in [0, 1]")

        confidence_mean = float(confidence.mean())
        tracking_gaps = float((confidence < self.min_tracking_confidence).mean())
        if len(positions) == 1:
            quality = float(np.clip(
                0.6 * confidence_mean + 0.4 * (1.0 - tracking_gaps), 0, 1
            ))
            return {
                "slam_tracking_confidence_mean": confidence_mean,
                "slam_tracking_gap_fraction": tracking_gaps,
                "slam_translation_jump_fraction": 0.0,
                "slam_rotation_jump_fraction": 0.0,
                "slam_step_scale_drift": 0.0,
                "slam_quality_score": quality,
            }

        translation = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        median = float(np.median(translation))
        mad = float(np.median(np.abs(translation - median)))
        robust_limit = max(
            median * self.translation_ratio,
            median + self.translation_mad_multiplier * mad,
            np.finfo(np.float64).eps,
        )
        if self.max_translation_step is not None:
            robust_limit = min(robust_limit, self.max_translation_step)
        translation_jumps = float((translation > robust_limit).mean())

        norms = np.linalg.norm(forwards, axis=1)
        if np.any(norms <= np.finfo(np.float64).eps):
            raise ValueError("camera forward axes must not contain zero-length vectors")
        unit = forwards / norms[:, None]
        dots = np.sum(unit[:-1] * unit[1:], axis=1)
        turns = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
        rotation_jumps = float((turns > self.max_turn_degrees).mean())

        if len(translation) == 1:
            scale_drift = 0.0
        else:
            midpoint = max(1, len(translation) // 2)
            early = float(np.median(translation[:midpoint]))
            late = float(np.median(translation[midpoint:]))
            scale = max(float(np.median(translation)), np.finfo(np.float64).eps)
            scale_drift = abs(late - early) / scale

        penalty = (
            0.35 * tracking_gaps
            + 0.25 * translation_jumps
            + 0.25 * rotation_jumps
            + 0.15 * min(scale_drift, 1.0)
        )
        quality = float(np.clip(confidence_mean * (1.0 - penalty), 0.0, 1.0))
        return {
            "slam_tracking_confidence_mean": confidence_mean,
            "slam_tracking_gap_fraction": tracking_gaps,
            "slam_translation_jump_fraction": translation_jumps,
            "slam_rotation_jump_fraction": rotation_jumps,
            "slam_step_scale_drift": scale_drift,
            "slam_quality_score": quality,
        }

    def transform_batch(self, batch, ctx=None):
        positions = np.asarray(
            batch[_input_token(ctx, POSITION_ROLE)], dtype=np.float64
        )
        forwards = np.asarray(
            batch[_input_token(ctx, FORWARD_ROLE)], dtype=np.float64
        )
        confidence = np.asarray(
            batch[_input_token(ctx, CONFIDENCE_ROLE)], dtype=np.float64
        )
        if positions.ndim == 4:
            if positions.shape[2] != 1:
                raise ValueError("SLAM quality requires exactly one camera entity per timestep")
            positions = positions[:, :, 0, :]
        if forwards.ndim == 4:
            if forwards.shape[2] != 1:
                raise ValueError("SLAM quality requires exactly one camera entity per timestep")
            forwards = forwards[:, :, 0, :]
        if confidence.ndim == 4:
            if confidence.shape[2:] != (1, 1):
                raise ValueError("SLAM quality requires scalar confidence for one camera entity")
            confidence = confidence[:, :, 0, 0]
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("camera positions must have batch shape (B, T, 3)")
        if positions.shape[0] < 1 or positions.shape[1] < 1:
            raise ValueError("SLAM quality windows must contain at least one timestep")
        if forwards.shape != positions.shape:
            raise ValueError("camera forward axes must match camera-position shape")
        if confidence.shape != positions.shape[:2]:
            raise ValueError("tracking confidence must have batch shape (B, T)")
        batch_size, steps, _ = positions.shape
        valid_steps = _valid_steps(ctx, batch_size=batch_size, padded_steps=steps)
        rows = [
            self._quality_row(
                positions[index, :length],
                forwards[index, :length],
                confidence[index, :length],
            )
            for index, length in enumerate(valid_steps)
        ]
        return {
            name: np.asarray([row[name] for row in rows], dtype=np.float64)
            for name in QUALITY_FEATURES
        }


def slam_quality(pipeline, *, min_score: float | None = None, **parameters):
    """Add reconstruction-quality features and optionally reject low-quality windows."""
    transformed = pipeline.transform("aintropy/slam-quality", **parameters)
    if min_score is None:
        return transformed
    return transformed.filter("slam_quality_score", ">=", _fraction(min_score, "min_score"))
