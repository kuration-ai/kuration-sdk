"""Bounded reconstruction-quality signals over aligned camera trajectories."""
from __future__ import annotations

from numbers import Real

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


def _positive(value, name: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) \
            or value <= 0 or (maximum is not None and value > maximum):
        suffix = f" at most {maximum:g}" if maximum is not None else ""
        raise ValueError(f"{name} must be a positive finite number{suffix}")
    return float(value)


def _fraction(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) \
            or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return float(value)


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

    def transform_batch(self, batch, ctx=None):
        positions = np.asarray(batch[f"field:{CAMERA_POSITION}"], dtype=np.float64)
        forwards = np.asarray(batch[f"field:{CAMERA_Z_AXIS}"], dtype=np.float64)
        confidence = np.asarray(batch[f"field:{TRACKING_CONFIDENCE}"], dtype=np.float64)
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("camera positions must have batch shape (B, T, 3)")
        if forwards.shape != positions.shape:
            raise ValueError("camera forward axes must match camera-position shape")
        if confidence.shape != positions.shape[:2]:
            raise ValueError("tracking confidence must have batch shape (B, T)")
        if not np.isfinite(positions).all() or not np.isfinite(forwards).all() \
                or not np.isfinite(confidence).all():
            raise ValueError("SLAM quality inputs must contain only finite values")
        if np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("tracking confidence must be in [0, 1]")

        batch_size, steps, _ = positions.shape
        confidence_mean = confidence.mean(axis=1)
        tracking_gaps = (confidence < self.min_tracking_confidence).mean(axis=1)
        if steps == 1:
            zeros = np.zeros(batch_size, dtype=np.float64)
            quality = np.clip(0.6 * confidence_mean + 0.4 * (1.0 - tracking_gaps), 0, 1)
            return {
                "slam_tracking_confidence_mean": confidence_mean,
                "slam_tracking_gap_fraction": tracking_gaps,
                "slam_translation_jump_fraction": zeros.copy(),
                "slam_rotation_jump_fraction": zeros.copy(),
                "slam_step_scale_drift": zeros.copy(),
                "slam_quality_score": quality,
            }

        translation = np.linalg.norm(np.diff(positions, axis=1), axis=2)
        median = np.median(translation, axis=1)
        mad = np.median(np.abs(translation - median[:, None]), axis=1)
        robust_limit = np.maximum(
            median * self.translation_ratio,
            median + self.translation_mad_multiplier * mad,
        )
        robust_limit = np.maximum(robust_limit, np.finfo(np.float64).eps)
        if self.max_translation_step is not None:
            robust_limit = np.minimum(robust_limit, self.max_translation_step)
        translation_jumps = (translation > robust_limit[:, None]).mean(axis=1)

        norms = np.linalg.norm(forwards, axis=2)
        if np.any(norms <= np.finfo(np.float64).eps):
            raise ValueError("camera forward axes must not contain zero-length vectors")
        unit = forwards / norms[..., None]
        dots = np.sum(unit[:, :-1] * unit[:, 1:], axis=2)
        turns = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
        rotation_jumps = (turns > self.max_turn_degrees).mean(axis=1)

        if translation.shape[1] == 1:
            scale_drift = np.zeros(batch_size, dtype=np.float64)
        else:
            midpoint = max(1, translation.shape[1] // 2)
            early = np.median(translation[:, :midpoint], axis=1)
            late = np.median(translation[:, midpoint:], axis=1)
            scale = np.maximum(np.median(translation, axis=1), np.finfo(np.float64).eps)
            scale_drift = np.abs(late - early) / scale

        penalty = (
            0.35 * tracking_gaps
            + 0.25 * translation_jumps
            + 0.25 * rotation_jumps
            + 0.15 * np.minimum(scale_drift, 1.0)
        )
        quality = np.clip(confidence_mean * (1.0 - penalty), 0.0, 1.0)
        return {
            "slam_tracking_confidence_mean": confidence_mean,
            "slam_tracking_gap_fraction": tracking_gaps,
            "slam_translation_jump_fraction": translation_jumps,
            "slam_rotation_jump_fraction": rotation_jumps,
            "slam_step_scale_drift": scale_drift,
            "slam_quality_score": quality,
        }


def slam_quality(pipeline, *, min_score: float | None = None, **parameters):
    """Add reconstruction-quality features and optionally reject low-quality windows."""
    transformed = pipeline.transform("aintropy/slam-quality", **parameters)
    if min_score is None:
        return transformed
    return transformed.filter("slam_quality_score", ">=", _fraction(min_score, "min_score"))
