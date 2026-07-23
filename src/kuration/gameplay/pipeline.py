"""Gameplay-specific fluent pipeline facade.

The generic :class:`kuration.core.pipeline.Pipeline` owns execution and registry semantics.
This subclass adds only typed gameplay spellings for maintained compositional operators.
"""
from __future__ import annotations

from typing import Any, cast

from ..core.pipeline import Pipeline
from ..core.pipeline import input as core_input


class GameplayPipeline(Pipeline):
    """A core pipeline with first-class gameplay operator methods."""

    def pixel_shift(
        self,
        *,
        controls=None,
        boundary: str = "exclude",
    ) -> "GameplayPipeline":
        """Measure isolated key/mouse pixel motion."""
        params: dict[str, Any] = {"boundary": boundary}
        if controls is not None:
            params["controls"] = controls
        return cast(GameplayPipeline, self._invoke_operator("aintropy/pixel-shift", params))

    def slam_motion(
        self,
        *,
        axis: str = "slam_path_length",
        bins: int = 8,
    ) -> "GameplayPipeline":
        """Add SLAM motion features and organize one diversity axis."""
        return cast(
            GameplayPipeline,
            self._invoke_operator(
                "aintropy/slam-motion-diversity",
                {"axis": axis, "bins": bins},
            ),
        )

    def slam_quality(
        self,
        *,
        min_score: float | None = None,
        min_tracking_confidence: float = 0.5,
        translation_ratio: float = 10.0,
        translation_mad_multiplier: float = 8.0,
        max_translation_step: float | None = None,
        max_turn_degrees: float = 90.0,
    ) -> "GameplayPipeline":
        """Add SLAM quality features and optionally reject weak windows."""
        params: dict[str, Any] = {
            "min_tracking_confidence": min_tracking_confidence,
            "translation_ratio": translation_ratio,
            "translation_mad_multiplier": translation_mad_multiplier,
            "max_turn_degrees": max_turn_degrees,
        }
        if min_score is not None:
            params["min_score"] = min_score
        if max_translation_step is not None:
            params["max_translation_step"] = max_translation_step
        return cast(GameplayPipeline, self._invoke_operator("aintropy/slam-quality", params))


def as_gameplay(pipeline: Pipeline) -> GameplayPipeline:
    """Return an immutable gameplay view over an existing core pipeline."""
    if not isinstance(pipeline, Pipeline):
        raise TypeError("as_gameplay() needs a kuration Pipeline")
    if isinstance(pipeline, GameplayPipeline):
        return pipeline
    result = object.__new__(GameplayPipeline)
    result.__dict__.update(pipeline._clone().__dict__)
    return result


def input(*args, **kwargs) -> GameplayPipeline:
    """Open gameplay data using the core input adapter machinery."""
    return as_gameplay(core_input(*args, **kwargs))


__all__ = ["GameplayPipeline", "as_gameplay", "input"]
