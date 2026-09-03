"""Built-in gameplay operator domain and reconstruction helpers.

Importing this module registers pixel-shift, action-state consistency, and SLAM methods on the
ordinary ``kuration.Pipeline`` while grouping their expert analysis/reconstruction helpers.
Heavy runtime frameworks stay lazy; install ``kuration-sdk[gameplay]`` before executing the
frame- or model-backed paths.
"""
from __future__ import annotations

from . import action_state_consistency, collision, pixel_shift, slam
from .action_state_consistency import (
    ActionStateConsistencyChecker,
    AnomalyChecker,
    load_baselines_from_reference,
)
from .collision import (
    AsgCollisionAdapter,
    CollisionMechanics,
    DirectionalCollisionAnalyzer,
    TeaPearceDust2Adapter,
    attach_directional_collision,
    fit_ramp_models,
    load_directional_arrays,
    load_ramp_models,
    save_ramp_models,
)
from .convenience import action_state_checker, slam_reconstruct
from .pixel_shift import PixelShift, PixelShiftReport
from .slam import (
    SLAMReconstructor,
    SLAMReconstruction,
    curate_point_cloud,
    curate_slam_gameplay,
    input_slam,
    preflight_slam,
)
from .video import VideoComparison, compare_videos


def register() -> None:
    """Register every maintained gameplay component, idempotently."""
    pixel_shift.register()
    action_state_consistency.register()
    slam.register()
    collision.register()
    from ..core.stats import register_stats_enricher
    from ._stats import (
        gameplay_stats,
        plot_gameplay_stats,
        render_gameplay_examples,
    )

    register_stats_enricher(
        "gameplay",
        gameplay_stats,
        plotter=plot_gameplay_stats,
        example_renderer=render_gameplay_examples,
    )


# Direct gameplay imports are immediately usable without a separate discovery call.
register()

__all__ = [
    "ActionStateConsistencyChecker",
    "AnomalyChecker",
    "AsgCollisionAdapter",
    "CollisionMechanics",
    "DirectionalCollisionAnalyzer",
    "PixelShift",
    "PixelShiftReport",
    "SLAMReconstructor",
    "SLAMReconstruction",
    "TeaPearceDust2Adapter",
    "VideoComparison",
    "action_state_checker",
    "action_state_consistency",
    "attach_directional_collision",
    "collision",
    "compare_videos",
    "curate_point_cloud",
    "curate_slam_gameplay",
    "fit_ramp_models",
    "input_slam",
    "load_baselines_from_reference",
    "load_directional_arrays",
    "load_ramp_models",
    "pixel_shift",
    "preflight_slam",
    "register",
    "save_ramp_models",
    "slam",
    "slam_reconstruct",
]
