"""Built-in gameplay operator domain and reconstruction helpers.

Importing this module registers pixel-shift and SLAM operators on the ordinary
``kuration.Pipeline`` while grouping action-state analysis and reconstruction helpers.
Heavy runtime frameworks stay lazy; install ``kuration-sdk[gameplay]`` before executing the
frame- or model-backed paths.
"""
from __future__ import annotations

from . import action_state_consistency, pixel_shift, slam
from .action_state_consistency import (
    ActionStateConsistencyChecker,
    AnomalyChecker,
    load_baselines_from_reference,
)
from .convenience import action_state_checker, slam_pipeline
from .pixel_shift import PixelShift, PixelShiftReport
from .slam import (
    SLAMReconstructor,
    SLAMReconstruction,
    curate_point_cloud,
    curate_slam_gameplay,
    input_slam,
    preflight_slam,
)


def register() -> None:
    """Register every maintained gameplay component, idempotently."""
    pixel_shift.register()
    action_state_consistency.register()
    slam.register()


# Direct gameplay imports are immediately usable without a separate discovery call.
register()

__all__ = [
    "ActionStateConsistencyChecker",
    "AnomalyChecker",
    "PixelShift",
    "PixelShiftReport",
    "SLAMReconstructor",
    "SLAMReconstruction",
    "action_state_checker",
    "action_state_consistency",
    "curate_point_cloud",
    "curate_slam_gameplay",
    "input_slam",
    "load_baselines_from_reference",
    "pixel_shift",
    "preflight_slam",
    "register",
    "slam",
    "slam_pipeline",
]
