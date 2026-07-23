"""Unified facade for Kuration's built-in gameplay capabilities.

The implementations remain separated by responsibility, while this module gives callers one
place to discover pixel-shift measurement, action-state analysis, and SLAM reconstruction.
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
    SLAMPipeline,
    SLAMReconstruction,
    curate_point_cloud,
    curate_slam_gameplay,
    input_slam,
    preflight_slam,
)

__all__ = [
    "ActionStateConsistencyChecker",
    "AnomalyChecker",
    "PixelShift",
    "PixelShiftReport",
    "SLAMPipeline",
    "SLAMReconstruction",
    "action_state_checker",
    "action_state_consistency",
    "curate_point_cloud",
    "curate_slam_gameplay",
    "input_slam",
    "load_baselines_from_reference",
    "pixel_shift",
    "preflight_slam",
    "slam",
    "slam_pipeline",
]
