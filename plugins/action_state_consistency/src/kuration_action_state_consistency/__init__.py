"""Separately packaged anomaly-detection extension for Kuration.

anomaly_checker.py is a generic, measurement-source-agnostic core (baseline validation,
the sign+magnitude consistency check, per-clip verdict aggregation) with no dependency on
kuration_pixel_shift at all. action_state.py is the "action_state" anomaly type built on
top of it -- Operator 2 of 2, checking kuration_pixel_shift's own pixel-shift
measurements against user-supplied baselines -- and is the ONLY place in this package
that ever references kuration_pixel_shift, lazily, only when actually invoked. Importing
this package never requires kuration-plugin-pixel-shift to be installed; only using the
action_state pieces does. See action_state.py's module docstring for the full mechanism.
"""
from __future__ import annotations

from .action_state import (
    ANOMALY_TYPES,
    ActionStateConsistencyChecker,
    load_baselines_from_reference,
    validate_action_state_baselines,
    validate_action_state_config,
)
from .anomaly_checker import AnomalyChecker, validate_baselines, validate_config

__all__ = [
    "ANOMALY_TYPES",
    "ActionStateConsistencyChecker",
    "AnomalyChecker",
    "load_baselines_from_reference",
    "validate_action_state_baselines",
    "validate_action_state_config",
    "validate_baselines",
    "validate_config",
]
