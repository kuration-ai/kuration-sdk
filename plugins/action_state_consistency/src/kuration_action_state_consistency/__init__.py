"""Separately packaged anomaly-detection extension for Kuration.

anomaly_checker.py is a generic, measurement-source-agnostic core (baseline validation,
the sign+magnitude consistency check, per-clip verdict aggregation) with no dependency on
kuration_pixel_shift at all. action_state.py is the "action_state" anomaly type built on
top of it -- Operator 2 of 2, checking kuration_pixel_shift's own pixel-shift
measurements against user-supplied baselines -- and is the ONLY place in this package
that ever references kuration_pixel_shift, lazily, only when actually invoked. Importing
this package never requires kuration-plugin-pixel-shift to be installed; only using the
action_state pieces does. See action_state.py's module docstring for the full mechanism.

register() exposes the real ActionStateConsistencyChecker through kuration-sdk's analyzer
registry. Analyzer factories are discoverable direct APIs for user-owned, cross-segment
state; they are not falsely represented as stateless pipeline transforms. Registration
does not construct the checker, so discovery remains safe without the optional
pixel-shift dependency installed.
"""
from __future__ import annotations

from .action_state import (
    ACTION_STATE_ANALYZER_PARAMETERS,
    ANOMALY_TYPES,
    ActionStateConsistencyChecker,
    action_state_consistency,
    load_baselines_from_reference,
    validate_action_state_baselines,
    validate_action_state_config,
)
from .anomaly_checker import (
    AnomalyChecker,
    validate_baselines,
    validate_config,
)

ANALYZER_NAME = "aintropy/action-state-consistency"


def register() -> None:
    """Idempotently register the real stateful action-state analyzer."""
    from kuration import default_registry, register_analyzer

    common = {
        "version": "1",
        "package": "kuration-plugin-action-state-consistency",
        "stability": "experimental",
        "install_hint": 'pip install "kuration-plugin-action-state-consistency[action_state]"',
        "documentation": "https://github.com/aintropy-ai/kuration-sdk/tree/main/"
                         "plugins/action_state_consistency#usage",
        "capabilities": {"stateful": True, "cross_segment": True},
    }
    if ANALYZER_NAME not in default_registry.names("analyzer"):
        register_analyzer(
            ANALYZER_NAME,
            action_state_consistency,
            description="Classify clips from cross-segment pixel-shift evidence against "
                        "user-supplied action baselines",
            parameters=ACTION_STATE_ANALYZER_PARAMETERS,
            **common,
        )


__all__ = [
    "ACTION_STATE_ANALYZER_PARAMETERS",
    "ANALYZER_NAME",
    "ANOMALY_TYPES",
    "ActionStateConsistencyChecker",
    "AnomalyChecker",
    "action_state_consistency",
    "load_baselines_from_reference",
    "register",
    "validate_action_state_baselines",
    "validate_action_state_config",
    "validate_baselines",
    "validate_config",
]
