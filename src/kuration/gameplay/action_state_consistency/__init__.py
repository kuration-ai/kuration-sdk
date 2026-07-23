"""Built-in action-state consistency analysis for Kuration gameplay.

anomaly_checker.py is a generic, measurement-source-agnostic core (baseline validation,
the sign+magnitude consistency check, per-clip verdict aggregation) with no dependency on
the pixel-shift implementation at all. action_state.py is the "action_state" anomaly type
built on top of it, checking the SDK's pixel-shift measurements against user-supplied
baselines.

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
        "package": "kuration-sdk",
        "stability": "experimental",
        "extra": "action-state",
        "install_hint": 'pip install "kuration-sdk[action-state]"',
        "documentation": "https://aintropy-ai.github.io/kuration-sdk/"
                         "action-state-consistency/",
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
