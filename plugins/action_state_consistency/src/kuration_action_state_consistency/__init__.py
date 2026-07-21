"""Separately packaged anomaly-detection extension for Kuration.

anomaly_checker.py is a generic, measurement-source-agnostic core (baseline validation,
the sign+magnitude consistency check, per-clip verdict aggregation) with no dependency on
kuration_pixel_shift at all. action_state.py is the "action_state" anomaly type built on
top of it -- Operator 2 of 2, checking kuration_pixel_shift's own pixel-shift
measurements against user-supplied baselines -- and is the ONLY place in this package
that ever references kuration_pixel_shift, lazily, only when actually invoked. Importing
this package never requires kuration-plugin-pixel-shift to be installed; only using the
action_state pieces does. See action_state.py's module docstring for the full mechanism.

register() registers ONLY the generic anomaly_checker.ConsistencyCheck component -- it
never references kuration_pixel_shift, so it succeeds in any install of this package,
with or without the `[action_state]` extra. That's deliberate: kuration-sdk's plugin
discovery contract requires every kuration.plugins entry point to register at least one
real component, and that check runs (and calls register()) unconditionally, regardless of
which optional extras happen to be installed -- so register() itself must never depend on
an optional extra either. ActionStateConsistencyChecker's clip-level verdict remains a
direct-API-only capability, never registered with kuration-sdk's registry, for the
cross-segment-state reason documented on the class itself.
"""
from __future__ import annotations

from .action_state import (
    ANOMALY_TYPES,
    ActionStateConsistencyChecker,
    load_baselines_from_reference,
    validate_action_state_baselines,
    validate_action_state_config,
)
from .anomaly_checker import (
    AnomalyChecker,
    CONSISTENCY_CHECK_PARAMETERS,
    ConsistencyCheck,
    consistency_check,
    validate_baselines,
    validate_config,
)

COMPONENT_NAME = "aintropy/consistency-check"


def register() -> None:
    """Idempotently register the generic ConsistencyCheck transform and operator."""
    from kuration import default_registry, register_batch_transform, register_operator

    common = {
        "version": "1",
        "package": "kuration-plugin-action-state-consistency",
        "stability": "experimental",
        "install_hint": "pip install kuration-plugin-action-state-consistency",
    }
    if COMPONENT_NAME not in default_registry.names("transform"):
        register_batch_transform(
            COMPONENT_NAME,
            ConsistencyCheck,
            description="Check a per-tick numeric field against an expected sign and "
                        "magnitude range",
            outputs=("n_consistent", "n_inconsistent"),
            inputs=("field:value",),
            parameters=CONSISTENCY_CHECK_PARAMETERS,
            **common,
        )
    if COMPONENT_NAME not in default_registry.names("operator"):
        register_operator(
            COMPONENT_NAME,
            consistency_check,
            description="Check a per-tick numeric field against an expected sign and "
                        "magnitude range",
            parameters=CONSISTENCY_CHECK_PARAMETERS,
            **common,
        )


__all__ = [
    "ANOMALY_TYPES",
    "ActionStateConsistencyChecker",
    "AnomalyChecker",
    "COMPONENT_NAME",
    "CONSISTENCY_CHECK_PARAMETERS",
    "ConsistencyCheck",
    "consistency_check",
    "load_baselines_from_reference",
    "register",
    "validate_action_state_baselines",
    "validate_action_state_config",
    "validate_baselines",
    "validate_config",
]
