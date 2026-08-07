"""Built-in action-state consistency analysis for Kuration gameplay.

anomaly_checker.py is a generic, measurement-source-agnostic core (baseline validation,
the sign+magnitude consistency check, per-clip verdict aggregation) with no dependency on
the pixel-shift implementation at all. action_state.py is the "action_state" anomaly type
built on top of it, checking the SDK's pixel-shift measurements against user-supplied
baselines.

register() exposes the real ActionStateConsistencyChecker through kuration-sdk's analyzer
registry and a companion pipeline operator that explicitly drives it and attaches the result.
The stateful check is not falsely represented as a stateless transform or selection feature.
Registration does not construct the checker, so discovery remains safe without the optional
pixel-shift dependency installed.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .action_state import (
    ACTION_STATE_ANALYZER_PARAMETERS,
    ANOMALY_TYPES,
    CALIBRATION_PARAMETERS,
    ActionStateConsistencyChecker,
    action_state_consistency,
    load_baselines_from_reference,
    validate_action_state_baselines,
    validate_action_state_config,
)
from .anomaly_checker import (
    DEFAULT_CALIBRATION,
    AnomalyChecker,
    derive_baselines,
    validate_baselines,
    validate_calibration,
    validate_config,
)

ANALYZER_NAME = "aintropy/action-state-consistency"
OPERATOR_NAME = ANALYZER_NAME

ACTION_STATE_OPERATOR_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "anomaly_type": {
            "type": "string",
            "const": "action_state",
            "default": "action_state",
        },
        # baselines OR controls is required (enforced by the operator handler, which
        # raises before running anything): baselines= is external calibration; controls=
        # alone selects self-calibrated outlier detection over the named controls.
        "baselines": deepcopy(
            ACTION_STATE_ANALYZER_PARAMETERS["properties"]["baselines"]
        ),
        "calibration": deepcopy(
            ACTION_STATE_ANALYZER_PARAMETERS["properties"]["calibration"]
        ),
        "controls": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "boundary": {
            "type": "string",
            "enum": ["exclude", "error", "assume-inactive"],
            "default": "exclude",
        },
        "mode": {"type": "string", "enum": ["stream", "dataset"]},
        "inputs": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "batch_size": {"type": "integer", "minimum": 1},
        "frames": {"type": "string", "minLength": 1},
        "actions": {"type": "string", "minLength": 1},
        "color_space": {"type": "string", "minLength": 1},
        "verify": {"type": "boolean"},
        "metadata_key": {
            "type": "string",
            "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]*$",
            "default": "action_state_consistency",
        },
        "replace": {"type": "boolean", "default": False},
    },
    "additionalProperties": False,
}


def attach_action_state_consistency(
    pipeline,
    *,
    baselines: dict | None = None,
    anomaly_type: str = "action_state",
    controls=None,
    calibration: dict | None = None,
    boundary: str = "exclude",
    mode: str | None = None,
    inputs=None,
    batch_size: int | None = None,
    frames: str | None = None,
    actions: str | None = None,
    color_space: str | None = None,
    verify: bool | None = None,
    metadata_key: str = "action_state_consistency",
    replace: bool = False,
):
    """Run action-state consistency and retain the verdict on a derived pipeline.

    ``baselines=`` checks clips against externally calibrated ground truth. Omitting it
    selects **self-calibrated** mode: clips are judged against the other clips'
    leave-one-out consensus for the named ``controls`` (relative outlier detection, tuned
    by ``calibration=``), and the derived consensus is recorded in the attached result.
    One of the two must be given -- there is nothing to check otherwise.
    """
    if baselines is None and controls is None:
        raise ValueError(
            "action-state consistency needs baselines= (external calibration) or "
            "controls= (self-calibrated outlier detection over the named controls)"
        )
    if baselines is not None and calibration is not None:
        raise ValueError(
            "calibration= applies only to self-calibrated mode; drop it or drop baselines="
        )
    resolved = pipeline.analysis_mode(mode)
    dataset_options = {
        "frames": frames,
        "actions": actions,
        "color_space": color_space,
        "verify": verify,
    }
    supplied_dataset_options = {
        key: value for key, value in dataset_options.items() if value is not None
    }
    if resolved == "stream" and supplied_dataset_options:
        raise ValueError(
            "frames/actions/color_space/verify apply only to action-state dataset mode; "
            "stream mode resolves media and action inputs from the pipeline schema"
        )
    options: dict[str, Any] = {"boundary": boundary, **supplied_dataset_options}
    if controls is not None:
        options["controls"] = controls
    parameters: dict[str, Any] = {"anomaly_type": anomaly_type}
    if baselines is not None:
        parameters["baselines"] = baselines
    if calibration is not None:
        parameters["calibration"] = calibration
    return pipeline.enrich(
        ANALYZER_NAME,
        parameters,
        namespace="gameplay",
        key=metadata_key,
        replace=replace,
        mode=mode,
        inputs=inputs,
        batch_size=batch_size,
        **options,
    )


def register() -> None:
    """Idempotently register the analyzer and pipeline-preserving companion operator."""
    from kuration import default_registry, register_analyzer, register_operator
    from .._registration import needs_registration

    common: dict[str, Any] = {
        "version": "1",
        "package": "kuration-sdk",
        "stability": "experimental",
        "extra": "action-state",
        "install_hint": 'pip install "kuration-sdk[action-state]"',
        "documentation": "https://aintropy-ai.github.io/kuration-sdk/"
                         "action-state-consistency/",
    }
    if needs_registration(
        default_registry,
        "analyzer",
        ANALYZER_NAME,
        action_state_consistency,
    ):
        register_analyzer(
            ANALYZER_NAME,
            action_state_consistency,
            description="Classify clips from cross-segment pixel-shift evidence against "
                        "user-supplied action baselines",
            parameters=ACTION_STATE_ANALYZER_PARAMETERS,
            # The tick-stream contract: the same input roles, media requirements, and boundary
            # context pixel-shift's transform declares, because instances are discovered through
            # the same measurement API. Only the action stream needs a lookbehind row (to prove a
            # press did not begin before the window), so declaring the count per input avoids
            # decoding one extra video frame per segment.
            inputs=("media:frame", "field:action"),
            media_requirements={"kind": "video-frame", "layout": "HWC"},
            lookbehind={"field:action": 1},
            lookahead=3,
            capabilities={"stateful": True, "cross_segment": True,
                          "tick_stream": True, "dataset": True},
            **common,
        )
    if needs_registration(
        default_registry,
        "operator",
        OPERATOR_NAME,
        attach_action_state_consistency,
    ):
        register_operator(
            OPERATOR_NAME,
            attach_action_state_consistency,
            description="Check action-state consistency and attach the verdict to pipeline "
                        "metadata",
            parameters=ACTION_STATE_OPERATOR_PARAMETERS,
            capabilities={
                "stateful": True,
                "cross_segment": True,
                "pipeline_metadata": True,
            },
            **common,
        )


__all__ = [
    "ACTION_STATE_ANALYZER_PARAMETERS",
    "ACTION_STATE_OPERATOR_PARAMETERS",
    "ANALYZER_NAME",
    "ANOMALY_TYPES",
    "CALIBRATION_PARAMETERS",
    "DEFAULT_CALIBRATION",
    "ActionStateConsistencyChecker",
    "AnomalyChecker",
    "OPERATOR_NAME",
    "action_state_consistency",
    "attach_action_state_consistency",
    "derive_baselines",
    "load_baselines_from_reference",
    "register",
    "validate_action_state_baselines",
    "validate_action_state_config",
    "validate_baselines",
    "validate_calibration",
    "validate_config",
]
