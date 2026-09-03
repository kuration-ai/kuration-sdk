"""Built-in action-state consistency analysis for Kuration gameplay.

anomaly_checker.py is a generic, measurement-source-agnostic core (baseline validation,
the sign+magnitude consistency check, per-clip verdict aggregation) with no dependency on
the pixel-shift implementation at all. action_state.py is the "action_state" anomaly type
built on top of it, checking the SDK's pixel-shift measurements against user-supplied
baselines.

register() exposes two independent surfaces:

- The ``scorer`` kind (``ActionStateAnomalyScorer``), reachable through the ordinary
  ``.score("aintropy/action-state-consistency", out=..., baselines=..., controls=...)``
  method like any other registered scorer. It reads each window's own
  ``Segment.records["aintropy/pixel-shift"]`` (populated by the pixel-shift transform) rather
  than re-deriving isolated presses from raw frames, so pixel-shift and action-state
  consistency actually share computation instead of redundantly duplicating it.
- The ``analyzer`` kind (``ActionStateConsistencyChecker``), for the dataset-mode audit only
  (``check_dataset``, over an already-written/curated dataset's shards) via
  ``attach_action_state_consistency``/``pipeline.action_state_consistency(...)``. Dataset mode
  never shared computation with a live pixel-shift transform in the first place -- it audits a
  finished dataset after the fact -- so it is unaffected by the scorer rehost and keeps its own
  registration. The tick-stream-driven analyzer path (redundantly re-deriving pixel-shift's own
  measurements from raw frames) is retired; the scorer is its replacement.

Registration does not construct anything, so discovery remains safe without the optional
pixel-shift dependency installed.
"""
from __future__ import annotations

from typing import Any

from .action_state import (
    ACTION_STATE_ANALYZER_PARAMETERS,
    ANOMALY_TYPES,
    CALIBRATION_PARAMETERS,
    PIXEL_SHIFT_RECORDS_KEY,
    ActionStateAnomalyScorer,
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
SCORER_NAME = ANALYZER_NAME
OPERATOR_NAME = ANALYZER_NAME

ACTION_STATE_SCORER_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        # baselines OR controls is required (enforced by ActionStateAnomalyScorer's
        # constructor, via resolve_controls()): baselines= is external calibration;
        # controls= alone selects self-calibrated outlier detection over the named controls.
        "baselines": ACTION_STATE_ANALYZER_PARAMETERS["properties"]["baselines"],
        "controls": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "calibration": ACTION_STATE_ANALYZER_PARAMETERS["properties"]["calibration"],
    },
    "additionalProperties": False,
}

ACTION_STATE_DATASET_OPERATOR_PARAMETERS: dict[str, Any] = {
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
        "baselines": ACTION_STATE_ANALYZER_PARAMETERS["properties"]["baselines"],
        "calibration": ACTION_STATE_ANALYZER_PARAMETERS["properties"]["calibration"],
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
    frames: str | None = None,
    actions: str | None = None,
    color_space: str | None = None,
    verify: bool | None = None,
    metadata_key: str = "action_state_consistency",
    replace: bool = False,
):
    """Audit an already-written, curated dataset's shards for action-state consistency and
    retain the per-clip verdict on a derived pipeline.

    This is the dataset-mode audit only -- checking a finished dataset after the fact, one
    shard at a time (see :meth:`ActionStateConsistencyChecker.check_dataset`). It never
    shared computation with a live pixel-shift transform, so it is unaffected by (and does
    not need) ``Segment.records``. For a live pipeline, use the per-window
    ``pipeline.score("aintropy/action-state-consistency", out=..., baselines=..., controls=...)``
    scorer instead -- it reads pixel-shift's own measurements directly rather than
    re-deriving them from raw frames.

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
    dataset_options = {
        "frames": frames,
        "actions": actions,
        "color_space": color_space,
        "verify": verify,
    }
    options: dict[str, Any] = {
        "boundary": boundary,
        **{key: value for key, value in dataset_options.items() if value is not None},
    }
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
        mode="dataset",
        **options,
    )


def register() -> None:
    """Idempotently register the dataset-mode analyzer, the per-window scorer, and the
    dataset-mode companion operator."""
    from kuration import default_registry, register_analyzer, register_operator, register_scorer
    from ...core._registration import needs_registration

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
            description="Audit an already-written dataset's shards against user-supplied "
                        "action baselines (dataset mode only; see the scorer of the same "
                        "name for live-pipeline use)",
            parameters=ACTION_STATE_ANALYZER_PARAMETERS,
            # No inputs=/media_requirements/lookbehind/lookahead: this analyzer supports
            # check_dataset only, deliberately not the tick-stream bounded-batch path (which
            # redundantly re-derived pixel-shift's own measurements from raw frames -- the
            # scorer below is its replacement).
            capabilities={"stateful": True, "cross_segment": True, "dataset": True},
            **common,
        )
    if needs_registration(
        default_registry,
        "scorer",
        SCORER_NAME,
        ActionStateAnomalyScorer,
    ):
        register_scorer(
            SCORER_NAME,
            ActionStateAnomalyScorer,
            description="Per-window count of pixel-shift instances inconsistent with the "
                        "configured (or self-calibrated) action-state baseline",
            parameters=ACTION_STATE_SCORER_PARAMETERS,
            capabilities={"stateful": True, "cross_segment": True},
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
            description="Audit an already-written dataset for action-state consistency and "
                        "attach the per-clip verdict to pipeline metadata (dataset mode only)",
            parameters=ACTION_STATE_DATASET_OPERATOR_PARAMETERS,
            capabilities={
                "stateful": True,
                "cross_segment": True,
                "pipeline_metadata": True,
                "dataset": True,
            },
            **common,
        )


__all__ = [
    "ACTION_STATE_ANALYZER_PARAMETERS",
    "ACTION_STATE_DATASET_OPERATOR_PARAMETERS",
    "ACTION_STATE_SCORER_PARAMETERS",
    "ANALYZER_NAME",
    "ANOMALY_TYPES",
    "CALIBRATION_PARAMETERS",
    "DEFAULT_CALIBRATION",
    "PIXEL_SHIFT_RECORDS_KEY",
    "ActionStateAnomalyScorer",
    "ActionStateConsistencyChecker",
    "AnomalyChecker",
    "OPERATOR_NAME",
    "SCORER_NAME",
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
