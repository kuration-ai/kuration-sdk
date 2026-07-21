"""The "action_state" anomaly-type adapter: the ONLY module in this package that ever
references kuration_pixel_shift, and even then only lazily, inside the functions/methods
that actually need it -- never at module import time. Merely importing this module (or
the package as a whole) never requires kuration-plugin-pixel-shift to be installed; only
actually constructing an ActionStateConsistencyChecker, calling its add_segment(), or
calling load_baselines_from_reference() does.

Builds on anomaly_checker.AnomalyChecker for everything generic (baseline validation, the
consistency check, clip aggregation, reporting) -- this module adds only what's specific
to checking pixel_shift's own measurements: validating baseline control names against
kuration_pixel_shift.ALL_CONTROLS, and discovering instances via kuration_pixel_shift's
own public iter_instances/validate_segment_inputs APIs rather than re-deriving isolated-press
detection, boundary handling, or optical-flow logic.
"""
from __future__ import annotations

from collections.abc import Sequence

from .anomaly_checker import AnomalyChecker, validate_baselines, validate_config

ANOMALY_TYPES = ("action_state",)


def _require_pixel_shift():
    try:
        import kuration_pixel_shift
    except ImportError as e:
        raise ImportError(
            "action_state anomaly detection requires kuration-plugin-pixel-shift. "
            "Install it with `pip install \"kuration-plugin-action-state-consistency[action_state]\"`."
        ) from e
    return kuration_pixel_shift


def validate_action_state_baselines(baselines) -> dict:
    """Same shape validation as anomaly_checker.validate_baselines, plus checking every
    control name against kuration_pixel_shift.ALL_CONTROLS -- this is the one place that
    ties baseline validation specifically to pixel_shift's DIAMOND control vocabulary."""
    validated = validate_baselines(baselines)
    all_controls = _require_pixel_shift().ALL_CONTROLS
    unknown = sorted(set(validated) - set(all_controls))
    if unknown:
        raise ValueError(f"unknown control(s) {unknown} in baselines; expected {all_controls}")
    return validated


def validate_action_state_config(config) -> dict:
    """Validate the full action_state operator config:
    ``{"anomaly_type": "action_state", "baselines": {...}}``."""
    validated = validate_config(config)
    if validated["anomaly_type"] not in ANOMALY_TYPES:
        raise ValueError(
            f"config['anomaly_type'] must be one of {ANOMALY_TYPES}, got {validated['anomaly_type']!r}"
        )
    validated["baselines"] = validate_action_state_baselines(config["baselines"])
    return validated


def load_baselines_from_reference(reference: dict, *, controls: Sequence[str] | None = None) -> dict:
    """Build a baselines dict from a reference JSON matching
    scripts/build_action_consistency_reference.py's ``cells`` shape
    (``{control_lower: {cell_key: {"reliable", "floor", "ceiling", "expected_sign", ...}}}``).

    A cell marked unreliable (not enough reference samples to trust) is skipped entirely --
    it becomes "no baseline configured for this cell" rather than being checked against an
    untrustworthy floor/ceiling. ``controls``, if given, restricts which controls are
    pulled from the reference (as pixel_shift-style names, e.g. "W", "MOUSE_X_POS").
    """
    all_controls = set(_require_pixel_shift().ALL_CONTROLS)
    wanted = None if controls is None else set(controls)
    baselines: dict = {}
    for control_lower, per_cell in reference.get("cells", {}).items():
        control = control_lower.upper()
        if control not in all_controls:
            continue
        if wanted is not None and control not in wanted:
            continue
        kept = {}
        for cell_key, cell in per_cell.items():
            if not cell.get("reliable"):
                continue
            kept[cell_key] = {
                "expected_sign": cell["expected_sign"],
                "min": cell["floor"],
                "max": cell["ceiling"],
            }
        if kept:
            baselines[control] = kept
    return baselines


class ActionStateConsistencyChecker(AnomalyChecker):
    """AnomalyChecker specialized for pixel_shift measurements. See this module's
    docstring for why kuration_pixel_shift is only ever imported lazily, inside
    add_segment() -- constructing this checker or calling clip_verdict()/summary()/
    print_report() never touches it; only add_segment() (and load_baselines_from_reference(),
    a free function) do."""

    def __init__(self, config):
        super().__init__(validate_action_state_config(config))

    def add_segment(
        self,
        frames,
        actions,
        segment_id,
        controls: Sequence[str] | None = None,
        color_space: str = "BGR",
        *,
        lookbehind_actions=None,
        lookahead_actions=None,
        lookahead_frames=None,
        boundary: str = "exclude",
    ) -> None:
        """Check every instance found in this one segment against the configured
        baselines, and record the result under its clip (segment_id's source_id).
        ``controls`` defaults to every control with a configured baseline; passing an
        explicit subset must still be a subset of the configured baselines -- there is
        nothing to check a control against without one."""
        from kuration_pixel_shift import (
            iter_instances,
            validate_boundary,
            validate_color_space,
            validate_segment_inputs,
        )

        controls = tuple(self.baselines) if controls is None else tuple(controls)
        unknown = sorted(set(controls) - set(self.baselines))
        if unknown:
            raise ValueError(
                f"no baseline configured for control(s) {unknown}; "
                f"configured: {sorted(self.baselines)}"
            )
        boundary = validate_boundary(boundary)
        color_space = validate_color_space(color_space)
        frames, actions, before, after_actions, after_frames = validate_segment_inputs(
            frames,
            actions,
            lookbehind_actions=lookbehind_actions,
            lookahead_actions=lookahead_actions,
            lookahead_frames=lookahead_frames,
        )
        source_id, start_idx, _end_idx = self.resolve_segment(segment_id, expected_length=len(actions))

        for control in controls:
            for cell_key, local_start, shift in iter_instances(
                frames,
                actions,
                control,
                color_space,
                lookbehind_actions=before,
                lookahead_actions=after_actions,
                lookahead_frames=after_frames,
                boundary=boundary,
            ):
                self.record_instance(source_id, control, cell_key, start_idx + local_start, shift)
