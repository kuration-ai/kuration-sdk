"""The ``action_state`` analyzer built on gameplay pixel-shift measurements.

Builds on anomaly_checker.AnomalyChecker for everything generic (baseline validation, the
consistency check, clip aggregation, reporting) -- this module adds only what's specific
to checking pixel_shift's own measurements: validating baseline control names against
``kuration.gameplay.pixel_shift.ALL_CONTROLS``, and discovering instances via pixel-shift's public
``iter_instances``/``validate_segment_inputs`` APIs rather than re-deriving isolated-press
detection, boundary handling, or optical-flow logic.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .anomaly_checker import AnomalyChecker, validate_baselines, validate_config

ANOMALY_TYPES = ("action_state",)
CALIBRATION_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "min_clips": {"type": "integer", "minimum": 2},
        "min_samples": {"type": "integer", "minimum": 2},
        "sign_consensus": {"type": "number", "exclusiveMinimum": 0.5, "maximum": 1},
        "tolerance": {"type": "number", "minimum": 1},
    },
    "additionalProperties": False,
}
ACTION_STATE_ANALYZER_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "anomaly_type": {"type": "string", "const": "action_state"},
        # Omitted entirely -> self-calibrated mode: clips are judged against the other
        # clips' leave-one-out consensus instead of an external reference.
        "baselines": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "expected_sign": {"type": "integer", "enum": [1, -1]},
                        "min": {"type": "number", "minimum": 0},
                        "max": {"type": "number", "minimum": 0},
                    },
                    "required": ["expected_sign", "min", "max"],
                    "additionalProperties": False,
                },
            },
        },
        "calibration": CALIBRATION_PARAMETERS,
    },
    "required": ["anomaly_type"],
    "additionalProperties": False,
}


def _require_pixel_shift():
    from .. import pixel_shift

    return pixel_shift


def _single_entity_actions(value):
    """Drop the schema-driven entity axis from one ``(T,E,D)`` action window.

    Core batch inputs are entity-shaped; this single-agent anomaly type consumes ``E == 1``.
    ``None`` (a source-edge boundary row with no context) passes straight through.
    """
    if value is None:
        return None
    import numpy as np

    actions = np.asarray(value)
    if actions.ndim == 3:
        if actions.shape[1] != 1:
            raise ValueError(
                "action-state consistency requires exactly one action entity per timestep")
        actions = actions[:, 0, :]
    return actions


def validate_action_state_baselines(baselines) -> dict:
    """Same shape validation as anomaly_checker.validate_baselines, plus checking every
    control name against ``kuration.gameplay.pixel_shift.ALL_CONTROLS`` -- this is the one place that
    ties baseline validation specifically to pixel_shift's DIAMOND control vocabulary."""
    validated = validate_baselines(baselines)
    all_controls = _require_pixel_shift().ALL_CONTROLS
    unknown = sorted(set(validated) - set(all_controls))
    if unknown:
        raise ValueError(f"unknown control(s) {unknown} in baselines; expected {all_controls}")
    return validated


def validate_action_state_config(config) -> dict:
    """Validate the full action_state operator config:
    ``{"anomaly_type": "action_state", "baselines": {...} | None}`` (plus optional
    ``calibration`` knobs when baselines is None -- see anomaly_checker.validate_config)."""
    validated = validate_config(config)
    if validated["anomaly_type"] not in ANOMALY_TYPES:
        raise ValueError(
            f"config['anomaly_type'] must be one of {ANOMALY_TYPES}, got {validated['anomaly_type']!r}"
        )
    if validated["baselines"] is not None:
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
    control_names = tuple(_require_pixel_shift().ALL_CONTROLS)
    all_controls = set(control_names)
    if controls is None:
        wanted = None
    else:
        if (
            isinstance(controls, (str, bytes))
            or not isinstance(controls, Sequence)
            or not controls
            or any(not isinstance(control, str) for control in controls)
        ):
            raise ValueError("controls must be a non-empty sequence of control names")
        unknown = sorted(set(controls) - all_controls)
        if unknown:
            raise ValueError(
                f"unknown action-state reference control(s) {unknown}; "
                f"expected {list(control_names)}"
            )
        if len(controls) != len(set(controls)):
            raise ValueError("controls must not contain duplicates")
        wanted = set(controls)
    if not isinstance(reference, dict):
        raise ValueError("reference must be a dict containing a non-empty 'cells' object")
    cells = reference.get("cells")
    if not isinstance(cells, dict) or not cells:
        raise ValueError("reference must contain a non-empty 'cells' object")
    baselines: dict = {}
    seen_controls = set()
    for control_lower, per_cell in cells.items():
        if not isinstance(control_lower, str) or not control_lower:
            raise ValueError("reference control keys must be non-empty strings")
        control = control_lower.upper()
        if control not in all_controls:
            raise ValueError(
                f"unknown control {control_lower!r} in reference; expected "
                f"{list(control_names)}"
            )
        if control in seen_controls:
            raise ValueError(
                f"reference contains duplicate case-insensitive control {control!r}"
            )
        seen_controls.add(control)
        if not isinstance(per_cell, dict) or not per_cell:
            raise ValueError(
                f"reference cells for control {control!r} must be a non-empty dict"
            )
        kept = {}
        for cell_key, cell in per_cell.items():
            if not isinstance(cell_key, str) or not cell_key:
                raise ValueError(
                    f"reference cell keys for control {control!r} must be non-empty strings"
                )
            if not isinstance(cell, dict):
                raise ValueError(
                    f"reference cell {control!r}/{cell_key!r} must be a dict"
                )
            reliable = cell.get("reliable")
            if not isinstance(reliable, bool):
                raise ValueError(
                    f"reference cell {control!r}/{cell_key!r} must declare boolean 'reliable'"
                )
            if not reliable:
                continue
            missing = {"expected_sign", "floor", "ceiling"} - set(cell)
            if missing:
                raise ValueError(
                    f"reliable reference cell {control!r}/{cell_key!r} "
                    f"missing {sorted(missing)}"
                )
            kept[cell_key] = {
                "expected_sign": cell["expected_sign"],
                "min": cell["floor"],
                "max": cell["ceiling"],
            }
        if kept:
            validated_cells = validate_baselines({control: kept})[control]
            if wanted is None or control in wanted:
                baselines[control] = validated_cells
    if not baselines:
        scope = "the requested controls" if wanted is not None else "the reference"
        raise ValueError(f"{scope} contain no reliable action-state baseline cells")
    return validate_action_state_baselines(baselines)


class ActionStateConsistencyChecker(AnomalyChecker):
    """AnomalyChecker specialized for the SDK's pixel-shift measurements."""

    def __init__(self, config):
        super().__init__(validate_action_state_config(config))

    def resolve_controls(self, controls: Sequence[str] | None) -> tuple[str, ...]:
        """Resolve/validate the control set to check, against configured baselines (or
        pixel-shift's vocabulary in self-calibrated mode where there is nothing configured
        to default to). ``controls`` defaults to every control with a configured baseline;
        an explicit subset must still be a subset of the configured baselines -- there is
        nothing to check a control against without one. In self-calibrated mode
        ``controls`` is required, since there is no configured set to default to."""
        from ..pixel_shift import ALL_CONTROLS

        if self.baselines is None:
            if controls is None:
                raise ValueError(
                    "self-calibrated action-state consistency needs controls=; without "
                    "baselines there is no configured set to default to"
                )
            controls = tuple(controls)
            unknown = sorted(set(controls) - set(ALL_CONTROLS))
            if unknown:
                raise ValueError(
                    f"unknown control(s) {unknown}; expected {list(ALL_CONTROLS)}"
                )
            return controls
        controls = tuple(self.baselines) if controls is None else tuple(controls)
        unknown = sorted(set(controls) - set(self.baselines))
        if unknown:
            raise ValueError(
                f"no baseline configured for control(s) {unknown}; "
                f"configured: {sorted(self.baselines)}"
            )
        return controls

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
        baselines, and record the result under its clip (segment_id's source_id). See
        :meth:`resolve_controls` for how ``controls`` defaults/validates."""
        from ..pixel_shift import (
            iter_instances,
            validate_boundary,
            validate_color_space,
            validate_segment_inputs,
        )

        controls = self.resolve_controls(controls)
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
                validated=True,
            ):
                self.record_instance(source_id, control, cell_key, start_idx + local_start, shift)

    #: Declared bounded-batch input roles, in registration order — the same tokens (and the
    #: same meaning) ``PixelShift`` declares, because instances are discovered through the same
    #: measurement API. A per-use ``inputs=`` remapping replaces the tokens but not the roles,
    #: so every batch lookup resolves through ``ctx.input_token``.
    FRAMES_ROLE = "media:frame"
    ACTIONS_ROLE = "field:action"

    def add_batch(
        self,
        batch,
        ctx,
        *,
        controls: Sequence[str] | None = None,
        boundary: str = "exclude",
    ) -> None:
        """Accumulate one bounded batch of segmented tick-stream windows.

        This is the tick-stream twin of :meth:`check_dataset`, and the reason it is worth
        having: ``segment_ids`` here are real ``(source_id, start_idx, end_idx)`` source
        windows, so the documented ``(source_id, control, cell_key, frame)`` de-duplication
        works in true source frames — the guarantee dataset mode has to give up. Frames,
        actions, and the declared lookbehind/lookahead context arrive already decoded from the
        SDK's one bounded media path; ``color_space`` comes from the declared media column
        rather than the direct-call BGR default.
        """
        import numpy as np

        frames_token = ctx.input_token(self.FRAMES_ROLE)
        actions_token = ctx.input_token(self.ACTIONS_ROLE)
        frames_batch = batch[frames_token]                      # (N, T, H, W, 3)
        actions_batch = np.asarray(batch[actions_token])
        if actions_batch.ndim == 4:
            # Schema-driven core inputs are (N,T,E,D); this single-agent analyzer consumes E=1.
            if actions_batch.shape[2] != 1:
                raise ValueError(
                    "action-state consistency requires exactly one action entity per timestep")
            actions_batch = actions_batch[:, :, 0, :]
        color_space = ctx["schema"].media[frames_token.split(":", 1)[1]].color_space
        segment_ids = list(batch["segment_ids"])
        valid_steps = ctx.valid_steps
        before_batch = ctx.before(actions_token)
        after_actions_batch = ctx.after(actions_token)
        after_frames_batch = ctx.after(frames_token)
        for index, segment_id in enumerate(segment_ids):
            # Ragged batches are padded to the widest window, so measure only the real steps.
            steps = int(valid_steps[index]) if valid_steps is not None else len(
                actions_batch[index])
            after_actions = after_actions_batch[index]
            after_frames = after_frames_batch[index]
            self.add_segment(
                frames_batch[index][:steps],
                _single_entity_actions(actions_batch[index][:steps]),
                segment_id,
                controls=controls,
                color_space=color_space,
                lookbehind_actions=_single_entity_actions(before_batch[index]),
                lookahead_actions=_single_entity_actions(after_actions),
                lookahead_frames=None if after_actions is None else after_frames,
                boundary=boundary,
            )

    def stream_result(self) -> dict:
        """Report the tick-stream verdict and the frame space its positions live in.

        A self-calibrated run additionally reports its ``calibration`` record (the knobs
        and the pooled consensus baselines), so a verdict judged against inferred
        normality says so in the attached result."""
        summary = self.summary()
        result = {
            "passed": not summary["inconsistent"],
            "frame_space": self.frame_space,
            "frame_space_note": self.frame_space_note,
            "cross_window_dedup": True,
        }
        calibration = self.calibration_report()
        if calibration is not None:
            result["calibration"] = calibration
        return result

    def check_dataset(
        self,
        dataset,
        *,
        frames: str = "frames",
        actions: str = "actions",
        controls: Sequence[str] | None = None,
        color_space: str = "BGR",
        boundary: str = "exclude",
        verify: bool = True,
    ) -> dict:
        """Run the analyzer over frame/action modalities in a curated dataset.

        Each shard remains a separate measurement segment so unrelated selected spans are
        never joined, and the manifest ``source_id`` supplies the clip-level aggregation key.

        **Frame numbers here are not source frames.** A dataset manifest entry records
        ``path``/``source_id``/``entity_id``/``split``/``shard_index``/``n_steps`` -- it does
        *not* record where a shard's rows started in its source, and a shard name only carries
        span and shard *indexes*, so a real source offset is not recoverable from the artifact.
        Instead every shard of a clip is placed after the previous one, and the returned
        ``frame_space`` is ``"shard-concatenation"``: positions count rows in manifest order
        within a clip. ``shard_ranges`` maps each shard path to the ``[start, end)`` it occupies
        there.

        Consequence, stated loudly because it inverts the per-segment guarantee:
        ``add_segment`` de-duplicates one physical press seen by two overlapping windows through
        its ``(source_id, control, cell_key, frame)`` key, and **dataset mode cannot** -- two
        curated shards covering overlapping source spans land at different concatenation
        offsets, so the same press is counted (and judged) once per shard. Verdicts are
        unaffected (one violation already condemns a clip), but instance counts and frame
        positions are shard-relative evidence, not deduplicated source-frame evidence.
        """
        if not isinstance(frames, str) or not frames:
            raise ValueError("frames must name a non-empty dataset modality")
        if not isinstance(actions, str) or not actions:
            raise ValueError("actions must name a non-empty dataset modality")
        if not isinstance(verify, bool):
            raise ValueError("verify must be true or false")
        from kuration.core.loaders import iter_shards

        self.declare_frame_space(
            "shard-concatenation",
            "positions count rows in manifest order within a clip, not source frames; the same "
            "press appearing in two curated shards with overlapping source spans is counted "
            "once per shard (no cross-shard de-duplication in dataset mode)",
        )
        offsets: dict[str, int] = {}
        shard_ranges: list[dict] = []
        shards = 0
        for arrays, entry in iter_shards(
            dataset,
            modalities=[frames, actions],
            with_meta=True,
            verify=verify,
        ):
            source_id = str(entry.get("source_id") or "")
            if not source_id:
                raise ValueError(
                    f"manifest shard {entry.get('path')!r} has no non-empty source_id"
                )
            frame_values = arrays[frames]
            action_values = arrays[actions]
            if len(frame_values) != len(action_values):
                raise ValueError(
                    f"shard {entry.get('path')!r} has {len(frame_values)} {frames!r} rows "
                    f"but {len(action_values)} {actions!r} rows"
                )
            # Synthetic, clip-local concatenation offsets: they only keep evidence from
            # different shards distinguishable. See this method's docstring for why real source
            # offsets are unavailable and what that costs.
            start = offsets.get(source_id, 0)
            end = start + len(action_values)
            self.add_segment(
                frame_values,
                action_values,
                (source_id, start, end),
                controls=controls,
                color_space=color_space,
                boundary=boundary,
            )
            offsets[source_id] = end
            shard_ranges.append({"path": entry.get("path"), "source_id": source_id,
                                 "start": start, "end": end})
            shards += 1
        summary = self.summary()
        result = {
            "passed": not summary["inconsistent"],
            "shards_checked": shards,
            "frame_space": self.frame_space,
            "frame_space_note": self.frame_space_note,
            "cross_shard_dedup": False,
            "shard_ranges": shard_ranges,
            "summary": summary,
        }
        calibration = self.calibration_report()
        if calibration is not None:
            result["calibration"] = calibration
        return result


def action_state_consistency(*, anomaly_type: str, baselines: dict | None = None,
                             calibration: dict | None = None):
    """Construct the discoverable, user-owned action-state analyzer.

    Without ``baselines`` the checker self-calibrates: each clip is judged against the
    other clips' leave-one-out consensus (relative outlier detection), tuned by the
    optional ``calibration`` knobs. Supplied baselines remain the externally-calibrated
    ground truth and reject ``calibration``."""
    config: dict[str, Any] = {"anomaly_type": anomaly_type, "baselines": baselines}
    if calibration is not None:
        config["calibration"] = calibration
    return ActionStateConsistencyChecker(config)


#: The pixel-shift transform/operator name that emits the ``Segment.records`` this scorer
#: consumes -- see ``PixelShift.transform_batch`` and ``segment_scalars(records_out=...)``.
PIXEL_SHIFT_RECORDS_KEY = "aintropy/pixel-shift"


class ActionStateAnomalyScorer:
    """SegmentScorer reporting, per window, how many of its OWN pixel-shift instances are
    inconsistent with the configured (or self-calibrated) baseline.

    This scores each segment independently from its own ``Segment.records`` content --
    unlike :meth:`ActionStateConsistencyChecker.add_segment`'s clip-level ledger, it does
    NOT de-duplicate a physical press across overlapping windows, since a press genuinely
    inside two overlapping windows is legitimately part of both windows' own content, and a
    per-segment score should reflect what that segment itself contains. Self-calibration
    (deriving consensus baselines) still needs the whole pool: it happens once inside
    :meth:`score`, which receives every segment in one call.

    Reducing this per-segment score back up to a clip-level verdict (matching the old
    Analyzer's ``consistent``/``inconsistent``/``inconclusive`` per source_id) is not yet a
    built-in capability -- see docs/architecture.md's extension-points notes. For now, use
    this scorer's output directly with ``.filter()``/``.bin()``/``.sample()`` per window, or
    construct :class:`ActionStateConsistencyChecker` directly (via
    :func:`action_state_consistency`) for the clip-level report.
    """

    needs_buffers = False

    def __init__(self, *, baselines: dict | None = None,
                 controls: Sequence[str] | None = None,
                 calibration: dict | None = None):
        config: dict[str, Any] = {"anomaly_type": "action_state", "baselines": baselines}
        if calibration is not None:
            config["calibration"] = calibration
        self._checker = ActionStateConsistencyChecker(config)
        self._controls = self._checker.resolve_controls(controls)

    def _records(self, segment) -> list[dict]:
        return [
            record for record in segment.records.get(PIXEL_SHIFT_RECORDS_KEY, [])
            if record["control"] in self._controls
        ]

    def score(self, segments, ctx):
        import numpy as np

        for segment in segments:
            self._checker.resolve_segment(
                (segment.source_id, segment.start_idx, segment.end_idx),
                expected_length=segment.end_idx - segment.start_idx)
            for record in self._records(segment):
                self._checker.record_instance(
                    segment.source_id, record["control"], record["cell_key"],
                    segment.start_idx + record["local_start"], record["shift"])

        scores = np.zeros(len(segments), dtype=np.float64)
        resolved_baselines: dict[str, dict] = {}
        for index, segment in enumerate(segments):
            records = self._records(segment)
            if not records:
                continue
            source_id = segment.source_id
            if source_id not in resolved_baselines:
                resolved_baselines[source_id] = (
                    self._checker.resolved_baselines_for_clip(source_id))
            baselines = resolved_baselines[source_id]
            scores[index] = float(sum(
                1 for record in records
                if self._checker.is_instance_consistent(
                    source_id, record["control"], record["cell_key"], record["shift"],
                    baselines=baselines,
                ) is False
            ))
        return scores
