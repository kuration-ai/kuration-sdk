"""Measure pixel shift for every isolated key or mouse press in a segment.

The implementation handles one-, two-, and three-frame key presses and isolated mouse taps. It reports what
happened -- makes NO judgment about whether it's "consistent." That is the action-state
analyzer's job: a separate, genuinely data-agnostic outlier-finding layer built on top of
this operator's output, kept deliberately out of this file.

Three things this operator produces:

1. Per-segment scalar features (so it composes inside a real kuration pipeline --
   .bin()/.filter()/.sample() on the results): for each of the 8 controls, 6 numbers --
   `{control}_shift_min`, `{control}_shift_max`, `{control}_shift_mean`,
   `{control}_shift_measured`, `{control}_n_instances`, and `{control}_requested` --
   collapsing across whatever press-lengths (W/S/A/D) or delta
   magnitudes (mouse controls) were found for that control in that segment. min/max are
   the true worst-case/best-case signed shift among however many instances were found (NOT
   an average that could hide one bad instance among several good ones) -- see the third
   deliverable below for the full per-instance detail these numbers are compressed from.

2. A console report (`PixelShiftReport.print_report()`, NOT a written file) -- for every
   (control, press_len or delta) combination, the exact instance count found across the
   whole dataset, an ASCII histogram of their shift values, and which clips each instance
   came from. Same style as docs/batch_results_sensitivity_w.md /
   batch_job_sensitivity_w.py's own report(), just clip-traceable.

3. `Segment.records["aintropy/pixel-shift"]` -- the raw per-instance
   `{"control", "cell_key", "local_start", "shift"}` dicts the scalars above collapse away,
   one list per segment, attached by the transform runner from `transform_batch`'s
   `"records"` output. This is the supported channel for a downstream operator (e.g.
   action-state consistency) that needs the individual presses rather than a per-segment
   summary -- unlike `PixelShiftReport` below, it is per-segment state carried on the
   `Segment` the runner already owns, not process-global accumulation.

`PixelShiftReport` is a plain, explicitly-instantiated object -- deliberately NOT wired
into `PixelShift.transform_batch`. An earlier version accumulated instance detail into a
module-level dict as a transform side effect; process-global accumulation is not a valid
contract regardless of what channel is available, since execution placement (worker,
process, spawn strategy) may change across plans. Build a `PixelShiftReport` yourself, by
calling `add_segment()` directly over your own segments, entirely outside pipeline
execution, if you want the whole-dataset report.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .diamond_action_decode import (
    ALL_CONTROLS,
    BOUNDARY_POLICIES,
    MOUSE_AXIS_OF,
    MOUSE_CONTROLS,
    _mouse_delta_value,
    _release_transition_displacement_validated,
    validate_actions,
    validate_boundary,
    validate_color_space,
    validate_frames,
)

KEY_CONTROLS = ["W", "S", "A", "D"]
MOUSE_CONTROLS_LIST = ["MOUSE_X_POS", "MOUSE_X_NEG", "MOUSE_Y_POS", "MOUSE_Y_NEG"]
PRESS_LENS = [1, 2, 3]

# `shift_measured` is the filterable 0/1 companion to the three shift statistics: the core
# transform contract admits only finite scalars, so an untested segment has to report *some*
# number for min/max/mean, and 0.0 alone is indistinguishable from a real zero-motion
# measurement. Filter or bin on `{control}_shift_measured == 1` (equivalently
# `{control}_n_instances >= 1`) before trusting a shift statistic.
STATS_PER_CONTROL = ("shift_min", "shift_max", "shift_mean", "shift_measured",
                     "n_instances", "requested")
OUTPUT_KEYS = tuple(
    f"{control.lower()}_{stat}" for control in ALL_CONTROLS for stat in STATS_PER_CONTROL
)
CONTROL_PARAMETERS = {
    "type": "object",
    "properties": {
        "controls": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "enum": list(ALL_CONTROLS)},
            "description": "Subset of W/S/A/D/MOUSE_X_POS/.../MOUSE_Y_NEG to measure; "
            "the default is all eight controls.",
        },
        "boundary": {
            "type": "string",
            "enum": list(BOUNDARY_POLICIES),
            "default": "exclude",
            "description": "How to handle a targeted press clipped by a segment edge. "
            "'exclude' is conservative; 'error' requires context; "
            "'assume-inactive' treats outside frames as inactive.",
        },
    },
    "additionalProperties": False,
}


def _validated_controls(controls) -> tuple[str, ...]:
    if controls is None:
        return tuple(ALL_CONTROLS)
    if isinstance(controls, (str, bytes)) or not isinstance(controls, Sequence) or not controls:
        raise ValueError("controls must be a non-empty sequence of control names")
    if not all(isinstance(control, str) for control in controls):
        raise ValueError("controls must contain only strings")
    unknown = sorted(set(controls) - set(ALL_CONTROLS))
    if unknown:
        raise ValueError(f"unknown pixel-shift control(s) {unknown}; expected {ALL_CONTROLS}")
    if len(controls) != len(set(controls)):
        raise ValueError("controls must not contain duplicates")
    return tuple(controls)


# Human-readable names for the report -- the DIAMOND action-vector codes (W/S/A/D/
# MOUSE_X_POS/...) are what the code and reference data key on, but not what a reader wants
# to see printed. Direction words follow the DIAMOND sign convention exactly (see
# asg2diamond/action_map.py: mouse_x positive = looking right, mouse_y positive = looking
# down), not guessed.
CONTROL_DESCRIPTIONS = {
    "W": "Move Forward (W)",
    "S": "Move Backward (S)",
    "A": "Strafe Left (A)",
    "D": "Strafe Right (D)",
    "MOUSE_X_POS": "Mouse Look Right (mouse_x > 0)",
    "MOUSE_X_NEG": "Mouse Look Left (mouse_x < 0)",
    "MOUSE_Y_POS": "Mouse Look Down (mouse_y > 0)",
    "MOUSE_Y_NEG": "Mouse Look Up (mouse_y < 0)",
}


def iter_instances(
    frames: np.ndarray,
    actions: np.ndarray,
    control: str,
    color_space: str = "BGR",
    *,
    lookbehind_actions: np.ndarray | None = None,
    lookahead_actions: np.ndarray | None = None,
    lookahead_frames: np.ndarray | None = None,
    boundary: str = "exclude",
    validated: bool = False,
):
    """Yield ``(cell_key, local_start, signed_shift)`` for one control.

    This is the public per-instance measurement contract for integrations that need more
    detail than :func:`segment_scalars`. Frames, actions, and optional boundary context are
    validated here, so a direct caller cannot get silent garbage out of malformed arrays.
    Pass ``validated=True`` only when the arrays already came from
    :func:`validate_segment_inputs` -- the one supported way to skip the (otherwise repeated)
    per-control revalidation when iterating several controls over the same segment.

    Pure generator: yields (cell_key, local_start, signed_shift) for every isolated-press
    instance of `control` found in this one segment -- no side effects, no shared state,
    safe to call from anywhere (a worker process, a plain loop, whatever). cell_key is
    press_len as a string ("1"/"2"/"3") for W/S/A/D, the observed delta value as a string
    (e.g. "2", "-1000") for mouse controls. local_start is the instance's press-sequence
    start frame, indexed within this segment (0-based) -- callers that know this segment's
    window offset can add it to get a globally meaningful frame number; see
    PixelShiftReport.add_segment. `color_space` (see diamond_action_decode._prep_gray)
    defaults to BGR, matching raw ASG/DIAMOND HDF5 arrays; pass the real declared
    color_space when `frames` came through a kuration-sdk media column instead."""
    control = _validated_controls((control,))[0]
    color_space = validate_color_space(color_space)
    boundary = validate_boundary(boundary)
    if not isinstance(validated, bool):
        raise ValueError("validated must be true or false")
    if not validated:
        (frames, actions, lookbehind_actions, lookahead_actions,
         lookahead_frames) = validate_segment_inputs(
            frames,
            actions,
            lookbehind_actions=lookbehind_actions,
            lookahead_actions=lookahead_actions,
            lookahead_frames=lookahead_frames,
        )
    if control in MOUSE_CONTROLS:
        axis = MOUSE_AXIS_OF[control]
        for start, px in _release_transition_displacement_validated(
            frames,
            actions,
            control,
            press_len=1,
            color_space=color_space,
            lookbehind_actions=lookbehind_actions,
            lookahead_actions=lookahead_actions,
            lookahead_frames=lookahead_frames,
            boundary=boundary,
        ):
            delta_value = int(_mouse_delta_value(actions[start : start + 1], axis)[0])
            yield str(delta_value), start, px
    else:
        for press_len in PRESS_LENS:
            for start, px in _release_transition_displacement_validated(
                frames,
                actions,
                control,
                press_len,
                color_space,
                lookbehind_actions=lookbehind_actions,
                lookahead_actions=lookahead_actions,
                lookahead_frames=lookahead_frames,
                boundary=boundary,
            ):
                yield str(press_len), start, px


def validate_segment_inputs(
    frames,
    actions,
    *,
    lookbehind_actions=None,
    lookahead_actions=None,
    lookahead_frames=None,
):
    """Validate one segment and its optional boundary context for instance measurement."""
    actions = validate_actions(actions)
    frames = validate_frames(frames, steps=len(actions))
    before = (
        None
        if lookbehind_actions is None
        else validate_actions(lookbehind_actions, name="lookbehind_actions")
    )
    if (lookahead_actions is None) != (lookahead_frames is None):
        raise ValueError("lookahead_actions and lookahead_frames must be provided together")
    after_actions = None
    after_frames = None
    if lookahead_actions is not None:
        after_actions = validate_actions(lookahead_actions, name="lookahead_actions")
        after_frames = validate_frames(
            lookahead_frames, steps=len(after_actions), name="lookahead_frames"
        )
        if after_frames.shape[1:] != frames.shape[1:]:
            raise ValueError("lookahead_frames must have the same spatial shape as frames")
        if after_frames.dtype != frames.dtype:
            raise ValueError("lookahead_frames must have the same dtype as frames")
    return frames, actions, before, after_actions, after_frames


# Compatibility aliases for the 0.1 development series. New integrations must use the public
# names above; keeping these aliases avoids breaking notebooks written before the API promotion.
_iter_instances = iter_instances
_validated_segment_inputs = validate_segment_inputs


def segment_scalars(
    frames: np.ndarray,
    actions: np.ndarray,
    controls: Sequence[str] | None = None,
    color_space: str = "BGR",
    *,
    lookbehind_actions=None,
    lookahead_actions=None,
    lookahead_frames=None,
    owned_length: int | None = None,
    boundary: str = "exclude",
    records_out: list | None = None,
) -> dict:
    """The 6-number-per-control summary for ONE segment -- true to the data (no reference,
    no consistency judgment): the actual worst/best/mean signed shift and how many
    instances were found, collapsed across whatever press-lengths/deltas occurred for each
    control in this segment. Pure function with one optional side effect: pass a list as
    `records_out` to also collect the raw per-instance
    `{"control", "cell_key", "local_start", "shift"}` dicts this scalar summary collapses
    away -- e.g. for ``Segment.records``, so a downstream consumer (like action-state
    consistency) can judge each press directly instead of re-deriving them from raw frames.
    Leave `records_out` as `None` (the default) for the original side-effect-free behavior.
    `color_space` defaults to BGR (raw ASG/DIAMOND HDF5 arrays); PixelShift.transform_batch
    passes the real declared color_space of the SDK media column instead (see
    diamond_action_decode._prep_gray)."""
    controls = _validated_controls(controls)
    boundary = validate_boundary(boundary)
    color_space = validate_color_space(color_space)
    frames, actions, before, after_actions, after_frames = validate_segment_inputs(
        frames,
        actions,
        lookbehind_actions=lookbehind_actions,
        lookahead_actions=lookahead_actions,
        lookahead_frames=lookahead_frames,
    )
    if owned_length is None:
        owned_length = len(actions)
    if (not isinstance(owned_length, (int, np.integer)) or isinstance(owned_length, bool)
            or not 0 <= int(owned_length) <= len(actions)):
        raise ValueError("owned_length must be an integer within the segment window")
    owned_length = int(owned_length)
    out = {}
    for control in ALL_CONTROLS:
        prefix = control.lower()
        if control not in controls:
            out[f"{prefix}_shift_min"] = 0.0
            out[f"{prefix}_shift_max"] = 0.0
            out[f"{prefix}_shift_mean"] = 0.0
            out[f"{prefix}_shift_measured"] = 0.0
            out[f"{prefix}_n_instances"] = 0.0
            out[f"{prefix}_requested"] = 0.0
            continue
        shifts = []
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
            if local_start >= owned_length:
                continue
            shifts.append(shift)
            if records_out is not None:
                records_out.append({
                    "control": control, "cell_key": cell_key,
                    "local_start": local_start, "shift": shift,
                })
        if shifts:
            out[f"{prefix}_shift_min"] = float(min(shifts))
            out[f"{prefix}_shift_max"] = float(max(shifts))
            out[f"{prefix}_shift_mean"] = float(sum(shifts) / len(shifts))
            out[f"{prefix}_shift_measured"] = 1.0
            out[f"{prefix}_n_instances"] = float(len(shifts))
        else:
            # 0.0 is a placeholder, not a claim of "zero shift". The core transform contract
            # admits only finite scalars, so NaN is not available as a sentinel;
            # `shift_measured` (0/1) and `n_instances` are the authoritative signals, and a
            # downstream consumer must check one of them -- e.g.
            # `.filter("w_shift_measured", ">=", 1)` -- before trusting min/max/mean.
            out[f"{prefix}_shift_min"] = 0.0
            out[f"{prefix}_shift_max"] = 0.0
            out[f"{prefix}_shift_mean"] = 0.0
            out[f"{prefix}_shift_measured"] = 0.0
            out[f"{prefix}_n_instances"] = 0.0
        out[f"{prefix}_requested"] = 1.0
    return out


# ---------------------------------------------------------------------------------------
# Console report -- printed, not written to a file. Same shape as
# docs/batch_results_sensitivity_w.md / batch_job_sensitivity_w.py's report(): per
# (control, cell_key), instance count + an ASCII histogram of shift values + which clips.
#
# PixelShiftReport is a plain object you instantiate yourself -- see this module's
# docstring for why it is deliberately never touched by PixelShift.transform_batch. Typical
# use: loop over your own segments (however you obtained them -- outside pipeline.collect()
# execution) calling add_segment() once per segment, then print_report() at the end.
# ---------------------------------------------------------------------------------------


def _ascii_histogram(values: list, bins: int = 10, width: int = 40) -> list:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [(f"{lo:.3f}", len(values), "#" * min(width, len(values)))]
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    peak = max(counts) if counts.max() > 0 else 1
    rows = []
    for i, count in enumerate(counts):
        bar_len = int(round(width * count / peak))
        rows.append((f"[{edges[i]:.3f}, {edges[i + 1]:.3f})", int(count), "#" * bar_len))
    return rows


class PixelShiftReport:
    """Explicit, user-owned accumulator for the full per-instance pixel-shift report --
    instance counts, histograms, and clip provenance, per (control, press_len/delta).
    Deliberately a plain object you construct yourself, not a module-level singleton: no
    reset_ledger()-style footgun, and no ambiguity about how it interacts with the SDK's
    parallel executors, because it never does -- add_segment() is meant to be called
    directly, in your own code, outside pipeline execution."""

    def __init__(self):
        self._ledger: dict = {}
        # A press's global frame position (start_idx + local_start) is the same
        # regardless of which window observed it -- so it doubles as the tie-breaker key
        # for presses "owned" by more than one overlapping window (a press's start can
        # fall within more than one window's local range when windows overlap; each
        # add_segment() call only sees its own window, with no visibility into what a
        # previous call already recorded, so this cross-call de-dup lives here instead).
        # The first add_segment() call to observe a given (source, control, cell_key,
        # frame) wins; later observations of the exact same global instance are skipped.
        self._seen_instances: set = set()

    def add_segment(
        self,
        frames: np.ndarray,
        actions: np.ndarray,
        segment_id,
        controls: Sequence[str] | None = None,
        color_space: str = "BGR",
        *,
        lookbehind_actions=None,
        lookahead_actions=None,
        lookahead_frames=None,
        boundary: str = "exclude",
    ) -> None:
        """Record every instance found in this one segment under its exact cell_key.
        `segment_id` is (source_id, start_idx, end_idx) -- the same shape the SDK's
        segment_ids batch entry provides. Each instance keeps its OWN precise
        (source_id, frame) location, computed from start_idx + the instance's actual
        local start frame within this window -- not one label shared by every instance
        found in this call, which would make every instance from the same window
        indistinguishable in the printed report (see this class's docstring)."""
        controls = _validated_controls(controls)
        boundary = validate_boundary(boundary)
        color_space = validate_color_space(color_space)
        frames, actions, before, after_actions, after_frames = validate_segment_inputs(
            frames,
            actions,
            lookbehind_actions=lookbehind_actions,
            lookahead_actions=lookahead_actions,
            lookahead_frames=lookahead_frames,
        )
        if (
            isinstance(segment_id, (str, bytes))
            or not isinstance(segment_id, Sequence)
            or len(segment_id) != 3
        ):
            raise ValueError("segment_id must be (source_id, start_idx, end_idx)")
        source_id, start_idx, end_idx = segment_id
        # An unvalidated source_id (None, "", a stray object) or a negative index would
        # still satisfy every check below, poisoning print_report()'s clip/frame
        # provenance with a plausible-looking but meaningless location.
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("segment_id source_id must be a non-empty string")
        if (
            isinstance(start_idx, bool)
            or isinstance(end_idx, bool)
            or not isinstance(start_idx, (int, np.integer))
            or not isinstance(end_idx, (int, np.integer))
            or start_idx < 0
            or end_idx < 0
        ):
            raise ValueError("segment_id start_idx and end_idx must be non-negative integers")
        start_idx, end_idx = int(start_idx), int(end_idx)
        if end_idx - start_idx != len(actions):
            raise ValueError("segment_id range length must match the number of action frames")
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
                frame = start_idx + local_start
                instance_key = (source_id, control, cell_key, frame)
                if instance_key in self._seen_instances:
                    continue
                self._seen_instances.add(instance_key)
                record = {"source_id": source_id, "frame": frame, "shift": shift}
                self._ledger.setdefault(control, {}).setdefault(cell_key, []).append(record)

    def to_records(self) -> list[dict]:
        """Export the accumulated ledger as flat, per-instance rows.

        The row schema is identical to ``AnomalyChecker.to_records()`` so callers can
        concatenate the two lists directly. Pixel shift measures instances without
        judging them, so ``consistent`` and ``verdict`` are ``None``.
        ``n_instances`` is the total number of recorded instances for the row's
        ``source_id``. PixelShiftReport positions are always real source frames.

        Rows are sorted by ``(source_id, frame, control, cell_key)``. Sources with no
        recorded instances are absent because this report only learns about a source
        when an instance is recorded.
        """
        rows = []
        instance_counts: dict[str, int] = {}
        for control, cells in self._ledger.items():
            for cell_key, entries in cells.items():
                for entry in entries:
                    source_id = entry["source_id"]
                    instance_counts[source_id] = instance_counts.get(source_id, 0) + 1
                    rows.append(
                        {
                            "source_id": source_id,
                            "control": control,
                            "cell_key": cell_key,
                            "frame": entry["frame"],
                            "shift": entry["shift"],
                            "consistent": None,
                            "verdict": None,
                            "n_instances": None,
                            "frame_space": "source",
                            "frame_space_note": "",
                        }
                    )
        for row in rows:
            row["n_instances"] = instance_counts[row["source_id"]]
        return sorted(
            rows,
            key=lambda item: (
                item["source_id"],
                item["frame"],
                item["control"],
                item["cell_key"],
            ),
        )

    def print_report(self, controls: Sequence[str] | None = None) -> None:
        """Print the full pixel-shift report accumulated so far -- instance counts,
        histograms, and clip provenance, per (control, press_len/delta). Distribution-only
        style: printed to the screen, nothing written to disk. See ``to_records()`` for
        this same data as plain, unprinted structured rows."""
        controls = _validated_controls(controls)
        print("=" * 78)
        print("PIXEL SHIFT REPORT -- isolated key/mouse presses found across the dataset")
        print("=" * 78)
        for control in controls:
            name = CONTROL_DESCRIPTIONS.get(control, control)
            cells = self._ledger.get(control, {})
            if not cells:
                print(f"\n{name}: no isolated instances found")
                continue
            print(f"\n{name}")
            for cell_key in sorted(cells, key=lambda k: (len(k), k)):
                entries = cells[cell_key]
                shifts = [e["shift"] for e in entries]
                # Each instance's own (source, frame) -- distinct real instances at
                # different frames never collapse into one entry, unlike a shared
                # per-segment label would; a genuine duplicate (same source+frame) still
                # legitimately dedupes.
                locations = sorted({(e["source_id"], e["frame"]) for e in entries})
                label = "held for" if control in KEY_CONTROLS else "mouse delta"
                unit = " frame(s)" if control in KEY_CONTROLS else ""
                print(
                    f"  {label} {cell_key}{unit}  --  n_instances={len(entries)}  "
                    f"mean={sum(shifts) / len(shifts):.4f}  min={min(shifts):.4f}  max={max(shifts):.4f}"
                )
                for edge_label, count, bar in _ascii_histogram(shifts):
                    print(f"    {edge_label:>24}  {count:>5}  {bar}")
                previews = [
                    f"clip '{source_id}' @ frame {frame}" for source_id, frame in locations[:8]
                ]
                clip_preview = "; ".join(previews) + (
                    f"; ... (+{len(locations) - 8} more)" if len(locations) > 8 else ""
                )
                print(f"    Clip ID(s): {clip_preview}")
        print()


def _single_action_entity(value, label: str):
    actions = np.asarray(value)
    if actions.ndim == 3:
        if actions.shape[1] != 1:
            raise ValueError(f"pixel-shift {label} require exactly one entity per timestep")
        actions = actions[:, 0, :]
    return actions


class PixelShift:
    """Kuration batch transform implementing the per-segment measurements."""

    def __init__(
        self,
        controls: Sequence[str] | None = None,
        boundary: str = "exclude",
    ):
        self.controls = _validated_controls(controls)
        self.boundary = validate_boundary(boundary)

    #: Declared input roles, in registration order. A per-use ``inputs=`` remapping (e.g.
    #: ``inputs=("media:frames2", "field:controls")``) replaces the tokens but not the roles,
    #: so every batch/context lookup below resolves the role through ``ctx.input_token`` --
    #: hardcoding the registration tokens would pass pipeline construction and then fail
    #: mid-run with a bare KeyError on a remapped use.
    FRAMES_ROLE = "media:frame"
    ACTIONS_ROLE = "field:action"

    def transform_batch(self, batch, ctx):
        typed = hasattr(ctx, "input_token")
        frames_token = ctx.input_token(self.FRAMES_ROLE) if typed else self.FRAMES_ROLE
        actions_token = ctx.input_token(self.ACTIONS_ROLE) if typed else self.ACTIONS_ROLE
        frames_batch = batch[frames_token]  # (N, T, H, W, 3)
        actions_batch = np.asarray(batch[actions_token])
        if actions_batch.ndim == 4:
            if actions_batch.shape[2] != 1:
                raise ValueError(
                    "pixel-shift requires exactly one action entity per timestep")
            actions_batch = actions_batch[:, :, 0, :]
        # Schema-driven core inputs are (N,T,E,D); this single-agent plugin consumes E=1.
        # Read the declared media color space rather than assuming the BGR direct-call default.
        color_space = ctx["schema"].media[frames_token.split(":", 1)[1]].color_space
        count = len(frames_batch)
        if hasattr(ctx, "before"):
            before_batch = ctx.before(actions_token)
            after_actions_batch = ctx.after(actions_token)
            after_frames_batch = ctx.after(frames_token)
            owned_lengths = ctx.owned_lengths
        else:  # Compatibility for direct callers still supplying a plain context mapping.
            before_batch = batch.get(f"lookbehind:{actions_token}", [None] * count)
            after_actions_batch = batch.get(f"lookahead:{actions_token}", [None] * count)
            after_frames_batch = batch.get(f"lookahead:{frames_token}", [None] * count)
            owned_lengths = batch.get("owned_lengths")
        before_batch = tuple(
            None if value is None else _single_action_entity(value, "lookbehind actions")
            for value in before_batch)
        after_actions_batch = tuple(
            None if value is None else _single_action_entity(value, "lookahead actions")
            for value in after_actions_batch)
        if owned_lengths is None:
            owned_lengths = [len(actions) for actions in actions_batch]
        records: list[list[dict]] = [[] for _ in range(count)]
        rows = [
            segment_scalars(
                frames,
                actions,
                self.controls,
                color_space,
                lookbehind_actions=before,
                lookahead_actions=after_actions,
                lookahead_frames=after_frames,
                owned_length=int(owned_length),
                boundary=self.boundary,
                records_out=row_records,
            )
            for frames, actions, before, after_actions, after_frames, owned_length, row_records
            in zip(frames_batch, actions_batch, before_batch, after_actions_batch,
                   after_frames_batch, owned_lengths, records, strict=True)
        ]
        result = {key: np.array([row[key] for row in rows], dtype=np.float64) for key in OUTPUT_KEYS}
        result["records"] = records
        return result


def pixel_shift(pipeline, *, controls=None, boundary: str = "exclude"):
    """Compose the registered transform into a pipeline."""
    params = {"boundary": boundary}
    if controls is not None:
        params["controls"] = controls
    return pipeline.transform("aintropy/pixel-shift", **params)
