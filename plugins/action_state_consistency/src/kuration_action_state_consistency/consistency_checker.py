"""Operator 2 of 2: checks kuration_pixel_shift's per-instance shift measurements against
user-supplied consistency baselines and classifies each clip as CONSISTENT, INCONSISTENT,
or INCONCLUSIVE. Pixel_shift itself makes NO consistency judgment ("no consistency
judgment, just what pixel motion happened") -- this module is exactly that missing
judgment layer, built on top of it, kept in a separate distribution.

Data-agnostic, like pixel_shift: every baseline (expected sign + a [min, max] magnitude
range, per control and per press-length/mouse-delta cell) is supplied by the caller as
config -- nothing here is hardcoded to ASG, DIAMOND, or any specific reference dataset.
`load_baselines_from_reference()` is a convenience for the common case of deriving those
baselines empirically from a reference JSON (see this repo's own
scripts/build_action_consistency_reference.py, which is dataset-specific tooling for
producing baseline VALUES kept out of this data-agnostic package), but any caller can
build a baselines dict by hand and it works identically.

Reuses kuration_pixel_shift's own instance-finding directly (_iter_instances,
_validated_segment_inputs) -- the same isolated-press detection, boundary handling, and
per-instance detail pixel_shift itself is built on -- rather than re-deriving any of it.
This module adds its own overlapping-window de-duplication (the same (source_id, control,
cell_key, frame) tie-breaker PixelShiftReport uses) since it accumulates its own
per-instance records independently of PixelShiftReport.
"""
from __future__ import annotations

from collections.abc import Sequence

from kuration_pixel_shift import ALL_CONTROLS
from kuration_pixel_shift.diamond_action_decode import validate_boundary, validate_color_space
from kuration_pixel_shift.pixel_shift_operator import _iter_instances, _validated_segment_inputs

ANOMALY_TYPES = ("action_state",)


def _validate_baseline_cell(control: str, cell_key: str, cell) -> dict:
    if not isinstance(cell, dict):
        raise ValueError(f"baseline for {control!r}/{cell_key!r} must be a dict")
    missing = {"expected_sign", "min", "max"} - cell.keys()
    if missing:
        raise ValueError(f"baseline for {control!r}/{cell_key!r} missing {sorted(missing)}")
    sign = cell["expected_sign"]
    if sign not in (1, -1, 1.0, -1.0) or isinstance(sign, bool):
        raise ValueError(f"baseline for {control!r}/{cell_key!r}: expected_sign must be 1 or -1")
    lo, hi = cell["min"], cell["max"]
    numeric = (int, float)
    if (
        not isinstance(lo, numeric)
        or not isinstance(hi, numeric)
        or isinstance(lo, bool)
        or isinstance(hi, bool)
    ):
        raise ValueError(f"baseline for {control!r}/{cell_key!r}: min/max must be numbers")
    if lo < 0 or hi < 0:
        raise ValueError(
            f"baseline for {control!r}/{cell_key!r}: min/max are magnitudes, must be non-negative"
        )
    if lo > hi:
        raise ValueError(f"baseline for {control!r}/{cell_key!r}: min must not exceed max")
    return {"expected_sign": int(sign), "min": float(lo), "max": float(hi)}


def validate_baselines(baselines) -> dict:
    """Validate a ``{control: {cell_key: {expected_sign, min, max}}}`` baseline dict.

    ``control`` must be one of ``kuration_pixel_shift.ALL_CONTROLS``. ``cell_key`` is
    press_len as a string ("1"/"2"/"3") for W/S/A/D, the observed mouse delta value as a
    string (e.g. "2", "-1000") for the mouse controls -- the exact same cell_key
    pixel_shift itself reports instances under. ``min``/``max`` are non-negative
    MAGNITUDE bounds (not signed), checked against ``abs(shift)``; direction is checked
    separately via ``expected_sign``.
    """
    if not isinstance(baselines, dict) or not baselines:
        raise ValueError("baselines must be a non-empty dict")
    out: dict = {}
    for control, cells in baselines.items():
        if control not in ALL_CONTROLS:
            raise ValueError(f"unknown control {control!r} in baselines; expected {ALL_CONTROLS}")
        if not isinstance(cells, dict) or not cells:
            raise ValueError(
                f"baselines[{control!r}] must be a non-empty dict of cell_key -> baseline"
            )
        out[control] = {}
        for cell_key, cell in cells.items():
            if not isinstance(cell_key, str):
                raise ValueError(
                    f"baseline cell keys must be strings, got {cell_key!r} for {control!r}"
                )
            out[control][cell_key] = _validate_baseline_cell(control, cell_key, cell)
    return out


def validate_config(config) -> dict:
    """Validate the full operator config: ``{"anomaly_type": "action_state", "baselines": {...}}``.

    ``anomaly_type`` is explicit rather than assumed, naming what this config means to
    check -- "action_state" is the only supported value right now; a differently-shaped
    anomaly check would need its own type name and its own validation, not silently reuse
    this one's baseline shape.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")
    anomaly_type = config.get("anomaly_type")
    if anomaly_type not in ANOMALY_TYPES:
        raise ValueError(f"config['anomaly_type'] must be one of {ANOMALY_TYPES}, got {anomaly_type!r}")
    if "baselines" not in config:
        raise ValueError("config must include 'baselines'")
    return {"anomaly_type": anomaly_type, "baselines": validate_baselines(config["baselines"])}


def _is_consistent(shift: float, baseline: dict) -> bool:
    """A shift is consistent with a baseline when its magnitude falls in [min, max] AND,
    for any nonzero shift, its sign matches expected_sign. A shift of exactly 0 has no
    sign to violate -- only the magnitude bound applies (which a min > 0 would already
    fail on its own, so this doesn't let a genuinely-missing press through)."""
    magnitude = abs(shift)
    if not (baseline["min"] <= magnitude <= baseline["max"]):
        return False
    if magnitude == 0:
        return True
    return (shift > 0) == (baseline["expected_sign"] == 1)


def _validate_segment_id(segment_id, actions) -> tuple[str, int, int]:
    """Same segment_id contract as kuration_pixel_shift.PixelShiftReport.add_segment:
    (source_id, start_idx, end_idx), source_id a non-empty string, both indexes
    non-negative integers, range length matching the number of action frames."""
    if (
        isinstance(segment_id, (str, bytes))
        or not isinstance(segment_id, Sequence)
        or len(segment_id) != 3
    ):
        raise ValueError("segment_id must be (source_id, start_idx, end_idx)")
    source_id, start_idx, end_idx = segment_id
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("segment_id source_id must be a non-empty string")
    if (
        isinstance(start_idx, bool)
        or isinstance(end_idx, bool)
        or not isinstance(start_idx, int)
        or not isinstance(end_idx, int)
        or start_idx < 0
        or end_idx < 0
    ):
        raise ValueError("segment_id start_idx and end_idx must be non-negative integers")
    if end_idx - start_idx != len(actions):
        raise ValueError("segment_id range length must match the number of action frames")
    return source_id, start_idx, end_idx


def load_baselines_from_reference(reference: dict, *, controls: Sequence[str] | None = None) -> dict:
    """Build a baselines dict from a reference JSON matching
    scripts/build_action_consistency_reference.py's ``cells`` shape
    (``{control_lower: {cell_key: {"reliable", "floor", "ceiling", "expected_sign", ...}}}``).

    A cell marked unreliable (not enough reference samples to trust) is skipped entirely --
    it becomes "no baseline configured for this cell" rather than being checked against an
    untrustworthy floor/ceiling. ``controls``, if given, restricts which controls are
    pulled from the reference (as pixel_shift-style names, e.g. "W", "MOUSE_X_POS").
    """
    wanted = None if controls is None else set(controls)
    baselines: dict = {}
    for control_lower, per_cell in reference.get("cells", {}).items():
        control = control_lower.upper()
        if control not in ALL_CONTROLS:
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


class ActionStateConsistencyChecker:
    """Explicit, user-owned accumulator (same pattern as
    kuration_pixel_shift.PixelShiftReport): construct it with a config, call
    ``add_segment()`` directly over your own segments -- outside any pipeline execution,
    for the same reason PixelShiftReport does: classifying a clip needs to aggregate
    across every segment belonging to it, which a stateless per-segment SDK transform
    cannot do.

    Verdict per clip (grouped by segment_id's source_id):
      - INCONSISTENT if ANY recorded instance (any baselined control/cell) violates its
        baseline -- one wrong-direction or out-of-range press makes the whole clip
        inconsistent, not just that one instance.
      - CONSISTENT if there is no violation AND at least one baselined-control instance
        was actually observed somewhere in the clip (real supporting evidence -- a clip
        where nothing relevant ever happened isn't "consistent", it's untested).
      - INCONCLUSIVE if no violation was found but also no instance of any baselined
        control was ever observed in the clip -- nothing to judge either way. This is
        deliberately NOT the same bar as "every configured control fired at least once";
        a clip that only ever exercised W, with no violations, is CONSISTENT even if the
        baselines also configured S/A/D and none of those ever appeared in this clip.
    """

    def __init__(self, config):
        validated = validate_config(config)
        self.anomaly_type = validated["anomaly_type"]
        self.baselines = validated["baselines"]
        # A press's global frame position can be "owned" by more than one overlapping
        # window (see kuration_pixel_shift's README, "Input and segment-boundary
        # contract") -- same tie-breaker PixelShiftReport uses, tracked independently
        # here since this accumulates its own per-instance records.
        self._seen_instances: set = set()
        self._clips_seen: set = set()
        self._clip_records: dict = {}

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
        controls = tuple(self.baselines) if controls is None else tuple(controls)
        unknown = sorted(set(controls) - set(self.baselines))
        if unknown:
            raise ValueError(
                f"no baseline configured for control(s) {unknown}; "
                f"configured: {sorted(self.baselines)}"
            )
        boundary = validate_boundary(boundary)
        color_space = validate_color_space(color_space)
        frames, actions, before, after_actions, after_frames = _validated_segment_inputs(
            frames,
            actions,
            lookbehind_actions=lookbehind_actions,
            lookahead_actions=lookahead_actions,
            lookahead_frames=lookahead_frames,
        )
        source_id, start_idx, _end_idx = _validate_segment_id(segment_id, actions)
        self._clips_seen.add(source_id)

        for control in controls:
            cells = self.baselines[control]
            for cell_key, local_start, shift in _iter_instances(
                frames,
                actions,
                control,
                color_space,
                lookbehind_actions=before,
                lookahead_actions=after_actions,
                lookahead_frames=after_frames,
                boundary=boundary,
            ):
                baseline = cells.get(cell_key)
                if baseline is None:
                    continue  # no baseline for this exact press_len/delta -- not checked
                frame = start_idx + local_start
                instance_key = (source_id, control, cell_key, frame)
                if instance_key in self._seen_instances:
                    continue
                self._seen_instances.add(instance_key)
                record = {
                    "control": control,
                    "cell_key": cell_key,
                    "frame": frame,
                    "shift": shift,
                    "consistent": _is_consistent(shift, baseline),
                }
                self._clip_records.setdefault(source_id, []).append(record)

    def clip_verdict(self, source_id: str) -> str:
        """"consistent" | "inconsistent" | "inconclusive" for one clip."""
        if source_id not in self._clips_seen:
            raise KeyError(f"no segments recorded for clip {source_id!r}")
        records = self._clip_records.get(source_id, [])
        if not records:
            return "inconclusive"
        if any(not r["consistent"] for r in records):
            return "inconsistent"
        return "consistent"

    def summary(self) -> dict:
        """``{"consistent": [...], "inconsistent": [...], "inconclusive": [...]}`` --
        every clip ever seen via add_segment(), sorted by source_id within each verdict."""
        verdicts: dict = {"consistent": [], "inconsistent": [], "inconclusive": []}
        for source_id in sorted(self._clips_seen):
            verdicts[self.clip_verdict(source_id)].append(source_id)
        return verdicts

    def print_report(self) -> None:
        """Print the full consistency report: counts and exact clip IDs per verdict,
        plus which control/cell/frame violated the baseline for each inconsistent clip.
        Printed to the screen, nothing written to disk (same convention as
        PixelShiftReport.print_report())."""
        verdicts = self.summary()
        print("=" * 78)
        print(f"ACTION STATE CONSISTENCY REPORT (anomaly_type={self.anomaly_type!r})")
        print("=" * 78)
        for verdict in ("consistent", "inconsistent", "inconclusive"):
            clip_ids = verdicts[verdict]
            print(f"\n{verdict.upper()}: {len(clip_ids)} clip(s)")
            for clip_id in clip_ids:
                if verdict == "inconsistent":
                    violations = [r for r in self._clip_records[clip_id] if not r["consistent"]]
                    detail = "; ".join(
                        f"{r['control']}/{r['cell_key']}@frame{r['frame']} shift={r['shift']:.4f}"
                        for r in violations
                    )
                    print(f"  - {clip_id}  ({len(violations)} violation(s): {detail})")
                else:
                    print(f"  - {clip_id}")
        print()
