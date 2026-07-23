"""Generic anomaly-detection core: baseline validation, the sign+magnitude consistency
check, and per-clip (source_id) verdict aggregation.

This module knows NOTHING about pixel_shift, the DIAMOND action-vector format, or any
specific measurement source -- it operates purely on (source_id, control, cell_key,
frame, shift) tuples handed to it by a caller. A different, future anomaly type that has
nothing to do with pixel shift could
reuse everything in this module without pixel_shift ever being imported; only a
type-specific adapter (like action_state.py) needs to know how to DISCOVER those tuples
from raw data, and only THAT adapter needs its own measurement source installed.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _finite_number(value, *, name: str) -> float:
    numeric = (int, float, np.integer, np.floating)
    if isinstance(value, bool) or not isinstance(value, numeric):
        raise ValueError(f"{name} must be a number")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _validate_baseline_cell(control: str, cell_key: str, cell) -> dict:
    if not isinstance(cell, dict):
        raise ValueError(f"baseline for {control!r}/{cell_key!r} must be a dict")
    missing = {"expected_sign", "min", "max"} - cell.keys()
    if missing:
        raise ValueError(f"baseline for {control!r}/{cell_key!r} missing {sorted(missing)}")
    sign = cell["expected_sign"]
    if sign not in (1, -1, 1.0, -1.0) or isinstance(sign, bool):
        raise ValueError(f"baseline for {control!r}/{cell_key!r}: expected_sign must be 1 or -1")
    prefix = f"baseline for {control!r}/{cell_key!r}"
    lo = _finite_number(cell["min"], name=f"{prefix}: min")
    hi = _finite_number(cell["max"], name=f"{prefix}: max")
    if lo < 0 or hi < 0:
        raise ValueError(
            f"baseline for {control!r}/{cell_key!r}: min/max are magnitudes, must be non-negative"
        )
    if lo > hi:
        raise ValueError(f"baseline for {control!r}/{cell_key!r}: min must not exceed max")
    return {"expected_sign": int(sign), "min": lo, "max": hi}


def validate_baselines(baselines) -> dict:
    """Validate a ``{control: {cell_key: {expected_sign, min, max}}}`` baseline dict,
    generically -- ``control``/``cell_key`` are just non-empty string labels here. A
    specific adapter (e.g. action_state.py) is responsible for checking they're valid
    values for its own vocabulary (e.g. ``kuration.gameplay.pixel_shift.ALL_CONTROLS``);
    this function
    has no opinion on what a "control" is.

    ``min``/``max`` are non-negative MAGNITUDE bounds (not signed), checked against
    ``abs(shift)``; direction is checked separately via ``expected_sign``.
    """
    if not isinstance(baselines, dict) or not baselines:
        raise ValueError("baselines must be a non-empty dict")
    out: dict = {}
    for control, cells in baselines.items():
        if not isinstance(control, str) or not control:
            raise ValueError(f"baseline control keys must be non-empty strings, got {control!r}")
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
    """Validate the generic config shape: ``{"anomaly_type": <non-empty string>,
    "baselines": {...}}``. Does not restrict which ``anomaly_type`` values are legal --
    that's entirely a specific adapter's concern (e.g. action_state.py's ANOMALY_TYPES),
    since the generic core has no opinion on what anomaly types exist.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")
    anomaly_type = config.get("anomaly_type")
    if not isinstance(anomaly_type, str) or not anomaly_type:
        raise ValueError("config['anomaly_type'] must be a non-empty string")
    if "baselines" not in config:
        raise ValueError("config must include 'baselines'")
    return {"anomaly_type": anomaly_type, "baselines": validate_baselines(config["baselines"])}


def _is_consistent(shift: float, baseline: dict) -> bool:
    """A shift is consistent with a baseline when its magnitude falls in [min, max] AND,
    for any nonzero shift, its sign matches expected_sign. A shift of exactly 0 has no
    sign to violate -- only the magnitude bound applies (which a min > 0 would already
    fail on its own, so this doesn't let a genuinely-missing press through)."""
    shift = _finite_number(shift, name="shift")
    magnitude = abs(shift)
    if not (baseline["min"] <= magnitude <= baseline["max"]):
        return False
    if magnitude == 0:
        return True
    return (shift > 0) == (baseline["expected_sign"] == 1)


def _validate_segment_id(segment_id, expected_length: int) -> tuple[str, int, int]:
    """(source_id, start_idx, end_idx): source_id a non-empty string, both indexes
    non-negative integers, range length matching ``expected_length`` (the number of
    frames/rows the caller's measurements actually cover)."""
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
    if end_idx - start_idx != expected_length:
        raise ValueError("segment_id range length must match the number of measured frames")
    return source_id, start_idx, end_idx


class AnomalyChecker:
    """Explicit, user-owned accumulator (same pattern as
    ``kuration.gameplay.pixel_shift.PixelShiftReport``): classifying a clip needs to
    aggregate across
    every segment belonging to it, which a stateless per-segment SDK transform cannot do.

    Generic -- has no opinion on where instances come from. Construct it with an already
    validated config (``anomaly_checker.validate_config`` plus whatever type-specific
    checks a caller wants layered on top -- see action_state.py for the concrete
    "action_state" adapter), then call ``resolve_segment()`` once per segment and
    ``record_instance()`` once per measured instance found in it.

    Verdict per clip (grouped by segment_id's source_id):
      - INCONSISTENT if ANY recorded instance (any baselined control/cell) violates its
        baseline -- one wrong-direction or out-of-range instance makes the whole clip
        inconsistent, not just that one instance.
      - CONSISTENT if there is no violation AND at least one baselined instance was
        actually observed somewhere in the clip (real supporting evidence -- a clip
        where nothing relevant ever happened isn't "consistent", it's untested).
      - INCONCLUSIVE if no violation was found but also no baselined instance was ever
        observed in the clip -- nothing to judge either way. This is deliberately NOT the
        same bar as "every configured control fired at least once".
    """

    def __init__(self, validated_config: dict):
        """``validated_config`` must already be ``{"anomaly_type": ..., "baselines":
        ...}`` in validated shape -- callers are responsible for calling
        ``validate_config`` (plus any type-specific checks) before constructing this."""
        self.anomaly_type = validated_config["anomaly_type"]
        self.baselines = validated_config["baselines"]
        # A press's global frame position can be "owned" by more than one overlapping
        # window (see the pixel-shift guide, "Input and segment-boundary
        # contract") -- same tie-breaker PixelShiftReport uses.
        self._seen_instances: set = set()
        self._clips_seen: set = set()
        self._clip_records: dict = {}

    def resolve_segment(self, segment_id, *, expected_length: int) -> tuple[str, int, int]:
        """Validate segment_id, mark its clip as seen, and return (source_id, start_idx,
        end_idx). Call this once per segment before recording any of its instances."""
        source_id, start_idx, end_idx = _validate_segment_id(segment_id, expected_length)
        self._clips_seen.add(source_id)
        return source_id, start_idx, end_idx

    def record_instance(self, source_id: str, control: str, cell_key: str, frame: int, shift: float) -> None:
        """Check one measured instance against its configured baseline and record the
        result under its clip. Silently does nothing if there's no baseline for this
        exact cell_key (matches segment_scalars' own "not requested" semantics) -- but
        raises if `control` itself was never configured at all, since a caller reaching
        this point with an unconfigured control is an adapter bug, not absent data."""
        cells = self.baselines.get(control)
        if cells is None:
            raise KeyError(f"no baseline configured for control {control!r}")
        baseline = cells.get(cell_key)
        if baseline is None:
            return  # no baseline for this exact cell -- not checked
        shift = _finite_number(shift, name="shift")
        instance_key = (source_id, control, cell_key, frame)
        if instance_key in self._seen_instances:
            return
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
        every clip ever seen via resolve_segment(), sorted by source_id within each
        verdict."""
        verdicts: dict = {"consistent": [], "inconsistent": [], "inconclusive": []}
        for source_id in sorted(self._clips_seen):
            verdicts[self.clip_verdict(source_id)].append(source_id)
        return verdicts

    def print_report(self) -> None:
        """Print the full anomaly report: counts and exact clip IDs per verdict, plus
        which control/cell/frame violated the baseline for each inconsistent clip.
        Printed to the screen, nothing written to disk (same convention as
        PixelShiftReport.print_report())."""
        verdicts = self.summary()
        print("=" * 78)
        print(f"ANOMALY REPORT (anomaly_type={self.anomaly_type!r})")
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
