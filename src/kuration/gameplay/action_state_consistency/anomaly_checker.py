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
            if not isinstance(cell_key, str) or not cell_key:
                raise ValueError(
                    f"baseline cell keys must be non-empty strings, got "
                    f"{cell_key!r} for {control!r}"
                )
            out[control][cell_key] = _validate_baseline_cell(control, cell_key, cell)
    return out


#: Self-calibration knobs and their defaults. ``min_clips``: the smallest pool that can
#: define a consensus at all; ``min_samples``: measurements a (control, cell) needs from
#: OTHER clips before it is judged; ``sign_consensus``: the fraction of nonzero shifts that
#: must agree on a direction (a contested cell is skipped, never guessed); ``tolerance``:
#: how far the derived magnitude bounds are widened beyond the observed 5th/95th percentile.
DEFAULT_CALIBRATION = {
    "min_clips": 3,
    "min_samples": 5,
    "sign_consensus": 0.8,
    "tolerance": 1.5,
}

#: The observed-magnitude percentiles the derived [min, max] bounds start from, before
#: ``tolerance`` widening. Not a knob: robust-bound percentiles are a statistical choice the
#: calibration should own, not a lever every caller re-decides.
_BOUND_QUANTILES = (0.05, 0.95)


def validate_calibration(calibration) -> dict:
    """Validate self-calibration knob overrides and fill in the documented defaults."""
    if calibration is None:
        return dict(DEFAULT_CALIBRATION)
    if not isinstance(calibration, dict):
        raise ValueError("calibration must be a dict of knob overrides")
    unknown = sorted(set(calibration) - set(DEFAULT_CALIBRATION))
    if unknown:
        raise ValueError(
            f"unknown calibration knob(s) {unknown}; expected {sorted(DEFAULT_CALIBRATION)}"
        )
    out = dict(DEFAULT_CALIBRATION)
    for key, value in calibration.items():
        if key in ("min_clips", "min_samples"):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2:
                raise ValueError(f"calibration {key} must be an integer >= 2")
            out[key] = int(value)
        elif key == "sign_consensus":
            consensus = _finite_number(value, name="calibration sign_consensus")
            if not 0.5 < consensus <= 1.0:
                raise ValueError("calibration sign_consensus must be in (0.5, 1.0]")
            out[key] = consensus
        else:  # tolerance
            tolerance = _finite_number(value, name="calibration tolerance")
            if tolerance < 1.0:
                raise ValueError("calibration tolerance must be >= 1")
            out[key] = tolerance
    return out


def derive_baselines(
    samples,
    *,
    # int() because DEFAULT_CALIBRATION mixes ints and floats, so its value type
    # widens to float; the count knobs stay ints without duplicating the defaults.
    min_samples: int = int(DEFAULT_CALIBRATION["min_samples"]),
    sign_consensus: float = DEFAULT_CALIBRATION["sign_consensus"],
    tolerance: float = DEFAULT_CALIBRATION["tolerance"],
) -> dict:
    """Derive consensus baselines from pooled measurements, skipping what can't be trusted.

    ``samples`` maps ``(control, cell_key)`` to the pooled shifts measured for that cell. A
    cell yields a baseline only when it has at least ``min_samples`` measurements, at least
    one of them nonzero, and at least ``sign_consensus`` of the nonzero ones agree on a
    direction; anything underpowered or contested is omitted (its instances stay unjudged)
    rather than judged against a guess. ``expected_sign`` is the consensus direction and
    ``min``/``max`` are the 5th/95th percentile of the cell's ``|shift|`` widened by
    ``tolerance`` — note the bounds include every sample, so a mostly-zero cell derives a
    zero floor and the magnitude check degrades to conservative rather than accusatory.
    """
    derived: dict = {}
    for (control, cell_key), shifts in samples.items():
        values = np.asarray([_finite_number(s, name="shift") for s in shifts], dtype=float)
        if len(values) < min_samples:
            continue
        nonzero = values[values != 0]
        if len(nonzero) == 0:
            continue
        positive = int((nonzero > 0).sum())
        negative = len(nonzero) - positive
        if max(positive, negative) / len(nonzero) < sign_consensus:
            continue
        magnitudes = np.abs(values)
        derived.setdefault(control, {})[cell_key] = {
            "expected_sign": 1 if positive >= negative else -1,
            "min": float(np.quantile(magnitudes, _BOUND_QUANTILES[0])) / tolerance,
            "max": float(np.quantile(magnitudes, _BOUND_QUANTILES[1])) * tolerance,
        }
    return derived


def validate_config(config) -> dict:
    """Validate the generic config shape: ``{"anomaly_type": <non-empty string>,
    "baselines": {...} | None}``. Does not restrict which ``anomaly_type`` values are legal --
    that's entirely a specific adapter's concern (e.g. action_state.py's ANOMALY_TYPES),
    since the generic core has no opinion on what anomaly types exist.

    ``baselines=None`` selects **self-calibrated** mode: every clip is judged against
    baselines derived from the other clips' own measurements (see ``AnomalyChecker``), tuned
    by the optional ``calibration`` knobs. Explicit baselines reject ``calibration`` loudly —
    the knobs have nothing to calibrate.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")
    anomaly_type = config.get("anomaly_type")
    if not isinstance(anomaly_type, str) or not anomaly_type:
        raise ValueError("config['anomaly_type'] must be a non-empty string")
    if "baselines" not in config:
        raise ValueError("config must include 'baselines' (a dict, or None to self-calibrate)")
    if config["baselines"] is None:
        return {
            "anomaly_type": anomaly_type,
            "baselines": None,
            "calibration": validate_calibration(config.get("calibration")),
        }
    if config.get("calibration") is not None:
        raise ValueError(
            "calibration applies only to self-calibrated mode; drop it or drop baselines"
        )
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
    frames/rows the caller's measurements actually cover).

    NumPy integers are accepted (an SDK-provided ``segment_ids`` entry carries them) and
    normalized to ``int`` so recorded frame keys have one canonical type.
    """
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
        or not isinstance(start_idx, (int, np.integer))
        or not isinstance(end_idx, (int, np.integer))
        or start_idx < 0
        or end_idx < 0
    ):
        raise ValueError("segment_id start_idx and end_idx must be non-negative integers")
    start_idx, end_idx = int(start_idx), int(end_idx)
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

    **Self-calibrated mode** (``baselines=None`` in the config): no external truth is
    supplied, so instances are recorded unjudged and every clip is judged at verdict time
    against baselines derived from the OTHER clips' measurements (leave-one-out — a clip
    never vouches for itself). This detects *relative* outliers: the clip that disagrees
    with the corpus. It structurally cannot catch a corpus-wide problem — if every clip is
    desynced the same way, the consensus IS the desync — which is why supplied baselines
    remain the externally-calibrated ground truth and self-calibration is a labeled mode,
    never a silent fallback. Cells without enough independent evidence (see
    ``DEFAULT_CALIBRATION``) stay unjudged, and pools smaller than ``min_clips`` refuse to
    issue verdicts at all.
    """

    def __init__(self, validated_config: dict):
        """``validated_config`` must already be ``{"anomaly_type": ..., "baselines":
        ...}`` in validated shape -- callers are responsible for calling
        ``validate_config`` (plus any type-specific checks) before constructing this."""
        self.anomaly_type = validated_config["anomaly_type"]
        self.baselines = validated_config["baselines"]
        # None is the supplied-baselines marker: there are no self-calibration knobs in
        # that mode. Otherwise fill the documented defaults, so a config that omits
        # "calibration" entirely still gets a usable dict rather than a None deref later.
        self.calibration: dict | None = (
            validate_calibration(validated_config.get("calibration"))
            if self.baselines is None
            else None
        )
        self._judged_cache: dict | None = None
        # A press's global frame position can be "owned" by more than one overlapping
        # window (see the pixel-shift guide, "Input and segment-boundary
        # contract") -- same tie-breaker PixelShiftReport uses.
        self._seen_instances: set = set()
        self._clips_seen: set = set()
        self._clip_records: dict = {}
        # What a recorded `frame` counts in. "source" means real source frame numbers, which is
        # what the per-segment API provides and what cross-window de-duplication relies on. A
        # caller measuring in some other coordinate space must say so (declare_frame_space), so
        # a printed report never presents synthetic positions as source frames.
        self.frame_space = "source"
        self.frame_space_note = ""

    def declare_frame_space(self, label: str, note: str) -> None:
        """Declare the coordinate space every recorded ``frame`` value belongs to.

        ``label`` is a short identifier ("source" for real source frame numbers) and ``note``
        is the caveat printed with the report. Changing the space once instances have been
        recorded would mix incomparable coordinates in one ledger, so that raises.
        """
        if not isinstance(label, str) or not label:
            raise ValueError("frame space label must be a non-empty string")
        if not isinstance(note, str) or (not note and label != "source"):
            raise ValueError("a non-source frame space must document its caveat")
        if label != self.frame_space and self._seen_instances:
            raise ValueError(
                f"cannot redeclare the frame space as {label!r} after recording instances in "
                f"{self.frame_space!r}")
        self.frame_space = label
        self.frame_space_note = note

    def resolve_segment(self, segment_id, *, expected_length: int) -> tuple[str, int, int]:
        """Validate segment_id, mark its clip as seen, and return (source_id, start_idx,
        end_idx). Call this once per segment before recording any of its instances."""
        source_id, start_idx, end_idx = _validate_segment_id(segment_id, expected_length)
        self._clips_seen.add(source_id)
        self._judged_cache = None
        return source_id, start_idx, end_idx

    @property
    def self_calibrated(self) -> bool:
        return self.baselines is None

    def record_instance(
        self,
        source_id: str,
        control: str,
        cell_key: str,
        frame: int,
        shift: float,
    ) -> None:
        """Check one measured instance against its configured baseline and record the
        result under its clip. Silently does nothing if there's no baseline for this
        exact cell_key (matches segment_scalars' own "not requested" semantics) -- but
        raises if `control` itself was never configured at all, since a caller reaching
        this point with an unconfigured control is an adapter bug, not absent data.

        In self-calibrated mode there is no baseline to check against yet: every instance
        is recorded unjudged (``consistent=None``) and judged leave-one-out at verdict
        time."""
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("instance source_id must be a non-empty string")
        if source_id not in self._clips_seen:
            raise ValueError(
                f"clip {source_id!r} was not resolved; call resolve_segment() "
                "before recording its instances"
            )
        if not isinstance(control, str) or not control:
            raise ValueError("instance control must be a non-empty string")
        if not isinstance(cell_key, str) or not cell_key:
            raise ValueError("instance cell_key must be a non-empty string")
        if (
            isinstance(frame, bool)
            or not isinstance(frame, (int, np.integer))
            or frame < 0
        ):
            raise ValueError("instance frame must be a non-negative integer")
        frame = int(frame)
        if self.baselines is None:
            baseline = None  # judged leave-one-out at verdict time
        else:
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
            "consistent": None if baseline is None else _is_consistent(shift, baseline),
        }
        self._judged_cache = None
        self._clip_records.setdefault(source_id, []).append(record)

    def _calibration_knobs(self) -> dict:
        """The active self-calibration knobs.

        Every caller is already past a ``self.baselines is not None`` guard, so this
        only narrows away the supplied-baselines ``None``; reaching it in that mode is a
        bug rather than a user error.
        """
        calibration = self.calibration
        if calibration is None:
            raise ValueError("baselines were supplied; there are no self-calibration knobs")
        return calibration

    def resolved_baselines_for_clip(self, source_id: str) -> dict:
        """The baseline dict this clip is actually judged against: the supplied baselines
        verbatim, or -- in self-calibrated mode -- the leave-one-out consensus derived from
        every *other* clip's measurements, so a clip never vouches for itself. Enforces the
        same ``min_clips`` floor as :meth:`summary`/:meth:`clip_verdict`. Exposed as its own
        method (not just inlined in :meth:`_judged_records`) so a caller that needs to judge
        something finer-grained than a whole clip -- e.g. one window's own instances -- can
        resolve the same baseline a clip-level verdict would use, without re-deriving it."""
        if self.baselines is not None:
            return self.baselines
        if source_id not in self._clips_seen:
            raise KeyError(f"no segments recorded for clip {source_id!r}")
        min_clips = self._calibration_knobs()["min_clips"]
        if len(self._clips_seen) < min_clips:
            raise ValueError(
                f"self-calibration needs at least {min_clips} clips to define a consensus; "
                f"saw {len(self._clips_seen)} -- supply baselines= for pools this small"
            )
        knobs = {key: self._calibration_knobs()[key]
                 for key in ("min_samples", "sign_consensus", "tolerance")}
        samples: dict = {}
        for other_id, other_records in self._clip_records.items():
            if other_id == source_id:
                continue
            for record in other_records:
                samples.setdefault(
                    (record["control"], record["cell_key"]), []
                ).append(record["shift"])
        return derive_baselines(samples, **knobs)

    def is_instance_consistent(
        self, source_id: str, control: str, cell_key: str, shift: float,
        *, baselines: dict | None = None,
    ) -> bool | None:
        """Whether one instance would be judged consistent under this clip's resolved
        baseline (supplied, or leave-one-out self-calibrated) -- without adding it to the
        ledger or affecting de-duplication. ``None`` if no baseline is configured/derivable
        for this exact ``(control, cell_key)``, matching :meth:`record_instance`'s own
        "not checked" semantics. Use this for judging something finer-grained than a whole
        clip (e.g. one window's own instances, independent of what other overlapping
        windows also happen to contain) -- :meth:`record_instance` is the right call
        instead when the goal is the clip-level ledger itself. Pass an already-resolved
        ``baselines`` (from :meth:`resolved_baselines_for_clip`) to judge several instances
        of the same clip without re-deriving self-calibrated consensus each time -- that
        derivation pools every other clip's records, so it is not free."""
        if baselines is None:
            baselines = self.resolved_baselines_for_clip(source_id)
        baseline = baselines.get(control, {}).get(cell_key)
        if baseline is None:
            return None
        return _is_consistent(shift, baseline)

    def _judged_records(self) -> dict:
        """Per-clip records with ``consistent`` resolved.

        Supplied-baselines mode returns the ledger as recorded (judged eagerly). In
        self-calibrated mode each clip is judged against baselines derived from every
        *other* clip's measurements, so one bad clip cannot vouch for itself; instances in
        cells without enough independent evidence keep ``consistent=None`` and carry no
        verdict weight.
        """
        if self.baselines is not None:
            return self._clip_records
        if self._judged_cache is not None:
            return self._judged_cache
        min_clips = self._calibration_knobs()["min_clips"]
        if len(self._clips_seen) < min_clips:
            raise ValueError(
                f"self-calibration needs at least {min_clips} clips to define a consensus; "
                f"saw {len(self._clips_seen)} -- supply baselines= for pools this small"
            )
        judged: dict = {}
        for source_id in self._clips_seen:
            records = self._clip_records.get(source_id, [])
            if not records:
                judged[source_id] = []
                continue
            baselines = self.resolved_baselines_for_clip(source_id)
            resolved = []
            for record in records:
                baseline = baselines.get(record["control"], {}).get(record["cell_key"])
                resolved.append({
                    **record,
                    "consistent": None if baseline is None
                    else _is_consistent(record["shift"], baseline),
                })
            judged[source_id] = resolved
        self._judged_cache = judged
        return judged

    def derived_baselines(self) -> dict:
        """The pooled (all-clips) consensus baselines in self-calibrated mode.

        This is the reportable summary of what "normal" was inferred to be; verdicts
        themselves are judged leave-one-out, so a clip's own measurements never appear in
        the baselines it is checked against. Raises with supplied baselines -- nothing was
        derived.
        """
        if self.baselines is not None:
            raise ValueError("baselines were supplied; nothing was derived")
        self._judged_records()  # enforce the min_clips floor before summarizing a consensus
        samples: dict = {}
        for records in self._clip_records.values():
            for record in records:
                samples.setdefault(
                    (record["control"], record["cell_key"]), []
                ).append(record["shift"])
        knobs = {key: self._calibration_knobs()[key]
                 for key in ("min_samples", "sign_consensus", "tolerance")}
        return derive_baselines(samples, **knobs)

    def calibration_report(self) -> dict | None:
        """The embeddable self-calibration record (knobs + pooled consensus), or ``None``
        when baselines were supplied. Results that judged clips against inferred normality
        must say so -- this is what analyzers attach next to their verdicts."""
        if self.baselines is not None:
            return None
        return {
            "mode": "self-calibrated",
            **self._calibration_knobs(),
            "derived_baselines": self.derived_baselines(),
        }

    def clip_verdict(self, source_id: str) -> str:
        """"consistent" | "inconsistent" | "inconclusive" for one clip.

        Only judged instances carry verdict weight: a self-calibrated instance whose cell
        had too little independent evidence (``consistent=None``) is not treated as
        supporting evidence, so a clip with nothing but unjudged instances stays
        inconclusive."""
        if source_id not in self._clips_seen:
            raise KeyError(f"no segments recorded for clip {source_id!r}")
        records = self._judged_records().get(source_id, [])
        judged = [r for r in records if r["consistent"] is not None]
        if not judged:
            return "inconclusive"
        if any(not r["consistent"] for r in judged):
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

    def to_records(self) -> list[dict]:
        """Export the accumulated ledger as flat, per-instance rows.

        The row schema is identical to ``PixelShiftReport.to_records()`` so callers can
        concatenate the two lists directly. ``consistent`` and ``verdict`` contain this
        checker's judgment; pixel-shift-only rows leave them ``None``. ``n_instances`` is
        the total number of recorded instances for the row's ``source_id``, repeated on
        each row so it has the same meaning in both exports.

        Rows are sorted by ``(source_id, frame, control, cell_key)``. A source with no
        recorded instance still gets exactly one inconclusive row with zero
        ``n_instances`` so it is not silently absent. ``frame_space`` and
        ``frame_space_note`` travel with every row because dataset-mode positions may be
        shard-relative rather than real source frames.

        Does not print anything; see ``print_report()`` for the human-readable report.
        """
        rows = []
        judged_records = self._judged_records()
        for source_id in sorted(self._clips_seen):
            records = judged_records.get(source_id, [])
            verdict = self.clip_verdict(source_id)
            if not records:
                rows.append(
                    {
                        "source_id": source_id,
                        "control": None,
                        "cell_key": None,
                        "frame": None,
                        "shift": None,
                        "consistent": None,
                        "verdict": verdict,
                        "n_instances": 0,
                        "frame_space": self.frame_space,
                        "frame_space_note": self.frame_space_note,
                    }
                )
                continue
            for record in sorted(
                records,
                key=lambda item: (item["frame"], item["control"], item["cell_key"]),
            ):
                rows.append(
                    {
                        "source_id": source_id,
                        "control": record["control"],
                        "cell_key": record["cell_key"],
                        "frame": record["frame"],
                        "shift": record["shift"],
                        "consistent": record["consistent"],
                        "verdict": verdict,
                        "n_instances": len(records),
                        "frame_space": self.frame_space,
                        "frame_space_note": self.frame_space_note,
                    }
                )
        return rows

    def print_report(self) -> None:
        """Print the full anomaly report: counts and exact clip IDs per verdict, plus
        which control/cell/frame violated the baseline for each inconsistent clip.
        Printed to the screen, nothing written to disk (same convention as
        PixelShiftReport.print_report()). See ``to_records()`` for this same data as
        plain, unprinted structured rows."""
        verdicts = self.summary()
        judged_records = self._judged_records()
        print("=" * 78)
        print(f"ANOMALY REPORT (anomaly_type={self.anomaly_type!r})")
        print("=" * 78)
        if self.baselines is None:
            print("baselines: self-calibrated -- clips judged against the other clips' "
                  "consensus (leave-one-out), not an external reference")
        if self.frame_space != "source":
            print(f"frame space: {self.frame_space} -- {self.frame_space_note}")
        for verdict in ("consistent", "inconsistent", "inconclusive"):
            clip_ids = verdicts[verdict]
            print(f"\n{verdict.upper()}: {len(clip_ids)} clip(s)")
            for clip_id in clip_ids:
                if verdict == "inconsistent":
                    violations = [r for r in judged_records[clip_id]
                                  if r["consistent"] is False]
                    # Outside the "source" frame space the number is not a source frame, so it
                    # is labeled with the space it actually belongs to.
                    position = ("frame" if self.frame_space == "source"
                                else f"{self.frame_space}-offset")
                    detail = "; ".join(
                        f"{r['control']}/{r['cell_key']}@{position}{r['frame']} "
                        f"shift={r['shift']:.4f}"
                        for r in violations
                    )
                    print(f"  - {clip_id}  ({len(violations)} violation(s): {detail})")
                else:
                    print(f"  - {clip_id}")
        print()
