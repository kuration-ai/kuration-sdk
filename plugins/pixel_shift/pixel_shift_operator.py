"""pixel_shift_operator -- Operator 1 of 2: measures pixel shift for every isolated key
press (once/twice/thrice) and every isolated mouse tap found in a dataset. Reports what
happened -- makes NO judgment about whether it's "consistent." That's Operator 2's job
(a separate, genuinely data-agnostic outlier-finding operator built on top of this one's
output), kept deliberately out of this file.

Two things this operator produces:

1. Per-segment scalar features (so it composes inside a real kuration pipeline --
   .bin()/.filter()/.sample() on the results): for each of the 8 controls, 4 numbers --
   `{control}_shift_min`, `{control}_shift_max`, `{control}_shift_mean`,
   `{control}_n_instances` -- collapsing across whatever press-lengths (W/S/A/D) or delta
   magnitudes (mouse controls) were found for that control in that segment. min/max are
   the true worst-case/best-case signed shift among however many instances were found (NOT
   an average that could hide one bad instance among several good ones) -- see the second
   deliverable below for the full per-instance detail these 4 numbers are compressed from.

2. A console report (print_report(), NOT a written file) -- for every (control, press_len
   or delta) combination, the exact instance count found across the whole dataset, an
   ASCII histogram of their shift values, and which clips each instance came from. Same
   style as docs/batch_results_sensitivity_w.md / batch_job_sensitivity_w.py's own
   report(), just clip-traceable and produced by a discoverable SDK operator instead of a
   one-off batch script.

CAVEAT on the ledger backing print_report(): it accumulates in a module-level dict across
`transform_batch` calls, which only works correctly for a single-process run
(`.options(workers=1)`, the default) -- an SDK executor with workers>1 would each hold
their own separate copy in a separate process, exactly the same limitation
batch_job_sensitivity_w.py works around with explicit per-worker-result merging (see its
`merge()`). Reset with `reset_ledger()` between separate runs in the same process.
"""
from __future__ import annotations

import numpy as np

from .diamond_action_decode import ALL_CONTROLS, release_transition_displacement

KEY_CONTROLS = ["W", "S", "A", "D"]
MOUSE_CONTROLS_LIST = ["MOUSE_X_POS", "MOUSE_X_NEG", "MOUSE_Y_POS", "MOUSE_Y_NEG"]
PRESS_LENS = [1, 2, 3]

STATS_PER_CONTROL = ("shift_min", "shift_max", "shift_mean", "n_instances")
OUTPUT_KEYS = tuple(f"{control.lower()}_{stat}" for control in ALL_CONTROLS for stat in STATS_PER_CONTROL)

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

# ---------------------------------------------------------------------------------------
# Ledger: every instance found, across the whole dataset, for print_report(). Separate from
# the per-segment scalar features above -- those are bound by the SDK's "one finite float
# per segment per output name" rule; this isn't a pipeline feature, so it isn't.
#   LEDGER[control][cell_key] -> list of {"clip": "source_id[start:end]", "shift": float}
# cell_key is press_len as a string ("1"/"2"/"3") for W/S/A/D, the observed delta value as
# a string (e.g. "2", "-1000") for mouse controls.
# ---------------------------------------------------------------------------------------
LEDGER: dict = {}


def reset_ledger() -> None:
    LEDGER.clear()


def _record(control: str, cell_key: str, clip_label: str, shift: float) -> None:
    LEDGER.setdefault(control, {}).setdefault(cell_key, []).append({"clip": clip_label, "shift": shift})


def _clip_label(segment_id) -> str:
    """A clip ID readable on its own: which source clip, and which frame range within it --
    not a bracket/slice-looking notation that reads like code."""
    source_id, start_idx, end_idx = segment_id
    return f"clip '{source_id}', frames {start_idx}-{end_idx}"


def instances_for_segment(frames: np.ndarray, actions: np.ndarray, control: str, clip_label: str,
                           color_space: str = "BGR") -> list:
    """Every instance of `control` found in this one segment -- records each into LEDGER
    under its exact cell_key (press_len or delta), and returns the flat list of signed
    shifts (collapsed across cell_keys) this control's 4 per-segment scalars are computed
    from. `color_space` (see diamond_action_decode._prep_gray) defaults to BGR, matching
    raw ASG/DIAMOND HDF5 arrays; pass the real declared color_space when `frames` came
    through a kuration-sdk media column instead."""
    from .diamond_action_decode import MOUSE_CONTROLS, MOUSE_AXIS_OF, mouse_delta_value

    shifts = []
    if control in MOUSE_CONTROLS:
        axis = MOUSE_AXIS_OF[control]
        for start, px in release_transition_displacement(frames, actions, control, press_len=1,
                                                           color_space=color_space):
            delta_value = int(mouse_delta_value(actions[start:start + 1], axis)[0])
            _record(control, str(delta_value), clip_label, px)
            shifts.append(px)
    else:
        for press_len in PRESS_LENS:
            for start, px in release_transition_displacement(frames, actions, control, press_len,
                                                               color_space=color_space):
                _record(control, str(press_len), clip_label, px)
                shifts.append(px)
    return shifts


def segment_scalars(frames: np.ndarray, actions: np.ndarray, clip_label: str,
                     controls: list = None, color_space: str = "BGR") -> dict:
    """The 4-number-per-control summary for ONE segment -- true to the data (no reference,
    no consistency judgment): the actual worst/best/mean signed shift and how many
    instances were found, collapsed across whatever press-lengths/deltas occurred for each
    control in this segment. `color_space` defaults to BGR (raw ASG/DIAMOND HDF5 arrays);
    PixelShift.transform_batch passes the real declared color_space of the SDK media
    column instead (see diamond_action_decode._prep_gray)."""
    controls = controls or ALL_CONTROLS
    out = {}
    for control in ALL_CONTROLS:
        prefix = control.lower()
        if control not in controls:
            out[f"{prefix}_shift_min"] = 0.0
            out[f"{prefix}_shift_max"] = 0.0
            out[f"{prefix}_shift_mean"] = 0.0
            out[f"{prefix}_n_instances"] = 0.0
            continue
        shifts = instances_for_segment(frames, actions, control, clip_label, color_space)
        if shifts:
            out[f"{prefix}_shift_min"] = float(min(shifts))
            out[f"{prefix}_shift_max"] = float(max(shifts))
            out[f"{prefix}_shift_mean"] = float(sum(shifts) / len(shifts))
            out[f"{prefix}_n_instances"] = float(len(shifts))
        else:
            # 0.0 is a placeholder, not a claim of "zero shift" -- n_instances=0.0 is the
            # authoritative signal; a downstream consumer must check it before trusting
            # min/max/mean, exactly like the earlier -1.0 sentinel pattern, just per-control
            # here instead of one dataset-wide flag.
            out[f"{prefix}_shift_min"] = 0.0
            out[f"{prefix}_shift_max"] = 0.0
            out[f"{prefix}_shift_mean"] = 0.0
            out[f"{prefix}_n_instances"] = 0.0
    return out


# ---------------------------------------------------------------------------------------
# Console report -- printed, not written to a file. Same shape as
# docs/batch_results_sensitivity_w.md / batch_job_sensitivity_w.py's report(): per
# (control, cell_key), instance count + an ASCII histogram of shift values + which clips.
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


def print_report(controls: list = None) -> None:
    """Print the full pixel-shift report accumulated in LEDGER so far -- instance counts,
    histograms, and clip provenance, per (control, press_len/delta). Distribution-only
    style: printed to the screen, nothing written to disk."""
    controls = controls or ALL_CONTROLS
    print("=" * 78)
    print("PIXEL SHIFT REPORT -- isolated key/mouse presses found across the dataset")
    print("=" * 78)
    for control in controls:
        name = CONTROL_DESCRIPTIONS.get(control, control)
        cells = LEDGER.get(control, {})
        if not cells:
            print(f"\n{name}: no isolated instances found")
            continue
        print(f"\n{name}")
        for cell_key in sorted(cells, key=lambda k: (len(k), k)):
            entries = cells[cell_key]
            shifts = [e["shift"] for e in entries]
            clip_ids = sorted({e["clip"] for e in entries})
            label = "held for" if control in KEY_CONTROLS else "mouse delta"
            unit = " frame(s)" if control in KEY_CONTROLS else ""
            print(f"  {label} {cell_key}{unit}  --  n_instances={len(entries)}  "
                  f"mean={sum(shifts) / len(shifts):.4f}  min={min(shifts):.4f}  max={max(shifts):.4f}")
            for edge_label, count, bar in _ascii_histogram(shifts):
                print(f"    {edge_label:>24}  {count:>5}  {bar}")
            clip_preview = "; ".join(clip_ids[:8]) + (f"; ... (+{len(clip_ids) - 8} more)" if len(clip_ids) > 8 else "")
            print(f"    Clip ID(s): {clip_preview}")
    print()


# ---------------------------------------------------------------------------------------
# kuration-sdk registration -- validated against the real installed package the same way
# action_state_consistency.py's operator was (pip install + list_components/
# describe_component + a real end-to-end run against a real MP4 + real ASG telemetry).
# ---------------------------------------------------------------------------------------

try:
    from kuration import register_batch_transform, register_operator
except ImportError:
    register_batch_transform = register_operator = None  # SDK not installed in this checkout


if register_batch_transform is not None:

    @register_batch_transform(
        "aintropy/pixel-shift",
        description="Per-segment pixel-shift statistics (min/max/mean/n_instances per "
                     "control) for every isolated key/mouse press found in the window -- "
                     "no consistency judgment, just what pixel motion happened",
        outputs=OUTPUT_KEYS,
        inputs=("media:frame", "field:action"),
        media_requirements={"kind": "video-frame", "layout": "HWC"},
        resources={"num_gpus": 0},
        version="1",
    )
    class PixelShift:
        def __init__(self, controls: list = None):
            self.controls = controls  # None = all 8

        def transform_batch(self, batch, ctx):
            frames_batch = batch["media:frame"]       # (N, T, H, W, 3)
            actions_batch = batch["field:action"]     # (N, T, action_dim)
            segment_ids = batch["segment_ids"]        # (N,) of (source_id, start_idx, end_idx)
            # The real declared color_space of the "frame" media column -- populated by the
            # SDK runtime (transforms.py's ctx["schema"]), not assumed. kuration-sdk's
            # .join_video() always tags this "RGB" (its PyAV provider decodes rgb24); a raw
            # BGR-sourced media column (e.g. hand-built HDF5 adapter) would say "BGR"
            # instead. Passed straight through to _prep_gray, never used to reorder frames.
            color_space = ctx["schema"].media["frame"].color_space
            rows = [
                segment_scalars(frames, actions, _clip_label(seg_id), self.controls, color_space)
                for frames, actions, seg_id in zip(frames_batch, actions_batch, segment_ids)
            ]
            return {key: np.array([row[key] for row in rows], dtype=np.float64) for key in OUTPUT_KEYS}

    @register_operator(
        "aintropy/pixel-shift",
        description="Measure pixel shift for every isolated key/mouse press -- no "
                     "consistency judgment, just what happened",
        parameters={
            "type": "object",
            "properties": {
                "controls": {"type": "array", "items": {"type": "string"},
                             "description": "subset of W/S/A/D/MOUSE_X_POS/.../MOUSE_Y_NEG "
                             "to check; default is all 8"},
            },
            "additionalProperties": False,
        },
        version="1",
    )
    def pixel_shift(pipeline, *, controls=None):
        return pipeline.transform("aintropy/pixel-shift", controls=controls)
