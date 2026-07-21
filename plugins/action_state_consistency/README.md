# Kuration Anomaly Detection

A generic clip-level anomaly-detection core, plus one concrete anomaly type built on top
of it: **action_state**, which checks `kuration-plugin-pixel-shift`'s per-instance pixel
shift measurements against baselines you supply, and classifies every clip in your dataset
as **CONSISTENT**, **INCONSISTENT**, or **INCONCLUSIVE**.

`kuration-plugin-pixel-shift` measures what pixel motion happened for every isolated
key/mouse press — deliberately making no judgment about whether that motion was
"consistent" with anything. `action_state` is that missing judgment layer. But this
package is **not** a wrapper around pixel_shift: the generic core
(`kuration_action_state_consistency.anomaly_checker`) has no dependency on pixel_shift at
all, and never imports it, anywhere, at any time. Only `action_state` — the specific
anomaly type that happens to check pixel-shift measurements — reaches for pixel_shift, and
only lazily, the moment you actually use it. Installing or importing this package never
requires `kuration-plugin-pixel-shift` to be installed; only constructing an
`ActionStateConsistencyChecker` (or calling `load_baselines_from_reference`) does.

## Install

```bash
# Generic anomaly-checking core only -- no pixel_shift dependency at all
pip install kuration-plugin-action-state-consistency

# With the action_state anomaly type (pulls in kuration-plugin-pixel-shift)
pip install "kuration-plugin-action-state-consistency[action_state]"
```

For repository development, install the SDK and both plugins together, then run only
this distribution's tests:

```bash
python -m pip install -e . -e 'plugins/pixel_shift[test]' -e 'plugins/action_state_consistency[test]'
python -m pytest plugins/action_state_consistency/tests -q
```

## Structure

```
kuration_action_state_consistency/
├── anomaly_checker.py   # generic core -- baseline validation, the consistency check,
│                        # clip-level verdict aggregation. Operates on plain
│                        # (source_id, control, cell_key, frame, shift) tuples. No
│                        # pixel_shift import anywhere in this file, ever.
└── action_state.py      # the "action_state" anomaly type: validates baselines against
                          # kuration_pixel_shift.ALL_CONTROLS, discovers instances via
                          # pixel_shift's own _iter_instances, and builds an
                          # AnomalyChecker on top. The ONLY file that references
                          # kuration_pixel_shift, and only inside function bodies.
```

If a future anomaly type has nothing to do with pixel shift, it can reuse
`anomaly_checker.AnomalyChecker` directly without ever installing pixel_shift — only an
`action_state`-shaped workload needs `[action_state]`.

## Config

```python
config = {
    "anomaly_type": "action_state",   # the only supported value right now
    "baselines": {
        "W": {
            "1": {"expected_sign": 1, "min": 0.001363, "max": 1.669654},
            "2": {"expected_sign": 1, "min": 0.00099,  "max": 1.632512},
            "3": {"expected_sign": 1, "min": 0.000227, "max": 1.818079},
        },
        "MOUSE_X_POS": {
            "2": {"expected_sign": 1, "min": 0.000001, "max": 0.535586},
            ...
        },
    },
}
```

- **Keys** are `kuration_pixel_shift.ALL_CONTROLS` names (`W`, `S`, `A`, `D`,
  `MOUSE_X_POS`, `MOUSE_X_NEG`, `MOUSE_Y_POS`, `MOUSE_Y_NEG`) -- checked by
  `validate_action_state_baselines`, the `action_state`-specific layer. The generic core's
  own `validate_baselines` accepts any non-empty string as a control name; it has no
  opinion on what a "control" is.
- **Cell keys** are the exact same `cell_key` pixel_shift itself reports instances
  under: press length as a string (`"1"`/`"2"`/`"3"`) for W/S/A/D, the observed mouse
  delta value as a string (e.g. `"2"`, `"-1000"`) for the mouse controls. A control/cell
  combination with no baseline configured is simply never checked — not an error, and not
  silently assumed consistent either (it just never contributes an instance to the clip's
  verdict).
- `min`/`max` are **magnitudes** (always non-negative), checked against `abs(shift)`.
  Direction is checked separately via `expected_sign` (`1` or `-1`), against `sign(shift)`
  for any nonzero shift — an exact-zero shift has no direction to violate, so only the
  magnitude bound applies to it.

### Deriving baselines empirically

`load_baselines_from_reference(reference)` builds a baselines dict directly from a
reference JSON shaped like this repo's own `scripts/build_action_consistency_reference.py`
output (`{control_lower: {cell_key: {"reliable", "floor", "ceiling", "expected_sign"}}}`):
`floor`/`ceiling` become `min`/`max`, and any cell marked `reliable: false` (not enough
reference samples to trust) is skipped entirely, rather than being checked against an
untrustworthy value. This function is part of `action_state` (it needs
`kuration_pixel_shift.ALL_CONTROLS` to map control names), so it requires the
`[action_state]` extra.

```python
import json
from kuration_action_state_consistency import load_baselines_from_reference

with open("action_consistency_reference.json") as f:
    reference = json.load(f)
baselines = load_baselines_from_reference(reference)  # or controls=["W", "S", "A", "D"]
```

That reference-building script is itself dataset-specific tooling (it derives values from
real ASG/TeaPearce reference gameplay) and is kept out of this data-agnostic plugin —
nothing stops any caller from building a baselines dict by hand instead.

## Usage

`ActionStateConsistencyChecker` is a plain, explicitly-instantiated object — same pattern
as `PixelShiftReport`, and for the same reason: classifying a clip means aggregating across
every segment that belongs to it, which a stateless per-segment SDK transform cannot do.
It is deliberately **not** registered as a kuration-sdk operator/transform.

```python
from kuration_action_state_consistency import ActionStateConsistencyChecker

checker = ActionStateConsistencyChecker(config)  # requires kuration-plugin-pixel-shift installed
for frames, actions, segment_id in your_own_segments:
    # segment_id is (source_id, start_idx, end_idx) -- same contract as
    # PixelShiftReport.add_segment. Segments sharing a source_id are one clip.
    checker.add_segment(frames, actions, segment_id, controls=["W", "S", "A", "D"])

checker.print_report()          # counts + exact clip IDs per verdict, printed to stdout
print(checker.summary())        # {"consistent": [...], "inconsistent": [...], "inconclusive": [...]}
print(checker.clip_verdict("clip_042"))  # "consistent" | "inconsistent" | "inconclusive"
```

`controls` defaults to every control with a configured baseline; passing an explicit
control not present in `baselines` raises immediately (there's nothing to check it
against). Every other keyword `add_segment()` accepts (`color_space`, `lookbehind_actions`,
`lookahead_actions`, `lookahead_frames`, `boundary`) is passed straight through to
pixel_shift's own instance-finding — see its README for what each means.

If `kuration-plugin-pixel-shift` isn't installed, constructing `ActionStateConsistencyChecker`
(or calling `load_baselines_from_reference`) raises a clear `ImportError` naming the
`[action_state]` extra to install — not a bare `ModuleNotFoundError`, and not a silent
failure elsewhere.

### Using the generic core directly

For a workload that isn't about pixel_shift at all, build on `AnomalyChecker` directly --
this never imports or requires `kuration-plugin-pixel-shift`:

```python
from kuration_action_state_consistency import AnomalyChecker, validate_config

config = validate_config({"anomaly_type": "my_other_check", "baselines": {...}})
checker = AnomalyChecker(config)
source_id, start_idx, _ = checker.resolve_segment(("clip1", 0, 5), expected_length=5)
checker.record_instance(source_id, "some_control", "some_cell", start_idx + 2, measured_value)
checker.print_report()
```

## Verdict rule

For each clip (segments grouped by `segment_id`'s `source_id`):

- **INCONSISTENT** if *any* recorded instance — any control, any cell — violates its
  baseline (wrong direction, or magnitude outside `[min, max]`). One bad press poisons the
  whole clip, not just that one instance.
- **CONSISTENT** if there is no violation *and* at least one baselined instance was
  actually observed somewhere in the clip. A clip where nothing relevant ever happened
  isn't "consistent" — it's untested.
- **INCONCLUSIVE** if no violation was found but also *no* instance of *any* baselined
  control was ever observed in the clip — nothing to judge either way.

This is deliberately **not** the same bar as "every configured control fired at least
once": a clip that only ever exercised W, with no violations, is CONSISTENT even if the
baselines also configured S/A/D and none of those ever appeared in this particular clip.

## Why the generic core and the action_state adapter are separate files (and why pixel_shift is optional)

pixel_shift and this package are **independent** — pixel_shift never depends on this
package, and this package's generic core never depends on pixel_shift. They only compose
together at the point where a caller specifically wants `action_state` anomaly detection.
An earlier version of this plugin imported `kuration_pixel_shift` at module top level and
listed it as a hard dependency, which meant merely installing or importing this package
pulled pixel_shift in regardless of whether you wanted action-state checking at all.
Splitting the generic verdict-aggregation logic (`anomaly_checker.py`) from the
pixel_shift-specific instance-discovery logic (`action_state.py`), and moving the
pixel_shift import inside the functions that actually use it, means a future anomaly type
unrelated to pixel shift can reuse the generic core with zero pixel_shift footprint, and
installing this package for that purpose never needs `kuration-plugin-pixel-shift` at all.
