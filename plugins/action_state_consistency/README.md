# Kuration Action State Consistency

Operator 2 of 2. `kuration-plugin-pixel-shift` measures what pixel motion happened for
every isolated key/mouse press — deliberately making no judgment about whether that
motion was "consistent" with anything. This plugin is that missing judgment layer: it
checks pixel_shift's per-instance measurements against baselines you supply, and
classifies every clip in your dataset as **CONSISTENT**, **INCONSISTENT**, or
**INCONCLUSIVE**.

Data-agnostic, like pixel_shift itself — nothing here is hardcoded to ASG, DIAMOND, or
any specific reference dataset. Every baseline (an expected direction and a `[min, max]`
magnitude range, per control and per press-length/mouse-delta cell) is config you supply.

## Install

```bash
pip install kuration-plugin-action-state-consistency
```

For repository development, install the SDK and both plugins together, then run only
this distribution's tests:

```bash
python -m pip install -e . -e 'plugins/pixel_shift[test]' -e 'plugins/action_state_consistency[test]'
python -m pytest plugins/action_state_consistency/tests -q
```

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
  `MOUSE_X_POS`, `MOUSE_X_NEG`, `MOUSE_Y_POS`, `MOUSE_Y_NEG`).
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
untrustworthy value.

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

checker = ActionStateConsistencyChecker(config)
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

## Why this is a separate distribution from pixel_shift

Same reasoning as `plugins/_template`'s convention: this plugin's whole purpose (baseline
comparison, clip-level judgment) is a distinct, optional capability layered on top of
pixel_shift's measurements, not something every pixel_shift user needs. It depends on
`kuration-plugin-pixel-shift` and reuses its instance-finding directly (`_iter_instances`,
`_validated_segment_inputs`) rather than re-deriving any of the isolated-press-detection,
boundary-handling, or optical-flow logic pixel_shift already got right.
