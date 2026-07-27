# Kuration Anomaly Detection

A generic clip-level anomaly-detection core, plus one concrete anomaly type built on top
of it: **action_state**, which checks Kuration's pixel-shift operator's per-instance pixel
shift measurements against baselines you supply, and classifies every clip in your dataset
as **CONSISTENT**, **INCONSISTENT**, or **INCONCLUSIVE**.

`kuration.gameplay.pixel_shift` measures what pixel motion happened for every isolated
key/mouse press — deliberately making no judgment about whether that motion was
"consistent" with anything. `action_state` is that missing judgment layer. But this
module is **not** a wrapper around pixel_shift: the generic core
(`kuration.gameplay.action_state_consistency.anomaly_checker`) has no dependency on optical flow.
Only `action_state` composes with the built-in pixel-shift measurement API.

## Install

```bash
pip install "kuration-sdk[action-state]"
```

Use `kuration-sdk[gameplay]` to install this analyzer together with Pixel Shift and SLAM.
The analyzer, validation, and aggregation code ships in the base SDK. The extra installs
OpenCV for measurements made from frames. For repository development:

```bash
python -m pip install -e ".[dev,action-state]"
python -m pytest tests/test_action_state_consistency.py tests/test_anomaly_checker.py -q
```

## What gets registered with kuration-sdk

The SDK registers this capability as the built-in analyzer
`aintropy/action-state-consistency`. An analyzer is a discoverable, user-owned stateful
object: unlike a batch transform, it may accumulate evidence across multiple segments
before `summary()` can return a result. Kuration validates and constructs it through
`get_analyzer()` but does not pretend it is an independently schedulable per-segment stage.

Discover it without reading a dataset:

```bash
kur-list analyzers
kur-describe analyzer aintropy/action-state-consistency --json
```

Registration only stores a dependency-light factory; it does not run optical flow.
Discovery and analyzer construction therefore work without OpenCV.

## Structure

```
kuration/gameplay/action_state_consistency/
├── action_state.py      # the "action_state" anomaly type: validates baselines against
│                        # kuration.gameplay.pixel_shift.ALL_CONTROLS, discovers instances via
│                        # pixel_shift's public iter_instances API, and builds an
│                        # AnomalyChecker on top.
└── anomaly_checker.py   # generic core -- baseline validation, the consistency check,
                         # clip-level verdict aggregation. Operates on plain
                         # (source_id, control, cell_key, frame, shift) tuples. No
                         # pixel_shift import anywhere in this file, ever.
```

If a future anomaly type has nothing to do with pixel shift, it can reuse
`anomaly_checker.AnomalyChecker` directly without installing OpenCV.

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
    },
}
```

- **Keys** are `kuration.gameplay.pixel_shift.ALL_CONTROLS` names (`W`, `S`, `A`, `D`,
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
`kuration.gameplay.pixel_shift.ALL_CONTROLS` to map control names).

```python
import json
from kuration.gameplay.action_state_consistency import load_baselines_from_reference

with open("action_consistency_reference.json") as f:
    reference = json.load(f)
baselines = load_baselines_from_reference(reference)  # or controls=["W", "S", "A", "D"]
```

That reference-building script is itself dataset-specific tooling (it derives values from
real ASG/TeaPearce reference gameplay) and is kept out of the reusable gameplay domain —
nothing stops any caller from building a baselines dict by hand instead.

## Usage

`ActionStateConsistencyChecker` is a user-owned analyzer — same ownership pattern as
`PixelShiftReport`, and for the same reason: classifying a clip means aggregating across
every segment that belongs to it, which a stateless per-segment SDK transform cannot do.
The normal Python path needs only the SDK import:

```python
import kuration.gameplay as gameplay

checker = gameplay.action_state_checker(**config)
for frames, actions, segment_id in your_own_segments:
    # segment_id is (source_id, start_idx, end_idx) -- same contract as
    # PixelShiftReport.add_segment. Segments sharing a source_id are one clip.
    checker.add_segment(frames, actions, segment_id)

checker.print_report()          # counts + exact clip IDs per verdict, printed to stdout
print(checker.summary())        # {"consistent": [...], "inconsistent": [...], "inconclusive": [...]}
print(checker.clip_verdict("clip_042"))  # "consistent" | "inconsistent" | "inconclusive"
```

### Running on a curated dataset from the CLI

If a written dataset contains frame and DIAMOND-action modalities, the same analyzer can be
driven by `kur-check`. Put the analyzer parameters shown above in `action-state.json`:

```bash
kur-check out/gameplay \
  --analyzer aintropy/action-state-consistency \
  --analyzer-config action-state.json \
  --json
```

The default modality names are `frames` and `actions`. For different names, create an input
options file such as `{"frames": "video", "actions": "input", "controls": ["W", "S"]}` and pass
it with `--analyzer-options`. Each shard is measured independently so selected spans are not
silently joined; verdicts are still aggregated by the manifest's `source_id`. An inconsistent
clip makes the analyzer check fail with exit code 2, while inconclusive clips remain visible in
the result without being treated as inconsistent.

**Dataset-mode positions are not source frames.** A manifest entry records
path/`source_id`/`entity_id`/`split`/`shard_index`/`n_steps` and no source start offset, so a
real source position cannot be recovered from the artifact. `check_dataset` therefore places each
shard of a clip after the previous one and reports `frame_space: "shard-concatenation"` plus a
`shard_ranges` list mapping every shard path to the `[start, end)` it occupies; the printed report
carries the same caveat and labels positions `@shard-concatenation-offset<N>`. The consequence is
stated explicitly rather than implied: `add_segment()` de-duplicates one physical press seen by two
overlapping windows through its `(source_id, control, cell_key, frame)` key, and dataset mode
**cannot** (`cross_shard_dedup: false`) — two curated shards covering overlapping source spans land
at different concatenation offsets, so such a press is judged once per shard. Verdicts are
unaffected (one violation already condemns a clip); instance counts and positions are
shard-relative evidence.

`action_state_checker()` delegates to the registered
`aintropy/action-state-consistency` analyzer, preserving discovery and parameter validation.
`kuration.get_analyzer("aintropy/action-state-consistency", **config)` remains available for
declarative or dynamically selected components.

Direct construction with
`kuration.gameplay.action_state_consistency.ActionStateConsistencyChecker(config)` remains available
for callers that do not need registry discovery.

`controls` defaults to every control with a configured baseline; passing an explicit
control not present in `baselines` raises immediately (there's nothing to check it
against). Every other keyword `add_segment()` accepts (`color_space`, `lookbehind_actions`,
`lookahead_actions`, `lookahead_frames`, `boundary`) is passed straight through to
pixel_shift's own instance-finding — see its README for what each means.

If OpenCV is absent, analyzer construction and baseline processing still work. An operation
that actually measures optical flow raises a clear error naming
`pip install "kuration-sdk[action-state]"`.

### Using the generic core directly

For a workload that isn't about pixel_shift at all, build on `AnomalyChecker` directly:

```python
from kuration.gameplay.action_state_consistency import AnomalyChecker, validate_config

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

## Why the generic core and the action_state adapter are separate files

The generic verdict-aggregation logic (`anomaly_checker.py`) remains independent from the
pixel-shift-specific instance-discovery logic (`action_state.py`). They ship in the same SDK
wheel, but the separation still lets other analyzers
reuse tuple aggregation without depending on the DIAMOND vocabulary or OpenCV runtime.
