# Kuration Anomaly Detection

A generic clip-level anomaly-detection core, plus one concrete anomaly type built on top
of it: **action_state**, which checks Kuration's pixel-shift operator's per-instance pixel
shift measurements against baselines you supply. In a live pipeline this is a per-window
anomaly count (see [Usage](#usage)); auditing an already-written dataset classifies every
clip as **CONSISTENT**, **INCONSISTENT**, or **INCONCLUSIVE**.

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

Use `[gameplay]` to install this analyzer together with Pixel Shift and SLAM.
The analyzer, validation, and aggregation code ships in the base SDK. The extra installs
OpenCV for measurements made from frames. For repository development:

```bash
python -m pip install -e ".[dev,action-state]"
python -m pytest tests/test_action_state_consistency.py tests/test_anomaly_checker.py -q
```

## What gets registered with kuration-sdk

The SDK registers `aintropy/action-state-consistency` three times, through distinct registry
kinds, split by what each one is actually for:

- as a **scorer** (`ActionStateAnomalyScorer`) — the live-pipeline path. Reachable through
  `.score("aintropy/action-state-consistency", out=..., baselines=..., controls=...)` like any
  other registered scorer, or directly as `pipeline.pixel_shift(...).score(...)`. It reads each
  window's own `Segment.records["aintropy/pixel-shift"]` (populated by the pixel-shift
  transform) rather than re-deriving isolated presses from raw frames, so pixel-shift and
  action-state consistency actually share computation. Its output is one anomaly count per
  *window*, not a per-clip verdict — see [Usage](#usage) below.
- as the stateful **analyzer** — dataset-mode only now. It audits an already-written, curated
  dataset's shards (`check_dataset`, identical to what `run_dataset_analyzer()`/`kur-check`
  drive). It never shared computation with a live pixel-shift transform in the first place, so
  it is unaffected by the scorer above and keeps the full per-clip
  CONSISTENT/INCONSISTENT/INCONCLUSIVE verdict. The tick-stream-driven mode (`add_batch` over a
  live pipeline, redundantly re-deriving pixel-shift's own measurements from raw frames) is
  retired; the scorer is its replacement.
- as the direct `.action_state_consistency(...)` **operator** — dataset-mode only, wrapping the
  analyzer above and attaching its per-clip verdict to pipeline metadata. Calling it on a
  live (non-written) pipeline raises the same "artifact-backed pipeline" error `.check()` gives
  under the same circumstance.

All three are also discoverable without reading a dataset, from the CLI (see
[Command-line interface](#command-line-interface)).

Registration only stores dependency-light scorer/analyzer/operator callables; it does not run
optical flow. Discovery and construction therefore work without OpenCV.

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
output (`{"cells": {control_lower: {cell_key: {"reliable", "floor", "ceiling", "expected_sign"}}}}`):
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

### Self-calibrated mode: no baselines at all

Omitting `baselines` (and naming `controls=` instead) switches the check to **relative
outlier detection**: every measured instance is pooled, and each clip is judged against
consensus baselines derived from *the other clips'* measurements — leave-one-out, so one
bad clip never vouches for itself.

```python
pipeline.action_state_consistency(controls=["W", "S"])   # no baselines anywhere
```

Be clear about what this mode means before reaching for it:

- **It finds the clip that disagrees with the corpus** — a desynced recording, a remapped
  key, doctored footage — without any reference measurement.
- **It structurally cannot find a corpus-wide problem.** If every clip is desynced the
  same way, the consensus *is* the desync and everything passes. Supplied baselines remain
  the externally calibrated ground truth; self-calibration is a labeled mode, never a
  silent fallback.
- **It refuses to guess from noise.** A `(control, press-length)` cell is judged only when
  enough independent measurements agree on a direction; pools below `min_clips` refuse to
  issue verdicts at all. The knobs (defaults in `DEFAULT_CALIBRATION`) come in via
  `calibration=`:

```python
pipeline.action_state_consistency(
    controls=["W", "S"],
    calibration={           # all optional; shown with their defaults
        "min_clips": 3,       # smallest pool that can define a consensus
        "min_samples": 5,     # measurements a cell needs from OTHER clips to be judged
        "sign_consensus": 0.8,  # fraction of nonzero shifts that must agree on direction
        "tolerance": 1.5,     # widening of the derived magnitude bounds
    },
)
```

Derived `expected_sign` is the consensus direction; `min`/`max` are the 5th/95th
percentile of the pooled `|shift|` widened by `tolerance`. The attached result records the
mode, the knobs, and the pooled consensus under `result["calibration"]`
(`{"mode": "self-calibrated", ..., "derived_baselines": {...}}`), so a verdict judged
against inferred normality says so in the artifact. Instances in cells that could not be
derived stay unjudged and carry no verdict weight — a clip with nothing but unjudged
instances is `inconclusive`, not blessed.

## Usage

### Score each window in a live pipeline

Importing `kuration.gameplay` registers `.pixel_shift(...)` and the
`aintropy/action-state-consistency` scorer on the ordinary core `Pipeline`. Measure pixel-shift
first, so each window's own presses are available in `Segment.records`, then score:

The download-free runnable companion is
`examples/07_action_state_consistency.py`;
it injects an in-memory media provider, exercises real optical flow, and shows both verdicts.

```python
import kuration.gameplay  # registers pixel-shift and the scorer
from kuration import input

dataset = (
    input({"clip1": rows}, schema=schema, fields={"action": action_columns}, time="t")
    .join_video({"clip1": "clip1.mp4"})            # lazy frames
    .window(64, stride=32)
    .pixel_shift(controls=["W"])
    .score("aintropy/action-state-consistency", out="action_state_anomaly",
          baselines=baselines, controls=["W"])
    .filter("w_shift_measured", ">=", 1)
    .bin("w_shift_mean", bins=8)
    .sample("equal", n=100)
    .output("out/gameplay")
)
```

Every selected segment carries `action_state_anomaly` — the count of that window's OWN
pixel-shift instances that violate the baseline (0 if none do, or if the window has no
recorded presses to judge). This is scored per window, not per clip: two overlapping windows
that both contain the same physical press are each scored independently from their own
content, since a window's score should reflect what that window itself contains. Reducing
this back up to a whole-clip verdict (the pre-scorer analyzer's CONSISTENT/INCONSISTENT/
INCONCLUSIVE) is not yet a built-in capability; for that, either audit an already-written
dataset (next section) or construct `ActionStateConsistencyChecker` directly (see
[Own the detailed checker](#own-the-detailed-checker)).

Self-calibration (omitting `baselines=` in favor of `controls=` + optional `calibration=`)
still needs the whole pool to derive its leave-one-out consensus, which `.score()` sees in one
call — the per-window judgment above is unaffected either way.

Baseline validation happens at scorer construction (schema/`resolve_controls` checks), so a bad
config fails when `.score(...)` is called, not partway through execution.

### Audit an already-written dataset

`.action_state_consistency(...)` and `.analyze("aintropy/action-state-consistency", ...)` are
dataset-mode only: they audit an already-written, curated dataset's shards and attach (or
return) the same per-clip verdict `check_dataset`/`run_dataset_analyzer()` produce. Calling
either on a live, non-written pipeline raises the same "artifact-backed pipeline" error
`.check()` gives under the same circumstance — use the scorer above for a live pipeline instead.

```python
adapter = CuratedDatasetAdapter("out/gameplay")
pipeline = input(adapter=adapter, schema=adapter.schema, sources=adapter.sources())

enriched = pipeline.action_state_consistency(baselines=baselines, controls=["W"])
result = enriched.metadata["gameplay"]["action_state_consistency"]["result"]
result["passed"]       # False if any clip is inconsistent
result["summary"]      # {"consistent": [...], "inconsistent": [...], "inconclusive": [...]}
result["frame_space"]  # "shard-concatenation" -- see below

# Or get the result directly, without attaching it:
result = pipeline.analyze(
    "aintropy/action-state-consistency",
    {"anomaly_type": "action_state", "baselines": baselines},
    controls=["W"],
)
```

The metadata entry also records the analyzer descriptor, invocation, resolved mode, and a digest
of the input basis it describes. Later selection and output calls preserve it. Use
`metadata_key=` to retain multiple labeled checks and `replace=True` only for an explicit
overwrite. Pass `frames=`/`actions=` for non-default modality names.

### Own the detailed checker

`ActionStateConsistencyChecker` remains the expert path for a detailed per-instance ledger or a
custom segment loop:

```python
import kuration.gameplay as gameplay

checker = gameplay.action_state_checker(**config)
for frames, actions, segment_id in your_own_segments:
    checker.add_segment(frames, actions, segment_id)

checker.print_report()
records = checker.to_records()
print(checker.summary())
print(checker.clip_verdict("clip_042"))
```

Each record uses the shared `source_id`/`control`/`cell_key`/`frame`/`shift` schema and adds
`consistent`, the source-level `verdict`, and source-level `n_instances`. `frame_space` and
`frame_space_note` prevent dataset-mode shard offsets from being mistaken for source frames.
A source with no measured instance still emits one inconclusive row.

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
that actually measures optical flow raises a clear error naming the `action-state` extra.

### Using the generic core directly

For a workload that isn't about pixel_shift at all, build on `AnomalyChecker` directly:

```python
from kuration.gameplay.action_state_consistency import AnomalyChecker, validate_config

config = validate_config({"anomaly_type": "my_other_check", "baselines": {...}})
checker = AnomalyChecker(config)
source_id, start_idx, _ = checker.resolve_segment(("clip1", 0, 5), expected_length=5)
checker.record_instance(source_id, "some_control", "some_cell", start_idx + 2, measured_value)
checker.print_report()
records = checker.to_records()
```

## Verdict rule

This section covers the per-clip verdict (`check_dataset`/`.analyze()`/`.action_state_consistency(...)`,
dataset-mode audit and the direct checker). The live-pipeline scorer scores per window instead
(see [Score each window in a live pipeline](#score-each-window-in-a-live-pipeline)); its
per-instance judgment uses the same consistency rule, just without the clip-level aggregation
below.

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

## Command-line interface

Discover any of the three registered kinds (scorer, analyzer, operator) without reading a
dataset:

```bash
kur-list scorers
kur-list analyzers
kur-describe scorer aintropy/action-state-consistency --json
kur-describe analyzer aintropy/action-state-consistency --json
kur-describe operator aintropy/action-state-consistency --json
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
