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

Use `[gameplay]` to install this analyzer together with Pixel Shift and SLAM.
The analyzer, validation, and aggregation code ships in the base SDK. The extra installs
OpenCV for measurements made from frames. For repository development:

```bash
python -m pip install -e ".[dev,action-state]"
python -m pytest tests/test_action_state_consistency.py tests/test_anomaly_checker.py -q
```

## What gets registered with kuration-sdk

The SDK registers `aintropy/action-state-consistency` twice through distinct registry kinds:

- as the stateful analyzer that owns cross-segment accumulation and both execution hooks; and
- as the direct `.action_state_consistency(...)` operator that drives that analyzer once and
  returns a pipeline carrying the result.

The operator is the normal input-backed API. The analyzer remains the single implementation and
the explicit escape hatch: `.analyze()` returns its detached result,
`run_dataset_analyzer()`/`kur-check` drive written artifacts, and a caller-owned checker can
still receive `add_segment()` calls. The analyzer descriptor advertises both driven modes
(`capabilities: {"stateful", "cross_segment", "tick_stream", "dataset"}`); the operator
advertises that it contributes pipeline metadata.

Discover it without reading a dataset:

```bash
kur-list analyzers
kur-describe analyzer aintropy/action-state-consistency --json
kur-describe operator aintropy/action-state-consistency --json
```

Registration only stores dependency-light analyzer/operator callables; it does not run optical
flow.
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

### Keep the result on the pipeline

Importing `kuration.gameplay` registers the direct method on the ordinary core `Pipeline`.
It decodes frames through the SDK's bounded media path, drives the analyzer once, and returns an
immutable pipeline carrying the verdict:

The download-free runnable companion is
`examples/07_action_state_consistency.py`;
it injects an in-memory media provider, exercises real optical flow, and shows both verdicts.

```python
import kuration.gameplay  # registers the direct method
from kuration import input

pipeline = (
    input({"clip1": rows}, schema=schema, fields={"action": action_columns}, time="t")
    .join_video({"clip1": "clip1.mp4"})            # lazy frames
    .window(64, stride=32)
    .options(streaming=True)
    .action_state_consistency(baselines=baselines, controls=["W"])
)

result = pipeline.metadata["gameplay"]["action_state_consistency"]["result"]
result["passed"]       # False if any clip is inconsistent
result["summary"]      # {"consistent": [...], "inconsistent": [...], "inconclusive": [...]}
result["frame_space"]  # "source" -- real source frames (see below)

# Keep working with the same value.
dataset = (
    pipeline
    .pixel_shift(controls=["W"])
    .filter("w_shift_measured", ">=", 1)
    .bin("w_shift_mean", bins=8)
    .sample("equal", n=100)
    .output("out/gameplay")
)
```

The metadata entry also records the analyzer descriptor, invocation, resolved mode, and a digest
of the input basis it describes. Later selection and output calls preserve it. Changing the
schema, video join, window, or registry after materialization fails loudly instead of leaving a
stale verdict attached. Use `metadata_key=` to retain multiple labeled checks and
`replace=True` only for an explicit overwrite.

The analyzer declares the same input contract as the pixel-shift transform
(`inputs=("media:frame", "field:action")`, one lookbehind action row, three lookahead rows,
`media_requirements={"kind": "video-frame", "layout": "HWC"}`), because instances are discovered
through the same measurement API. Consequences worth knowing:

- **Positions are real source frames.** `segment_ids` carry `(source_id, start_idx, end_idx)` of
  the source window, so the documented `(source_id, control, cell_key, frame)` de-duplication
  holds: one physical press seen by two overlapping windows is recorded once
  (`cross_window_dedup: true`). This is exactly the guarantee dataset mode has to give up.
- **One bounded pass.** Only a batch of windows and its halo are resident, so `streaming=True`
  and long clips are fine. `batch_size=` sizes the batch; `inputs=` remaps the tokens if your
  schema names its frames/actions differently.
- **Options are the `add_segment` options.** `controls` and `boundary` pass through per batch;
  `color_space` is read from the declared media column instead of assuming BGR.

An artifact-backed pipeline (`input(adapter="curated", ...)`) instead resolves to dataset mode
and attaches exactly what `run_dataset_analyzer()` returns. Pass `frames=`/`actions=` for
non-default modality names or `mode="stream"`/`mode="dataset"` to choose explicitly.

### Return the result directly

`.analyze()` uses the same analyzer driver but is intentionally terminal:

```python
result = pipeline.analyze(
    "aintropy/action-state-consistency",
    {"anomaly_type": "action_state", "baselines": baselines},
    controls=["W"],
)
```

Use it when the result itself is the goal and no later pipeline work needs to retain it.

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
