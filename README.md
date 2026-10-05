# kuration-sdk

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue?style=flat)](./LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange?style=flat)](#project-status)

[![Docs](https://img.shields.io/badge/docs-live-brightgreen?style=flat)](https://aintropy-solutions.github.io/kuration-sdk/)
[![Blog](https://img.shields.io/badge/blog-aintropy.ai%2Fblog-0D9488?style=flat)](https://aintropy.ai/blog)
[![Examples](https://img.shields.io/badge/examples-runnable-blue?style=flat)](./examples/)
[![Roadmap](https://img.shields.io/badge/roadmap-planned-lightgrey?style=flat)](./ROADMAP.md)
[![Changelog](https://img.shields.io/badge/changelog-latest-blue?style=flat)](./CHANGELOG.md)
[![Contributing](https://img.shields.io/badge/contributing-welcome-brightgreen?style=flat)](./CONTRIBUTING.md)
[![Support](https://img.shields.io/badge/support-help-blue?style=flat)](./SUPPORT.md)
[![Security](https://img.shields.io/badge/security-policy-red?style=flat)](./SECURITY.md)
[![Design notes](https://img.shields.io/badge/design-notes-lightgrey?style=flat)](./DESIGN.md)
[![Issues](https://img.shields.io/badge/issues-open-blue?style=flat)](https://github.com/aintropy-solutions/kuration-sdk/issues)

**Kuration curates large Physical AI datasets into training-ready data for world
models.** Point it at raw gameplay clips or time-series data or video and Kuration picks the windows worth
training on, writes them as shards, and drops a config alongside so the same run can
be rebuilt.

The SDK is built around **operators** — named pipeline steps that measure a per-window feature, filter windows out, or select which windows to keep. The gameplay operators target three failure modes that recur when training a world model from gameplay:

- **The action doesn't reliably cause the state change.** `pixel_shift` measures the
  actual screen motion behind every isolated key or mouse press (Farneback optical flow
  projected onto the axis that control should move the camera on). `action_state_consistency`
  compares those measurements against a percentile-pooled baseline and issues a per-clip
  verdict — consistent, inconsistent, or inconclusive.
- **The interaction moments a world model needs are the ones good players avoid.**
  `collision` finds and classifies wall-hits (`head_on`, `angled`, `W_slide`, `WA_slide`,
  `slide_end`, `slide_constrained`) so they can be oversampled relative to their natural,
  very low, base rate.
- **The footage captures motion, but not reliably where that motion occurs in the world.**
  `slam_reconstruct` reconstructs a camera's 3D path from raw video with loop closure.
  `slam_quality` scores each window for gaps, impossible motion, and scale.
  `slam_motion_diversity` bins windows by path length, turning, and climb.

`join_video` (core, always available) pairs an action log with a video, frame by frame.

## Contents

1. [Getting started](#getting-started)
2. [Optional extras](#optional-extras)
3. [Choose your path](#choose-your-path)
4. [Use cases](#use-cases)
    - [Action-state consistency (gameplay)](#action-state-consistency-gameplay)
    - [Collision awareness (gameplay)](#collision-awareness-gameplay)
    - [SLAM motion diversity and quality (gameplay)](#slam-motion-diversity-and-quality-gameplay)
    - [Spatial coverage (general)](#spatial-coverage-general)
    - [Rare-event coverage (general)](#rare-event-coverage-general)
    - [Diversity sampling (general)](#diversity-sampling-general)
5. [Examples](#examples)
6. [Beyond the gallery](#beyond-the-gallery)
    - [Sampling methods](#sampling-methods)
    - [Data source adapters](#data-source-adapters)
    - [Output formats and presets](#output-formats-and-presets)
    - [Pre-selection filters](#pre-selection-filters)
    - [Pipeline methods](#pipeline-methods)
7. [Reproducibility](#reproducibility)
8. [Writing your own operation](#writing-your-own-operation)
9. [Reference](#reference)
    - [CLI reference](#cli-reference)
10. [Research blogs](#research-blogs)
11. [Project status](#project-status)
12. [Contributing](#contributing)

## Getting started

Install with pip:

```bash
pip install "kuration-sdk[parquet]"
```

Runnable quickstart against a small sample dataset bundled with the package — no downloads,
no data of your own required:

```python
from kuration import input
from kuration.sample_data import sample_sessions_path

dataset = (
    input(sample_sessions_path(), time="t", entity="device", hz=1)
    .window(12, stride=6)
    .measure("activity", "std", of="signal")
    .measure("fault_rate", "rate", of="fault")
    .bin("activity", bins=3)
    .bin("fault_rate", bins=2)
    .split(test=1/3)
    .sample("equal", n=12)
    .options(seed=7)
)

print(dataset.explain().render())
dataset.output("out/quickstart-out")
```

The result is a train/test set of shards plus a `config.json` recording the schema
and pipeline so the same run can be rebuilt. Every shard has a SHA-256 checksum in
the manifest, so resume and verification are safe.

<p align="center">
  <img src="docs/assets/curation-flow.svg" alt="Sources are cut into windows, scored and filtered, and the selected windows become training shards" width="95%">
</p>

## Optional extras

Kuration's core has NumPy-only dependencies and handles in-memory, CSV, JSON, and NPZ workflows. Add only the integrations a pipeline needs:

```bash
pip install "kuration-sdk[methods,parquet]"  # selection methods + Parquet
pip install "kuration-sdk[video]"            # PyAV video frames
pip install "kuration-sdk[stats]"            # Matplotlib stats plots
pip install "kuration-sdk[gameplay]"         # all gameplay operators (pixel_shift, action_state_consistency, slam_*, collision)
pip install "kuration-sdk[ray]"              # distributed source execution
pip install "kuration-sdk[s3]"               # fsspec + the S3 backend
pip install "kuration-sdk[all]"              # every runtime integration/backend
```

`pipeline.explain()` reports the extras required by a planned run before it reads the full
dataset. The provider-specific cloud extras are `s3`, `gcs`, and `azure`; `cloud` installs only
the backend-neutral fsspec layer for environments that manage provider packages separately.

`kuration.gameplay` ships seven gameplay operators: `join_video`, `pixel_shift`,
`action_state_consistency`, `slam_reconstruct`, `slam_quality`, `slam_motion_diversity`,
and `collision`. Importing the module registers them on the standard `Pipeline` class, so
an ordinary pipeline can call them as methods — it doesn't create a separate `Pipeline`
class. See each operator's worked example and clip in [Use cases](#use-cases) below.

## Choose your path

| I want to… | Start here |
|---|---|
| Build my first curated dataset | [Quickstart](https://aintropy-solutions.github.io/kuration-sdk/quickstart/) |
| Understand the product and its data model | [Core concepts](https://aintropy-solutions.github.io/kuration-sdk/overview/) |
| Find the right pipeline operation | [Pipeline guide](https://aintropy-solutions.github.io/kuration-sdk/pipeline/) |
| Solve a task such as streaming, cloud output, or comparison | [Recipe cookbook](https://aintropy-solutions.github.io/kuration-sdk/usage/) |
| Integrate a data format | [Adapters](https://aintropy-solutions.github.io/kuration-sdk/adapters/) |
| Build a reusable operator or transform | [Operators](https://aintropy-solutions.github.io/kuration-sdk/operators/) |
| Run checked JSON jobs and CLIs | [Job and CLI reference](https://aintropy-solutions.github.io/kuration-sdk/jobs/) |
| Operate or audit generated artifacts | [Output and integrity](https://aintropy-solutions.github.io/kuration-sdk/output/) |
| Look up an exact Python signature | [API reference](https://aintropy-solutions.github.io/kuration-sdk/api/) |
| Understand internals and extension boundaries | [Architecture](https://aintropy-solutions.github.io/kuration-sdk/architecture/) |

The **[documentation home](https://aintropy-solutions.github.io/kuration-sdk/)** groups the complete documentation by getting
started, guides, extensions, operations, and reference.

## Use cases

An **operator** is a named, discoverable pipeline step (an analyzer, a transform, a
scorer, a filter, ...) that you chain onto a `kuration.input(...)` pipeline; see
[Operators](docs/operators.md) for the full registry. The six use cases below are ready-made
curation jobs built from those operators — three gameplay/video-specific, three general-purpose
selection recipes built into the SDK (`kuration.selection`).
Each links to a runnable example and, where one exists, a notebook; the `[gameplay]` extra is
only needed by `action-state-consistency` and `collision-aware`.

#### Gameplay

| Use case | What it does | Runnable | Notebook |
|---|---|---|---|
| [Action-state consistency](#action-state-consistency-gameplay) | Verdict on whether a key press produced the state change the training data claims — consistent / inconsistent / inconclusive. | [`07_action_state_consistency.py`](https://github.com/aintropy-solutions/kuration-sdk/blob/main/examples/07_action_state_consistency.py) | — |
| [Collision awareness](#collision-awareness-gameplay) | Finds and classifies wall-hits so they can be oversampled relative to their natural, very low base rate. | — | — |
| [SLAM motion diversity and quality](#slam-motion-diversity-and-quality-gameplay) | Reconstructs a camera's 3D path from raw video; scores windows for gaps/impossible motion/scale and bins by path length, turning, and climb. | [`06_gameplay_operator_methods.py`](https://github.com/aintropy-solutions/kuration-sdk/blob/main/examples/06_gameplay_operator_methods.py), [`slam_kurious.ipynb`](https://github.com/aintropy-solutions/kuration-sdk/blob/main/examples/notebooks/slam_kurious.ipynb) | — |

#### General

| Use case | What it does | Runnable | Notebook |
|---|---|---|---|
| [Spatial coverage](#spatial-coverage-general) | Samples equally across a chosen axis (turning, path length, ...) so harder motions aren't crowded out. | `pipeline.balanced_selector(feature=..., of=..., bins=..., count=...)` | — |
| [Rare-event coverage](#rare-event-coverage-general) | Oversamples rare events (like collisions) relative to their natural base rate. | `pipeline.rare_event_coverage(of=..., count=...)` | — |
| [Diversity sampling](#diversity-sampling-general) | Clusters in feature space and samples equally from each cluster so redundant windows don't crowd out diverse ones. | `pipeline.diverse_subset(features=..., count=...)` | — |

### Action-state consistency (gameplay)

If "W" was pressed but the camera barely moved, we wouldn't want the model learning that
behavior. The action doesn't reliably cause the state change. `pixel_shift` measures the actual screen
motion behind every isolated key or mouse press (Farneback optical flow projected onto the axis
that control should move the camera on). `action_state_consistency` compares those measurements
against a percentile-pooled baseline and issues a per-clip verdict — consistent, inconsistent, or
inconclusive. Continuing from any existing windowed pipeline, once
`import kuration.gameplay as gameplay` has registered the operator:

```python
import kuration.gameplay as gameplay  # registers .action_state_consistency(...)

# `pipeline`: any already-windowed Pipeline (e.g. from the quickstart above).
# `baselines`: a percentile-pooled baseline, typically built from gameplay.pixel_shift(...).
pipeline = pipeline.action_state_consistency(baselines=baselines, controls=["W"])
verdict = pipeline.metadata["gameplay"]["action_state_consistency"]["result"]
```

The pipeline you get back is the same immutable pipeline. You can keep chaining `.sample()`,
`.output()`, or any other operator from it. Use `.analyze(...)` instead if you just want the
analyzer result on its own, without the pipeline.

<p align="center">
  <img src="assets/hojepa-collision-flagged.gif" alt="The clip is flagged by the model as it has two instances where the player collides with the wall." width="100%">
  <br>
  <sub><em>The clip is flagged by the model as it has two instances where the player collides with the wall.</em></sub>
</p>

### Collision awareness (gameplay)

A player bumps into a wall and the camera just stops — if that moment is rare in the training
data, the model never learns that walls are solid. The interaction moments a world model needs
are the ones good players avoid. `collision` finds
and classifies wall-hits (`head_on`, `angled`, `W_slide`, `WA_slide`, `slide_end`,
`slide_constrained`) so they can be oversampled relative to their natural, very low, base rate.

<p align="center">
  <img src="assets/diamond-world-model-blur.gif" alt="DIAMOND's CS:GO world model blurring out on contact with solid objects (car, crates, wall)." width="100%">
  <br>
  <sub><em>DIAMOND's CS:GO world model blurring out on contact with solid objects (car, crates, wall).</em></sub>
</p>

### SLAM motion diversity and quality (gameplay)

Two clips can look nearly identical frame-by-frame — a tight corridor crawl and a wide-open
sprint — and without knowing where the camera actually went, a model can't tell them apart. The
footage captures motion, but not reliably where that motion occurs in the world. `slam_reconstruct` reconstructs a camera's 3D path from
raw video with loop closure. `slam_quality` scores each window for gaps, impossible motion, and
scale. `slam_motion_diversity` bins windows by path length, turning, and climb:

```python
import kuration.gameplay as gameplay  # registers the slam_* operators

pipeline = (
    gameplay.slam_reconstruct("slam-export/manifest.json")
    .window(32, stride=16)
    .slam_motion_diversity()
    .slam_quality(min_score=0.6)
    .options(progress="auto")
)

report = pipeline.stats()                 # rich HTML in Jupyter
report.plot()                             # automatically chooses the gameplay plot
evidence = report.examples(
    metric="slam_quality_score",
    compare="selected-vs-rejected",
)
```

`report.available()` lists the plots and evidence modes for a given report. On a live video
pipeline, the evidence panel shows real frames and the action stream around each example.
`gameplay.compare_videos(before, after)` renders synchronized players and can save the comparison
as a standalone HTML file. `join_video` (core, always available) pairs an action log with a video,
frame by frame — the primitive `pixel_shift`, `action_state_consistency`, and the SLAM operators
above all build on it.

### Spatial coverage (general)

If most of your clips are the character walking forward on flat ground, that's what the model
will think the world mostly is. Most gameplay is one kind of movement — walking forward, standard turns, level ground. Sample naively on that and your training set barely touches the harder motions: tight turns, sharp climbs, high-speed segments. The Intro blog puts the fix plainly: **coverage has to be measured on the axis you actually care about.**

`BalancedSelector` does exactly that. It measures a per-window metric on your chosen axis (turning, path length, whatever), bins the values across the observed range, and samples an equal count from every bin. It's the SDK's `slam_motion_diversity` pattern (default axis `slam_path_length`; the Intro blog example switches to `slam_turn_degrees`), generalized to any per-window column.

### Rare-event coverage (general)

A player crashing into another object might happen once every few hundred clips — sample your
data evenly and the model barely ever sees it, so it never learns the event is even possible. A
world model only learns the physics that show up consistently in its training data. And the interactions it most needs — a moving object hitting a solid one, a player-object contact — are the exact ones good players avoid. That's why DIAMOND's CS:GO world model lets a player double- and triple-jump against a wall: successive jumps simply don't appear often enough in the training data for the model to learn that they shouldn't be possible.

The fix isn't collecting more footage. It's finding the events already buried in the footage you have and oversampling them relative to their natural, very low, base rate. `RareEventCoverage` measures the rare event's per-window rate and weights the sampler so those windows show up far more often than they do in the wild.

### Diversity sampling (general)

A frantic firefight and a slow walk look nothing alike, but if you only sample on one feature you
can end up with a training set that's accidentally all firefights or all walks. Most
activity-based metrics move together. A chaotic firefight is high on every axis at once; an idle stretch is low on all of them. Pick one axis to sample on and you inherit whatever the other axes happen to correlate with.

Cluster the windows in a multi-axis feature space and sample equally from each cluster, and the redundant part of the corpus stops crowding out the parts that actually differ. The Intro blog names this "diversity sampling" — one of the checks that "all get easier on a dataset that previously had no notion of space." `DiverseSubset` clusters over any set of pre-measured features and takes an equal share from every cluster. An optional dedup pre-pass filters near-duplicates first.

Each of the three general-purpose recipes above is also available as a direct dataclass
(`kuration.selection.BalancedSelector`, `RareEventCoverage`, `DiverseSubset`) for standalone use,
or as a `{"type": "BalancedSelector", ...}`-style config-dict entry for
`kuration.selection.run(...)` — see [`docs/selection.md`](docs/selection.md) for the full
parameter reference and the config-driven runner.

## Examples

| Example | What it does |
|---|---|
| [`quickstart.py`](https://github.com/aintropy-solutions/kuration-sdk/blob/main/examples/quickstart.py) | Runs the ten-minute quick start: builds six in-memory sensor sessions, selects 12 diverse windows, writes NumPy shards, reads one back. |
| [`02_robotics_action_aware.py`](https://github.com/aintropy-solutions/kuration-sdk/blob/main/examples/02_robotics_action_aware.py) | Robotics: combines a learned action-informativeness score with geometric end-effector-speed coverage, plus coverage across motion phases. Needs the `[methods]` extra. |
| [`05_custom_operators.py`](https://github.com/aintropy-solutions/kuration-sdk/blob/main/examples/05_custom_operators.py) | Shows two ways to write your own operator: compositional (a function that chains existing operators) and transform-backed (custom batch code that produces a new feature). |
| [`06_gameplay_operator_methods.py`](https://github.com/aintropy-solutions/kuration-sdk/blob/main/examples/06_gameplay_operator_methods.py) | Uses the gameplay SLAM trajectory operators on synthetic precomputed camera trajectories. NumPy core only — no video decoder or GPU needed. |
| [`07_action_state_consistency.py`](https://github.com/aintropy-solutions/kuration-sdk/blob/main/examples/07_action_state_consistency.py) | Runs `.action_state_consistency(...)` on a synthetic clip via an in-memory fake video provider. Shows the verdict (consistent / inconsistent / inconclusive) attached to the pipeline metadata. |
| [`gameplay_stats.ipynb`](https://github.com/aintropy-solutions/kuration-sdk/blob/main/examples/notebooks/gameplay_stats.ipynb) | JupyterLab notebook: runs `slam_motion_diversity` and `slam_quality` on synthetic camera trajectories, shows candidate stats, plots, and selected-vs-rejected evidence. Optional gameplay-video comparison at the end. |
| [`slam_kurious.ipynb`](https://github.com/aintropy-solutions/kuration-sdk/blob/main/examples/notebooks/slam_kurious.ipynb) | JupyterLab notebook demonstrating the SLAM operators end-to-end: reconstructs a trajectory from raw video, then curates windows by quality and motion diversity. |
| [TeaPearce Counter-Strike](https://github.com/aintropy-solutions/kuration-sdk/blob/main/examples/public_data/teapearce_csgo.md) | Curates human Counter-Strike gameplay for behavioral-cloning or world-model training. Reads the public TeaPearce HDF5 dataset, covers keyboard/mouse activity, reserves budget for rare kill/death events. |
| [Open P2P](https://github.com/aintropy-solutions/kuration-sdk/blob/main/examples/public_data/open_p2p.md) | Curates Elefant AI's Open P2P dataset across games, action patterns, corrections, and text instructions. Reads Protobuf annotations without decoding video. |

## Beyond the gallery

The three shipped selection recipes are curated shortcuts. The Kuration SDK itself is a
broader pipeline library with 48 registered components, and you can drop into the raw pipeline API
for anything the gallery doesn't cover.

### Sampling methods

Eleven methods, invoked via `.sample("name", ...)`.

- Coverage: `equal` / `uniform-bins`, `cluster-balanced`, `k-center`, `facility-location`
- Weighting and ranking: `weighted`, `top-k`, `target-distribution`, `dsir`
- Rarity: `rare` (density-inverse), `typical` (prototype-favoring)
- Predicate-only: `filter`

### Data source adapters

Eleven registered adapters, invoked via `input(...)`:

- `timeseries`: CSV files, in-memory dict of rows, list of dicts
- `parquet-dataset`, `jsonl`, `hdf5`, `npz`, `zarr`: file-based formats
- `huggingface`: Hugging Face datasets
- `curated`: re-curate a previously-curated Kuration dataset
- `aintropy/slam-trajectory`, `aintropy/collision-asg`, `aintropy/collision-teapearce`: gameplay-specific

### Output formats and presets

Seven total: `numpy-timeseries`, `pytorch-sequences`, `tabular` (flat Parquet), plus four raw shard containers (`npz`, `json`, `parquet`, `hdf5`).

### Pre-selection filters

Two: `anomaly` (Isolation Forest), `dedup` (SemDeDup).

### Pipeline methods

Core methods used by the shipped operations: `.window()`, `.measure()`, `.filter()`, `.bin()`, `.sample()`, `.split()`, `.output()`, `.options()`, `.describe()`, `.collect()`. Pipeline exposes additional methods for tagging, scoring, video joins, previewing, and lifecycle stats.

Full inventory with parameters, extras, and runtime dependencies at [`docs/operator-inventory.md`](./docs/operator-inventory.md).

Every gallery operation is transparent about the primitives it calls. Run `.describe()` on the resulting pipeline and you see the exact chain. Drop the `kuration.selection` imports and your custom pipelines still work. No lock-in.

## Reproducibility

Every run of Kuration SDK is deterministic. Commit a config file alongside your training code, and a teammate re-running it against the same package versions a year from now produces the same curated dataset.

Same operation, three equivalent invocation styles. The three are guaranteed to produce identical results, enforced by test.

```python
from kuration.selection import run, BalancedSelector, RareEventCoverage

# `pipeline`: any already-windowed Pipeline with "turn_rate" and "collision" measured.

# 1. Direct: one operation, one line
result = BalancedSelector(feature="motion", of="turn_rate").apply(pipeline).collect()

# 2. Chained instances: pipeline in code
result = run(pipeline, [
    RareEventCoverage(of="collision"),
    BalancedSelector(feature="motion", of="turn_rate"),
]).collect()

# 3. Config dict: pipeline in YAML, JSON, or TOML
config = [
    {"type": "RareEventCoverage", "of": "collision"},
    {"type": "BalancedSelector", "feature": "motion", "of": "turn_rate"},
]
result = run(pipeline, config).collect()
```

The config-dict form is the artifact you commit. Same package versions plus the same `curate.yaml` produce the same curated dataset.

## Writing your own operation

Any Python class with `.apply(pipeline) -> pipeline` is a valid operation. No inheritance, no registration required. Six lines for a domain-specific example:

```python
from dataclasses import dataclass
from kuration.selection import run, BalancedSelector

@dataclass(frozen=True)
class OnlyActiveWindows:
    threshold: float = 0.1

    def apply(self, pipeline):
        return (pipeline
                .measure("turn_activity", "std", of="turn_rate")
                .filter("turn_activity", ">", self.threshold))

# Compose it with the shipped operations.
# `pipeline`: any already-windowed Pipeline with "turn_rate" measured.
result = run(pipeline, [
    OnlyActiveWindows(threshold=0.15),
    BalancedSelector(feature="motion", of="turn_rate", bins=4, count=20),
]).options(seed=1).collect()
```

Register it to also drive it from a config dict:

```python
from kuration.selection import register_operation_type
register_operation_type("OnlyActiveWindows", OnlyActiveWindows)
```

For the fuller extension model — compositional vs. transform-backed operators, discovery,
contract testing — see [`docs/operators.md`](docs/operators.md) and
[`docs/customization.md`](docs/customization.md); for the config-driven runner shown above, see
[`docs/selection.md`](docs/selection.md).

## Reference

- [Issue tracker](https://github.com/aintropy-solutions/kuration-sdk/issues) for feature requests, bug reports, roadmap discussion.
- [ROADMAP.md](https://github.com/aintropy-solutions/kuration-sdk/blob/main/ROADMAP.md) for what's shipped, next, and planned.

### CLI reference

`kur-list` and `kur-describe` are installed alongside the library and cover every registered
component kind, not just operators.

#### `kur-list`

```bash
kur-list operators
```

Lists available operators (built-in plus any you have registered), annotated with which extras
each one needs at runtime. Pass `adapters`, `filters`, `presets`, etc. for other component kinds.

#### `kur-describe`

```bash
kur-describe operator kuration/balanced-selector
```

Shows one component's docstring and parameter contract.

## Research blogs

Long-form blog posts on physical-AI data curation — these are external writing, not additional
SDK documentation:

- **[Why raw gameplays don't make good world models — and how to fix the Virtual2Real gap](https://aintropy.ai/blog)**. The failure modes that make raw gameplay footage a poor training corpus for a world model — actions that don't produce a consistent state change, state changes with no action, rare player-object contact, missing pose or location, eval metrics that look fine when the physics is visibly wrong.
- **[Quickly convert raw corpus into high-quality world-model training sets](https://aintropy.ai/blog)**. A line-by-line walkthrough of the Pipeline API — `input`, `window`, `measure`, `filter`, `bin`, `split`, `sample`, `output` — on a small runner-telemetry example.
- **[Action-state consistency with Kuration SDK](https://aintropy.ai/blog)**. How the `action_state_consistency` operator uses Farneback optical flow to catch action-state mismatches — wall collisions, coyote time, input buffering, engine glitches — and returns a per-clip verdict (consistent, inconsistent, inconclusive).
- **[SLAM with Kuration SDK](https://aintropy.ai/blog)**. Adding scene and trajectory reconstruction to any video or image dataset. Built on DepthAnything V3 with plug-and-play VPR support for loop closure; stitches long clips into one 3D map.

## Project status

Kuration currently targets Python 3.11 through 3.14 — this range tracks CI's version matrix
(`.github/workflows/ci-versions.yml`) and moves forward as new interpreter versions are
qualified there, so treat it as a living statement rather than a hard ceiling. Optional
integrations are tested at
both their declared minimum versions and their current releases. CI runs over 1,500 test
functions across 97 test files, including property-based invariants, wheel installation,
every CLI entry point, and integration paths for Zarr, PyTorch, Hugging Face, PyAV, and Ray.

The SDK is a `0.x` beta. The supported surface and deprecation rules are documented in the
**[compatibility policy](https://aintropy-solutions.github.io/kuration-sdk/compatibility/)**. See
[SECURITY.md](https://github.com/aintropy-solutions/kuration-sdk/blob/main/SECURITY.md) for private
reporting and runtime trust boundaries,
[CHANGELOG.md](https://github.com/aintropy-solutions/kuration-sdk/blob/main/CHANGELOG.md) for notable
changes, and [ROADMAP.md](https://github.com/aintropy-solutions/kuration-sdk/blob/main/ROADMAP.md) for the
remaining public-release gates.

## Contributing

Contributions to the core SDK or the maintained gameplay module should follow
[CONTRIBUTING.md](https://github.com/aintropy-solutions/kuration-sdk/blob/main/CONTRIBUTING.md).
The gameplay operators ship inside this repo.

```bash
python -m pip install -e ".[dev,methods,parquet,array,s3,gcs,azure,p2p]"
python -m pytest tests/ -q
python -m ruff check src tests examples
python -m mypy
```

Read [CONTRIBUTING.md](https://github.com/aintropy-solutions/kuration-sdk/blob/main/CONTRIBUTING.md)
before proposing a component or public contract change.
