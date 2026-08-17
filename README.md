<p align="center">
  <img src="assets/kuration-operators-hero.svg" alt="Three of the Kuration SDK's shipped operators — action-state consistency, collision, and coverage-based sampling — showing what each produces on real gameplay recordings" width="100%">
</p>

# kuration-sdk

**Build reproducible training datasets from time-series, telemetry, trajectories, and
synchronized video.** Kuration segments raw sources, measures what matters, selects a useful
subset, and writes verified training shards with the decisions needed to reproduce them.

```python
from kuration import input

dataset = (
    input("sessions/*.parquet", time="timestamp", entity="device")
    .window(4, stride=2, unit="seconds")
    .measure("activity", "std", of="signal")
    .measure("fault_rate", "rate", of="fault")
    .bin("activity", bins=8)
    .bin("fault_rate", bins=4)
    .split(test=0.1)
    .sample("equal", n=5_000)
    .options(mode="strict", seed=7)
)

print(dataset.explain().render())
dataset.output("out/training-v1", preset="numpy-timeseries")
```

The result is not just a directory of arrays. It is a portable dataset artifact with train/test
shards, source identity, selection metadata, a resolved configuration, component provenance,
and SHA-256 integrity metadata.

## Contents

- [Why teams use kuration](#why-teams-use-kuration)
- [Install](#install)
- [Try it in five minutes](#try-it-in-five-minutes)
- [A dataset you can inspect and defend](#a-dataset-you-can-inspect-and-defend)
- [Choose your path](#choose-your-path)
- [Realistic examples](#realistic-examples)
- [kuration-ai — the companion package](#kuration-ai--the-companion-package)
  - [What's in here](#whats-in-here)
  - [Run any notebook in a browser](#run-any-notebook-in-a-browser)
  - [Getting started](#getting-started)
  - [Use cases](#use-cases)
    - [Balanced sensor shards](#balanced-sensor-shards)
    - [Rare-event coverage](#rare-event-coverage)
    - [Diverse subset](#diverse-subset)
  - [Beyond the gallery](#beyond-the-gallery)
    - [Sampling methods](#sampling-methods)
    - [Data source adapters](#data-source-adapters)
    - [Output formats and presets](#output-formats-and-presets)
    - [Pre-selection filters](#pre-selection-filters)
    - [Fluent pipeline methods](#fluent-pipeline-methods)
  - [Reproducibility](#reproducibility)
  - [Writing your own operation](#writing-your-own-operation)
  - [CLI reference](#cli-reference)
    - [`kai list`](#kai-list)
    - [`kai describe`](#kai-describe)
  - [Coming next](#coming-next)
  - [Reference](#reference)
  - [Contributing](#contributing-1)
  - [License](#license)
- [Reliability and project status](#reliability-and-project-status)
- [Contributing](#contributing)

## Why teams use kuration

| Need | What kuration provides |
|---|---|
| Make a large corpus trainable | Count- or duration-budget selection over fixed or overlapping windows |
| Preserve useful diversity | Quantile bins, tag coverage, weighted ranking, filters, and feature-space methods |
| Avoid train/test leakage | Source- and group-aware deterministic splits |
| Understand what changed | Lifecycle stats, candidate-versus-selected review, artifact comparison, plots, and static HTML reports |
| Resume long runs safely | Staged writes, fail-closed commits, complete run fingerprints, and checksum-verified shard reuse |
| Add domain intelligence | Discoverable operators, analyzers, metrics, scorers, adapters, transforms, encoders, and presets |
| Move from laptop to scale | Streaming sources, bounded Arrow checkpoints, process workers, cloud paths, and optional Ray execution |

<p align="center">
  <img src="docs/assets/curation-flow.svg" alt="Sources flow through windows, features, selection, and verified dataset artifacts" width="95%">
</p>

## Install

Until the public PyPI launch gates in the roadmap are complete, install from a source checkout:

```bash
git clone https://github.com/aintropy-ai/kuration-sdk.git
cd kuration-sdk
python -m pip install -e .
```

The NumPy-only core handles in-memory, CSV, JSON, and NPZ workflows. Add only the integrations a
pipeline needs:

```bash
python -m pip install -e '.[methods,parquet]'  # selection methods + Parquet
python -m pip install -e '.[video]'            # PyAV video frames
python -m pip install -e '.[stats]'            # Matplotlib stats plots
python -m pip install -e '.[gameplay]'         # pixel shift, action-state, and SLAM
python -m pip install -e '.[ray]'              # distributed source execution
python -m pip install -e '.[s3]'               # fsspec + the S3 backend
python -m pip install -e '.[all]'              # every runtime integration/backend
```

`pipeline.explain()` reports the extras required by a planned run before it reads the full
dataset. The provider-specific cloud extras are `s3`, `gcs`, and `azure`; `cloud` installs only
the backend-neutral fsspec layer for environments that manage provider packages separately.

Domain APIs are grouped without changing the core execution engine:

```python
import kuration.gameplay as gameplay

pipeline = (
    gameplay.slam_pipeline("slam-export/manifest.json")
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

`kuration.gameplay` is an operator domain, not a second pipeline implementation. Importing it
registers gameplay methods and relevant stats on the ordinary `kuration.Pipeline`; the core
stats call remains unchanged. `report.available()` discovers the plots, evidence modes, and
media support present on that concrete report. On a video-backed live pipeline, evidence renders
bounded frame strips and action context automatically. For whole-video review,
`gameplay.compare_videos(before, after)` renders synchronized players and can save a standalone
HTML comparison.

<p align="center">
  <img src="assets/diamond-world-model-blur.gif" alt="DIAMOND's CS:GO world model blurring out on contact with solid objects (car, crates, wall)." width="100%">
  <br>
  <sub><em>DIAMOND's CS:GO world model blurring out on contact with solid objects (car, crates, wall).</em></sub>
</p>

<p align="center">
  <img src="assets/hojepa-collision-flagged.gif" alt="The clip is flagged by the model as it has two instances where the player collides with the wall." width="100%">
  <br>
  <sub><em>The clip is flagged by the model as it has two instances where the player collides with the wall.</em></sub>
</p>

Stateful input checks follow the same pipeline-first rule:

```python
pipeline = pipeline.action_state_consistency(baselines=baselines, controls=["W"])
verdict = pipeline.metadata["gameplay"]["action_state_consistency"]["result"]
```

The returned value is still the ordinary immutable pipeline, so selection, inspection, and
output can continue from it. `.analyze(...)` remains available when a detached result is the
intended endpoint.

## Try it in five minutes

The repository includes a deterministic, download-free sensor example using only NumPy:

```bash
git clone https://github.com/aintropy-ai/kuration-sdk.git
cd kuration-sdk
python -m pip install .
python examples/quickstart.py --out quickstart-out
kur-inspect quickstart-out --verify
```

It creates six tiny sessions, scores 42 candidate windows, selects 12 across activity and fault
patterns, writes train/test NPZ shards, and reads one back. Follow the
**[guided quickstart](https://aintropy-ai.github.io/kuration-sdk/quickstart/)** for the
operator-by-operator explanation.

## A dataset you can inspect and defend

```text
out/training-v1/
├── config.json                 # resolved job, pipeline, versions, and provenance
├── dataset_manifest.json       # shard index, splits, shapes, sizes, and SHA-256
├── dataset.commit.json         # final publication state and metadata checksums
├── train/
│   └── <source>_<entity>_<hash>_shard0000.npz
└── test/
    └── <source>_<entity>_<hash>_shard0000.npz
```

```bash
kur-inspect out/training-v1 --verify
kur-stats out/training-v1 --html stats.html
kur-diff out/training-v0 out/training-v1
kur-check out/training-v1 --policy quality.json
kur-export out/training-v1 --out arrays/ --format parquet
```

Starting from raw files with no job spec yet? Infer the schema, scaffold a job, and look at
the data before running anything:

```bash
kur-infer   telemetry.csv --out schema.json
kur-init    telemetry.csv --out job.json --dataset-out out/training-v1
kur-preview job.json
kur-curate  job.json --review
```

Every command is also a `kur` subcommand (`kur curate job.json`); `kur --help` lists them and
`kur completion zsh` prints a completion script.

Already have a supported dataset? Detect its adapter and import it without a strategy or
selection:

```bash
kur-import gameplay.npz --out out/gameplay
kur-import telemetry.csv --out out/telemetry
kur-inspect out/gameplay --verify
```

- Selection is deterministic across supported worker counts and executors.
- `resume=True` reuses a shard only when its prior byte size and SHA-256 still match.
- Jobs, artifact formats, and installed components are versioned and fail loudly when
  incompatible.
- Every optional integration remains outside the NumPy-only core.

## Choose your path

| I want to… | Start here |
|---|---|
| Build my first curated dataset | [Quickstart](https://aintropy-ai.github.io/kuration-sdk/quickstart/) |
| Understand the product and its data model | [Core concepts](https://aintropy-ai.github.io/kuration-sdk/overview/) |
| Find the right pipeline operation | [Fluent pipeline guide](https://aintropy-ai.github.io/kuration-sdk/pipeline/) |
| Solve a task such as streaming, cloud output, or comparison | [Recipe cookbook](https://aintropy-ai.github.io/kuration-sdk/usage/) |
| Integrate a data format | [Adapters](https://aintropy-ai.github.io/kuration-sdk/adapters/) |
| Build a reusable operator or transform | [Operators](https://aintropy-ai.github.io/kuration-sdk/operators/) |
| Run checked JSON jobs and CLIs | [Job and CLI reference](https://aintropy-ai.github.io/kuration-sdk/jobs/) |
| Operate or audit generated artifacts | [Output and integrity](https://aintropy-ai.github.io/kuration-sdk/output/) |
| Look up an exact Python signature | [API reference](https://aintropy-ai.github.io/kuration-sdk/api/) |
| Understand internals and extension boundaries | [Architecture](https://aintropy-ai.github.io/kuration-sdk/architecture/) |

The **[documentation home](https://aintropy-ai.github.io/kuration-sdk/)** groups the complete documentation by getting
started, guides, extensions, operations, and reference.

## Realistic examples

| Example | Demonstrates |
|---|---|
| [`quickstart.py`](https://github.com/aintropy-ai/kuration-sdk/blob/main/examples/quickstart.py) | Schema inference, measures, diversity bins, train/test output, and inspection |
| [`02_robotics_action_aware.py`](https://github.com/aintropy-ai/kuration-sdk/blob/main/examples/02_robotics_action_aware.py) | Action-aware scoring composed with geometric and phase coverage |
| [`05_custom_operators.py`](https://github.com/aintropy-ai/kuration-sdk/blob/main/examples/05_custom_operators.py) | Custom compositional and transform-backed operators called through generated fluent methods |
| [`06_gameplay_operator_methods.py`](https://github.com/aintropy-ai/kuration-sdk/blob/main/examples/06_gameplay_operator_methods.py) | Built-in gameplay operators on the core `Pipeline`, including namespace-qualified syntax |
| [`gameplay_stats.ipynb`](https://github.com/aintropy-ai/kuration-sdk/blob/main/examples/notebooks/gameplay_stats.ipynb) | JupyterLab progress, lifecycle stats, gameplay plots, output inspection, and synchronized video review |
| [TeaPearce Counter-Strike](https://github.com/aintropy-ai/kuration-sdk/blob/main/examples/public_data/teapearce_csgo.md) | HDF5 adaptation, rare-event coverage, and optional image output |
| [Open P2P](https://github.com/aintropy-ai/kuration-sdk/blob/main/examples/public_data/open_p2p.md) | Protobuf annotations, cross-game action coverage, corrections, instructions, and video joins |

## kuration-ai — the companion package

The notebooks above walk through the Kuration SDK's gameplay operators against Counter-Strike telemetry. The `kuration-ai` companion package that ships alongside is the same shape of tool for a more general problem: reproducible training-data curation for machine-learning engineers working with telemetry, sensor logs, and other row-oriented data. It ships three operations (`BalancedSelector`, `RareEventCoverage`, `DiverseSubset`) plus a small pattern for writing your own. Runs from a config file you commit alongside your training code, so a curated subset is re-derivable months later against the same package versions.

![](https://img.shields.io/badge/20.9%25-LPIPS_improvement_over_DIAMOND-0D9488?style=for-the-badge)
![](https://img.shields.io/badge/62%25-JEDi_reduction_across_600_epochs-0D9488?style=for-the-badge)
![](https://img.shields.io/badge/<18%25-training_budget_to_reach_near--baseline-0D9488?style=for-the-badge)

![](https://img.shields.io/badge/4-runnable_notebooks-0D9488)
![](https://img.shields.io/badge/1,090-test_episodes_scored-0D9488)
![](https://img.shields.io/badge/4--second_segments-16_fps-0D9488)
![](https://img.shields.io/badge/51--dim_action-20_categories-0D9488)

![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat)
[![Tests](https://github.com/kuration-ai/python-sdk/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/kuration-ai/python-sdk/actions/workflows/tests.yml)
![License](https://img.shields.io/badge/license-Apache_2.0-blue?style=flat)
![Status](https://img.shields.io/badge/status-alpha-orange?style=flat)

### What's in here

- **[`kuration-ai/notebooks/`](./kuration-ai/notebooks/)**. One-click Binder and Colab launchers. The first cell in each notebook installs the SDK.
- **The `kuration-ai` companion package**: source at [`kuration-ai/src/kuration_ai/`](./kuration-ai/src/kuration_ai/), examples at [`kuration-ai/examples/`](./kuration-ai/examples/), design notes at [`kuration-ai/design/`](./kuration-ai/design/), tests at [`kuration-ai/tests/`](./kuration-ai/tests/), package config at [`kuration-ai/pyproject.toml`](./kuration-ai/pyproject.toml), release notes at [`kuration-ai/CHANGELOG.md`](./kuration-ai/CHANGELOG.md). `kai` CLI plus three shipped operations (`BalancedSelector`, `DiverseSubset`, `RareEventCoverage`). Mirrored from `kuration-ai/python-sdk`.

Long-form blog posts on physical-AI data curation live on [aintropy.ai/blog](https://aintropy.ai/blog), published directly from Drive through the content pipeline.

### Getting started

Install:

```bash
pip install kuration-ai
```

Runs on Python 3.11 or later. Depends on `kuration-sdk[methods,parquet]`.

Your first curated result, in seven lines:

```python
from kuration_ai import input, BalancedSelector

sessions = {"device-a": [{"t": float(i), "signal": float(i * 3 % 7)} for i in range(100)]}
pipeline = input(sessions, time="t", hz=10.0).window(8, stride=4)
result = (BalancedSelector(feature="activity", of="signal", bins=4, count=20)
          .apply(pipeline).options(seed=1).collect())
print(f"selected {len(result.selected)} windows")
```

Prints `selected 20 windows`. Replace the in-memory `sessions` dict with a folder of CSVs (`input("logs/*.csv", time="t")`), a Parquet dataset, or a Hugging Face dataset. The operation stays identical.

### Run any notebook in a browser

Every notebook has a Colab and a Binder badge. Click, wait, run.

> **Launcher status.** The URLs below point at `aintropy-solutions/kuration-sdk`. Colab and Binder can only clone public repositories, so the badges activate once this repo goes public. In the meantime, clone and run each notebook locally. See the notebook's own README for the exact install command (the `[gameplay]` extra is only needed by `action-state-consistency` and `collision-aware`).

| Notebook | Read | Launch |
|---|---|---|
| `starter` | [README](./kuration-ai/notebooks/starter/) | [![Colab](https://img.shields.io/badge/Colab-open-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aintropy-solutions/kuration-sdk/blob/main/kuration-ai/notebooks/starter/starter.ipynb) [![Binder](https://img.shields.io/badge/Binder-launch-579ACA?logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/aintropy-solutions/kuration-sdk/HEAD?labpath=kuration-ai%2Fnotebooks%2Fstarter%2Fstarter.ipynb) |
| `action-state-consistency` | [README](./kuration-ai/notebooks/action-state-consistency/) | [![Colab](https://img.shields.io/badge/Colab-open-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aintropy-solutions/kuration-sdk/blob/main/kuration-ai/notebooks/action-state-consistency/action-state-consistency.ipynb) [![Binder](https://img.shields.io/badge/Binder-launch-579ACA?logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/aintropy-solutions/kuration-sdk/HEAD?labpath=kuration-ai%2Fnotebooks%2Faction-state-consistency%2Faction-state-consistency.ipynb) |
| `collision-aware` | [README](./kuration-ai/notebooks/collision-aware/) | [![Colab](https://img.shields.io/badge/Colab-open-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aintropy-solutions/kuration-sdk/blob/main/kuration-ai/notebooks/collision-aware/collision-aware.ipynb) [![Binder](https://img.shields.io/badge/Binder-launch-579ACA?logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/aintropy-solutions/kuration-sdk/HEAD?labpath=kuration-ai%2Fnotebooks%2Fcollision-aware%2Fcollision-aware.ipynb) |
| `coverage-based-curation` | [README](./kuration-ai/notebooks/coverage-based-curation/) | [![Colab](https://img.shields.io/badge/Colab-open-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aintropy-solutions/kuration-sdk/blob/main/kuration-ai/notebooks/coverage-based-curation/coverage-based-curation.ipynb) [![Binder](https://img.shields.io/badge/Binder-launch-579ACA?logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/aintropy-solutions/kuration-sdk/HEAD?labpath=kuration-ai%2Fnotebooks%2Fcoverage-based-curation%2Fcoverage-based-curation.ipynb) |

### Use cases

Three curation jobs come with the library. Each has its own runnable example folder with the expected output baked into the folder's README.

#### Balanced sensor shards

Sensor data across multiple devices tends to have uneven activity. Some devices are busy, some are quiet, and row-uniform sampling produces a training set dominated by the busy ones. If the model needs to work on both regimes, that skew is a problem.

`BalancedSelector` measures activity per window (default: standard deviation of a chosen signal), quantile-bins the range, then samples equally from each bin. Quiet and busy windows get equal representation regardless of their volume in the raw data.

The typical alternative is manually oversampling the underrepresented segments in pandas. Works, but doesn't leave a reproducible trail.

Runnable example: [`kuration-ai/examples/balanced-sensor-shards/`](./kuration-ai/examples/balanced-sensor-shards/)

#### Rare-event coverage

Fault detection, safety-critical monitoring, and anomaly detection have the same shape: the rare event is what matters, and row-uniform sampling drops most of it. A model trained without correction learns to predict "no event" and looks fine on aggregate accuracy while failing on the events that actually matter.

`RareEventCoverage` measures the per-window rate of a target column (a boolean or 0/1 field like `fault`, `alarm`, or `dropout`), then samples with weighted importance toward windows containing the event.

Standard training-time alternatives are SMOTE, class-weighted loss, or focal loss. This is a data-selection-time alternative and composes with any of them.

Runnable example: [`kuration-ai/examples/rare-event-coverage/`](./kuration-ai/examples/rare-event-coverage/)

#### Diverse subset

Corpora with heavy near-duplication (same device replaying similar conditions, same content re-scraped across sources, same images with augmentations already applied) cost training compute without adding new signal. Row-uniform sampling returns redundant windows.

`DiverseSubset` clusters in feature space over pre-measured columns, optionally applies a cosine-similarity dedup filter, then samples equally from each cluster. Similar to the SemDeDup and k-means diversity techniques used in modern LLM pretraining pipelines (Anthropic, Meta, DINOv2), packaged as one operation.

Runnable example: [`kuration-ai/examples/diverse-subset/`](./kuration-ai/examples/diverse-subset/)

Each folder holds one runnable script, a README with the exact expected output, and a "what to try next" section for parameter adjustments.

### Beyond the gallery

The three shipped operations are curated shortcuts. Kuration AI sits on top of a broader pipeline library with 45 registered components, and you can drop into the raw fluent API for anything the gallery doesn't cover.

#### Sampling methods

Eleven methods, invoked via `.sample("name", ...)`.

- Coverage: `equal` / `uniform-bins`, `cluster-balanced`, `k-center`, `facility-location`
- Weighting and ranking: `weighted`, `top-k`, `target-distribution`, `dsir`
- Rarity: `rare` (density-inverse), `typical` (prototype-favoring)
- Predicate-only: `filter`

#### Data source adapters

Eleven registered adapters, invoked via `input(...)`:

- `timeseries`: CSV files, in-memory dict of rows, list of dicts
- `parquet-dataset`, `jsonl`, `hdf5`, `npz`, `zarr`: file-based formats
- `huggingface`: Hugging Face datasets
- `curated`: re-curate a previously-curated Kuration dataset
- `aintropy/slam-trajectory`, `aintropy/collision-asg`, `aintropy/collision-teapearce`: gameplay-specific

#### Output formats and presets

Seven total: `numpy-timeseries`, `pytorch-sequences`, `tabular` (flat Parquet), plus four raw shard containers (`npz`, `json`, `parquet`, `hdf5`).

#### Pre-selection filters

Two: `anomaly` (Isolation Forest), `dedup` (SemDeDup).

#### Fluent pipeline methods

Core methods used by the shipped operations: `.window()`, `.measure()`, `.filter()`, `.bin()`, `.sample()`, `.split()`, `.output()`, `.options()`, `.describe()`, `.collect()`. Pipeline exposes additional methods for tagging, scoring, video joins, previewing, and lifecycle stats.

Full inventory with parameters, extras, and runtime dependencies at [`kuration-ai/docs/history/sdk_operation_inventory.md`](./kuration-ai/docs/history/sdk_operation_inventory.md).

Every gallery operation is transparent about the primitives it calls. Run `.describe()` on the resulting pipeline and you see the exact chain. Drop the `kuration_ai` imports and your custom pipelines still work. No lock-in.

### Reproducibility

Every run of Kuration AI is deterministic. Commit a config file alongside your training code, and a teammate re-running it against the same package versions a year from now produces the same curated dataset.

Same operation, three equivalent invocation styles. The three are guaranteed to produce identical results, enforced by test.

```python
# 1. Direct: one operation, one line
result = BalancedSelector(feature="activity", of="signal").apply(pipeline).collect()

# 2. Chained instances: pipeline in code
result = run(pipeline, [
    RareEventCoverage(of="fault"),
    BalancedSelector(feature="activity", of="signal"),
]).collect()

# 3. Config dict: pipeline in YAML, JSON, or TOML
config = [
    {"type": "RareEventCoverage", "of": "fault"},
    {"type": "BalancedSelector", "feature": "activity", "of": "signal"},
]
result = run(pipeline, config).collect()
```

The config-dict form is the artifact you commit. Same package versions plus the same `curate.yaml` produce the same curated dataset.

### Writing your own operation

Any Python class with `.apply(pipeline) -> pipeline` is a valid operation. No inheritance, no registration required. Six lines for a domain-specific example:

```python
from dataclasses import dataclass
from kuration_ai import run, BalancedSelector

@dataclass(frozen=True)
class OnlyActiveWindows:
    threshold: float = 0.1

    def apply(self, pipeline):
        return (pipeline
                .measure("signal_activity", "std", of="signal")
                .filter("signal_activity", ">", self.threshold))

# Compose it with the shipped operations
result = run(pipeline, [
    OnlyActiveWindows(threshold=0.15),
    BalancedSelector(feature="activity", of="signal", bins=4, count=20),
]).options(seed=1).collect()
```

Register it to also drive it from a config dict:

```python
from kuration_ai import register
register("OnlyActiveWindows", OnlyActiveWindows)
```

For the full extension contract, see [`kuration-ai/design/04_extension_seam.md`](./kuration-ai/design/04_extension_seam.md).

### CLI reference

The `kai` command is installed alongside the library.

#### `kai list`

```bash
kai list
```

Lists available operations (built-in plus any you have registered), annotated with which extras each one needs at runtime.

#### `kai describe`

```bash
kai describe BalancedSelector
```

Shows one operation's docstring and parameter contract.

### Coming next

Not in the initial release, on the roadmap:

- **Gameplay operators**: SLAM motion diversity, SLAM quality, pixel-shift, action-state consistency, and collision detection (both regular and directional variants) for game and video telemetry. The notebook set above previews some of these.
- **`kai init <profile>`**: project scaffolder for new users starting from scratch.
- **Property-based test suite**: the initial release ships with a 52-test smoke suite; property-based tests follow.

### Reference

- [`kuration-ai/design/`](./kuration-ai/design/): five design decisions behind the API (operation abstraction, composition, config model, extension seam, workarounds).
- [`kuration-ai/CHANGELOG.md`](./kuration-ai/CHANGELOG.md): release notes for `kuration-ai`.
- [`kuration-ai/MIGRATION_NOTE.md`](./kuration-ai/MIGRATION_NOTE.md): where this content came from and why.
- [Issue tracker](https://github.com/kuration-ai/python-sdk/issues) for feature requests, bug reports, roadmap discussion.
- [`kuration-ai/examples/`](./kuration-ai/examples/): the gallery of runnable use cases.

### Contributing

Two contributing scopes coexist in this repo:

- [`CONTRIBUTING.md`](./CONTRIBUTING.md) at the repo root: for the `kuration-sdk` repo as a whole (the pipeline primitive and its docs, examples, plugins, and CI).
- [`kuration-ai/CONTRIBUTING.md`](./kuration-ai/CONTRIBUTING.md): for the `kuration-ai` companion package specifically (dev install, tests, adding operations).

### License

[Apache 2.0](./LICENSE).

## Reliability and project status

Kuration currently targets Python 3.11–3.14 and tests optional integrations at both declared
minimums and current versions. CI covers more than 500 tests, property-based invariants, clean
wheel installation, all CLI entry points, and real Zarr, PyTorch, Hugging Face, PyAV, and Ray
integration paths.

The SDK is a `0.x` beta. The supported surface and deprecation rules are documented in the
**[compatibility policy](https://aintropy-ai.github.io/kuration-sdk/compatibility/)**. See
[SECURITY.md](https://github.com/aintropy-ai/kuration-sdk/blob/main/SECURITY.md) for private
reporting and runtime trust boundaries,
[CHANGELOG.md](https://github.com/aintropy-ai/kuration-sdk/blob/main/CHANGELOG.md) for notable
changes, and [ROADMAP.md](https://github.com/aintropy-ai/kuration-sdk/blob/main/ROADMAP.md) for the
remaining public-release gates.

## Contributing

Core and maintained domain changes follow
[CONTRIBUTING.md](https://github.com/aintropy-ai/kuration-sdk/blob/main/CONTRIBUTING.md).
First-party domain capabilities ship in the SDK; independently distributed extensions can start
from the template in [`plugins/`](https://github.com/aintropy-ai/kuration-sdk/tree/main/plugins).

```bash
python -m pip install -e '.[dev,methods,parquet,array,s3,gcs,azure,p2p]'
python -m pytest tests/ -q
python -m ruff check src tests examples
python -m mypy
```

Validate one repository plugin in the same isolated environment used by CI:

```bash
python scripts/check_plugin.py plugins/<plugin-name>
```

Read [CONTRIBUTING.md](https://github.com/aintropy-ai/kuration-sdk/blob/main/CONTRIBUTING.md)
before proposing a component or public contract change.
