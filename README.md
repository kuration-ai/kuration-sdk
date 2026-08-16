<p align="center">
  <img src="assets/beta-solutions-hero.svg" alt="Kuration SDK wrapper and companion package" width="100%">
</p>

# The Kuration SDK, walked through end to end

Runnable notebooks that showcase the Kuration SDK's gameplay operators, plus the companion `kuration-ai` Python package for the reproducible-curation side of the same story. Everything lives in one folder.

![](https://img.shields.io/badge/20.9%25-LPIPS_improvement_over_DIAMOND-0D9488?style=for-the-badge)
![](https://img.shields.io/badge/62%25-JEDi_reduction_across_600_epochs-0D9488?style=for-the-badge)
![](https://img.shields.io/badge/<18%25-training_budget_to_reach_near--baseline-0D9488?style=for-the-badge)

![](https://img.shields.io/badge/4-runnable_notebooks-0D9488)
![](https://img.shields.io/badge/1,090-test_episodes_scored-0D9488)
![](https://img.shields.io/badge/4--second_segments-16_fps-0D9488)
![](https://img.shields.io/badge/51--dim_action-20_categories-0D9488)

## Contents

- [What's in here](#whats-in-here)
- [Run any notebook in a browser](#run-any-notebook-in-a-browser)
- [The kuration-ai companion package](#the-kuration-ai-companion-package)
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
- [Contributing](#contributing)
- [License](#license)

## What's in here

- **[`notebooks/`](./notebooks/)**. One-click Binder and Colab launchers. The first cell in each notebook installs the SDK.
- **The `kuration-ai` companion package**: source at [`src/kuration_ai/`](./src/kuration_ai/), examples at [`examples/`](./examples/), design notes at [`design/`](./design/), tests at [`tests/`](./tests/), package config at [`pyproject.toml`](./pyproject.toml), release notes at [`CHANGELOG.md`](./CHANGELOG.md). `kai` CLI plus three shipped operations (`BalancedSelector`, `DiverseSubset`, `RareEventCoverage`). Mirrored from `kuration-ai/python-sdk`.

Long-form blog posts on physical-AI data curation live on [aintropy.ai/blog](https://aintropy.ai/blog), published directly from Drive through the content pipeline.

## Run any notebook in a browser

Every notebook has a Colab and a Binder badge. Click, wait, run.

> **Launcher status.** The URLs below point at `aintropy-solutions/beta-solutions`. Colab and Binder can only clone public repositories, so the badges activate once this repo goes public. In the meantime, clone and run each notebook locally. See the notebook's own README for the exact install command (the `[gameplay]` extra is only needed by `action-state-consistency` and `collision-aware`).


| Notebook | Read | Launch |
|---|---|---|
| `starter` | [README](./notebooks/starter/) | [![Colab](https://img.shields.io/badge/Colab-open-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aintropy-solutions/beta-solutions/blob/main/kuration-wrapper-and-guide/notebooks/starter/starter.ipynb) [![Binder](https://img.shields.io/badge/Binder-launch-579ACA?logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/aintropy-solutions/beta-solutions/HEAD?labpath=kuration-wrapper-and-guide%2Fnotebooks%2Fstarter%2Fstarter.ipynb) |
| `action-state-consistency` | [README](./notebooks/action-state-consistency/) | [![Colab](https://img.shields.io/badge/Colab-open-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aintropy-solutions/beta-solutions/blob/main/kuration-wrapper-and-guide/notebooks/action-state-consistency/action-state-consistency.ipynb) [![Binder](https://img.shields.io/badge/Binder-launch-579ACA?logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/aintropy-solutions/beta-solutions/HEAD?labpath=kuration-wrapper-and-guide%2Fnotebooks%2Faction-state-consistency%2Faction-state-consistency.ipynb) |
| `collision-aware` | [README](./notebooks/collision-aware/) | [![Colab](https://img.shields.io/badge/Colab-open-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aintropy-solutions/beta-solutions/blob/main/kuration-wrapper-and-guide/notebooks/collision-aware/collision-aware.ipynb) [![Binder](https://img.shields.io/badge/Binder-launch-579ACA?logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/aintropy-solutions/beta-solutions/HEAD?labpath=kuration-wrapper-and-guide%2Fnotebooks%2Fcollision-aware%2Fcollision-aware.ipynb) |
| `coverage-based-curation` | [README](./notebooks/coverage-based-curation/) | [![Colab](https://img.shields.io/badge/Colab-open-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aintropy-solutions/beta-solutions/blob/main/kuration-wrapper-and-guide/notebooks/coverage-based-curation/coverage-based-curation.ipynb) [![Binder](https://img.shields.io/badge/Binder-launch-579ACA?logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/aintropy-solutions/beta-solutions/HEAD?labpath=kuration-wrapper-and-guide%2Fnotebooks%2Fcoverage-based-curation%2Fcoverage-based-curation.ipynb) |

## The `kuration-ai` companion package

The notebooks above walk through the Kuration SDK's gameplay operators against Counter-Strike telemetry. The `kuration-ai` companion package that ships alongside is the same shape of tool for a more general problem: reproducible training-data curation for machine-learning engineers working with telemetry, sensor logs, and other row-oriented data. It ships three operations (`BalancedSelector`, `RareEventCoverage`, `DiverseSubset`) plus a small pattern for writing your own. Runs from a config file you commit alongside your training code, so a curated subset is re-derivable months later against the same package versions.

![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat)
[![Tests](https://github.com/kuration-ai/python-sdk/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/kuration-ai/python-sdk/actions/workflows/tests.yml)
![License](https://img.shields.io/badge/license-Apache_2.0-blue?style=flat)
![Status](https://img.shields.io/badge/status-alpha-orange?style=flat)

### Getting started

Install:

```bash
pip install kuration-ai
```

Runs on Python 3.11 or later.

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

### Use cases

Three curation jobs come with the library. Each has its own runnable example folder with the expected output baked into the folder's README.

#### Balanced sensor shards

Sensor data across multiple devices tends to have uneven activity. Some devices are busy, some are quiet, and row-uniform sampling produces a training set dominated by the busy ones. If the model needs to work on both regimes, that skew is a problem.

`BalancedSelector` measures activity per window (default: standard deviation of a chosen signal), quantile-bins the range, then samples equally from each bin. Quiet and busy windows get equal representation regardless of their volume in the raw data.

The typical alternative is manually oversampling the underrepresented segments in pandas. Works, but doesn't leave a reproducible trail.

Runnable example: [`examples/balanced-sensor-shards/`](./examples/balanced-sensor-shards/)

#### Rare-event coverage

Fault detection, safety-critical monitoring, and anomaly detection have the same shape: the rare event is what matters, and row-uniform sampling drops most of it. A model trained without correction learns to predict "no event" and looks fine on aggregate accuracy while failing on the events that actually matter.

`RareEventCoverage` measures the per-window rate of a target column (a boolean or 0/1 field like `fault`, `alarm`, or `dropout`), then samples with weighted importance toward windows containing the event.

Standard training-time alternatives are SMOTE, class-weighted loss, or focal loss. This is a data-selection-time alternative and composes with any of them.

Runnable example: [`examples/rare-event-coverage/`](./examples/rare-event-coverage/)

#### Diverse subset

Corpora with heavy near-duplication (same device replaying similar conditions, same content re-scraped across sources, same images with augmentations already applied) cost training compute without adding new signal. Row-uniform sampling returns redundant windows.

`DiverseSubset` clusters in feature space over pre-measured columns, optionally applies a cosine-similarity dedup filter, then samples equally from each cluster. Similar to the SemDeDup and k-means diversity techniques used in modern LLM pretraining pipelines (Anthropic, Meta, DINOv2), packaged as one operation.

Runnable example: [`examples/diverse-subset/`](./examples/diverse-subset/)

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

Full inventory with parameters, extras, and runtime dependencies at [`docs/history/sdk_operation_inventory.md`](./docs/history/sdk_operation_inventory.md).

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

For the full extension contract, see [`design/04_extension_seam.md`](./design/04_extension_seam.md).

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

## Reference

- [`design/`](./design/): five design decisions behind the API (operation abstraction, composition, config model, extension seam, workarounds).
- [`CHANGELOG.md`](./CHANGELOG.md): release notes for `kuration-ai`.
- [`MIGRATION_NOTE.md`](./MIGRATION_NOTE.md): where this content came from and why.
- [Issue tracker](https://github.com/kuration-ai/python-sdk/issues) for feature requests, bug reports, roadmap discussion.
- [`examples/`](./examples/): the gallery of runnable use cases.

## Contributing

Two contributing scopes coexist in this repo:

- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) at the repo root: for `beta-solutions` as a whole (notebooks, sdr-agent).
- [`./CONTRIBUTING.md`](./CONTRIBUTING.md) here: for the `kuration-ai` companion package specifically (dev install, tests, adding operations).

## License

[Apache 2.0](./LICENSE).
