<p align="center">
  <img src="../assets/kuration-operators-hero.svg" alt="Kuration SDK wrapper and companion package" width="100%">
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
    - [Motion-diversity coverage](#motion-diversity-coverage)
    - [Rare-event coverage](#rare-event-coverage)
    - [Diversity sampling](#diversity-sampling)
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

> **Launcher status.** The URLs below point at `aintropy-solutions/kuration-sdk`. Colab and Binder can only clone public repositories, so the badges activate once this repo goes public. In the meantime, clone and run each notebook locally. See the notebook's own README for the exact install command (the `[gameplay]` extra is only needed by `action-state-consistency` and `collision-aware`).


| Notebook | Read | Launch |
|---|---|---|
| `starter` | [README](./notebooks/starter/) | [![Colab](https://img.shields.io/badge/Colab-open-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aintropy-solutions/kuration-sdk/blob/main/kuration-ai/notebooks/starter/starter.ipynb) [![Binder](https://img.shields.io/badge/Binder-launch-579ACA?logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/aintropy-solutions/kuration-sdk/HEAD?labpath=kuration-ai%2Fnotebooks%2Fstarter%2Fstarter.ipynb) |
| `action-state-consistency` | [README](./notebooks/action-state-consistency/) | [![Colab](https://img.shields.io/badge/Colab-open-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aintropy-solutions/kuration-sdk/blob/main/kuration-ai/notebooks/action-state-consistency/action-state-consistency.ipynb) [![Binder](https://img.shields.io/badge/Binder-launch-579ACA?logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/aintropy-solutions/kuration-sdk/HEAD?labpath=kuration-ai%2Fnotebooks%2Faction-state-consistency%2Faction-state-consistency.ipynb) |
| `collision-aware` | [README](./notebooks/collision-aware/) | [![Colab](https://img.shields.io/badge/Colab-open-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aintropy-solutions/kuration-sdk/blob/main/kuration-ai/notebooks/collision-aware/collision-aware.ipynb) [![Binder](https://img.shields.io/badge/Binder-launch-579ACA?logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/aintropy-solutions/kuration-sdk/HEAD?labpath=kuration-ai%2Fnotebooks%2Fcollision-aware%2Fcollision-aware.ipynb) |
| `coverage-based-curation` | [README](./notebooks/coverage-based-curation/) | [![Colab](https://img.shields.io/badge/Colab-open-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aintropy-solutions/kuration-sdk/blob/main/kuration-ai/notebooks/coverage-based-curation/coverage-based-curation.ipynb) [![Binder](https://img.shields.io/badge/Binder-launch-579ACA?logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/aintropy-solutions/kuration-sdk/HEAD?labpath=kuration-ai%2Fnotebooks%2Fcoverage-based-curation%2Fcoverage-based-curation.ipynb) |

## The `kuration-ai` companion package

The notebooks above walk through the Kuration SDK's gameplay operators against Counter-Strike telemetry. The `kuration-ai` companion package that ships alongside is the same shape of tool for the sampling half of world-model curation. It ships three composable operations: `BalancedSelector` for motion-diversity coverage, `RareEventCoverage` for oversampling collisions and other rare events, and `DiverseSubset` for multi-axis diversity sampling. Plus a small pattern for writing your own. Runs from a config file you commit alongside your training code, so a curated subset is re-derivable months later against the same package versions.

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
from kuration_ai import input, RareEventCoverage

sessions = {"clip-1": [{"t": float(i), "collision": 1 if i % 30 == 0 else 0} for i in range(200)]}
pipeline = input(sessions, time="t", hz=10.0).window(8, stride=4)
result = (RareEventCoverage(of="collision", count=15, weight=2.0)
          .apply(pipeline).options(seed=1).collect())
print(f"selected {len(result.selected)} windows")
```

Prints `selected 15 windows`. Replace the in-memory `sessions` dict with a folder of CSVs (`input("clips/*.csv", time="t")`), a Parquet dataset, or a Hugging Face dataset. The operation stays identical.

### Use cases

Three curation jobs come with the library. Each has its own runnable example folder with the expected output baked into the folder's README.

#### Motion-diversity coverage

Most gameplay is the same movement: walking forward, standard turns, level ground. Sample naively and the training set inherits that natural distribution and barely touches the harder motions (tight turns, sharp climbs, high-speed segments). Quantile binning matches the distribution, which is exactly what discards rare events. Coverage has to be measured on the axis you actually care about.

`BalancedSelector` measures a per-window metric on one axis (turning, path length, whatever), bins the values evenly across the observed range, then samples the same count from every bin. That's the pattern the SDK's `slam_motion_diversity` uses (default axis `slam_path_length`; the Intro blog example switches to `slam_turn_degrees`), generalized to any per-window column.

Runnable example: [`examples/motion-coverage/`](./examples/motion-coverage/)

#### Rare-event coverage

A world model only learns the physics that show up consistently in its training data. And the interactions it most needs to learn (a moving object hitting a solid one, a player-object contact) are the ones good players avoid. That's why DIAMOND's CS:GO world model lets players double- and triple-jump against walls: successive jumps don't appear often enough in real play for the model to learn they shouldn't be possible.

The fix isn't more footage. It's finding the events already in the footage you have and oversampling them past their natural base rate. `RareEventCoverage` measures the rare event's per-window rate, then weights the sampler so those windows show up far more often than they do in the wild.

Runnable example: [`examples/rare-event-coverage/`](./examples/rare-event-coverage/)

#### Diversity sampling

Most activity-based metrics move together. A chaotic firefight is high on every axis at once; an idle stretch is low on all of them. Pick a single axis to sample on and you inherit whatever the other axes happen to correlate with. Cluster the windows in a multi-axis feature space and sample equally from each cluster, and the redundant part of the corpus stops crowding out the parts that actually differ from it.

The Intro World-Model-Data-Curation blog names this "diversity sampling", one of the things "collision detection, kinematics, action-state checks, and diversity sampling all get easier on a dataset that previously had no notion of space." `DiverseSubset` clusters over any set of pre-measured features and takes an equal share from every cluster. Optional dedup pre-pass filters near-duplicates first.

Runnable example: [`examples/diversity-sampling/`](./examples/diversity-sampling/)

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
                .measure("turn_activity", "std", of="turn_rate")
                .filter("turn_activity", ">", self.threshold))

# Compose it with the shipped operations
result = run(pipeline, [
    OnlyActiveWindows(threshold=0.15),
    BalancedSelector(feature="motion", of="turn_rate", bins=4, count=20),
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

- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) at the repo root: for the `kuration-sdk` repo as a whole (the pipeline primitive and its docs, examples, plugins, and CI).
- [`./CONTRIBUTING.md`](./CONTRIBUTING.md) here: for the `kuration-ai` companion package specifically (dev install, tests, adding operations).

## License

[Apache 2.0](./LICENSE).
