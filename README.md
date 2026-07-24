<p align="center">
  <img src="https://raw.githubusercontent.com/aintropy-ai/kuration-sdk/main/docs/assets/kuration-hero.svg" alt="Kuration turns raw sequential data into selected, verified training shards" width="100%">
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

## Why teams use kuration

| Need | What kuration provides |
|---|---|
| Make a large corpus trainable | Count- or duration-budget selection over fixed or overlapping windows |
| Preserve useful diversity | Quantile bins, tag coverage, weighted ranking, filters, and feature-space methods |
| Avoid train/test leakage | Source- and group-aware deterministic splits |
| Understand what changed | Input previews, candidate-versus-selected review, artifact comparison, and static HTML reports |
| Resume long runs safely | Atomic writes, complete run fingerprints, and checksum-verified shard reuse |
| Add domain intelligence | Discoverable operators, analyzers, metrics, scorers, adapters, transforms, encoders, and presets |
| Move from laptop to scale | Streaming sources, bounded Arrow checkpoints, process workers, cloud paths, and optional Ray execution |

<p align="center">
  <img src="https://raw.githubusercontent.com/aintropy-ai/kuration-sdk/main/docs/assets/curation-flow.svg" alt="Sources flow through windows, features, selection, and verified dataset artifacts" width="95%">
</p>

## Install

```bash
python -m pip install kuration-sdk
```

The NumPy-only core handles in-memory, CSV, JSON, and NPZ workflows. Add only the integrations a
pipeline needs:

```bash
python -m pip install 'kuration-sdk[methods,parquet]'  # selection methods + Parquet
python -m pip install 'kuration-sdk[video]'            # PyAV video frames
python -m pip install 'kuration-sdk[gameplay]'         # pixel shift, action-state, and SLAM
python -m pip install 'kuration-sdk[ray]'              # distributed source execution
python -m pip install 'kuration-sdk[s3]'               # fsspec + the S3 backend
python -m pip install 'kuration-sdk[all]'              # every runtime integration/backend
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
)
```

`kuration.gameplay` is an operator domain, not a second pipeline implementation. Importing it
registers gameplay methods on the ordinary `kuration.Pipeline`.

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
├── dataset.commit.json         # final atomic-publication state and metadata checksums
├── train/
│   └── <source>_<entity>_<hash>_shard0000.npz
└── test/
    └── <source>_<entity>_<hash>_shard0000.npz
```

```bash
kur-inspect out/training-v1 --verify
kur-diff out/training-v0 out/training-v1
kur-check out/training-v1 --policy quality.json
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
| [`05_custom_operators.py`](https://github.com/aintropy-ai/kuration-sdk/blob/main/examples/05_custom_operators.py) | A compositional operator and a transform-backed operator with discovery metadata |
| [TeaPearce Counter-Strike](https://github.com/aintropy-ai/kuration-sdk/blob/main/examples/public_data/teapearce_csgo.md) | HDF5 adaptation, rare-event coverage, and optional image output |
| [Open P2P](https://github.com/aintropy-ai/kuration-sdk/blob/main/examples/public_data/open_p2p.md) | Protobuf annotations, cross-game action coverage, corrections, instructions, and video joins |

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

Core changes follow [CONTRIBUTING.md](https://github.com/aintropy-ai/kuration-sdk/blob/main/CONTRIBUTING.md).
New maintained extensions start from the independent-distribution template in
[`plugins/`](https://github.com/aintropy-ai/kuration-sdk/tree/main/plugins).

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
