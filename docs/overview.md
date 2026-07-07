# kuration — overview & end-to-end example

`kuration` is a dataset-agnostic curation SDK: it turns a large time-series corpus into a
curated, training-ready set of shards — *raw source → curated subset → encoded shards* — for
**any** sequential dataset (game telemetry, robot trajectories, sensor logs, generic
multivariate series). The core carries **no domain knowledge**; a dataset is described by a
`SchemaSpec` (data) and reached through an `InputAdapter` / `OutputEncoder` (the only code
seams).

> **Status.** The full pipeline is implemented: schema + metric DSL; input adapters (timeseries
> CSV/Parquet, HDF5, NPZ, JSONL, curated-dataset-as-source, partitioned Parquet, HuggingFace
> datasets, Zarr); segmentation; feature engine;
> binning; strategies (base + feature-space methods + filter + anomaly/dedup pre-pass filters);
> a learned-scorer seam; the output layer (npz/json/hdf5/parquet containers, sub-track spans,
> hours-budget fill); the read side (`iter_shards` numpy-only, `kuration.torch.ShardDataset`
> for PyTorch); declarative tag-coverage; a stats report and a multi-arm experiment
> harness; parallel / skip-and-log / resume / **out-of-core** execution (`streaming=True` handles
> a single source larger than RAM); path-agnostic local + cloud I/O; and the
> `kur-validate` / `kur-curate` / `kur-experiment` / `kur-report` CLIs (+ a shipped JSON
> Schema). The remaining scale item is a
> spill/external tier for a *total segment pool* larger than RAM. See `../PLAN.md`.

## The pipeline

```
 raw source ─[InputAdapter]→ canonical Tick stream
   ─[Segmenter]→ overlapping windows (steps or seconds)
   ─[FeatureEngine: closed-grammar DSL over the SchemaSpec]→ per-segment features
   ─[bin_segments]→ quantile bin coordinates
   ─[Strategy + selection / hours-budget fill]→ chosen segments / tracks
   ─[OutputSpec + OutputEncoder + ShardWriter]→ shards + manifests + reproducible config
```

Two entry points sit on top:
- **`curate(...)`** — score + select a target **count** of segments (returns a `CurateResult`).
- **`write_dataset(...)`** — score + fill an **hour budget** (train/test) + write shards to disk.

## Install

```bash
pip install -e .                    # core (numpy only)
pip install -e '.[methods]'         # + scikit-learn (feature-space selection, filters, scorers)
pip install -e '.[parquet,array]'   # + pyarrow / h5py (Parquet & HDF5 containers/adapters)
pip install -e '.[cloud]'           # + fsspec (gs:// / s3:// / az:// I/O; add a backend)
```

## End-to-end example (a generic dataset, zero domain code)

Curating drone-flight logs (CSV) into 64-step position/attitude shards, preferring dynamic,
high-turn-rate windows:

```python
from kuration import (SchemaSpec, TimeseriesAdapter, Segmenter,
                      load_strategy, OutputSpec, write_dataset)

# 1. Declare the dataset's observable vocabulary (data, not code).
schema = SchemaSpec.from_dict({
    "name": "drones", "entity_axis": "drone", "sample_hz": 20.0,
    "fields": {
        "pos": {"kind": "vector", "dims": ["x", "y", "z"],
                "semantics": "position", "max_rate": 50},      # speed-clamp = 50 units/s
        "att": {"kind": "vector", "dims": ["roll", "pitch", "yaw"], "semantics": "angular"},
        "battery": {"kind": "scalar"},
    },
})

# 2. Map raw columns to fields — the generic CSV adapter needs no per-dataset code.
adapter = TimeseriesAdapter(
    schema,
    fields={"pos": ["x", "y", "z"], "att": ["roll", "pitch", "yaw"], "battery": "batt"},
    time_column="t", entity_column="drone_id",
)

# 3. Curation strategy as data: rank windows by mean speed + attitude change.
strategy = load_strategy({
    "name": "dynamic",
    "define": {
        "speed": {"source": "drone.pos", "map": "speed", "reduce": "mean"},
        "turn":  {"source": "drone.att", "map": "angle_delta", "reduce": "mean"},
    },
    "binning": {"axes": [{"metric": "speed", "bins": 10}, {"metric": "turn", "bins": 10}]},
    "selection": {"method": "top-k", "weights": {"speed": 1.0, "turn": 0.5}},
}, schema)

# 4. Declare the output: each timestep is a 3-vector position + 3-vector attitude.
spec = OutputSpec.from_dict({
    "shard_length": 64, "container": "npz",
    "modalities": {
        "pos": {"dtype": "float32", "shape": [3], "source": "field:pos"},
        "att": {"dtype": "float32", "shape": [3], "source": "field:att"},
    },
})

# 5. Curate to an hour budget and write shards (the emitted set is the strategy's best 10h).
write_dataset(
    "out/", schema=schema, adapter=adapter,
    sources=["flight1.csv", "flight2.csv", "flight3.csv"],
    strategy=strategy,
    segmenter=Segmenter(4, 2, unit="seconds", sample_hz=20),   # 4s windows, 2s stride
    spec=spec, train_hours=10, test_hours=1, seed=0,
)
# -> out/train/<src>_<entity>_<hash8>_shard0000.npz, ..., dataset_manifest.json, config.json
```

`config.json` embeds the **resolved schema + strategy + output spec**, so the dataset
reproduces from its own directory alone.

## Reference docs

- [usage.md](usage.md) — cookbook: task-oriented recipes across every capability.
- [architecture.md](architecture.md) — design deep-dive: layers, data contracts, decisions.
- [schema.md](schema.md) — authoring a `SchemaSpec` (fields, kinds, semantics tags).
- [dsl.md](dsl.md) — the closed metric grammar (`SOURCE→MAP→FILTER→REDUCE×SCOPE`).
- [adapters.md](adapters.md) — input adapters; the generic timeseries adapter; writing your own.
- [strategies.md](strategies.md) — declarative strategies, selection methods, binning, splits.
- [output.md](output.md) — output specs, encoders, containers, the hours-budget fill.
- [api.md](api.md) — API reference: every public name with its real signature.
- [jobs.md](jobs.md) — job-spec reference: every JSON key the `kur-*` CLIs accept.

## Design invariants

- **Closed grammar, loud validation.** Every metric token is enumerated and validated against
  the schema at load; unknown tokens raise. No silent fallbacks.
- **Reproducible from the artifact.** Resolved schema + strategy + output spec live in
  `config.json`.
- **Deterministic under parallelism.** Segments are sorted to a canonical order before
  binning/selection, so seeded results are worker-count-independent.
- **Domain code only at the seams.** `InputAdapter`, `OutputEncoder`, a registered Python
  metric, or a `SegmentScorer` — never in the core.
