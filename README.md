# kuration-sdk

A **dataset-agnostic curation SDK**: turn a large time-series corpus into a curated,
training-ready set of shards — *raw source → curated subset → encoded shards* — for any
sequential dataset (game telemetry, robot trajectories, sensor logs, generic multivariate
series).

The core carries **no domain knowledge**. A dataset is described by a `SchemaSpec` (data) and
reached through an `InputAdapter` / `OutputEncoder` (the only code seams). The metric language
is a **closed grammar** validated against the schema, selection is a **declarative strategy**,
and the curated set + its full configuration are written reproducibly.

```bash
pip install -e .                          # core (numpy only)
pip install -e '.[methods]'               # + scikit-learn  (feature-space selection + filters)
pip install -e '.[parquet,array]'         # + pyarrow / h5py  (Parquet & HDF5 containers/adapters)
pip install -e '.[cloud]'                 # + fsspec  (gs:// / s3:// / az:// I/O; add a backend)
```

## At a glance

```python
from kuration import (SchemaSpec, TimeseriesAdapter, Segmenter,
                      load_strategy, OutputSpec, write_dataset)

schema   = SchemaSpec.from_dict({...})                       # declare fields + semantics
adapter  = TimeseriesAdapter(schema, fields={...},           # map columns -> fields (no code)
                             time_column="t", entity_column="id")
strategy = load_strategy({...}, schema)                      # metrics + selection, as data
spec     = OutputSpec.from_dict({...})                       # declared output modalities

write_dataset("out/", schema=schema, adapter=adapter, sources=[...],
              strategy=strategy, segmenter=Segmenter(4, 2, unit="seconds", sample_hz=20),
              spec=spec, train_hours=10, test_hours=1, seed=0)
```

See **[docs/overview.md](docs/overview.md)** for the full end-to-end example.

## Pipeline

```
raw source ─[InputAdapter]→ canonical ticks ─[Segmenter]→ windows
   ─[FeatureEngine: closed-grammar DSL over the schema]→ per-segment features
   ─[binning]→ quantile bins ─[Strategy selection / hours-budget fill]→ chosen segments/tracks
   ─[OutputSpec + OutputEncoder + ShardWriter]→ shards + manifests + reproducible config
```

## Documentation

| doc | covers |
|---|---|
| [docs/overview.md](docs/overview.md) | the pipeline, install, a full worked example |
| [docs/architecture.md](docs/architecture.md) | design deep-dive: layers, data contracts, decisions, extension points |
| [docs/schema.md](docs/schema.md) | authoring a `SchemaSpec` (fields, kinds, semantics tags) |
| [docs/dsl.md](docs/dsl.md) | the closed metric grammar (`SOURCE→MAP→FILTER→REDUCE×SCOPE`) |
| [docs/adapters.md](docs/adapters.md) | input adapters; the generic timeseries adapter; writing your own |
| [docs/strategies.md](docs/strategies.md) | declarative strategies, selection methods, binning, splits |
| [docs/output.md](docs/output.md) | output specs, encoders, containers, the hours-budget fill |

## Scale & robustness

- **Parallel feature pass** — `workers>1` on `curate` / `run_experiment` / `write_dataset`
  scores sources over a process pool (workers recompile the DSL from schema + `define`),
  deterministic regardless of worker count.
- **Skip-and-log** — `on_error="skip"` (default) logs + skips a source that fails to read and
  collects it in the result's `errors`; `on_error="raise"` is fail-fast.
- **Resume** — `write_dataset(resume=True)` skips re-encoding shards that already exist, so an
  interrupted run only does the missing work.
- **Path-agnostic I/O** — any source/shard/manifest path may be local or a cloud URL
  (`gs://`, `s3://`, `az://`, …) via fsspec (the `[cloud]` extra + a backend like `gcsfs`);
  local needs nothing.

## Design invariants

- **Closed grammar, loud validation** — every metric token is enumerated and checked against
  the schema at load; unknown tokens raise. No silent fallbacks.
- **Reproducible from the artifact** — resolved schema + strategy + output spec live in
  `config.json`.
- **Deterministic under parallelism** — canonical segment ordering before binning/selection
  makes seeded results worker-count-independent.
- **Domain code only at the seams** — `InputAdapter`, `OutputEncoder`, or a registered Python
  metric; never in the core.

## CLIs

A whole run is one JSON job spec (schema + adapter + sources + segmenter + strategy + output):

```bash
kur-curate job.json                       # sources -> hours-budget shards
kur-experiment job.json --out cmp.json    # compare several strategy "arms" over the same sources
kur-report job.json --out report.json     # profile a dataset's metric distributions (no selection)
```

## Status

Implemented: schema + metric DSL, input adapters (timeseries CSV/Parquet, **HDF5**, **NPZ**,
**JSONL**, and **curated-dataset-as-source** — re-curate kuration's own output), segmentation,
feature engine, binning, declarative strategies (uniform-bins / weighted / top-k /
target-distribution / **filter** + feature-space **cluster-balanced / k-center /
facility-location / dsir / rare / typical**, plus **anomaly/dedup pre-pass filters** — the
`[methods]` extra), the output layer (specs, encoders, **npz / json / hdf5 / parquet**
containers, **sub-track spans**, hours-budget fill, `write_dataset`), **declarative
tag-coverage**, a single-dataset **stats report**, a first-class **learned-scorer seam** (whole-pool/action-aware metrics), a multi-arm **experiment/comparison** harness,
**parallel/skip-and-log/resume** execution, and **path-agnostic I/O** (local + `gs://`/`s3://`/
`az://` via fsspec), driven by the `kur-*` CLIs. 122 tests (`pytest tests/`).

Planned: migration of the reference CS2 pipeline onto the SDK (Phase G). See
`../kuration-sdk-plan.md`.

## Tests

```bash
pip install -e . && pytest tests/ -q
```
