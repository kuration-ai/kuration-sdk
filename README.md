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
pip install -e .                 # core (numpy only)
pip install -e '.[methods]'      # + scikit-learn   (feature-space selection — planned)
pip install -e '.[parquet]'      # + pyarrow         (Parquet I/O)
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
```

## Status

Implemented: schema + metric DSL, input adapters (timeseries CSV/Parquet, **HDF5**, and
**curated-dataset-as-source** — re-curate kuration's own output), segmentation, feature engine,
binning, declarative strategies (uniform-bins / weighted / top-k / target-distribution /
**filter** + feature-space **cluster-balanced / k-center / facility-location / dsir / rare /
typical**, plus **anomaly/dedup pre-pass filters** — the `[methods]` extra), the output layer
(specs, encoders, **npz / json / hdf5 / parquet** containers, hours-budget fill,
`write_dataset`), a multi-arm **experiment/comparison** harness, and the `kur-*` CLIs. 98 tests
(`pytest tests/`).

Planned: migration of the reference CS2 pipeline onto the SDK (Phase G). See
`../kuration-sdk-plan.md`.

## Tests

```bash
pip install -e . && pytest tests/ -q
```
