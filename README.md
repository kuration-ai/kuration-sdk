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
pip install -e '.[torch]'                 # + torch   (kuration.torch.ShardDataset; read side)
pip install -e '.[hf,zarr]'               # + datasets / zarr  (HuggingFace & Zarr input adapters)
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

# read it back for training — numpy core, or kuration.torch.ShardDataset for PyTorch
from kuration import iter_shards
for arr in iter_shards("out/", split="train"):   # modalities + boolean arr["_present"]
    ...
```

See **[docs/overview.md](docs/overview.md)** for the full end-to-end example, the
[docs/usage.md](docs/usage.md) cookbook for recipes, and **[examples/](examples/)** for
runnable, self-contained demos (`python examples/01_timeseries_to_shards.py`).

## Fluent pipeline (no explicit schema/adapter/strategy)

For the common case, `input(...)` sniffs a source, infers a `SchemaSpec`, and returns a
chainable `Pipeline` — every operator validates itself loudly and returns a new `Pipeline`:

```python
from kuration import input

(input("logs/*.csv", time="t", entity="agent")
    .window(4, unit="seconds")
    .measure("speed", "mean", of="pos")
    .filter("speed", ">=", 1.0)              # universal predicate, any selection method
    .organize("speed", bins=8)
    .curate("equal", n=5000)
    .output("out/run1"))
```

See **[docs/pipeline.md](docs/pipeline.md)** for the full operator reference and
`examples/04_operator_pipeline.py` for a runnable chain. The classic API below (explicit
`SchemaSpec` + `InputAdapter` + `load_strategy`) remains the way to reach custom adapters,
encoders, or `workers>1` over in-memory data.

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
| [docs/pipeline.md](docs/pipeline.md) | the fluent `input(...).operator(...)` API — inference rules, every operator, terminals |
| [docs/usage.md](docs/usage.md) | **cookbook** — task-oriented recipes across every capability |
| [docs/architecture.md](docs/architecture.md) | design deep-dive: layers, data contracts, decisions, extension points |
| [docs/schema.md](docs/schema.md) | authoring a `SchemaSpec` (fields, kinds, semantics tags) |
| [docs/dsl.md](docs/dsl.md) | the closed metric grammar (`SOURCE→MAP→FILTER→REDUCE×SCOPE`) |
| [docs/adapters.md](docs/adapters.md) | input adapters; the generic timeseries adapter; writing your own |
| [docs/strategies.md](docs/strategies.md) | declarative strategies, selection methods, binning, splits |
| [docs/output.md](docs/output.md) | output specs, encoders, containers, the hours-budget fill |
| [docs/api.md](docs/api.md) | **API reference** — every public name with its real signature |
| [docs/jobs.md](docs/jobs.md) | **job-spec reference** — every JSON key the `kur-*` CLIs accept |

Hitting a loud validation error? [docs/usage.md § Troubleshooting](docs/usage.md#17--troubleshooting--common-loud-failures)
maps the common ones to causes and fixes.

## Scale & robustness

- **Parallel feature pass** — `workers>1` on `curate` / `run_experiment` / `write_dataset`
  scores sources over a process pool (workers recompile the DSL from schema + `define`),
  deterministic regardless of worker count.
- **Skip-and-log** — `on_error="skip"` (default) logs + skips a source that fails to read and
  collects it in the result's `errors`; `on_error="raise"` is fail-fast.
- **Resume** — `write_dataset(resume=True)` skips re-encoding shards that already exist, so an
  interrupted run only does the missing work. Safe by construction: shards are written
  atomically (a crash never leaves a partial file at the final path), and resuming with a
  different run/source/component fingerprint raises instead of reusing stale shards.
- **Plugins + progress** — jobs load registration modules through `plugins`, installed packages
  use the `kuration.plugins` entry-point group, and all execution APIs emit structured progress
  events (rendered automatically by run CLIs on interactive stderr).
- **Out-of-core** — `streaming=True` reads each source as a lazy tick stream on both the
  feature pass and shard write for streaming-capable adapters. CSV, JSONL, single-file Parquet,
  HDF5, Zarr, and HuggingFace inputs support it; NPZ and partitioned-Parquet `dataset` mode
  reject it because their compressed/global-sort semantics cannot guarantee bounded memory.
- **Path-agnostic I/O** — any source/shard/manifest path may be local or a cloud URL
  (`gs://`, `s3://`, `az://`, …) via fsspec (the `[cloud]` extra + a backend like `gcsfs`);
  local needs nothing.

## Design invariants

- **Closed grammar, loud validation** — every metric token is enumerated and checked against
  the schema at load; unknown tokens raise. No silent fallbacks.
- **Reproducible from the artifact** — the resolved job, source/component provenance,
  timestamps, budget accounting, and output spec live in `config.json`.
- **Deterministic under parallelism** — canonical segment ordering before binning/selection
  makes seeded results worker-count-independent.
- **Domain code only at the seams** — `InputAdapter`, `OutputEncoder`, a registered Python
  metric, or a `SegmentScorer`; never in the core.

## CLIs

A whole run is one JSON job spec (schema + adapter + sources + segmenter + strategy + output):

```bash
kur-validate job.json                     # preflight a job spec in ms — no data read
kur-curate job.json                       # sources -> hours-budget shards
kur-experiment job.json --out cmp.json    # compare several strategy "arms" over the same sources
kur-report job.json --out report.json     # profile a dataset's metric distributions (no selection)
```

`kur-validate` runs every load-time validator the real run would (schema, adapter field maps,
strategy DSL/tags/scorers/metric references, segmenter, output spec) and fails loudly on the
first problem — so a typo fails instantly instead of after a parallel pass. `--check-sources`
also stats that each source exists **and sniffs its header/schema against the column map**
(every built-in adapter: CSV/JSONL headers, Parquet/HDF5/NPZ/Zarr metadata, HuggingFace
builder features — a column/key typo fails at preflight, no data rows read);
`--print-schema` emits the shipped JSON Schema (point your
editor's `$schema` at it for autocomplete), and `--json-schema` cross-checks against it.
Every CLI prints a one-line `error:` on failure and exits 1; `--debug` shows the traceback.

## Status

Implemented: schema + metric DSL, input adapters (timeseries CSV/Parquet, **HDF5**, **NPZ**,
**JSONL**, **curated-dataset-as-source** — re-curate kuration's own output — plus
**partitioned Parquet**, **HuggingFace datasets**, and **Zarr**), segmentation,
feature engine, binning, declarative strategies (uniform-bins / weighted / top-k /
target-distribution / **filter** + feature-space **cluster-balanced / k-center /
facility-location / dsir / rare / typical**, plus **anomaly/dedup pre-pass filters** — the
`[methods]` extra), the output layer (specs, encoders, **npz / json / hdf5 / parquet**
containers, **sub-track spans**, hours-budget fill, `write_dataset`), the **read side**
(`iter_shards` — numpy-only — and `kuration.torch.ShardDataset` for PyTorch), **declarative
tag-coverage**, a single-dataset **stats report**, a first-class **learned-scorer seam**
(whole-pool/action-aware metrics), a multi-arm **experiment/comparison** harness,
**parallel / skip-and-log / resume / out-of-core** execution, **path-agnostic I/O**
(local + `gs://`/`s3://`/`az://` via fsspec), a **`kur-validate`** preflight (+ shipped JSON
Schema), driven by the `kur-*` CLIs, **schema inference** (`kuration.infer`) and a **fluent
`input(...).operator(...)` pipeline** over it with universal pre-filter predicates (any
selection method, count or hours path), complete resume provenance, default presence masks,
budget accounting, plugin discovery, and structured progress. 375 passed / 2 optional-Zarr
skipped in the development environment (`pytest tests/`).

Planned: migration of the reference CS2 pipeline onto the SDK (Phase G), and a spill/external
tier for *total* segment pools larger than RAM. See `PLAN.md`.

## Tests

```bash
pip install -e . && pytest tests/ -q
```
