# kuration — design & build log

Status: **built and standalone.** This is the design rationale + what shipped, kept as the
project's roadmap record. For live API detail see `docs/` (architecture, schema, dsl, adapters,
strategies, output) and `examples/`; for orientation see `README.md` and `docs/overview.md`.

`kuration` was extracted from a CS2→world-model curation pipeline (`asg2diamond`) and
generalized into a dataset-agnostic SDK. The original `asg2diamond` is a separate codebase and
is **out of scope** here; this repo is the standalone engine.

---

## 1. Goal & scope

Turn a large time-series corpus into a curated, training-ready set of shards — *raw source →
curated subset → encoded shards* — for **any** sequential dataset (game telemetry, robot
trajectories, AV/sensor logs, generic multivariate series, video-via-encoder).

**Core principle: no domain knowledge in the core.** Datasets are described by a `SchemaSpec`
(data) and reached through documented input, media, transform, and output seams. The metric
language is a closed grammar validated against the schema; selection is a declarative strategy;
everything is written reproducibly.

**Non-goal:** a generic ML data-loader or feature store. `kuration` is a *curation +
shard-emission* engine: decide which spans of a corpus to keep, and write them in a declared
format, reproducibly.

---

## 2. Design principles (non-negotiable)

1. **Closed grammar, loud validation.** Metrics/strategies are data in an enumerated grammar;
   unknown tokens fail at load with an actionable error. No silent fallbacks.
2. **Auditable from the artifact.** Every run writes `config.json` embedding the resolved
   schema + strategy + output spec. Replay requires the same immutable/content-addressed sources
   and compatible implementations (including the pinned model for a model-backed scorer).
3. **Determinism under parallelism.** Segments are sorted to a canonical `(source_id,
   start_idx)` order before binning/selection → worker-count-independent results.
4. **Right tool, measured.** Vectorize hot paths (binning); parallelize only where per-worker
   compute ≫ serialization (the feature pass).
5. **Minimal core deps.** Core = numpy. scikit-learn (`[methods]`), pyarrow (`[parquet]`),
   h5py (`[array]`) and fsspec/provider backends (`[s3]`, `[gcs]`, `[azure]`) are optional extras,
   lazily imported, loud-on-missing.
6. **Path-agnostic I/O.** Local and `gs://`/`s3://`/`az://` behind one interface (`io.py`).
7. **Plugins are the only seams.** Domain code enters via an `InputAdapter`, `MediaProvider`,
   `OutputEncoder`, registered metric, `BatchTransform`, or `SegmentScorer` — never the core.

---

## 3. Architecture & pipeline

```
raw source ─[InputAdapter.ticks]→ Tick stream ─[Segmenter]→ Segment[]
   ─[FeatureEngine: closed-grammar DSL + registered metrics over SchemaSpec]→ per-segment features
   ─[BatchTransform? bounded field/media batches]→ features[…]
   ─[SegmentScorer? whole-pool/learned]→ features[…]
   ─[bin_segments]→ quantile bins
   ─[Strategy: select (count) | fill_to_hours (hours)]→ chosen segments / (source,span) items
   ─[tracks_from_ticks + OutputEncoder + ShardWriter]→ shards + manifests + config.json
```

Four data contracts connect the stages (and *are* the architecture): `Tick` (canonical
per-timestep record — the adapter's only output), `WindowBuffer` (one window as `(T,E,D)`
arrays — all a compiled metric sees), `Segment` (window metadata + generic
`features`/`bin`/`tags`/`group`), `Track` (one entity's timestep sequence — the output unit).
Depth: `docs/architecture.md`.

---

## 4. The output layer

Selected segments → training shards, no format baked in. Mirrors the input side: declarative
`OutputSpec` + pluggable `OutputEncoder` + generic `ShardWriter`.

- **`OutputSpec`** declares modalities; each routes via `field:<name>` (straight from the
  record), `segment:<attr>` (selected/aggregated segment feature), or `encoder:<hook>` (an `OutputEncoder` — where
  domain logic lives), plus `media:<name>` for lazy provider-backed columns. Plus `shard_length`, `short_policy`
  (keep/pad/drop), `container`.
- **Containers** (`output/containers.py`): `npz` + `json` (dep-free), `hdf5` (`[array]`),
  `parquet` (`[parquet]`) — uniform semantics (arrays as-shaped + in-band `_n_steps`); all four
  re-ingestable via the curated-dataset adapter.
- **`ShardWriter`** chunks a track into shards, applies short-policy, stages to local/cloud,
  emits the dataset manifest. **Sub-track spans** (`span_steps`/`span_seconds`) let the budget
  fill keep the best *parts* of a long source.
- Concrete media decode is optional and provider-backed; action-bin encodings and other
  application-specific tensors remain in `OutputEncoder`s.

---

## 5. What shipped (phase log)

The original A–F phasing, plus work added beyond the initial plan. The canonical count is kept
in `README.md` and verified by CI rather than duplicated here.

| phase | delivered |
|---|---|
| A | `SchemaSpec`/`FieldSpec` (semantics tags), `WindowBuffer`, schema-driven `compile_metric` (closed grammar) |
| B | `InputAdapter` (`ticks → Tick`) + generic `Tick`→buffer builder; `Segmenter`; timeseries adapter |
| C | binning, `Strategy` + base methods (uniform/weighted/top-k/target/**filter**), `select`, source-level splits, manifests + `config.json` |
| D | output layer: `OutputSpec`/`OutputEncoder`/`ShardWriter`, npz/json/hdf5/parquet containers, hours-budget fill, **sub-track spans** |
| E | feature-space methods (cluster/k-center/facility/dsir/rare/typical) + anomaly/dedup filters; multi-arm `experiment` (JS comparison) |
| F | `kur-curate` / `kur-experiment` / `kur-report` CLIs; full `docs/` + runnable `examples/` |
| G | lazy video media columns, bounded registered transforms, batch output encoding, ordered local/Ray execution |

**Beyond the original plan:**
- **Scale & robustness** — parallel feature pass (`workers`), skip-and-log (`on_error`), resume
  (`write_dataset(resume=)`), and **out-of-core streaming** (`streaming=True`: lazy tick
  iterator on both the feature pass and the shard write — a single source larger than RAM is
  fine).
- **Path-agnostic I/O** — local + `gs://`/`s3://`/`az://` via fsspec.
- **Declarative tag-coverage** — `tags` block derives `Segment.tags` from data; coverage
  enforced in both the count path and the hours-fill reserve.
- **More adapters** — HDF5, NPZ, JSONL, and **curated-dataset-as-source** (re-ingest the SDK's
  own output to compose/split/filter).
- **`SegmentScorer` seam** — first-class whole-pool / learned / action-aware metrics
  (obs→action residual reference scorer), composing *with* geometric coverage rather than
  replacing it. The robotics-credibility answer: quality and diversity as orthogonal weighted
  axes.
- **`dataset_report`** — single-dataset distribution profile.

---

## 6. Lineage

The curation core was lifted from `asg2diamond`'s already-domain-agnostic modules and
generalized (`object`→`tag`, `map_name`→`group`, CS2 constants → schema `semantics`/`max_rate`,
`data.json` shape → `InputAdapter`+`Tick`, `gcs_io` → `io.py`). That repo is separate; migrating
it to *consume* this SDK ("Phase G") is possible future work but **out of scope** here.

---

## 7. Open / deferred (by explicit decision)

- **Total-pool spill** — handling a *total segment pool* (across all sources, distinct from one
  big source) larger than RAM. Design fork identified: compact-columnar in-memory pool (drop
  unreferenced features, ~10× headroom) vs. true disk-external (multi-pass streaming
  binning/selection; feature-space methods can't go external) vs. both. **Deferred** — current
  scale (single-source streaming + parallel) judged "good enough" for now.
- **Public-release metadata** — LICENSE remains pending the corporate decision. Packaging
  metadata, `py.typed`, CI, lint, and type-check configuration are complete.

## 8. Decisions settled during the build

- Standalone repository; the import and distribution names remain intentionally distinct.
- Dist `kuration-sdk`, import `kuration`, console scripts `kur-*`.
- Output model = declared modalities + encoder hooks + container (the chosen altitude).
- Streaming shipped as **opt-in** (`streaming=True`); in-memory stays the default.
- Tensor containers: HDF5 + NPZ (npz dep-free); Parquet for tabular; Zarr is available as an
  optional input adapter.
