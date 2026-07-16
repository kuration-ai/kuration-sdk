# kuration-sdk

A **dataset-agnostic curation SDK**: turn a large time-series corpus into a curated,
training-ready set of shards — *raw source → curated subset → encoded shards* — for any
sequential dataset (game telemetry, robot trajectories, sensor logs, generic multivariate
series).

The core carries **no domain knowledge**. A dataset is described by a `SchemaSpec` (data) and
reached through documented input, media, transform, and output seams. The metric language
is a **closed grammar** validated against the schema, selection is a **declarative strategy**,
and the curated set + its full configuration are written reproducibly.

Start with the NumPy-only core. Install an extra only when preflight says the pipeline needs it:

```bash
pip install -e .                 # recommended first install
pip install -e '.[all]'          # every optional integration
```

Common extras are `methods`, `parquet`, `array`, `scale`, `cloud`, `torch`, `hf`, `zarr`,
`video`, and `ray`. The relevant guide names the extra at its point of use, while
`pipeline.explain()` reports extras implied by a fluent pipeline.

## Quick start

From the repository root, make a complete first run with only Python 3.11+ and NumPy:

```bash
python -m pip install -e .
python examples/quickstart.py --out quickstart-out
```

This creates six tiny sensor sessions locally, scores 42 candidate windows, selects 12 that
cover different activity and fault patterns, writes train/test NPZ shards, and reads one back.
The run is deterministic and requires no download or optional dependency.

```text
preflight:
source: in-memory rows (6 source(s))
schema: inferred
window: 12 steps (stride 6)
selection: uniform-bins (count=12)
execution: local; errors=raise; total-pool materialization=none
optional extras: none (NumPy core)
output: npz; shard_length=12; modalities=['fault', 'signal', 'temperature']
estimate: about 42 candidate windows (bounded source sample)
estimate: about 12 output shards; before short-policy drops and per-source split constraints
input: 6 sessions, 288 rows
curation: 42 candidate windows -> 12 selected
output: 12 shards in quickstart-out
artifact: 12 shards; splits=['test', 'train']; metadata valid
first shard: split=test, modalities=['_present', 'fault', 'signal', 'temperature'], signal_shape=(12,)
reproducible config: quickstart-out/config.json
```

Follow the **[ten-minute walkthrough](docs/quickstart.md)** to understand each operator and
inspect the resulting manifest, configuration, and arrays.

## Choose the shortest path

| goal | start here |
|---|---|
| Try kuration or curate ordinary CSV/Parquet/row data | Fluent `input(...).window(...).sample(...).output(...)` and the [quickstart](docs/quickstart.md) |
| Decide whether a planned run will fit and what it needs | `pipeline.output(path, dry_run=True, inspect_sources=True)` before writing |
| Run a checked, reproducible JSON job | `kur-validate job.json --check-sources`, then `kur-explain job.json` and `kur-curate job.json` |
| Add a proprietary format, model, metric, or container | The [customization guide](docs/customization.md) and classic explicit API |
| Find a task-specific variation | The [usage cookbook](docs/usage.md) |

For development, prefer `.options(mode="strict", seed=...)`: it fails on the first bad source
instead of completing with skipped-source errors.

Then see the bounded
**[TeaPearce Counter-Strike worked example](examples/public_data/teapearce_csgo.md)** for a real public
dataset, a custom HDF5 adapter, rare-event coverage, and optional image output.
The companion **[Open P2P example](examples/public_data/open_p2p.md)** curates keyboard, mouse,
correction, environment, and text-instruction coverage from `annotation.proto` files.

## Fluent pipeline (no explicit schema/adapter/strategy)

For the common case, `input(...)` sniffs a source, infers a `SchemaSpec`, and returns a
chainable `Pipeline` — every operator validates itself loudly and returns a new `Pipeline`:

```python
from kuration import input

(input("logs/*.csv", time="t", entity="agent")
    .as_position("pos", dims=["x", "y", "z"], max_rate=20)
    .at_rate(20)
    .window(4, unit="seconds")
    .measure("speed", "mean", of="pos")
    .filter("speed", ">=", 1.0)              # universal predicate, any selection method
    .bin("speed", bins=8)
    .sample("equal", n=5000)
    .options(mode="strict")
    .output("out/run1", preset="numpy-timeseries", categorical="omit"))
```

### Operator map

Every operator returns a new immutable `Pipeline`.

| operators | purpose | recommended compute |
|---|---|---|
| `input(...)` | load files, rows, arrays, or a custom adapter; infer the schema | CPU; storage bandwidth matters for large sources |
| `.join_video()` | align one lazy MP4 media column to each ordinary source by timestamp or frame index | CPU metadata scan; frames remain undecoded |
| `.semantic()`, `.as_position()`, `.as_angular()`, `.as_bitmask()`, `.at_rate()` | refine inferred meaning and sample rate | negligible |
| `.window()` | divide each source into candidate segments | CPU |
| `.measure()`, `.score()`, `.transform()`, `.tag()` | calculate DSL metrics, whole-pool scores, bounded field/media transform features, and categorical coverage | CPU by default; transforms may explicitly request an accelerator |
| `.filter()` | apply predicates or anomaly/dedup filters | CPU and RAM; `[methods]` filters may parallelize |
| `.bin()`, `.group()`, `.split()` | create diversity bins, leakage groups, and train/test splits | CPU and RAM |
| `.apply()` | expand a registered, reusable sequence of ordinary operators | depends on the expanded operators |
| `.sample()` | choose a count or duration budget using a selection method | CPU and RAM; feature-space methods use `[methods]` |
| `.options()` | set strictness, local/Ray executor, workers, seed, streaming, resume, and progress | use `workers` locally or `executor="ray"` for optional Ray source execution |
| `.head()`, `.describe()`, `.explain()` | inspect data or the execution plan before running | negligible; `.head()` reads a small source sample |
| `.collect()`, `.output()`, `.report()`, `.experiment()` | select, write shards, profile data, or compare strategies | CPU plus storage I/O; experiments reuse one scoring pass |

Extend behavior with registered components; package a reusable workflow with
`register_operator()` and `.apply("name", ...)`. See the
[pipeline guide](docs/pipeline.md) for accepted forms and the [API reference](docs/api.md) for
exact signatures.

See **[docs/pipeline.md](docs/pipeline.md)** for the full operator reference and
`examples/04_operator_pipeline.py` for a runnable chain. Custom adapters and encoders plug
directly into the fluent pipeline through `input(adapter=..., schema=..., sources=...)` and
`.output(encoder=...)`. The classic explicit `SchemaSpec` + `InputAdapter` + `load_strategy`
API remains available for compatibility and declarative job specifications.

## Pipeline

```
raw source ─[InputAdapter]→ canonical ticks ─[Segmenter]→ windows
   ─[FeatureEngine DSL + optional bounded BatchTransform]→ per-segment features
   ─[binning]→ quantile bins ─[Strategy selection / hours-budget fill]→ chosen segments/tracks
   ─[OutputSpec + OutputEncoder + ShardWriter]→ shards + manifests + reproducible config
```

## Documentation

| doc | covers |
|---|---|
| [docs/quickstart.md](docs/quickstart.md) | a tested, network-independent first curated dataset |
| [docs/overview.md](docs/overview.md) | the pipeline, install, a full worked example |
| [docs/pipeline.md](docs/pipeline.md) | the fluent `input(...).operator(...)` API — inference rules, every operator, terminals |
| [docs/usage.md](docs/usage.md) | **cookbook** — task-oriented recipes across every capability |
| [docs/architecture.md](docs/architecture.md) | design deep-dive: layers, data contracts, decisions, extension points |
| [docs/schema.md](docs/schema.md) | authoring a `SchemaSpec` (fields, media, kinds, semantics) |
| [docs/dsl.md](docs/dsl.md) | the closed metric grammar (`SOURCE→MAP→FILTER→REDUCE×SCOPE`) |
| [docs/adapters.md](docs/adapters.md) | input adapters; the generic timeseries adapter; writing your own |
| [docs/strategies.md](docs/strategies.md) | declarative strategies, selection methods, binning, splits |
| [docs/output.md](docs/output.md) | output specs, encoders, containers, the hours-budget fill |
| [docs/api.md](docs/api.md) | **API reference** — every public name with its real signature |
| [docs/jobs.md](docs/jobs.md) | **job-spec reference** — every JSON key the `kur-*` CLIs accept |
| [docs/customization.md](docs/customization.md) | registries, descriptors, presets, versioning, and extension authoring |
| [docs/documentation-contract.md](docs/documentation-contract.md) | documentation audiences, entry paths, information architecture, and public API boundary |
| [docs/releasing.md](docs/releasing.md) | build, trusted-publishing setup, and public PyPI release checklist |
| [CONTRIBUTING.md](CONTRIBUTING.md) | contributor setup, design rules, checks, documentation, and pull requests |
| [ROADMAP.md](ROADMAP.md) | release gates, delivered optional execution extensions, scale items, and path to 1.0 |

Hitting a loud validation error? [docs/usage.md § Troubleshooting](docs/usage.md#17--troubleshooting--common-loud-failures)
maps the common ones to causes and fixes.

## Scale & robustness

- **Automatic bounded count output** — supported exact top-k and uniform-bin count plans stream fused
  segmentation/metrics/predicates/transforms into checksummed Arrow partitions and select from
  a disk-backed index. The selected `k`, rather than the total segment pool, crosses back into
  Python objects. `config.json` records the physical plan and committed partitions.
- **External hours metadata** — span candidates, bounds, aggregate metadata, tags, and budget
  ordering live in a temporary disk-backed index; only admitted span metadata returns to the
  driver. Exact top-k, weighted, and uniform-bin hours plans automatically use checkpointed
  bounded features and external priorities. Model-backed transforms are constructed once per
  bounded stage worker and skipped entirely when all matching feature partitions are reusable.
- **Backpressure and telemetry** — local and Ray batch schedulers cap in-flight work while
  preserving input order. Written configs report checkpoint hits/misses, bytes, rows, batch size,
  and queue depth. Exact disk-backed feature-hash dedup is available separately from approximate
  near-dedup.
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
- **Domain code only at the seams** — `InputAdapter`, `MediaProvider`, `BatchTransform`,
  `OutputEncoder`, a registered metric, or a `SegmentScorer`; orchestration remains generic.

## CLIs

A whole run is one JSON job spec (schema + adapter + sources + segmenter + strategy + output):

```bash
kur-validate job.json                     # preflight a job spec in ms — no data read
kur-curate job.json                       # sources -> hours-budget shards
kur-experiment job.json --out cmp.json    # compare several strategy "arms" over the same sources
kur-report job.json --out report.json     # profile a dataset's metric distributions (no selection)
kur-list methods                          # discover built-ins and installed plugins
kur-list operators                        # discover compositional pipeline operators
kur-describe scorer action_residual       # parameter schema, capabilities, origin
kur-explain job.json --json               # structured execution plan, no row reads
kur-migrate old-job.json --out job.json   # deterministic format migration
kur-inspect out/run1 --verify             # summarize an artifact and decode-check every shard
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

## Project status

The fluent pipeline, explicit extension APIs, bounded execution paths, artifact inspection, and
documented CLIs are implemented and tested on Python 3.11–3.13. See the
[API reference](docs/api.md) for the supported surface and [ROADMAP.md](ROADMAP.md) for deferred
work.

## Tests

```bash
pip install -e . && pytest tests/ -q
```
