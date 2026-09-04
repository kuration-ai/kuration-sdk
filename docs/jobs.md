# Job specs — the CLI reference

The `kur-*` CLIs run a whole pipeline from **one JSON document**. This page enumerates every
recognized key per job kind. The same information ships machine-readably as a JSON Schema:
`kur-validate --print-schema` dumps it (point your editor's `$schema` at it for autocomplete),
and `kur-validate job.json` preflights any job in milliseconds — no data read.

```bash
kur-infer      SOURCE [--out schema.json] [--force] [--name NAME] [--time C] [--entity C]
               [--hz N] [--kinds JSON|PATH] [--sample N] [--json]
kur-init       SOURCE... --out job.json [--dataset-out DIR] [--window N] [--stride N]
               [--train-hours N] [--test-hours N] [--preset NAME] [--seed N] [--name NAME]
               [--time C] [--entity C] [--hz N] [--kinds JSON|PATH] [--sample N] [--force] [--json]
kur-preview    job.json | --infer SOURCE... [--max-sources N] [--max-ticks N] [--examples N]
               [--time C] [--entity C] [--hz N] [--kinds JSON|PATH] [--json] [--html PATH]
kur-validate   job.json [--check-sources] [--json-schema] [--print-schema]
kur-explain    job.json [--json]           # budget, execution, extras, warnings + fixes
kur-curate     job.json [--force] [--review] [--review-json PATH] [--review-html PATH]
kur-experiment job.json [--out cmp.json] [--force] # experiment -> per-arm JS-vs-baseline
kur-report     job.json [--out rep.json] [--force] # report -> metric distribution profile
kur-inspect    DATASET [--verify] [--json] [--html PATH] # artifact summary/integrity check
kur-stats      DATASET [--verify] [--no-content] [--hist-bins N] [--max-values N]
               [--examples N] [--capabilities] [--json] [--html PATH]
kur-diff       LEFT RIGHT [--verify] [--json] [--html PATH] # compare two artifacts
kur-check      DATASET [--policy POLICY] [--analyzer NAME] [--json] # gates/analyzers
kur-export     DATASET --out DIR [--force] [--format npz|npy|parquet] [--split S]
               [--modalities A,B] [--no-trim] [--no-verify] [--json]
kur-import     SOURCE... --out DATASET [--force] [--adapter NAME] [--json] # detected import
kur-list       KIND [--json]                  # list one plural component kind
kur-plugin-check [ENTRY-POINT|MODULE] [--json] # validate plugin loading and descriptors
kur-describe   KIND NAME [--json]             # inspect one component contract
kur-checkpoint list|verify|clear ROOT [--stage S] [--identity I] [--json] # checkpoint maintenance
kur-migrate    JOB_OR_CONFIG [--out PATH] [--force] # migrate a versioned JSON document
```

Every command is also reachable through the `kur` dispatcher — `kur curate job.json` is
`kur-curate job.json`. `kur --help` lists what is installed, including CLIs contributed by
plugin packages, and `kur completion bash|zsh|fish` prints a completion script:

```bash
eval "$(kur completion zsh)"     # add to ~/.zshrc
```

`kur-list` accepts `adapters`, `analyzers`, `containers`, `encoders`, `filters`, `methods`,
`metrics`, `operators`, `presets`, `scorers`, and `transforms`. `kur-describe` accepts the
corresponding singular or plural kind. Analyzers are discoverable here, but they are user-owned
direct APIs rather than job-spec or implicit pipeline stages; construct one with
`get_analyzer()` and drive its documented update methods in Python. An analyzer that also
implements the dataset-check hook can run directly on a written artifact with `kur-check`.

Standalone workflow CLIs normally accept `--debug`: failures print a one-line `error: ...` and
exit 1, while `--debug` re-raises for the full traceback. Dispatcher and domain command groups
document their own debugging options in `--help`.

File-producing authoring commands refuse an existing output by default. Pass `--force` only when
replacement is intentional; report, experiment, inference, initialization, migration, import,
export, and curation commands still reject an output that overlaps one of their inputs.

`kur-inspect`, `kur-stats`, `kur-diff`, `kur-check`, `kur-export`, `kur-checkpoint`, and
`kur-plugin-check` are exceptions to the job-file model: they take artifact paths, checkpoint
roots, or plugin names directly. The artifact commands need neither the original jobs nor input
sources. `kur-inspect` exits 2 when `--verify` finds a missing or corrupt shard;
`kur-checkpoint verify` exits 2 when a committed partition fails its checksum; `kur-check` and
`kur-plugin-check` exit 2 when a policy or plugin contract fails. A quality policy is a JSON
object using the expectations listed in [usage.md](usage.md).

## Starting without a job (`kur-infer`, `kur-init`, `kur-preview`)

These three cover the gap before a job spec exists. `kur-infer` samples one source and prints
the schema it deduces; `--out` writes a schema document a job can reference by path
(`"schema": "schema.json"`). `kur-init` goes further and writes a whole job — inferred schema
and adapter mapping, a binning axis over the first scalar field, and an output preset — then
resolves it before writing, so a scaffold that lands on disk is one `kur-validate` accepts.
`kur-preview` decodes a bounded prefix of the sources and reports field ranges, missingness,
and example ticks; unlike `kur-explain` (which reads no rows) it shows what the data contains.

`kur-infer` and `kur-init` sample 200 rows by default; use `--sample` to change that bound and
`--name` to set the inferred schema name. `kur-init` defaults to a 64-step window whose stride
equals the window, a one-hour train budget, zero test hours, the `numpy-timeseries` preset, and
seed zero. Override those choices with `--window`, `--stride`, `--train-hours`, `--test-hours`,
`--preset`, and `--seed`. For raw-source previews, `--max-sources`, `--max-ticks`, and
`--examples` independently bound source count, ticks per source, and retained example rows.

```bash
kur-infer   telemetry.csv --out schema.json     # inspect the inferred schema first
kur-init    telemetry.csv --out job.json --window 128 --dataset-out out/curated
kur-preview job.json                            # what is actually in the corpus
kur-validate job.json && kur-explain job.json   # preflight, no rows read
kur-curate  job.json --review                   # run, then explain the hours budget
```

The scaffold is a starting point, not a recommendation: it picks the first scalar field as a
binning axis because it cannot know which metric matters. Edit `strategy` before trusting a
run. When a schema has no scalar field, `kur-init` refuses to guess and names the block to
write by hand.

## Artifact statistics and export (`kur-stats`, `kur-export`)

`kur-inspect` answers "is this artifact intact"; `kur-stats` answers "what is in it" —
per-modality distributions, split composition, and any section a domain package contributes
through a registered stats enricher. `--capabilities` lists the plots and evidence a given
report supports. `--hist-bins` controls histogram resolution, `--max-values` bounds the values
sampled per modality, and `--examples` bounds embedded selection evidence when the artifact
contains it. `--no-content` reads only manifest metadata; combine it with `--verify` for a
checksum pass without distribution decoding.

`kur-export` walks the manifest and writes shards as plain array files for consumers outside
Python, alongside an `export_index.json` carrying the manifest metadata (source, entity,
split, `n_steps`) that bare arrays would lose. `--format npy` writes one file per shard and
modality; `npz` and `parquet` write one file per shard. Pad rows are trimmed unless
`--no-trim` is given. `--modalities` accepts a comma-separated subset, and `--no-verify`
explicitly skips source-artifact checksum verification for trusted inputs. Every indexed payload
has a SHA-256 and byte size. Payloads are staged
under an owned generation namespace and `export_index.json` is committed last; `--force`
atomically points the index at a new complete generation while leaving unrelated files alone.
One destination lease prevents concurrent exporters from interleaving.

```bash
kur-stats  out/curated --html stats.html
kur-export out/curated --out arrays/ --format parquet --split train
```

## Checkpoint maintenance (`kur-checkpoint`)

A job's `checkpoint_dir` accumulates per-stage partition commits. `kur-checkpoint list`
summarizes the stages, their partition and row counts, whether each stage was finalized, and
whether stale attempt files remain. `verify` re-checksums every committed data file and exits
2 on a mismatch — it reads the committed metadata rather than reconstructing the run's
`CheckpointIdentity`, so it works from the checkpoint tree alone. `clear` deletes state and
requires `--yes`; `--attempts-only` removes leftover attempts while keeping valid commits.
`--identity` restricts an operation to one run-fingerprint namespace (`legacy` selects the
pre-fingerprint layout). Clearing a stage that exists under multiple identities fails unless
the identity is explicit.

`kur-check` accepts `--policy`, `--analyzer`, or both. Analyzer construction parameters and
dataset-driving options are separate JSON objects:

```bash
kur-check out/candidate \
  --analyzer aintropy/action-state-consistency \
  --analyzer-config action-state.json \
  --analyzer-options action-state-input.json \
  --json
```

Here `action-state.json` contains the ordinary `anomaly_type` and `baselines` parameters.
`action-state-input.json` may select `frames`, `actions`, `controls`, `color_space`, `boundary`,
and `verify`. The default modality names are `frames` and `actions`. When both a policy and an
analyzer are supplied, the JSON result contains `quality` and `analyzer` members and either
failure produces exit code 2. `--html` currently renders quality-policy results and therefore
requires `--policy`.

## Smart dataset import

`kur-import` detects a registered input adapter and wraps the complete input as a Kuration
artifact without scoring, selection, resampling, or row loss:

```bash
kur-import gameplay.npz --out out/gameplay
kur-import telemetry.csv --out out/telemetry
kur-import logs/ --out out/logs                         # partitioned Parquet
kur-import episodes/ --out out/gameplay --shard-length 4096
```

Built-in detection covers CSV, JSONL, single-file Parquet, partitioned-Parquet directories,
NPZ, HDF5, and Zarr. NPZ/HDF5/Zarr adapters expose a lossless array-import hook: every top-level
array becomes a modality and retains its per-row shape and dtype. A directory containing NPZ
episodes is searched recursively and each file becomes one logical source. Array files must
share a leading row dimension and multiple files must have the same modality contract.

Before schema or modality-contract validation, local files and directory stores are copied into
private descriptor-pinned snapshots; later pathname or ancestor swaps cannot change the bytes
published. Smart import therefore requires POSIX descriptor-relative open support; platforms
without it fail before plugins or adapter detection rather than accepting an ancestry race.
Generic fsspec reads cannot bind the download to the ETag/version returned by a separate metadata
request, so remote import likewise fails closed rather than trusting an ABA-vulnerable
metadata/read/metadata sequence. Download the exact immutable object generation locally first.

Tabular adapters use the same conservative sampling, schema inference, time/entity detection,
and canonical `Tick` conversion as `kuration.input()`. Numeric output is stored as float64 and
categorical columns receive a deterministic integer vocabulary recorded in `config.json`.
Ambiguous columns fail with the ordinary inference remedy instead of being guessed.

Import never drops data silently — these cases fail loudly with a remedy instead:
HDF5/Zarr files whose datasets live inside nested groups (the top-level import would miss
them; use an explicit configured adapter), schemas with `event_streams` (shards cannot carry
events yet), `--time`/`--entity`/`--kinds` on array sources that have no columns to apply them
to, generic remote files or prefixes that cannot be generation-pinned (download the exact
immutable generation locally first), and a destination whose previous import failed part-way
(rerun with `--force` to clear it — this also removes the stale shards).

Registered plugin adapters participate by implementing
`detect_source(uri) -> bool | confidence`. An adapter can additionally implement
`import_arrays(uri) -> {modality: ndarray}` for lossless tensor import. Otherwise pass an
explicit schema and adapter options so `kur-import` can construct it through the normal adapter
registry:

```bash
kur-import capture.custom --out out/capture \
  --adapter acme/capture \
  --schema capture-schema.json \
  --fields capture-fields.json \
  --adapter-options capture-options.json
```

Use `--adapter NAME` whenever detection is ambiguous, or `--schema`, `--fields`, `--time`,
`--entity`, `--hz`, and `--kinds` to override safe inference. Detection must agree across every
supplied source. By default each logical source/track remains one shard, which avoids introducing
artificial boundaries for sequence analyzers; use `--shard-length` when bounded shard sizes
matter more.

Use `--source-id ID` to override the ID for a single file, `--split NAME` to select the manifest
split, and `--container` to choose `npz`, `json`, `hdf5`, or `parquet`. Existing Kuration output
is protected unless `--force` is explicit. An already-curated Kuration source is detected and
reported as ready to use rather than redundantly imported. Unsafe arrays, empty inputs, mixed
adapter kinds, ambiguous detectors, and cross-file contract drift fail before publication.
The generated artifact works with `kur-inspect`, `kur-check`, `iter_shards()`, and
curated-dataset adapters.

The job **kind** is inferred: `arms` present → experiment; `output`/`train_hours` present →
curate; otherwise → report.

Recommended development loop:

```bash
kur-validate job.json --check-sources   # schema, mappings, and source metadata
kur-explain job.json                    # confirm budget/execution/dependencies; no rows read
kur-curate job.json                     # run only after the preflight looks right
```

`kur-explain --json` emits the same information for automation. Its text form highlights
error policy and optional extras, and every warning includes a concrete fix. Use
`on_error: "raise"` or `mode: "strict"` while developing; switch to skip-and-report only when
partial source failure is an intentional operational policy.

## Keys shared by all job kinds

| key | required | value |
|---|---|---|
| `job_schema_version` | no (legacy accepted) | serialized job contract version; current `1.0` |
| `schema` | unless the adapter derives one | inline `SchemaSpec` dict **or a path** to a JSON file ([schema.md](schema.md)). The curated adapter derives its own — omit there. |
| `adapter` | yes | `{"type": <registered name>, ...kwargs}` — type is `timeseries`, `jsonl`, `hdf5`, `npz`, `parquet-dataset`, `huggingface`, `zarr`, `curated`, or your own `register_adapter` name; the remaining keys are that adapter's constructor kwargs ([adapters.md](adapters.md)) |
| `sources` | no | explicit list of source URIs (local paths or `gs://`/`s3://`/`az://`). Omit to use the adapter's own `sources()` (curated / parquet-dataset / huggingface enumerate) |
| `plugins` | no | importable registration modules loaded before adapter/strategy construction |
| `include_plugin_entry_points` | no (true) | discover installed `kuration.plugins` entry points; set false for a job isolated to built-ins plus its explicit `plugins` list |
| `segmenter` | yes | `{"window": W, "stride": S, "unit": "steps"\|"seconds", "sample_hz"?: hz}` — `sample_hz` defaults to the schema's |
| `strategy` | yes (per-arm for experiments) | inline strategy dict or a path to one ([strategies.md](strategies.md)) |
| `seed` | no (0) | RNG seed — same seed + sources → same result, regardless of `workers` |
| `workers` | no (1) | parallel feature pass over a process pool |
| `executor` | no (`"local"`) | ordered `"local"` or optional `[ray]` source execution |
| `on_error` | no (`"skip"`) | `"skip"` logs + skips an unreadable source; `"raise"` fails fast. Either way an **empty** pool raises |
| `streaming` | no (false) | out-of-core for streaming-capable adapters; NPZ and partitioned-Parquet dataset mode reject it |
| `mode` | no | `"explore"` or `"strict"`; explicit options override mode defaults |

Run CLIs show throttled phase progress when stderr is an interactive terminal. Redirected output
stays clean; pass `--quiet` to suppress interactive progress explicitly.

## Curate jobs (`kur-curate`)

| key | required | value |
|---|---|---|
| `output` / `output_preset` | one | an explicit `OutputSpec`, or a registered preset name / `{name, ...parameters}` |
| `train_hours` | yes | train budget in hours (the emitted set is the strategy's *best* N hours) |
| `test_hours` | no (0) | test budget; sources split leakage-free |
| `out` | yes | output directory — local path or cloud URL |
| `resume` | no (false) | verify and reuse checksummed shards only when the complete run/source provenance fingerprint matches; rebuild missing or corrupt shards |
| `force` | no (false) | explicitly replace a committed or interrupted destination; cannot be combined with `resume` (`kur-curate --force` is an equivalent one-run override) |
| `encoder` | no | `{"name": registered_name, "params": {...}}` |
| `span_steps` / `span_seconds` | no | declarative sub-track spans (mutually exclusive) |
| `clip_aggregate` | no (`"max"`) | `max`, `min`, or `mean` |
| `memory_budget` | no | positive byte budget that requires an exact bounded execution plan |
| `batch_size` | no (1024) | maximum segment rows in one bounded feature batch |
| `max_in_flight` | no (2) | maximum queued producer/consumer batches |
| `checkpoint_dir` / `spill_dir` | no | explicit non-empty local paths for reusable partitions and bounded scratch data |
| `approximation` | no (`"exact"`) | `"exact"` or `"approximate"`; unsupported approximate plans fail rather than changing semantics |

**Python-API-only knobs:** arbitrary `group_of` / `tags_of` callables and encoder instances.
`.join_video()` and `.transform()` calls are likewise recorded in Python pipeline
provenance but are not serialized as job keys; declarative jobs can still use plugin adapters,
scorers, and encoders.
Registered/data-defined equivalents use names plus parameters. If you need arbitrary code, call
`write_dataset(...)` directly — every supported job-spec execution key maps onto its kwargs, so graduating
from a job file to Python is mechanical.

## Experiment jobs (`kur-experiment`)

| key | required | value |
|---|---|---|
| `arms` | yes | `[{"name": str, "strategy": <inline/path/name>}, ...]` — names must be unique |
| `target_size` | yes | segments selected per arm (count-based, apples-to-apples) |
| `baseline` | no | arm name used as the JS-divergence reference; must match an arm |

Scoring happens **once** over the union of all arms' metrics; each arm then selects on the
shared pool. Output (stdout, and `--out` JSON): per-arm `n_selected` + per-metric `mean` and
`js_vs_baseline`.

## Report jobs (`kur-report`)

No extra keys — the shared keys above. Scores every segment and prints/writes per-metric
`mean/std/p50/p95/min/max` + histogram, plus tag and group counts. No selection is run;
configured whole-pool scorers are executed and included like other metrics.

## A complete curate job

```json
{
  "schema": {"name": "demo", "entity_axis": "agent", "sample_hz": 20,
             "fields": {"pos": {"kind": "vector", "dims": ["x","y","z"], "semantics": "position"}}},
  "adapter": {"type": "timeseries", "fields": {"pos": ["x","y","z"]}, "time_column": "t",
              "entity_column": "agent_id"},
  "sources": ["a.csv", "b.csv"],
  "segmenter": {"window": 4, "stride": 2, "unit": "seconds"},
  "strategy": {"define": {"speed": {"source": "agent.pos", "map": "speed", "reduce": "mean"}},
               "selection": {"method": "top-k", "weights": {"speed": 1.0}}},
  "output": {"shard_length": 64, "container": "npz",
             "modalities": {"pos": {"dtype": "float32", "shape": [3], "source": "field:pos"}}},
  "train_hours": 10, "test_hours": 1,
  "out": "out/run1", "seed": 0, "executor": "local", "workers": 8,
  "on_error": "skip", "resume": false
}
```

Habit: `kur-validate job.json` first — every load-time error (schema, adapter field maps,
strategy DSL/tags/scorers/references, segmenter, output spec) surfaces before any data is
read. `--check-sources` additionally stats that each source exists and runs the adapter's
`check_source` preflight — every built-in adapter has one: CSV/JSONL header sniff,
Parquet footer / HDF5 / NPZ-header / Zarr metadata (keys, step counts, and per-step widths
checked against the schema), HuggingFace builder features, curated-manifest membership —
still without reading data rows. (This is the only mode that touches adapters whose source
enumeration is expensive, e.g. HuggingFace.)
