# Job specs — the CLI reference

The `kur-*` CLIs run a whole pipeline from **one JSON document**. This page enumerates every
recognized key per job kind. The same information ships machine-readably as a JSON Schema:
`kur-validate --print-schema` dumps it (point your editor's `$schema` at it for autocomplete),
and `kur-validate job.json` preflights any job in milliseconds — no data read.

```bash
kur-validate   job.json [--check-sources] [--json-schema] [--print-schema]
kur-curate     job.json                    # curate job -> hours-budget shards
kur-experiment job.json [--out cmp.json]   # experiment job -> per-arm JS-vs-baseline
kur-report     job.json [--out rep.json]   # report job -> metric distribution profile
```

Every CLI accepts `--debug`: failures normally print a one-line `error: ...` and exit 1;
`--debug` re-raises for the full traceback.

The job **kind** is inferred: `arms` present → experiment; `output`/`train_hours` present →
curate; otherwise → report.

## Keys shared by all job kinds

| key | required | value |
|---|---|---|
| `schema` | unless the adapter derives one | inline `SchemaSpec` dict **or a path** to a JSON file ([schema.md](schema.md)). The curated adapter derives its own — omit there. |
| `adapter` | yes | `{"type": <registered name>, ...kwargs}` — type is `timeseries`, `jsonl`, `hdf5`, `npz`, `parquet-dataset`, `huggingface`, `zarr`, `curated`, or your own `register_adapter` name; the remaining keys are that adapter's constructor kwargs ([adapters.md](adapters.md)) |
| `sources` | no | explicit list of source URIs (local paths or `gs://`/`s3://`/`az://`). Omit to use the adapter's own `sources()` (curated / parquet-dataset / huggingface enumerate) |
| `plugins` | no | importable registration modules loaded before adapter/strategy construction |
| `segmenter` | yes | `{"window": W, "stride": S, "unit": "steps"\|"seconds", "sample_hz"?: hz}` — `sample_hz` defaults to the schema's |
| `strategy` | yes (per-arm for experiments) | inline strategy dict or a path to one ([strategies.md](strategies.md)) |
| `seed` | no (0) | RNG seed — same seed + sources → same result, regardless of `workers` |
| `workers` | no (1) | parallel feature pass over a process pool |
| `on_error` | no (`"skip"`) | `"skip"` logs + skips an unreadable source; `"raise"` fails fast. Either way an **empty** pool raises |
| `streaming` | no (false) | out-of-core for streaming-capable adapters; NPZ and partitioned-Parquet dataset mode reject it |

Run CLIs show throttled phase progress when stderr is an interactive terminal. Redirected output
stays clean; pass `--quiet` to suppress interactive progress explicitly.

## Curate jobs (`kur-curate`)

| key | required | value |
|---|---|---|
| `output` | yes | an `OutputSpec` dict: `{"shard_length", "short_policy"?, "container"?, "presence_mask"?, "modalities": {...}}` ([output.md](output.md)) |
| `train_hours` | yes | train budget in hours (the emitted set is the strategy's *best* N hours) |
| `test_hours` | no (0) | test budget; sources split leakage-free |
| `out` | yes | output directory — local path or cloud URL |
| `resume` | no (false) | skip existing shards only when the complete run/source provenance fingerprint matches |

**Python-API-only knobs** (not JSON-expressible or not wired into the job spec):
`group_of` / `tags_of` (callables), `clip_aggregate`, `span_steps` / `span_seconds` (sub-track
spans), `encoder=` instances, and `sample_hz` override. If you need those, call
`write_dataset(...)` directly — every job-spec key maps 1:1 onto its kwargs, so graduating
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
`mean/std/p50/p95/min/max` + histogram, plus tag and group counts. No selection, no scorers.

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
  "out": "out/run1", "seed": 0, "workers": 8, "on_error": "skip", "resume": false
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
