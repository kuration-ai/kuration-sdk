# The fluent pipeline — `input(...)` → operators → terminals

A second, higher-level way to drive the SDK for the common case: no explicit `SchemaSpec`,
no explicit adapter, no raw strategy dict. `kuration.input(...)` sniffs a source, infers a
schema (via [`kuration.infer`](#schema-inference)), and returns an immutable `Pipeline`; every
operator returns a *new* `Pipeline` (nothing is mutated in place); a terminal compiles the
chain into the classic strategy dict and calls the existing engine
(`curate` / `write_dataset` / `dataset_report` / `run_experiment` — see [api.md](api.md)).
Nothing here replaces the classic API in `strategies.md`/`schema.md`/`adapters.md` — it's a
thin, validated facade over it, and every compiled chain is inspectable via `.describe()`
before you run anything.

```python
from kuration import input

result = (
    input("logs/*.csv", time="t", entity="agent")
    .window(4, unit="seconds")
    .measure("speed", "mean", of="pos")
    .filter("speed", ">=", 1.0)
    .organize("speed", bins=8)
    .curate("equal", n=5000)
    .output("out/run1")
)
```

## `input(...)` — the source factory

`input(src, *, schema=None, adapter=None, sources=None, fields=None, time=None, entity=None,
hz=None, kinds=None, sample=200, globals=None, **adapter_kw)` sniffs `src`'s shape and builds
the right adapter, eagerly — a bad source fails at `input()` time, not mid-pipeline.

| `src` form | resolves to |
|---|---|
| a path / glob string (`"a.csv"`, `"logs/*.parquet"`) | one sniffed adapter over the matched file(s); mixed extensions raise |
| a list of path strings | same, all must sniff to the same kind |
| a directory of `.parquet` parts | `ParquetDatasetAdapter` |
| a pandas `DataFrame` (duck-typed via `.to_dict("records")`) | in-memory rows, one source `"mem"` |
| `list[dict]` (row objects) | in-memory rows, one source `"mem"` |
| `list[Tick]` | in-memory ticks, one source `"mem"` |
| `dict` of columns (all-scalar/array values, equal length) | in-memory rows, one source `"mem"` |
| `{source_id: list[dict]}` | in-memory rows, **multi-source** |
| `{source_id: list[Tick]}` | in-memory ticks, **multi-source** |
| `input(adapter=<InputAdapter or registered name>, sources=[...], schema=...)` | escape hatch — no inference, `schema=` is required |

File-kind sniffing (`kuration.infer.sniff`): `.csv`/`.parquet` → timeseries, `.jsonl` → JSONL
(one row-object per line), `.json` → a JSON array of row objects *or* a columnar dict of
equal-length lists, `.npz` → NpzAdapter, `.h5`/`.hdf5` → HDF5Adapter, a directory containing
`.parquet` parts → `ParquetDatasetAdapter`. Anything else raises with a pointer at
`adapter=`.

Notes:
- `schema=` (a `SchemaSpec` or an equivalent dict) skips inference entirely; `fields=` then
  overrides the column mapping instead of the identity default.
- In-memory sources set `Pipeline._in_memory = True`; workers can't rebuild them per-process,
  so `.options(workers=2)` on in-memory input raises at `collect()`/`output()`/`report()`/
  `experiment()` time (write the data to files first, or keep `workers=1`).
- `time=`/`entity=`/`hz=`/`kinds=` steer inference (see below) without forcing a full explicit
  schema.

## Schema inference (`kuration.infer`)

`input()` calls `infer_schema(sample, ...)` on a small sample (`sample=200` rows by default)
pulled by `sample_source`. It is deliberately conservative and loud: an ambiguous column
raises rather than guessing. Summary (full behavior: [`kuration/infer.py`](../kuration/infer.py),
concepts: [schema.md](schema.md)):

- **Time column** — first case-insensitive match among `t, time, timestamp, ts, step, frame,
  tick`, or the explicit `time=`. If numeric, `sample_hz` is derived from the median gap
  between distinct sorted values (or pass `hz=` to force it).
- **Entity column** — first match among `entity, entity_id, id, agent, player, device,
  object`, or the explicit `entity=`; becomes `schema.entity_axis`.
- **Field kind** — `bool`/`str` → `categorical`; `int`/`float` → `scalar`; a list/tuple of
  numbers (fixed-length across the sample) → `vector` with dims `d0..dN`. Mixed types in one
  column, empty/ragged list cells, non-numeric list elements, or an all-`None` column all
  raise; `kinds={"col": "scalar"|"categorical"}` disambiguates a column (not vectors — those
  need an explicit `schema=`).
- Column names may not contain `.` (the metric DSL parses sources on it).
- dict-of-arrays / `list[Tick]` samples classify the same way over 1-D/2-D numpy arrays or
  tick `entities` fields respectively.

`infer_schema(sample, *, name="inferred", time=None, entity=None, hz=None, kinds=None) ->
(SchemaSpec, hints)` is public if you want the inferred schema without going through
`input()`; `hints["fields"]` is exactly the `fields=` mapping `TimeseriesAdapter` expects.

## Operators (each returns a new `Pipeline`)

| operator | signature | compiles to |
|---|---|---|
| `.window` | `(window, stride=None, unit="steps")` | the `Segmenter` (stride defaults to `window`) |
| `.measure` | `(name, spec=None, *, of=None, map=None, filter=None, scope=None)` | a `strategy.define[name]` entry — 4 forms below |
| `.filter` | `("anomaly"/"dedup", **knobs)` **or** `(metric, op, value)` | a pre-pass filter entry, or a `selection.predicates` entry |
| `.organize` | `(metric, bins=4)` (or positional `(metric, n)` / compat `(metric, "bin", n)`) | a `binning.axes` entry |
| `.score` | `(name, out=None, **params)` | a `strategy.scorers` entry (output name defaults to `name`) |
| `.tag` | `(*sources, min_frac=None)` | a `tags.from` entry; `min_frac` sets `selection.tag_coverage` |
| `.split` | `(test=0.2)` | source-level test split (count path: `test_frac`; hours path: budget share) |
| `.group` | `(fn)` | `group_of(source_id) -> key` (group cap + stratified splits) |
| `.options` | `(workers=, on_error=, seed=, streaming=, resume=, sample_hz=)` | execution kwargs passed to the terminal |
| `.curate` | `(method=None, *, n=None, hours=None, test_hours=None, weights=None, target=None, normalize=None, group_cap=None, **method_params)` | `selection.method` + knobs; exactly one of `n=`/`hours=` |

`.measure`'s four forms:
```python
.measure("hp")                            # bare name -> mean of the auto-qualified field
.measure("hp_p95", "p95", of="hp")        # reduce string + of=/map=/filter=/scope= kwargs
.measure("hp_max", {"source": "e.hp", "reduce": "max"})   # a full DSL dict, passed through
.measure("wlen", lambda w: float(w.T))    # a callable(WindowBuffer) -> float (registered)
```
A bare token auto-qualifies against the schema: an entity field becomes `<entity_axis>.field`,
a global stays bare, an event-stream name raises (name a field on it: `"stream.field"`).

`.curate`'s method resolution: an explicit `method=` (case-insensitive alias, e.g. `"equal"` →
`uniform-bins`, `"top"`/`"topk"` → `top-k`) wins; otherwise `.organize(...)` axes present →
`uniform-bins`; otherwise `weights=` given → `top-k`; otherwise it raises (name a method).

`.filter`'s two shapes are disambiguated by the first argument: `"anomaly"`/`"dedup"` (with
knob kwargs, no `op`/`value`) is a pre-pass filter; anything else needs `.filter(metric, op,
value)` and the metric must already be a declared `.measure`/`.score` output. `op` accepts
aliases case-insensitively (`le`/`lte`/`leq`/`<=`, `ge`/`gte`/`geq`/`>=`, `lt`/`less`/`<`,
`gt`/`greater`/`>`, `eq`/`equal`/`=`/`==`, `ne`/`neq`/`!=`). **Predicates are a universal
pre-filter**: they gate every selection method (not only `"filter"`), on both the count and
hours paths — see [strategies.md](strategies.md).

## Inspectors (cheap, no data touched / minimal read)

- `.head(k=5)` — the first `k` `Tick`s of the first source (reads only what it needs).
- `.describe()` — `{"schema", "strategy", "output": None, "segmenter", "options", "sources"}`:
  the pipeline's fully resolved state, without running anything. `"strategy"` is the same raw
  dict a terminal would compile and load.

## Terminals

| terminal | signature | routes to |
|---|---|---|
| `.collect` | `(n=None)` | `curate()` — count-path selection → `CurateResult` |
| `.output` | `(path, *, container=None, shard_length=64, short_policy="keep", modalities=None, include=None, exclude=None, encoder=None, span_steps=None, span_seconds=None)` | `.curate(hours=...)` → `write_dataset`; `.curate(n=...)` → select then write each segment as a sub-track |
| `.report` | `(**kw)` | feature pass only (no selection) → `dataset_report` |
| `.experiment` | `(arms, n, baseline=None)` | `run_experiment`; `arms` maps name → `Pipeline` (its compiled strategy) or a raw strategy dict |

`.output` details:
- `container` defaults from the path's suffix (`.hdf5`/`.h5` → hdf5, `.npz` → npz, `.parquet`
  → parquet, `.json` → json; otherwise npz) or pass it explicitly.
- Default `modalities` cover every **numeric** schema field (categorical fields are skipped
  with a printed note — pass `modalities=` to encode one); `include=`/`exclude=` filter the
  set by name (unknown names raise).
- `config.json` gains a `"pipeline"` key: the operator chain (`_ops`) as a JSON list of
  `{"op", "args", "kwargs"}`, in order — so a written dataset records how it was *asked for*,
  not just what was resolved into `"strategy"`.
- The count path rewrites its selection deterministically per run, so `resume=True` on it
  raises (hours path only).

## Loud-error philosophy

Every operator validates eagerly against the schema and the chain's own declared names — the
same "closed grammar, loud validation" invariant as the classic API (see CLAUDE.md). Examples
you'll hit directly:
- `.measure(name, ...)` on an unknown field/undeclared name, a bad DSL dict, or a duplicate
  `name` — raises immediately, not at `collect()`/`output()` time.
- `.filter(metric, op, value)` on a metric that isn't a declared measure/callable/scorer
  output, or an unrecognized `op` — raises with the known aliases listed.
- `.organize(metric, ...)` on an unknown or already-used axis metric — raises.
- `.curate()` with both `n=` and `hours=` (or neither), an unresolvable method, or an unknown
  method alias — raises.
- `.options(**kw)` with an unknown key, or `on_error` outside `{"skip", "raise"}` — raises.
- `.output(..., include=[...])`/`exclude=[...]` naming an unknown modality — raises.
- `workers > 1` with in-memory input, or `resume=True` on the count path — raise at the
  terminal, with the reason and the fix.

## Worked example

```python
from kuration import input, iter_shards

result = (
    input({"s0": rows0, "s1": rows1}, time="t")   # multi-source in-memory rows
    .window(16)
    .measure("hp")
    .filter("hp", "<=", 50)                        # universal predicate: drop bad windows
    .measure("action_entropy", "entropy", of="action")
    .organize("action_entropy", bins=4)             # diversity axis
    .curate("equal", n=12)                          # uniform-bins, count path
    .options(on_error="raise")
    .output("out/run1", container="hdf5")           # needs the [array] extra
)

for arrays in iter_shards("out/run1", split="train"):
    arrays["hp"]         # (16,) float32
```

See `examples/04_operator_pipeline.py` for this exact chain, run end to end (plus an
`hours=` variant, `.describe()`, and reading `config.json`'s `"pipeline"` provenance).

## When to drop to the classic API

Reach for `SchemaSpec` / `load_strategy` / an explicit `InputAdapter` / `curate` /
`write_dataset` directly (see [strategies.md](strategies.md), [schema.md](schema.md),
[adapters.md](adapters.md)) instead of `input(...)` when:
- you need a **custom `InputAdapter`** or **`OutputEncoder`** (a bespoke raw format or a
  derived output tensor) — the pipeline builds only the shipped adapters;
- you need **`workers > 1`** over **in-memory** data — parallel workers rebuild adapters by
  name in each process, so they can't see in-process Python objects; write the data to files
  first, or use `input(adapter=..., sources=[...])` with a registered adapter;
  file-backed `input(...)` sources work fine with `workers>1`;
- inference would be ambiguous or wrong for your data (mixed-type columns, vectors from a
  dict-of-arrays store with unusual dtypes, a time/entity column inference can't find) — pass
  an explicit `schema=` (and `fields=` if needed);
- you want a strategy or output spec authored as a **JSON job file** for the `kur-*` CLIs
  (see [jobs.md](jobs.md)) rather than a Python chain.
