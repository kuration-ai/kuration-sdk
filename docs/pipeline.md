# The pipeline — `input(...)` → operators → terminals

The primary way to drive the SDK for the common case: no explicit `SchemaSpec`,
no explicit adapter, no raw strategy dict. `kuration.input(...)` sniffs a source, infers a
schema (via [`kuration.core.infer`](#schema-inference-kurationcoreinfer)), and returns an immutable `Pipeline`; every
operator returns a *new* `Pipeline` (nothing is mutated in place); a terminal compiles the
chain into the classic strategy dict and calls the existing engine
(`curate` / `write_dataset` / `dataset_report` / `run_experiment` — see [api.md](api.md)).
The explicit APIs in `strategies.md`/`schema.md`/`adapters.md` remain available for custom and
declarative workflows. Every compiled chain is inspectable via `.describe()` before you run it.

```python
from kuration import input

result = (
    input("logs/*.csv", time="t", entity="agent")
    .window(4, unit="seconds")
    .measure("speed", "mean", of="pos")
    .filter("speed", ">=", 1.0)
    .bin("speed", bins=8)
    .sample("equal", n=5000)
    .output("out/run1")
)
```

## `input(...)` — the source factory

`input(src, *, schema=None, adapter=None, sources=None, fields=None, time=None, entity=None,
hz=None, kinds=None, sample=200, globals=None, registry=None, **adapter_kw)` sniffs `src`'s shape and builds
the right adapter, eagerly — a bad source fails at `input()` time, not mid-pipeline.

| `src` form | resolves to |
|---|---|
| a path / glob string (`"a.csv"`, `"logs/*.parquet"`) | one sniffed adapter over the matched file(s); mixed extensions raise |
| a list of path strings | same, all must sniff to the same kind |
| an MP4/M4V/MOV path, glob, or list | `VideoAdapter`; one source per video with lazy `media:frame` references (`[video]`) |
| a directory of `.parquet` parts | `ParquetDatasetAdapter` |
| a pandas `DataFrame` (duck-typed via `.to_dict("records")`) | in-memory rows, one source `"mem"` |
| `list[dict]` (row objects) | in-memory rows, one source `"mem"` |
| `list[Tick]` | in-memory ticks, one source `"mem"` |
| `dict` of columns (all-scalar/array values, equal length) | in-memory rows, one source `"mem"` |
| `{source_id: list[dict]}` | in-memory rows, **multi-source** |
| `{source_id: list[Tick]}` | in-memory ticks, **multi-source** |
| `input(adapter=<InputAdapter or registered name>, sources=[...], schema=..., registry=...)` | escape hatch — no inference, `schema=` is required; a named adapter resolves only through the supplied registry snapshot |

File-kind sniffing (`kuration.core.infer.sniff`): `.csv`/`.parquet` → timeseries, `.jsonl` → JSONL
(one row-object per line), `.json` → a JSON array of row objects *or* a columnar dict of
equal-length lists, `.npz` → NpzAdapter, `.h5`/`.hdf5` → HDF5Adapter, a directory containing
`.parquet` parts → `ParquetDatasetAdapter`; `.mp4`/`.m4v`/`.mov` → `VideoAdapter`. Anything else raises with a pointer at
`adapter=`.

Notes:
- `registry=` creates an isolated snapshot for adapter lookup and every later extension lookup;
  ambient entry-point discovery does not mutate that pipeline.
- `schema=` (a `SchemaSpec` or an equivalent dict) skips inference entirely; `fields=` then
  overrides the column mapping instead of the identity default.
- In-memory sources set `Pipeline._in_memory = True`; workers can't rebuild them per-process,
  so `.options(workers=2)` on in-memory input raises at `collect()`/`output()`/`report()`/
  `experiment()` time (write the data to files first, or keep `workers=1`).
- `time=`/`entity=`/`hz=`/`kinds=` steer inference (see below) without forcing a full explicit
  schema.

## Schema inference (`kuration.core.infer`)

`input()` calls `infer_schema(sample, ...)` on a small sample (`sample=200` rows by default)
pulled by `sample_source`. It is deliberately conservative and loud: an ambiguous column
raises rather than guessing. Summary (full behavior: [`src/kuration/core/infer.py`](https://github.com/aintropy-solutions/kuration-sdk/blob/main/src/kuration/core/infer.py),
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

## Operator reference

Each operator returns a new `Pipeline`; the pipeline being extended is never mutated.
Importing `kuration.gameplay` or `kuration.selection` registers the domain methods below on this
same core type; custom domains and plugins follow the same registered-operator mechanism.

| operator | signature | compiles to |
|---|---|---|
| `.join_video` | `(videos, *, column="frame", on="timestamp", tolerance=None, mismatch="error", provider=None)` | a `JoinedMediaAdapter` with a resolved source→video mapping and lazy media column |
| `.pixel_shift` | `(*, controls=None, boundary="exclude")` | the built-in `aintropy/pixel-shift` operator |
| `.action_state_consistency` | `(*, baselines=None, anomaly_type="action_state", controls=None, calibration=None, boundary="exclude", frames=None, actions=None, color_space=None, verify=None, metadata_key="action_state_consistency", replace=False)` | dataset-mode only (audits an already-written dataset's shards); runs the built-in analyzer once and returns a pipeline carrying its per-clip verdict under `metadata["gameplay"][metadata_key]`. For a live pipeline, use `.pixel_shift(...).score("aintropy/action-state-consistency", out=..., baselines=..., controls=...)` instead -- see [action-state-consistency.md](action-state-consistency.md) |
| `.collision` | `(*, map_for_source, map_registry, dist_path, **thresholds)` | the built-in `aintropy/collision` operator (per-window CS2 wall-collision counts raycast against map geometry) |
| `.collision_directional` | `(*, map_registry, ramp_models_path, map_for_source=None, default_map=None, fallback_ramp=None, inputs=None, batch_size=None, metadata_key="directional_collision", replace=False, **thresholds)` | runs the built-in `aintropy/collision-directional` analyzer once over the tick stream and returns a pipeline carrying its verdict under `metadata["gameplay"][metadata_key]` |
| `.slam_motion_diversity` | `(*, axis="slam_path_length", bins=8)` | the built-in `aintropy/slam-motion-diversity` operator |
| `.slam_quality` | `(*, min_score=None, **quality_options)` | the built-in `aintropy/slam-quality` operator |
| `.balanced_selector` | `(feature, of, kind="std", bins=8, count=100)` | the built-in `kuration/balanced-selector` operator (`.measure(...).bin(...).sample("equal", ...)`) |
| `.rare_event_coverage` | `(of, feature="event_rate", count=100, weight=1.0)` | the built-in `kuration/rare-event-coverage` operator (`.measure(..., "rate", ...).sample("weighted", ...)`) |
| `.diverse_subset` | `(features, count=100, n_clusters=32, dedup_threshold=None)` | the built-in `kuration/diverse-subset` operator (optional `.filter("dedup", ...)` then `.sample("cluster-balanced", ...)`) |
| `.window` | `(window, stride=None, unit="steps")` | the `Segmenter` (stride defaults to `window`) |
| `.measure` | `(name, spec=None, *, of=None, map=None, filter=None, scope=None)` | a `strategy.define[name]` entry — 4 forms below |
| `.transform` | `(name, *, using=None, inputs=None, batch_size=None, device="cpu", **params)` | a registered bounded batch transform whose scalar outputs become selection features |
| `.filter` | `("anomaly"/"dedup", **knobs)` **or** `(metric, op, value)` | a pre-pass filter entry, or a `selection.predicates` entry |
| `.bin` | `(metric, bins=4)` (or positional `(metric, n)`) | a `binning.axes` entry; `.organize` is a compatibility alias |
| `.score` | `(name, out=None, **params)` | a `strategy.scorers` entry (output name defaults to `name`) |
| `.tag` | `(*sources, min_frac=None)` | a `tags.from` entry; `min_frac` sets `selection.tag_coverage` |
| `.split` | `(test=0.2)` | source-level test split (count path: `test_frac`; hours path: budget share) |
| `.group` | `(fn)` | `group_of(source_id) -> key` (group cap + stratified splits) |
| `.options` | `(mode=, executor=, workers=, on_error=, seed=, streaming=, resume=, sample_hz=, progress=)` | execution kwargs passed to the terminal; executor is `"local"` or `"ray"` |
| `.sample` | `(method=None, *, n=None, hours=None, test_hours=None, weights=None, target=None, normalize=None, group_cap=None, **method_params)` | `selection.method` + knobs; exactly one of `n=`/`hours=`; `.curate` is a compatibility alias |
| `.semantic` | `(field, "position"|"angular"|"bitmask", *, dims=None, bits=None, max_rate=None)` | a new validated `SchemaSpec` |
| `.as_position/.as_angular/.as_bitmask` | semantic convenience wrappers | `.semantic(...)` |
| `.at_rate` | `(hz)` | `SchemaSpec.sample_hz` |
| `.using` | `(Registry)` | explicit component-resolution overlay |
| `.with_metadata` | `(namespace, key, value, *, replace=False)` | caller-owned, namespaced strict-JSON metadata on a new pipeline |
| `.enrich` | `(analyzer_name, parameters=None, *, namespace=None, key=None, replace=False, mode=None, inputs=None, batch_size=None, analyzer=None, **options)` | runs a registered analyzer once and attaches its result, descriptor, request, and input-basis fingerprint to a new pipeline |
| `.apply` | `(name, **params)` | compatibility-only string dispatch; prefer the generated direct method |

`.measure`'s four forms:
```python
pipeline.measure("hp")                            # bare name -> mean of the auto-qualified field
pipeline.measure("hp_p95", "p95", of="hp")        # reduce string + of=/map=/filter=/scope= kwargs
pipeline.measure("hp_max", {"source": "e.hp", "reduce": "max"})  # full DSL dict
pipeline.measure("wlen", lambda w: float(w.T))    # callable(WindowBuffer) -> float
```
A bare token auto-qualifies against the schema: an entity field becomes `<entity_axis>.field`,
a global stays bare, an event-stream name raises (name a field on it: `"stream.field"`).

### Media joins and batch transforms

`input("clips/*.mp4")` produces media-only ticks with lazy `FrameRef` values. For multimodal
sources, `.join_video()` accepts either a concrete `{source_id: video_uri}` mapping or a callable;
the callable is resolved immediately and the concrete mapping is recorded in pipeline provenance.
`on="frame_index"` is an exact one-to-one join. `on="timestamp"` chooses the nearest presentation
timestamp within the required `tolerance`; drift and missing mappings fail unless
`mismatch="drop"` is explicitly selected.

```python
pipeline = (
    input("telemetry/*.parquet", time="timestamp")
    .join_video(lambda source: source.replace(".parquet", ".mp4"), tolerance=0.020)
    .window(64, stride=32)
    .transform("acme/visual-novelty", inputs=["media:frame"],
               batch_size=16, device="cuda")
    .bin("visual_novelty", bins=8)
    .sample("equal", n=5000)
    .options(executor="ray", workers=32)
)
```

Transforms are registered with `register_batch_transform()`/`register_transform()`. Inputs use
`field:`, `feature:`, or `media:` prefixes; declared media semantics are checked before decode.
Each output must be one finite scalar per stable segment ID, after which it is accepted anywhere
a measure/scorer feature is accepted: predicates, bins, weights, and feature-space methods.
Transforms that declare bounded lookbehind/lookahead read neighboring field/media inputs through
the typed `TransformContext`: `ctx.before(token)` and `ctx.after(token)`. The `rightmost-start`
ownership policy supplies immutable `ctx.owned_lengths`, one prefix length per window, so
overlapping windows produce deterministic event ownership. The rc2 batch keys remain available
during the release-candidate compatibility window. `inputs=` remaps a transform's declared input
roles for one use; the transform resolves them through `ctx.input_token(role)`/`ctx.inputs`, and
`ctx.valid_steps` reports each row's real timestep count within the NaN-padded `(B,T,E,D)` field
batch.
Framework imports and model loading remain inside the optional transform implementation; core
import and ordinary pipelines stay NumPy-only.

`.sample`'s method resolution: an explicit `method=` (case-insensitive alias, e.g. `"equal"` →
`uniform-bins`, `"top"`/`"topk"` → `top-k`) wins; otherwise `.bin(...)` axes present →
`uniform-bins`; otherwise `weights=` given → `top-k`; otherwise it raises (name a method).

`.filter`'s two shapes are disambiguated by the first argument: `"anomaly"`/`"dedup"` (with
knob kwargs, no `op`/`value`) is a pre-pass filter; anything else needs `.filter(metric, op,
value)` and the metric must already be a declared `.measure`/`.score`/`.transform` output. `op` accepts
aliases case-insensitively (`le`/`lte`/`leq`/`<=`, `ge`/`gte`/`geq`/`>=`, `lt`/`less`/`<`,
`gt`/`greater`/`>`, `eq`/`equal`/`=`/`==`, `ne`/`neq`/`!=`). **Predicates are a universal
pre-filter**: they gate every selection method (not only `"filter"`), on both the count and
hours paths — see [strategies.md](strategies.md).

### Reusable compositions as direct methods

`register_operator()` packages a validated sequence of existing operators under a discoverable
name. The handler receives a cloned pipeline and keyword parameters, and must return a pipeline
derived from that clone with at least one ordinary operator added:

```python
from kuration import register_operator

@register_operator(
    "acme/balance-motion",
    parameters={
        "type": "object",
        "properties": {
            "field": {"type": "string"},
            "bins": {"type": "integer", "minimum": 1},
        },
        "required": ["field"],
        "additionalProperties": False,
    },
)
def balance_motion(pipeline, *, field, bins=8):
    return (pipeline
            .measure("motion", "mean", of=field)
            .bin("motion", bins=bins))

pipeline = pipeline.balance_motion(field="velocity", bins=8)
```

Parameter names and required keys are checked before expansion; the ordinary operators then
perform their usual schema and value validation. `config.json` records the named call, component
descriptor/code fingerprint, parameters, and nested expansion. Operators resolve through the
pipeline's active registry, so `.using(local_registry)` and plugin entry points work normally.
Hyphens and dots in the local component name become underscores. When two namespaces register
the same local spelling, use the qualified form such as
`pipeline.acme__balance_motion(...)`. `Pipeline.apply(name, **params)` remains a compatibility
wrapper for serialized integrations, but normal Python code does not need it.

This is deliberately a composition seam, not a native execution-stage seam: a handler cannot
introduce a new terminal or bypass the validation and execution contracts of existing operators.
Use registered metrics, scorers, filters, methods, adapters, encoders, containers, or presets
when the behavior belongs to one of those stages.

See [building operators](operators.md) for transform-backed computation, contract tests,
discovery, and plugin packaging.

## Inspectors (cheap, no data touched / minimal read)

- `.head(k=5)` — the first `k` `Tick`s of the first source (reads only what it needs).
- `.preview()` — bounded source/field quality: time ranges, entity counts, missingness, numeric
  ranges, categorical values, and representative ticks. `.render()` gives terminal text.
- `.describe()` — `{"schema", "strategy", "output": None, "segmenter", "options", "sources"}`:
  the pipeline's fully resolved state, without running anything. `"strategy"` is the same raw
  dict a terminal would compile and load.
- `.explain()` — a typed, mapping-compatible `ExecutionPlan` containing the backend-neutral
  physical plan: fused operators, pipeline/data-parallel mode, barriers and reasons, pass counts,
  materialization status, resources, checkpoint policy, implicit choices, and actionable
  warnings. `.render(detailed=True)` gives the stage-by-stage form.
- `.explain(inspect_sources=True)` — additionally samples a bounded number of ticks to estimate
  candidate windows. This reads source data; ordinary `.explain()` remains metadata-only.
- `.expect(...)` — attach quality gates for previews and reviews. Input, selection, hours-budget,
  and artifact expectations share one closed grammar; each report evaluates the applicable keys.

Execution tuning lives in `.options(...)`: `memory_budget` is a byte budget used by physical
planning, `batch_size` bounds each segment batch, `max_in_flight` bounds producer/consumer
queue depth, `checkpoint_dir` and `spill_dir`
override artifact locations, and `approximation` is `"exact"` or `"approximate"`. Exact remains
the default; an operator without a declared approximate implementation never silently changes
algorithms. Setting `memory_budget` makes bounded execution an enforced contract: output rejects
unsupported exact global methods, global filters/scorers, unavailable Arrow support, and the
not-yet-bounded Ray feature-island path before reading source rows. Without a budget, those plans
retain the compatible reference implementation and appear as materialization warnings in
`.explain()`.

## Terminals

| terminal | signature | routes to |
|---|---|---|
| `.collect` | `(n=None)` | `curate()` — count-path selection → `CurateResult` |
| `.review` | `(n=None, *, path=None, examples=10, hist_bins=10, **output_options)` | count selection review; hours selection writes to `path=` and reviews admission/budget decisions without rescoring |
| `.output` | `(path, *, preset=None, spec=None, container=None, shard_length=64, short_policy="keep", modalities=None, include=None, exclude=None, encoder=None, span_steps=None, span_seconds=None, categorical=None, preset_params=None, force=False, dry_run=False, inspect_sources=False)` | validate/estimate with `dry_run=True`, otherwise write through the ordinary no-clobber `OutputSpec` path; `force=True` explicitly replaces an existing destination |
| `.report` | `(**kw)` | feature pass only (no selection) → `dataset_report` |
| `.stats` | `(**kwargs)` | `kuration.stats(pipeline, **kwargs)` — lifecycle statistics for the pipeline's current stage |
| `.analyze` | `(name, parameters=None, *, mode=None, inputs=None, batch_size=None, analyzer=None, **options)` | a registered stateful analyzer: `run_stream_analyzer` over the tick stream, or `run_dataset_analyzer`'s `check_dataset` hook for an artifact-backed pipeline |
| `.check` | `(policy=None, *, verify=False, **expectations)` | `evaluate_quality` over the cheapest evidence this pipeline can produce — the pipeline-API equivalent of `kur-check --policy` |
| `.experiment` | `(arms, n, baseline=None)` | `run_experiment`; `arms` maps name → `Pipeline` (its compiled strategy) or a raw strategy dict |

### Stateful analyzers: retain or return the result

An analyzer accumulates evidence across segments, so it cannot honestly be a per-segment
selection stage. A domain method such as `.collision_directional(...)` is the normal
pipeline-preserving spelling: it drives the analyzer once and returns a new pipeline with the
result attached.

```python
pipeline = (
    input({"clip1": rows})
    .window(64, stride=32)
    .options(streaming=True)
    .collision_directional(map_registry=map_registry, ramp_models_path=ramp_models_path)
)
verdict = pipeline.metadata["gameplay"]["directional_collision"]["result"]
print(verdict["total_collisions"], verdict["per_source"])
```

Not every domain analyzer stays tick-stream drivable this way — `aintropy/action-state-consistency`
is dataset-mode only (see [action-state-consistency.md](action-state-consistency.md)); for a live
pipeline it is a `SegmentScorer` instead, reached through `.score(...)` like any other scorer.

The pipeline remains usable: continue with `.measure()`, `.filter()`, `.sample()`, `.output()`,
or any other downstream operation. Caller metadata uses the same carrier:
`pipeline.with_metadata("project", "review", {"owner": "training"})`. Metadata is defensively
copied, strict JSON, preserved by immutable chaining, and included in written `config.json`
after private path values are made publishable. A repeated namespace/key is a loud error unless
`replace=True`.

Materialized analyzer evidence is bound to the input/schema/media/window/registry state that
produced it. Downstream selection and output calls preserve it; attempts to change that upstream
basis afterwards raise instead of silently making the result stale.

Use generic `.enrich(name, ...)` when a domain has no convenience operator. Use `.analyze(...)`
when the detached result itself is the goal:

```python
result = pipeline.analyze(
    "aintropy/collision-directional",
    {"map_registry": map_registry, "ramp_models_path": ramp_models_path},
)
```

Both return policies use the same driver:

- **Tick-stream mode** (default for a raw pipeline) drives the analyzer's `add_batch` hook over
  this pipeline's segmented tick stream, using the same bounded batch/decode machinery as
  `.transform(...)`: declared `field:`/`media:` inputs, per-input lookbehind/lookahead context,
  lazy media decoded once per batch. It is a single pass with one batch of windows resident, so
  it works under `.options(streaming=True)`. `segment_ids` are real
  `(source_id, start_idx, end_idx)` source coordinates.
- **Dataset mode** (default when the source *is* a written kuration dataset, i.e. the adapter
  names its artifacts) drives `check_dataset` and returns exactly what `run_dataset_analyzer()`
  and `kur-check --analyzer` return.
- `parameters` are the analyzer's construction parameters, validated against its registered
  descriptor; `**options` belong to the hook being driven; `inputs=`/`batch_size=` remap and size
  the stream path; `analyzer=` drives an instance you already own, so its accumulated evidence
  (per-instance ledgers, printed reports) stays yours after the call.
- Loud, never silent: an unknown analyzer raises the registry error, an analyzer missing the hook
  for the resolved mode raises a `ValueError` naming the hook it needs *and* the mode it does
  support, and an analyzer that measured zero segments raises instead of reporting a vacuous pass.

See [customization.md](customization.md#stateful-analyzers) for the registration contract and
[action-state-consistency.md](action-state-consistency.md) for the bundled gameplay analyzer.

### `.check()` — quality policies in Python

`.expect(...)` declares gates; `.check()` evaluates them and returns a `QualityGateResult`
without raising (`check_quality()` is the raising spelling). An artifact-backed pipeline is
inspected and gated exactly as `kur-check --policy` gates it; any other pipeline is gated on a
bounded input preview. Expectations merge from `.expect(...)`, a `policy` mapping (the
`{"expectations": {...}}` policy-file shape is accepted), then keywords.

```python
gates = input("logs/*.parquet").expect(max_missing={"hp": 0.01}).check(min_sources=8)
print(gates.render())
```

Selection, budget, and shard gates need selection or artifact evidence: run those through
`.review()` (which raises on failure) or gate the written dataset.

`.output` details:
- `dry_run=True` resolves and validates the actual output declaration without creating the
  destination. Add `inspect_sources=True` for bounded candidate-window and shard estimates;
  estimates state their sampling assumptions in the returned `ExecutionPlan`.
- `container` defaults from the path's suffix (`.hdf5`/`.h5` → hdf5, `.npz` → npz, `.parquet`
  → parquet, `.json` → json; otherwise npz) or pass it explicitly.
- Default `modalities` cover every **numeric** schema field (categorical fields are skipped
  with a message during a real write — pass `modalities=` to encode one); `include=`/`exclude=` filter the
  set by name (unknown names raise).
- `config.json` gains a `"pipeline"` key: the operator chain (`_ops`) as a JSON list of
  `{"op", "args", "kwargs"}`, in order — so a written dataset records how it was *asked for*,
  not just what was resolved into `"strategy"`.
- Presets are `numpy-timeseries`, `pytorch-sequences`, and `tabular`. Categorical policy is
  explicit: `error`, `omit`, `integer-vocab` (with `categorical_vocabs`), or `encoder`.
- `.options(mode="explore"|"strict")` resolves named bundles into ordinary options. Explicit
  options win; strict mode raises on source errors and implicit categorical omission.
- The count path rewrites its selection deterministically per run, so `resume=True` on it
  raises (hours path only).
- Output destinations are protected by default. Pass `force=True` to replace an existing
  committed or interrupted artifact; selection completes before the old commit is invalidated.
- A destination-scoped writer lease prevents concurrent pipeline, classic, or import publishers
  from interleaving. Container dependencies and count-output memory minima are checked before
  publication begins.

`.review()` intentionally materializes the complete candidate pool so it can compare it with the
selection. Run the output dry run first for a large corpus; use `preview()` when you only need a
bounded look at raw input quality. An hours review instead reuses the output writer's bounded
decision summary:

```python
review = (
    pipeline.sample("top-k", hours=10, test_hours=1, weights={"quality": 1})
    .expect(min_budget_utilization=0.98, max_skipped_sources=0)
    .review(path="out/run1", preset="numpy-timeseries", categorical="omit")
)
print(review.render())
```

## Loud-error philosophy

Every operator validates eagerly against the schema and the chain's own declared names — the
same "closed grammar, loud validation" invariant as the classic API (see the
[architecture invariants](architecture.md#invariants-dont-break-these)). Examples
you'll hit directly:
- `.measure(name, ...)` on an unknown field/undeclared name, a bad DSL dict, or a duplicate
  `name` — raises immediately, not at `collect()`/`output()` time.
- `.filter(metric, op, value)` on a metric that isn't a declared measure/callable/scorer/transform
  output, or an unrecognized `op` — raises with the known aliases listed.
- `.bin(metric, ...)` on an unknown or already-used axis metric — raises.
- `.sample()` with both `n=` and `hours=` (or neither), an unresolvable method, or an unknown
  method alias — raises.
- `.options(**kw)` with an unknown key, `executor` outside `{"local", "ray"}`, or `on_error`
  outside `{"skip", "raise"}` — raises.
- `.output(..., include=[...])`/`exclude=[...]` naming an unknown modality — raises.
- `workers > 1` with in-memory input, or `resume=True` on the count path — raise at the
  terminal, with the reason and the fix.

## Worked example

```python
from kuration import input, iter_shards

pipeline = (
    input({"s0": rows0, "s1": rows1}, time="t")   # multi-source in-memory rows
    .window(16)
    .measure("hp")
    .filter("hp", "<=", 50)                        # universal predicate: drop bad windows
    .measure("action_entropy", "entropy", of="action")
    .bin("action_entropy", bins=4)                  # diversity axis
    .sample("equal", n=12)                          # uniform-bins, count path
    .options(on_error="raise")
)

# Validate the output choice and estimate work before writing anything.
print(
    pipeline.output("out/run1", container="hdf5", dry_run=True,
                    inspect_sources=True).render()
)

result = pipeline.output("out/run1", container="hdf5")  # needs the [array] extra

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
