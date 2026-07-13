# API reference — the public Python surface

Every name importable from `kuration` (plus `kuration.torch`), grouped by layer, with its real
signature. Optional-extra requirements are tagged (`[methods]`, `[parquet]`, `[array]`,
`[cloud]`, `[torch]`, `[hf]`, `[zarr]`). For concepts and recipes see the per-layer docs
([schema](schema.md) · [dsl](dsl.md) · [adapters](adapters.md) · [strategies](strategies.md) ·
[output](output.md)) and the [usage cookbook](usage.md).

## Entry points (start here)

| name | signature | does |
|---|---|---|
| `curate` | `(*, schema, adapter, sources, strategy, segmenter, target_size, seed=0, group_of=None, tags_of=None, test_frac=0.0, workers=1, on_error="skip", streaming=False) -> CurateResult` | score all sources, select a target **count** of segments |
| `write_dataset` | `(out_dir, *, schema, adapter, sources, strategy, segmenter, spec, encoder=None, train_hours, test_hours=0.0, sample_hz=None, seed=0, group_of=None, tags_of=None, clip_aggregate="max", span_steps=None, span_seconds=None, workers=1, on_error="skip", resume=False, streaming=False) -> dict` | score, fill a train/test **hour budget**, write shards + manifests + config |
| `run_experiment` | `(*, schema, adapter, sources, arms, segmenter, target_size, seed=0, baseline=None, group_of=None, tags_of=None, workers=1, on_error="skip", streaming=False) -> dict` | score once over the arms' union, select per arm, JS-divergence vs `baseline` |
| `dataset_report` | `(segments, *, metrics=None, hist_bins=10) -> dict` | per-metric distribution profile (+ tag/group counts) of a scored pool |

Shared kwargs: `group_of(source_id) -> key` sets each segment/track's `group` (for the group
cap and stratified splits); `tags_of(segment) -> iterable` overrides declarative tags;
`workers` parallelizes the feature pass (deterministic); `on_error` is `"skip"` (log + collect)
or `"raise"`; `streaming=True` is the out-of-core path.

- `CurateResult` — dataclass: `selected`, `all_segments`, `schema`, `strategy`, `splits`, `errors`.
- `write_dataset` returns `{"shards": [entries], "split": {source -> "train"/"test"}, "admitted": set, "results": {...counts}, "errors": [...]}`.
- `Arm(name, strategy)` — one experiment arm; `union_define(arms) -> dict` merges their `define`s.

## Fluent pipeline

The higher-level facade over the entry points above — no explicit schema/adapter/strategy
required; see [pipeline.md](pipeline.md) for the full operator reference.

| name | signature | does |
|---|---|---|
| `input` | `(src=None, *, schema=None, adapter=None, sources=None, fields=None, time=None, entity=None, hz=None, kinds=None, sample=200, globals=None, **adapter_kw) -> Pipeline` | sniff `src`, infer a schema (unless `schema=`), build the adapter — eagerly, loudly |
| `infer_schema` | `(sample, *, name="inferred", time=None, entity=None, hz=None, kinds=None) -> (SchemaSpec, hints)` | classify a sample (rows/dict-of-arrays/ticks) into a schema + adapter `fields=` hints |
| `Pipeline.window` | `(window, stride=None, unit="steps") -> Pipeline` | set the `Segmenter` |
| `Pipeline.measure` | `(name, spec=None, *, of=None, map=None, filter=None, scope=None) -> Pipeline` | declare a per-segment metric (bare name / reduce string / DSL dict / callable) |
| `Pipeline.filter` | `(what, op=None, value=None, **kw) -> Pipeline` | `("anomaly"/"dedup", **knobs)` pre-pass filter, or `(metric, op, value)` universal predicate |
| `Pipeline.organize` | `(metric, *args, bins=None) -> Pipeline` | add a quantile-bin axis |
| `Pipeline.score` | `(name, out=None, **params) -> Pipeline` | add a whole-pool `SegmentScorer` |
| `Pipeline.tag` | `(*sources, min_frac=None) -> Pipeline` | declarative tag sources (+ optional tag-coverage floor) |
| `Pipeline.split` | `(test=0.2) -> Pipeline` | source-level test split |
| `Pipeline.group` | `(fn) -> Pipeline` | `group_of(source_id) -> key` |
| `Pipeline.options` | `(**kw) -> Pipeline` | `workers, on_error, seed, streaming, resume, sample_hz` |
| `Pipeline.curate` | `(method=None, *, n=None, hours=None, test_hours=None, weights=None, target=None, normalize=None, group_cap=None, **method_params) -> Pipeline` | configure selection; exactly one of `n=`/`hours=` |
| `Pipeline.collect` | `(n=None) -> CurateResult` | terminal — count-path selection |
| `Pipeline.output` | `(path, *, container=None, shard_length=64, short_policy="keep", modalities=None, include=None, exclude=None, encoder=None, span_steps=None, span_seconds=None) -> dict` | terminal — write shards + manifests + `config.json` (embeds `"pipeline"` provenance) |
| `Pipeline.report` | `(**kw) -> dict` | terminal — feature pass only → `dataset_report` |
| `Pipeline.experiment` | `(arms: dict, n: int, baseline=None) -> dict` | terminal — multi-arm comparison (`arms` maps name → `Pipeline` or raw strategy dict) |
| `Pipeline.head` | `(k=5) -> list[Tick]` | inspector — first `k` ticks of the first source |
| `Pipeline.describe` | `() -> dict` | inspector — resolved `{schema, strategy, output, segmenter, options, sources}`, no data touched |

## Schema & data model

| name | signature | does |
|---|---|---|
| `SchemaSpec` | `.from_dict(d)` / `.to_dict()`; fields: `name, fields, entity_axis, sample_hz, event_streams, globals` | the declared vocabulary ([schema.md](schema.md)) |
| `FieldSpec` | `(name, kind, dims=(), semantics=None, bits={}, max_rate=None)` | one field: `kind` scalar/vector/categorical + semantics tag |
| `Tick` | `(time, entities, events={}, globals={})` | canonical per-timestep record — the adapter contract |
| `WindowBuffer` | `.numeric_field(name) -> (T,E,D)`, `.categorical_field(name) -> (codes, vocab)`, `.valid (T,E)`, `.events`, `.globals` | one window's arrays — what metrics see |
| `build_window` | `(schema, ticks) -> WindowBuffer` | generic, schema-driven extraction |
| `empty_like` | `(schema, T, E) -> WindowBuffer` | NaN/-1-initialized buffer (adapters/tests) |
| `Segment` | `(source_id, start_idx, end_idx, time_range, features={}, bin=(), tags=frozenset(), group=None)` | one scored window |
| `Segmenter` | `(window, stride, *, unit="steps", sample_hz=None)`; `.segments(source_id, ticks)`, `.stream(source_id, tick_iter)` | fixed-width overlapping windows (steps or seconds) |

## Metrics

| name | signature | does |
|---|---|---|
| `compile_metric` | `(spec, schema) -> fn(WindowBuffer) -> float` | compile + validate one closed-grammar spec ([dsl.md](dsl.md)) |
| `register_metric` | `@register_metric("name")` on `fn(WindowBuffer) -> float` | escape hatch: a Python per-segment metric |
| `register_compiled` | `(name, fn)` | same, non-decorator form |
| `FeatureEngine` | `(schema, *, define=None, registered=None, tags_spec=None)`; `.compute(ticks, segments)`, `.score_window(seg, window_ticks)` | runs all metrics over one buffer per segment |
| `score_sources` | `(*, schema, adapter, sources, define, registered, segmenter, tags_spec=None, group_of=None, tags_of=None, workers=1, on_error="skip", streaming=False) -> (segments, errors, source_tracks)` | the shared parallel feature pass (raises on an empty pool) |

## Input adapters

All take `schema` first and expose `.ticks(source_uri)`, plus a metadata-only
`.check_source(source_uri)` preflight (used by `kur-validate --check-sources`); see
[adapters.md](adapters.md).

| name | key kwargs | reads |
|---|---|---|
| `TimeseriesAdapter` | `fields, time_column=None, entity_column=None, globals=None` | CSV / single Parquet (`[parquet]`) / `from_rows(rows)` |
| `JsonlAdapter` | same as timeseries | `.jsonl` (one row-object per line) |
| `HDF5Adapter` | `fields, globals=None` | `.h5`, one file = one source/entity (`[array]`) |
| `NpzAdapter` | `fields, globals=None` | `.npz` (dep-free) |
| `ParquetDatasetAdapter` | `fields, path, source_per="part", time_column=None, entity_column=None, globals=None`; `.sources()` | a directory/glob of Parquet parts (`[parquet]`) |
| `HuggingFaceAdapter` | `fields, dataset, split="train", config=None, time_column=None, entity_column=None, globals=None, source_column=None, load_kwargs=None`; `.sources()` | a HF dataset (`[hf]`) |
| `ZarrAdapter` | `fields, globals=None` | a Zarr store (`[zarr]`; remote via `[cloud]`) |
| `CuratedDatasetAdapter` | `(dataset_dir_or_list, schema=None)`; `.sources()`, `.dataset_of` | kuration's own output, re-curatable |
| `schema_from_output` | `(output_spec_dict, *, name="curated", sample_hz=None, entity_axis="e") -> SchemaSpec` | derive a schema from a dataset's `config.json` output block |
| `register_adapter` / `get_adapter` | `(name, factory)` / `(name, **kwargs)` | registry (workers rebuild adapters by name) |
| `InputAdapter` | Protocol: `schema` + `ticks(uri)`; optional `check_source(uri)` + `sources_are_paths` | the seam to implement for bespoke formats |

## Selection

| name | signature | does |
|---|---|---|
| `load_strategy` | `(dict_or_path_or_builtin_name, schema) -> Strategy` | parse + validate a declarative strategy ([strategies.md](strategies.md)) |
| `Strategy` | fields: `define, axes, method, weights, normalize, target, tag_coverage, group_cap, predicates, feature_space, method_params, filters, tags, scorers, resolved` | the loaded strategy (its `.resolved` dict lands in `config.json`) |
| `bin_segments` | `(segments, axes) -> None` | assign quantile-bin tuples in place (tie-robust, NaN-loud) |
| `quantile_bins` | `(values, n_bins) -> edges` | raw equal-frequency edges |
| `select` | `(segments, strategy, *, target_size, tag_vocab=(), seed=0, workers=1) -> list[Segment]` | count-based selection + tag-coverage/group-cap post-passes |
| `segment_priorities` | `(segments, strategy, *, seed=0, workers=1) -> np.ndarray` | per-segment priority (what the hours fill consumes) |
| `assign_splits` | `(source_groups, test_frac, seed) -> {source -> "train"/"test"}` | leakage-free, group-stratified source split |
| `apply_filters` | `(segments, strategy, *, seed=0, workers=1) -> list[Segment]` | anomaly/dedup pre-pass (`[methods]`) |
| `method_priorities` | `(segments, strategy, *, seed=0, workers=1) -> np.ndarray` | feature-space methods (`[methods]`) |

## Scorers (whole-pool / learned)

| name | signature | does |
|---|---|---|
| `SegmentScorer` | Protocol: `score(segments, ctx) -> array`; optional `needs_buffers=True` | the learned/whole-pool seam |
| `register_scorer` / `get_scorer` | `(name, factory)` / `(name, **kwargs)` | registry (`action_residual` ships, `[methods]`) |
| `apply_scorers` | `(entries, segments, *, schema, adapter, seed=0) -> None` | run parsed scorers, write outputs into `seg.features` |

## Output

| name | signature | does |
|---|---|---|
| `OutputSpec` | `.from_dict({shard_length, short_policy?, container?, modalities})` | declared modalities + chunking ([output.md](output.md)) |
| `Modality` | `(name, dtype, shape, src_kind, src_arg)` | one declared modality |
| `OutputEncoder` / `PassthroughEncoder` | `timestep(record, ctx) -> {modality: array}` | the domain seam for derived tensors |
| `register_encoder` / `get_encoder` | `(name, factory)` / `(name, **kwargs)` | encoder registry |
| `Track` | `(source_id, entity_id, records, meta={})` | one entity's timestep sequence — the output unit |
| `tracks_from_ticks` | `(schema, ticks, source_id) -> list[Track]` | one track per entity |
| `write_track` | `(out_dir, track, spec, *, encoder=None, schema=None, split="all", resume=False, name_suffix="") -> [manifest entries]` | low-level: write one track's shards |
| `fill_to_hours` | `(items, *, train_hours, test_hours, max_group_overshoot=1.5, tag_coverage=None, seed=0) -> (admitted_ids, split)` | the hours-budget fill (rejects non-finite priorities) |

## Read side

| name | signature | does |
|---|---|---|
| `iter_shards` | `(dataset, *, split=None, modalities=None, trim=True, container=None, with_meta=False) -> Iterator[dict]` | shards back as `{modality -> (T, *shape)}` numpy arrays |
| `shard_entries` | `(dataset, *, split=None, container=None) -> (entries, container)` | manifest entries (paths rebased if the dataset moved) |
| `read_shard_arrays` | `(entry, container, *, modalities=None, trim=True) -> {name: array}` | one shard: trim pad, reshape, restore dtype |
| `kuration.torch.ShardDataset` | `(dataset, *, split=None, modalities=None, trim=True, container=None, transform=None, include_meta=False)` | map-style PyTorch `Dataset` over the above (`[torch]`) |

`dataset` is a dataset directory, a `dataset_manifest.json` path, or a loaded manifest dict.

## Manifests & I/O helpers

| name | signature | does |
|---|---|---|
| `build_config` / `write_config` | `(schema, strategy, *, results, seed, ...) -> dict` / `(path, config)` | the reproducibility record (`config.json`) |
| `dump_segment_manifest` | `(path, segments)` | JSON manifest of a scored/selected pool |
| `read_text` / `write_text` / `write_bytes` / `exists` | `(path, ...)` | path-agnostic I/O (local or `gs://`/`s3://`/`az://` via `[cloud]`) |
| `local_copy` | context manager: `(path) -> local file path` | stage a remote object for libs needing a real path |
