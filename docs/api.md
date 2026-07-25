# API reference — the public Python surface

Every name importable from `kuration` (plus the public `kuration.core`, `kuration.core.torch`,
and `kuration.gameplay` domain),
grouped by layer, with its real
signature. Optional-extra requirements are tagged (`[methods]`, `[parquet]`, `[array]`,
`[video]`, `[ray]`,
`[cloud]`, `[torch]`, `[hf]`, `[zarr]`, `[gameplay]`, `[pixel-shift]`, `[action-state]`, `[slam]`,
`[slam-viz]`). For concepts and recipes see the per-layer docs
([schema](schema.md) · [dsl](dsl.md) · [adapters](adapters.md) · [strategies](strategies.md) ·
[output](output.md)) and the [usage cookbook](usage.md).

Signatures below use `/` and `*` with their normal Python meanings. Unless a row says otherwise,
invalid names, shapes, ranges, or incompatible options raise `ValueError`; reaching an optional
capability without its extra raises `ImportError` with the install command; source/storage errors
propagate as the corresponding I/O or parser exception. User callbacks, registered components,
and encoders may raise their own exceptions. Terminal return values are typed or mapping-compatible
objects as noted below.

## Entry points (start here)

| name | signature | does |
|---|---|---|
| `__version__` | `str` | installed `kuration-sdk` distribution version |
| `curate` | `(*, schema, adapter, sources, strategy, segmenter, target_size, seed=0, group_of=None, tags_of=None, test_frac=0.0, workers=1, on_error="skip", streaming=False, progress=None, transforms=(), registry=None, executor="local") -> CurateResult` | score/transform all sources, select a target **count** of segments |
| `write_dataset` | `(out_dir, *, schema, adapter, sources, strategy, segmenter, spec, encoder=None, train_hours, test_hours=0.0, sample_hz=None, seed=0, group_of=None, tags_of=None, clip_aggregate="max", span_steps=None, span_seconds=None, workers=1, on_error="skip", resume=False, streaming=False, progress=None, transforms=(), registry=None, executor="local", memory_budget=None, batch_size=1024, checkpoint_dir=None, spill_dir=None, approximation="exact", max_in_flight=2) -> dict` | score/transform, fill a train/test **hour budget**, write shards + manifests + config; a memory budget selects the exact bounded checkpoint path |
| `run_experiment` | `(*, schema, adapter, sources, arms, segmenter, target_size, seed=0, baseline=None, group_of=None, tags_of=None, workers=1, on_error="skip", streaming=False, progress=None, executor="local") -> dict` | score once over the arms' union, select per arm, JS-divergence vs `baseline` |
| `dataset_report` | `(segments, *, metrics=None, hist_bins=10) -> dict` | per-metric distribution profile (+ tag/group counts) of a scored pool |
| `stats` | `(subject, *, hist_bins=10, max_sources=3, max_ticks_per_source=10000, examples=5, max_values=100000, content=True, verify=False, progress=None, registry=None) -> StatsReport` | unified statistics for a pipeline, count selection, write result, typed report, or existing artifact path |
| `progress_callback` | `(progress="auto") -> ProgressCallback \| None` | resolve `"auto"`, `"notebook"`, `"terminal"`, `"cli"`, `"off"`, a boolean, or an existing callback; pipeline users normally select this through `.options(progress=...)` |
| `kuration.core` | module | explicit import facade for the domain-agnostic top-level API |
| `kuration.gameplay.slam_pipeline` | `(source, output=None, *, depth_model=None, embedding_model=None, embeddings=None, depth_model_id=None, embedding_model_id=None, runtime_factory=None, join_video=False, tolerance=None, mismatch="error", provider=None, run_if_needed=True, decoder_options=None, export_options=None, resume=False, cache_key=None, progress=None, preflight=True, require_gpu=False, **runtime_options) -> Pipeline` | convenience function—not a class—that reconstructs raw input when needed or opens an existing SLAM manifest, then returns the sole curation pipeline type |
| `kuration.gameplay.action_state_checker` | `(*, baselines, anomaly_type="action_state") -> Analyzer` | construct the action-state consistency analyzer without a component-name string |
| `kuration.gameplay.compare_videos` | `(left, right, *, labels=None, provider=None) -> VideoComparison` | create a synchronized notebook/HTML gameplay review without eagerly importing PyAV or Matplotlib |

Shared kwargs: `group_of(source_id) -> key` sets each segment/track's `group` (for the group
cap and stratified splits); `tags_of(segment) -> iterable` overrides declarative tags;
`sources` contains non-empty string identifiers/URIs (use `str(path)` for `pathlib.Path` values);
`executor` selects ordered local or optional Ray source work; `workers` caps concurrent local
workers or submitted Ray tasks (deterministic); `on_error` is `"skip"` (log + collect)
or `"raise"`; `streaming=True` is the out-of-core path; `progress(event_dict)` receives phase
events without coupling execution to a particular UI.

- `CurateResult` — dataclass: `selected`, `all_segments`, `schema`, `strategy`, `splits`, `errors`,
  scored `source_tracks`, optional bounded-execution metadata in `execution`, and `seed`.
- `write_dataset` returns `{"dataset": out_dir, "shards": [entries], "split": {source -> "train"/"test"}, "admitted": set, "results": {...counts}, "errors": [...]}`.
- `Arm(name, strategy)` — one experiment arm; `union_define(arms) -> dict` merges their `define`s.

## Fluent pipeline

The higher-level facade over the entry points above — no explicit schema/adapter/strategy
required; see [pipeline.md](pipeline.md) for the full operator reference.

| name | signature | does |
|---|---|---|
| `input` | `(src=None, *, schema=None, adapter=None, sources=None, fields=None, time=None, entity=None, hz=None, kinds=None, sample=200, globals=None, registry=None, **adapter_kw) -> Pipeline` | sniff `src`, infer a schema (unless `schema=`), build the adapter — eagerly, loudly; an explicit registry also resolves named adapters in isolation |
| `infer_schema` | `(sample, *, name="inferred", time=None, entity=None, hz=None, kinds=None) -> (SchemaSpec, hints)` | classify a sample (rows/dict-of-arrays/ticks) into a schema + adapter `fields=` hints |
| `Pipeline.window` | `(window, stride=None, unit="steps") -> Pipeline` | set the `Segmenter` |
| `Pipeline.join_video` | `(videos, *, column="frame", on="timestamp", tolerance=None, mismatch="error", provider=None) -> Pipeline` | attach a strictly aligned lazy video media column |
| `Pipeline.pixel_shift` | `(*, controls=None, boundary="exclude") -> Pipeline` | registered by `kuration.gameplay` as `aintropy/pixel-shift`; add the DIAMOND action/pixel-motion features (`[pixel-shift]` when frames are measured) |
| `Pipeline.slam_motion_diversity` | `(*, axis="slam_path_length", bins=8) -> Pipeline` | registered by `kuration.gameplay` as `aintropy/slam-motion-diversity`; add SLAM motion features and bin one diversity axis |
| `Pipeline.slam_quality` | `(*, min_score=None, min_tracking_confidence=0.5, translation_ratio=10.0, translation_mad_multiplier=8.0, max_translation_step=None, max_turn_degrees=90.0) -> Pipeline` | registered by `kuration.gameplay` as `aintropy/slam-quality`; add SLAM quality features and optionally filter weak windows |
| `Pipeline.transform` | `(name, *, using=None, inputs=None, batch_size=None, device="cpu", **params) -> Pipeline` | add a registered bounded scalar-feature transform |
| `Pipeline.using` | `(registry) -> Pipeline` | resolve adapters, metrics, scorers, encoders, methods, filters, containers, and presets through an explicit `Registry` overlay |
| registered operator methods | `(**params) -> Pipeline` | every operator is callable by normalized local name (`acme/signal-diversity` → `signal_diversity`) or qualified name (`acme__signal_diversity`); custom modules and plugins use the same flow |
| `Pipeline.apply` | `(name, **params) -> Pipeline` | compatibility-only string dispatch; normal Python code uses the generated operator method |
| `Pipeline.semantic` | `(field, semantics, *, dims=None, bits=None, max_rate=None) -> Pipeline` | refine an inferred field as `"position"`, `"angular"`, or `"bitmask"` |
| `Pipeline.as_position` | `(field, *, dims=None, max_rate=None) -> Pipeline` | position-semantic convenience wrapper |
| `Pipeline.as_angular` | `(field, *, max_rate=None) -> Pipeline` | angular-semantic convenience wrapper |
| `Pipeline.as_bitmask` | `(field, *, bits) -> Pipeline` | bitmask-semantic convenience wrapper |
| `Pipeline.at_rate` | `(hz) -> Pipeline` | set `SchemaSpec.sample_hz` when inference could not determine it |
| `Pipeline.measure` | `(name, spec=None, *, of=None, map=None, filter=None, scope=None) -> Pipeline` | declare a per-segment metric (bare name / reduce string / DSL dict / callable) |
| `Pipeline.filter` | `(what, op=None, value=None, **kw) -> Pipeline` | `("anomaly"/"dedup", **knobs)` pre-pass filter, or `(metric, op, value)` universal predicate |
| `Pipeline.bin` | `(metric, *args, bins=None) -> Pipeline` | add a quantile-bin axis (`organize` remains as a compatibility alias) |
| `Pipeline.organize` | `(metric, *args, bins=None) -> Pipeline` | compatibility alias for `Pipeline.bin` |
| `Pipeline.score` | `(name, out=None, **params) -> Pipeline` | add a whole-pool `SegmentScorer` |
| `Pipeline.tag` | `(*sources, min_frac=None) -> Pipeline` | declarative tag sources (+ optional tag-coverage floor) |
| `Pipeline.split` | `(test=0.2) -> Pipeline` | source-level test split |
| `Pipeline.group` | `(fn) -> Pipeline` | `group_of(source_id) -> key` |
| `Pipeline.options` | `(**kw) -> Pipeline` | `mode, executor, workers, on_error, seed, streaming, resume, sample_hz, progress`; progress accepts a callback or `"auto"`, `"notebook"`, `"terminal"`, `"cli"`, `"off"` |
| `Pipeline.expect` | `(**expectations) -> Pipeline` | attach closed-grammar quality gates evaluated by `preview()` or `review()` |
| `Pipeline.sample` | `(method=None, *, n=None, hours=None, test_hours=None, weights=None, target=None, normalize=None, group_cap=None, **method_params) -> Pipeline` | configure selection; exactly one of `n=`/`hours=` (`curate` remains as a compatibility alias) |
| `Pipeline.curate` | `(method=None, **selection_options) -> Pipeline` | compatibility alias for `Pipeline.sample` |
| `Pipeline.collect` | `(n=None) -> CurateResult` | terminal — count-path selection |
| `Pipeline.output` | `(path, *, preset=None, spec=None, container=None, shard_length=64, short_policy="keep", modalities=None, include=None, exclude=None, encoder=None, span_steps=None, span_seconds=None, categorical=None, preset_params=None, dry_run=False, inspect_sources=False) -> DatasetWriteResult \| ExecutionPlan` | terminal — write shards + manifests + `config.json`; with `dry_run=True`, validate and estimate without writing |
| `Pipeline.report` | `(**kw) -> dict` | terminal — feature pass only → `dataset_report` |
| `Pipeline.stats` | `(*, hist_bins=10, max_sources=3, max_ticks_per_source=10000, examples=5) -> StatsReport` | terminal — input/candidate statistics before selection, candidate-versus-selected statistics after `sample(n=...)`, or a planned-hours report before output |
| `Pipeline.experiment` | `(arms: dict, n: int, baseline=None) -> dict` | terminal — multi-arm comparison (`arms` maps name → `Pipeline` or raw strategy dict) |
| `Pipeline.head` | `(k=5) -> list[Tick]` | inspector — first `k` ticks of the first source |
| `Pipeline.describe` | `() -> dict` | inspector — resolved `{schema, strategy, output, segmenter, options, sources, expectations}`, no data touched |
| `Pipeline.preview` | `(*, max_sources=3, max_ticks_per_source=10000, examples=5) -> DatasetPreview` | inspector — bounded field quality, source shape, and representative tick sample |
| `Pipeline.review` | `(n=None, *, path=None, examples=10, hist_bins=10, **output_options) -> SelectionReview \| HoursSelectionReview` | terminal — compare a complete count-selection pool, or write/review an hours selection when `path=` is supplied |
| `Pipeline.explain` | `(*, inspect_sources=False) -> ExecutionPlan` | inspector — typed stage plan, requirements, implicit choices, and warnings; optionally preflight source metadata |

All operators are immutable: each returns a new pipeline. `semantic` fails if a semantic is
already set; `at_rate` fails if `sample_hz` is already known; and `using` requires a `Registry`.
See [pipeline.md](pipeline.md) for accepted operator forms and validation behavior.

## Typed results and execution plans

| name | signature / stable fields | does |
|---|---|---|
| `CurateResult` | `(selected, all_segments, schema, strategy, splits={}, errors=[], source_tracks={}, execution=None, seed=0)`; `.stats()` | typed count-path result, including the scored source roster/length contract and bounded execution metadata when applicable |
| `DatasetWriteResult` | dict-compatible; `dataset` path, `.stats()`, `.to_dict()`, and attribute access | dataset write result returned by output paths |
| `ExperimentResult` | dict-compatible; `.to_dict()` and attribute access | multi-arm experiment result |
| `DatasetReport` | dict-compatible; `.to_dict()` and attribute access | distribution report result |
| `StatsReport` | dict-compatible; `.render()`, `.available()`, `.plot(metric=None, kind=None, ax=None)`, `.examples(mode="auto", metric=None, compare=None, n=6, frames=3)`, `.save(path)`, rich HTML display, and attribute access | unified input, selection, or output statistics; `plot()` automatically prefers a registered applicable domain plot, and plotting requires `[stats]` |
| `StatsExamples` | dict-compatible; stable `groups`, `metric`, `compare`, `bounded`, and `frames_per_example`; `.render()` and rich HTML display | bounded human-review evidence; live reports retain transient pipeline access for domain media rendering while `.to_dict()` remains portable |
| `DatasetInspection` | dict-compatible; `.to_dict()` and attribute access | artifact summary and optional shard-integrity result |
| `DatasetPreview` | dict-compatible; `.render()`, `.to_dict()`, and attribute access | bounded source/field-quality preview |
| `SelectionReview` | dict-compatible; `.render()`, `.to_dict()`, and attribute access | candidate-versus-selected metrics, coverage, rationale, and examples |
| `HoursSelectionReview` | dict-compatible; `.render()`, `.to_dict()`, and attribute access | hours admission/rejection reasons, priorities, budget utilization, and examples |
| `QualityGateResult` | dict-compatible; `.render()`, `.to_dict()`, and attribute access | complete pass/fail checks for a quality policy |
| `QualityGateError` | exception carrying `.result` | raised by `check_quality` and failing fluent expectations |
| `DatasetComparison` | dict-compatible; `.render()`, `.to_dict()`, and attribute access | structured differences between two curated artifacts |
| `ValidationResult` | dict-compatible; `.to_dict()` and attribute access | programmatic validation result used by the CLI layer |
| `ExecutionPlan` | `(source, schema, window, strategy, output, options, stages=(), warnings=(), physical_plan=None)` | immutable mapping returned by `Pipeline.explain`; `.to_dict()` serializes and `.render(detailed=False)` formats text |
| `inspect_dataset` | `(dataset, *, verify=False, container=None) -> DatasetInspection` | inspect manifests/config without original sources; `verify=True` decodes every shard |
| `preview_dataset` | `(*, schema, adapter, sources, max_sources=3, max_ticks_per_source=10000, examples=5) -> DatasetPreview` | classic-API form of the bounded data preview |
| `selection_review` | `(result, *, seed=0, examples=10, hist_bins=10) -> SelectionReview` | review an existing `CurateResult` |
| `hours_selection_review` | `(result) -> HoursSelectionReview` | review an hours-path `DatasetWriteResult` without rescoring |
| `validate_expectations` | `(expectations) -> dict` | validate and copy the closed quality-policy grammar |
| `evaluate_quality` | `(report, expectations=None, **kwargs) -> QualityGateResult` | evaluate every applicable gate without raising |
| `check_quality` | `(report, expectations=None, **kwargs) -> QualityGateResult` | evaluate gates and raise `QualityGateError` on failure |
| `compare_datasets` | `(left, right, *, verify=False) -> DatasetComparison` | compare artifact schema, output, pipeline, provenance, size, and integrity |
| `render_html_report` | `(*reports, title="kuration review") -> str` | render typed/mapping reports as dependency-free static HTML |
| `write_html_report` | `(path, *reports, title="kuration review") -> str` | write the static HTML report locally or through configured path I/O |
| `Arm` | `(name, strategy)` | one named `run_experiment` strategy arm |
| `union_define` | `(arms) -> dict` | merge arm metric definitions, rejecting incompatible duplicate names |

The dict-compatible result types preserve ordinary key access while adding attribute access
during the typed-result transition. Consumers should use documented keys, not the concrete `dict`
subclass implementation.

## Schema & data model

| name | signature | does |
|---|---|---|
| `SchemaSpec` | `.from_dict(d)` / `.to_dict()`; fields: `name, fields, entity_axis, sample_hz, event_streams, globals, media` | the declared vocabulary ([schema.md](schema.md)); fields or media may independently be non-empty |
| `FieldSpec` | `(name, kind, dims=(), semantics=None, bits={}, max_rate=None)` | one field: `kind` scalar/vector/categorical + semantics tag |
| `MediaSpec` | `(name, kind, dtype, shape, layout="HWC", color_space="RGB", timing="pts", capabilities={})` | one lazy auxiliary media-column contract |
| `Tick` | `(time, entities, events={}, globals={}, media={})` | canonical per-timestep record — the adapter contract |
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
| `score_sources` | `(*, schema, adapter, sources, define, registered, segmenter, tags_spec=None, group_of=None, tags_of=None, workers=1, on_error="skip", streaming=False, progress=None, executor="local", registry=None, required_fields=()) -> (segments, errors, source_tracks)` | the shared ordered feature pass (raises on an empty pool) |

## Input adapters

All take `schema` first and expose `.ticks(source_uri)`, plus a metadata-only
`.check_source(source_uri)` preflight (used by `kur-validate --check-sources`); see
[adapters.md](adapters.md).

| name | constructor | reads |
|---|---|---|
| `TimeseriesAdapter` | `(schema, *, fields, time_column=None, entity_column=None, globals=None)` | CSV / single Parquet (`[parquet]`) / `from_rows(rows)` |
| `JsonlAdapter` | `(schema, *, fields, time_column=None, entity_column=None, globals=None)` | `.jsonl` (one row-object per line) |
| `HDF5Adapter` | `(schema, *, fields, globals=None)` | `.h5`, one file = one source/entity (`[array]`) |
| `NpzAdapter` | `(schema, *, fields, globals=None)` | `.npz` (dep-free) |
| `ParquetDatasetAdapter` | `(schema, *, fields, path, source_per="part", time_column=None, entity_column=None, globals=None)`; `.sources()` | a directory/glob of Parquet parts (`[parquet]`) |
| `HuggingFaceAdapter` | `(schema, *, fields, dataset, split="train", config=None, time_column=None, entity_column=None, globals=None, source_column=None, load_kwargs=None)`; `.sources()` | a HF dataset (`[hf]`) |
| `ZarrAdapter` | `(schema, *, fields, globals=None)` | a Zarr store (`[zarr]`; remote via `[gcs]`, `[s3]`, or `[azure]`) |
| `CuratedDatasetAdapter` | `(dataset_dir_or_list, schema=None)`; `.sources()`, `.dataset_of` | kuration's own output, re-curatable |
| `schema_from_output` | `(output_spec_dict, *, name="curated", sample_hz=None, entity_axis="e") -> SchemaSpec` | derive a schema from a dataset's `config.json` output block |
| `register_adapter` / `get_adapter` | `(name, factory, **metadata) -> None` / `(name, *, registry=None, **kwargs) -> InputAdapter` | declarative adapter registry |
| `InputAdapter` | Protocol: `schema` + `ticks(uri)`; optional lazy `iter_ticks(uri)`, `check_source(uri)`, and `sources_are_paths` | the seam to implement for bespoke formats |
| `VideoAdapter` / `JoinedMediaAdapter` | video-only sources / strict media join over another adapter | lazy MP4 media columns (`[video]` for PyAV provider) |

## Selection

| name | signature | does |
|---|---|---|
| `load_strategy` | `(dict_or_json_path, schema, *, registry=None, available_metrics=()) -> Strategy` | parse + validate a declarative strategy ([strategies.md](strategies.md)) |
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
| `register_scorer` / `get_scorer` | `(name, factory, **metadata)` / `(name, *, registry=None, **kwargs)` | registry (`action_residual` ships, `[methods]`) |
| `apply_scorers` | `(entries, segments, *, schema, adapter, seed=0, progress=None, registry=None) -> None` | run parsed scorers, write outputs into `seg.features` |

## Plugins and progress

| name | signature | does |
|---|---|---|
| `load_plugins` | `(modules=(), *, include_entry_points=True) -> list[dict]` | load registration modules and installed entry points; ambient failures are rolled back and returned as `kind="error"` records with a warning |
| `loaded_plugin_modules` | `() -> tuple[str, ...]` | modules propagated to spawned scoring workers |
| `plugin_failures` | `() -> tuple[dict, ...]` | persistent diagnostics for ambient entry points that failed discovery; also reported by discovery CLIs |
| `plugin_provenance` | `(*, registry=None) -> list[dict]` | loaded implementation fingerprints and stable ambient-failure markers used by resume provenance; an explicit registry scopes successful components |
| `check_plugins` | `(target=None) -> list[PluginCheckResult]` | transactionally validate all installed plugin entry points, one entry-point name, or an explicit registration module |
| `PluginCheckResult` | immutable result with `.passed`, `.to_dict()`, and `.render()` | plugin load, descriptor, package, namespace, parameter-schema, and transform-contract diagnostics |
| `ProgressCallback` | `Callable[[dict], None]` | receives structured `{phase, event, ...}` execution events |
| `ProgressSpec` | `ProgressCallback \| str \| bool` | user selection accepted by `progress_callback` and fluent pipeline options |
| `notebook_progress` / `cli_progress` | `() -> ProgressCallback` | explicit rich notebook or throttled terminal renderer |
| `progress_callback` | `(progress="auto") -> ProgressCallback \| None` | environment-aware renderer selection; callbacks pass through unchanged |
| `register_stats_enricher` | `(name, enricher, *, plotter=None, example_renderer=None) -> None` | advanced domain hook in `kuration.core.stats`; adds a section, optional plot, and optional rich evidence renderer without changing the ordinary stats API |

## Component registries and discovery

| name | signature | does |
|---|---|---|
| `Registry` | `(parent=None)` | isolated component registry; `.child()` creates an overlay whose entries shadow its parent |
| `default_registry` | `Registry` instance | process-wide registry used when no explicit overlay is supplied |
| `ComponentDescriptor` | `(name, kind, description="", parameters=<factory>, capabilities=<factory>, execution_traits=None, extra=None, install_hint=None, package=None, version=None, schema_version=1, documentation=None, stability="stable", module=None, fingerprint=None)` | JSON-safe component metadata; `.to_dict()` omits `None` values |
| `list_components` | `(kind=None, *, registry=default_registry) -> list[dict]` | discover installed component descriptors without reading datasets |
| `describe_component` | `(kind, name, *, registry=default_registry) -> dict` | return one descriptor; raises for an unknown kind or name |
| `register_method` | `(name, fn, **metadata) -> None` | register a selection-priority method |
| `register_filter` | `(name, fn, **metadata) -> None` | register a pre-pass filter |
| `register_operator` | `(name, fn=None, **metadata)` | register a `(pipeline, **params) -> Pipeline` composition directly or as a decorator |
| `detect_adapter` | `(source_uri, *, registry=None) -> str` | select the uniquely highest-confidence registered adapter whose optional `detect_source()` hook recognizes a source |
| `Analyzer` | protocol requiring `summary()` | user-owned stateful direct-API analysis result |
| `DatasetAnalyzer` | protocol requiring `summary()` and `check_dataset(dataset, **options)` | optional analyzer contract for artifact-level `kur-check` execution |
| `SplitLeakageAnalyzer` | `(*, check_groups=True)` | built-in source/group split auditor; `check_groups=False` suppresses group leakage and conflict checks, but never source checks |
| `SplitLeakageAnalyzer.add_assignment` | `(source_id, split, *, group=None) -> None` | record one observation; repeating the same assignment is allowed |
| `SplitLeakageAnalyzer.add_assignments` | `(source_splits, *, source_groups=None) -> None` | record a `{source_id: split}` mapping and optional `{source_id: group}` mapping |
| `SplitLeakageAnalyzer.summary` | `() -> dict` | return the deterministic, JSON-safe leakage report described below |
| `register_analyzer` | `(name, factory=None, **metadata)` | register a discoverable analyzer factory directly or as a decorator |
| `get_analyzer` | `(name, *, registry=None, **params) -> Analyzer` | validate parameters and construct a user-owned analyzer instance |
| `run_dataset_analyzer` | `(name, dataset, *, parameters=None, options=None, registry=None) -> dict` | construct a dataset-capable analyzer, run its hook, and validate the boolean `passed` result |
| `import_dataset` | `(sources, out, *, adapter=None, schema=None, fields=None, time=None, entity=None, hz=None, kinds=None, adapter_options=None, source_id=None, split="train", shard_length=None, container="npz", force=False, registry=None) -> dict` | detect/construct a registered adapter and import its complete array or canonical-tick dataset into a committed Kuration artifact |

`SplitLeakageAnalyzer` requires `source_id`, `split`, and each supplied `group` to be non-empty
strings; `group=None` means that no group is known. Split names are not limited to train/test, so
validation or other named partitions can be audited. `source_groups` may omit sources, but cannot
contain a source absent from `source_splits`. Invalid constructor values, assignments, or mappings
raise `ValueError`. An empty analyzer passes because no conflicting evidence has been observed.

Its `summary()` mapping has these stable fields:

| field | value |
|---|---|
| `passed` | false when any enabled source leak, group leak, or source/group conflict exists |
| `check_groups` | the configured group-checking mode |
| `observations` | number of calls/rows recorded, including exact repetitions |
| `unique_assignments` | number of distinct `(source_id, split, group)` observations |
| `sources`, `groups` | counts of distinct observed source and non-null group identifiers |
| `splits` | sorted distinct split names |
| `source_leaks` | sorted `{"source_id": str, "splits": [str, ...]}` records |
| `group_leaks` | sorted `{"group": str, "splits": [str, ...], "sources": [str, ...]}` records; empty when group checks are disabled |
| `source_group_conflicts` | sorted `{"source_id": str, "groups": [str, ...]}` records; empty when group checks are disabled |

Registry kinds are `adapter`, `metric`, `scorer`, `encoder`, `method`, `filter`, `container`,
`preset`, `transform`, compositional `operator`, and stateful direct-API `analyzer`. Names accept `name` or `namespace/name`; duplicate registrations raise unless the
lower-level `Registry.register(..., replace=True)` API is used. See
[customization.md](customization.md) for descriptors, parameter schemas, and plugin packaging.

## Physical planning, columnar batches, and checkpoints

PyArrow-backed APIs require the optional `[scale]` extra. They keep feature metadata in bounded
record batches and atomically publish reusable stage partitions; ordinary `Segment` objects
remain the Python convenience boundary.

| name | signature | does |
|---|---|---|
| `OperatorResources` | `(cpu=1.0, gpu=0.0, memory_bytes=None, preferred_batch_size=None, max_batch_size=None)` | validated portable resource and batch bounds |
| `OperatorTraits` | `(scope, shape, fusible, passes="one", state_bound="constant", required_columns=(), produced_columns=(), resources=<factory>, determinism="stable-key", checkpoint="partition-reusable")` | closed machine-readable logical-operator execution contract |
| `PhysicalOperator` | `(id, name, traits, parameters={}, component=None)` | one normalized operator in a physical plan |
| `PhysicalStage` | `(id, mode, operators, passes, materializes_input, checkpoint, barrier_reason=None, estimated_peak_memory=None, resources=<factory>)` | fused pipeline island or explicit barrier stage |
| `PhysicalPlan` | `(version, backend, stages, memory_budget=None, spill_path=None, warnings=())` | deterministic mapping-compatible backend-neutral plan |
| `compile_physical_plan` | `(pipeline, *, memory_budget=None, spill_path=None) -> PhysicalPlan` | resolve operators, fuse bounded maps, and isolate global/multipass barriers without reading sources |
| `IDENTITY_COLUMNS` | `tuple[str, ...]` | required stable identity/metadata columns in every `SegmentBatch` |
| `SegmentBatch` | `(record_batch)`; `.from_segments(...)`, `.read_ipc(...)` | immutable bounded Arrow record batch with projection, slicing, NumPy views, and atomic IPC |
| `iter_segment_batches` | `(segments, batch_size, **kwargs) -> Iterator[SegmentBatch]` | convert an iterable while retaining at most one configured Python batch |
| `CheckpointIdentity` | `(logical_plan_hash, physical_plan_hash, input_fingerprints=(), schema={}, components=(), parameters={})` | complete reuse identity for a physical stage |
| `PartitionCommit` | `(partition_id, identity, data_file, rows, sha256, columns, ...)` | checksum-protected committed partition metadata |
| `StageCheckpoint` | `(root, stage_id, identity)` | validate, reuse, publish, and finalize independent partition commits |
| `canonical_hash` | `(value) -> str` | deterministic SHA-256 over canonical JSON |
| `FeatureStageResult` | `(commits, errors, source_tracks, telemetry={})` | bounded feature-island result without a total Python segment pool |
| `checkpoint_feature_stage` | `(*, schema, adapter, sources, define, registered, segmenter, checkpoint, batch_size=1024, ...) -> FeatureStageResult` | stream segmentation, metrics, predicates, and transforms into checkpoint partitions |
| `external_quantile_edges` | `(partitions, feature, n_bins, *, spill_dir=None) -> ndarray` | exact NumPy-compatible quantiles with disk-proportional state |
| `external_top_k` | `(partitions, *, weights, target_size, normalize="quantile", spill_dir=None) -> list[Segment]` | exact stable top-k that materializes only selected segments |
| `external_uniform_bins` | `(partitions, *, axes, target_size, seed=0, spill_dir=None) -> list[Segment]` | exact multipass quantile organization and deterministic disk-backed equal-bin selection |
| `iter_external_uniform_priorities` | `(partitions, *, axes, seed=0, spill_dir=None) -> Iterator[tuple[Segment, float]]` | exact disk-backed uniform-bin priorities for hours filling |
| `iter_external_top_k_priorities` | `(partitions, *, weights, normalize="quantile", spill_dir=None)` | stream every canonical segment with its exact disk-backed normalized priority |
| `iter_external_span_items` | `(prioritized_segments, source_tracks, *, sample_hz, span_len=None, clip_aggregate="max", group_of=None, spill_dir=None)` | aggregate overlapping segment priorities/features into a bounded span-item stream |
| `iter_external_weighted_priorities` | `(partitions, *, weights, normalize="quantile", seed=0, spill_dir=None)` | stream exact canonical Gumbel priorities from one disk-backed score index and one partition-at-a-time materialization pass |
| `external_weighted` | `(partitions, *, weights, target_size, normalize="quantile", seed=0, spill_dir=None) -> list[Segment]` | select exact hours-path Gumbel priorities while retaining only the selected heap; count-weighted sampling has a distinct NumPy contract |
| `iter_external_hash_dedup` | `(partitions, *, feature_names=None, per_group=False, spill_dir=None)` | exact disk-backed feature-vector dedup with full-payload collision resolution and canonical retention |
| `TransformContext` | mapping-compatible typed context; `.before(token)`, `.after(token)`, `.owned_lengths` | typed per-batch execution metadata and boundary access; ordinary `ctx["schema"]` access remains compatible and owned lengths are read-only |
| `apply_transforms_to_windows` | `(calls, segments, windows, *, adapter, registry, contexts=None, progress=None)` | bounded transform execution over extracted windows; optional contexts carry lookbehind/lookahead ticks and owned lengths |

## Output

Media contracts include `MediaSpec`, `FrameRef`, `VideoMetadata`, `MediaProvider`,
`PyAVMediaProvider`, `VideoAdapter`, and `JoinedMediaAdapter`. Transform and execution contracts
include `BatchTransform`, `TransformContext`, `register_batch_transform` (`register_transform`
alias), `get_transform`, `ExecutionBackend`, `ResourceSpec`, and the
ordered `LocalExecutionBackend` / `RayExecutionBackend` implementations. `apply_transforms`
executes resolved calls over canonical segments; `get_execution_backend` resolves a backend.

| name | signature | does |
|---|---|---|
| `OutputSpec` | `.from_dict({shard_length, short_policy?, container?, presence_mask?, modalities})` | declared modalities + chunking; `_present` enabled by default ([output.md](output.md)) |
| `Modality` | `(name, dtype, shape, src_kind, src_arg)` | one declared modality |
| `OutputEncoder` / `PassthroughEncoder` | `timestep(record, ctx)` and optional `encode_batch(records, ctx)` | the domain seam for derived tensors |
| `register_encoder` / `get_encoder` | `(name, factory)` / `(name, **kwargs)` | encoder registry |
| `Track` | `(source_id, entity_id, records, meta={})` | one entity's timestep sequence — the output unit |
| `tracks_from_ticks` | `(schema, ticks, source_id) -> list[Track]` | one track per entity |
| `write_track` | `(out_dir, track, spec, *, encoder=None, schema=None, split="all", resume=False, name_suffix="", media_provider=None, resume_entries=None, registry=None) -> [manifest entries]` | low-level: write one track's shards; `resume_entries` supplies a previously committed checksummed manifest for verified reuse and `registry` resolves overlay containers |
| `fill_to_hours` | `(items, *, train_hours, test_hours, max_group_overshoot=None, tag_coverage=None, seed=0, return_stats=False)` | hours-budget fill; optional accounting as a third return value |
| `Container` | `(extension, writer, reader)` | one shard-container implementation |
| `register_container` | `(name, container=None, *, extension=None, writer=None, reader=None, **metadata)` | register a `Container` instance or its three parts |
| `output_preset` | `(name, *, schema, registry=default_registry, **params) -> OutputSpec` | resolve a named output preset against a schema |
| `register_preset` | `(name, factory, **metadata) -> None` | register an output-preset factory |

Built-in presets are `numpy-timeseries`, `pytorch-sequences`, and `tabular`. Preset parameters
are validated through their component descriptors; unknown presets and parameters raise before
any output is written.

## Built-in domain operators and analyzers

The `kuration.gameplay` domain ships in the SDK wheel. Its heavy
measurement runtimes remain optional.

| name | signature | does |
|---|---|---|
| `kuration.gameplay.pixel_shift.pixel_shift` | `(pipeline, *, controls=None, boundary="exclude") -> Pipeline` | compositional `aintropy/pixel-shift` operator backed by the transform of the same name (`[pixel-shift]` for OpenCV execution) |
| `PixelShift` | `(*, controls=None, boundary="exclude")` | bounded batch transform producing per-control shift min/max/mean/count/requested features |
| `PixelShiftReport` | `()`; `.add_segment(...)`, `.summary()`, `.print_report()` | user-owned cross-segment instance report |
| `kuration.gameplay.action_state_consistency.action_state_consistency` | `(*, anomaly_type, baselines) -> ActionStateConsistencyChecker` | factory registered as analyzer `aintropy/action-state-consistency` |
| `ActionStateConsistencyChecker` | `(config)`; `.add_segment(...)`, `.clip_verdict(source_id)`, `.summary()`, `.print_report()` | classify clips against pixel-shift baselines (`[action-state]` for frame measurement) |
| `AnomalyChecker` | `(validated_config)` | measurement-agnostic tuple aggregation used by the action-state adapter |
| `VideoComparison` | `(left, right, *, labels=None, provider=None)`; `.render()`, `.save()`, `.probe()`, `.plot(frames=4)`, rich HTML display | synchronized side-by-side video review; `[video]` for default metadata/frame decoding and `[stats]` for contact sheets |
| `kuration.gameplay.compare_videos` | `(left, right, *, labels=None, provider=None) -> VideoComparison` | ergonomic constructor for `VideoComparison` |

`StatsReport.available()` returns `plots`, `default_plot`, `example_modes`, `comparisons`,
`media_examples`, `html`, and `save`. Count-selection evidence modes are `selected`, `rejected`,
`strongest`, and `weakest`; comparisons are `selected-vs-rejected` and
`strongest-vs-weakest`. `n` is bounded to 1–100 and gameplay `frames` to 1–12.

The associated validation and decoding helpers are documented in
[pixel-shift.md](pixel-shift.md) and
[action-state-consistency.md](action-state-consistency.md).

## SLAM reconstruction and curation

`kuration.gameplay.slam` is a public built-in module. Import and precomputed trajectory curation remain
NumPy-only while live model dependencies stay optional. See [slam.md](slam.md) for the end-to-end
workflow.

| name | signature | does |
|---|---|---|
| `kuration.gameplay.slam.SLAMReconstructor` | `(depth_model=None, *, embedding_model=None, embeddings=None, depth_model_id=None, embedding_model_id=None, runtime_factory=None, **runtime_options)` | global reconstruction coordinator; raw execution needs `[slam]` unless a compatible runtime is injected |
| `SLAMReconstructor.run` / `SLAMReconstructor.__call__` | `(data, *, progress=None) -> SLAMOutput` | notebook-style in-memory reconstruction from `{clip: (T,H,W,3)}` frames |
| `SLAMReconstructor.reconstruct` | `(source, out_dir=None, *, run_if_needed=True, decoder_options=None, export_options=None, resume=False, cache_key=None, progress=None, preflight=True, require_gpu=False) -> SLAMReconstruction` | reuse/export/run as needed and return a portable manifest artifact |
| `SLAMReconstruction` | `(manifest: Path)`; `.to_pipeline(join_video=False, tolerance=None, mismatch="error", provider=None)` | typed result that opens the reconstruction as a normal fluent `Pipeline` |
| `SLAMOperator` | `(depth_model, embedding_model=None, embeddings=None, output_pointcloud=True, devices=None, window_size=48, overlap=4, bridge_size=24, process_res=504, conf_thresh=20.0, num_max_points=1000000, top_k=5, similarity_threshold=0.5, seed=0, verbose=False)` | low-level intersection/chunk/depth/stitch engine (`[slam]`) |
| `CameraPose` / `SLAMOutput` | dataclasses containing camera basis/frustum data and trajectory/pose/point-cloud arrays | stable in-memory reconstruction data contract |
| `input_slam` | `(manifest, *, join_video=False, tolerance=None, mismatch="error", provider=None) -> Pipeline` | open a precomputed manifest without loading the SLAM runtime |
| `prepare_slam_input` | `(source, out_dir=None, *, run_if_needed=True, runner_options=None, decoder_options=None, export_options=None, resume=False, cache_key=None, progress=None, preflight=True, require_gpu=False) -> Path` | normalize a manifest, completed output, frame mapping, or video mapping |
| `run_slam_reconstruction` | `(data, *, depth_model=None, embedding_model=None, embeddings=None, depth_model_id=None, embedding_model_id=None, runtime_factory=None, progress=None, **runtime_options) -> SLAMOutput` | lower-level injected-runtime execution boundary |
| `decode_video_clips` | `(videos, *, frame_skip=1, batch_size=64, color_space="RGB", provider=None, progress=None) -> (frames, timestamps)` | decode named clips while preserving processed-frame timestamps (`[slam]` unless a provider is injected) |
| `export_slam_output` | `(output, out_dir, *, timestamps=None, sample_hz=None, videos=None, point_attributes=None, tracking_confidence=None, provenance=None) -> Path` | validate and write the portable trajectory/scene manifest |
| `preflight_slam` | `(source, out_dir=None, *, run_if_needed=True, runner_options=None, decoder_options=None, require_gpu=False) -> SLAMPreflightResult` | side-effect-free input/runtime/output readiness report |
| `slam_provenance` | `(options) -> dict` | stable JSON-safe engine, model, and option identity without live-object reprs |
| `curate_slam_gameplay` | `(manifest, output, *, window=32, stride=16, n=2000, min_quality=0.5, quality_options=None, feature_space=None, selection_method="uniform-bins", bins=4, seed=0, progress=None) -> SLAMCurationResult` | one-call trajectory quality/diversity selection workflow |
| `PointCloudArtifact` | `(points, colors=None, attributes={})`; `.select(...)`, `.save(...)` | validated immutable point cloud |
| `load_point_cloud` / `curate_point_cloud` | `(source)` / `(source, *, method="voxel", max_points=None, voxel_size=None, min_confidence=None, seed=0, output=None)` | load or deterministically reduce an exported scene |
| `build_pointcloud_scene` / `load_slam_output` | visualization helpers | build a notebook scene (`[slam-viz]`) or load a duck-typed output view |
| `slam_schema` | `(*, sample_hz=None) -> SchemaSpec` | canonical camera trajectory schema |

The public constants name the manifest/point-cloud/selection format versions, registered
component names, camera fields, and transform output features. `SLAMMotion`, `SLAMQuality`, and
`SLAMTrajectoryAdapter` are the concrete public transform/adapter implementations. Live
reconstruction is not a registered compositional operator; the registered operator names only
compose bounded trajectory transforms.

## Read side

| name | signature | does |
|---|---|---|
| `iter_shards` | `(dataset, *, split=None, modalities=None, trim=True, container=None, with_meta=False, registry=None, verify=True) -> Iterator[dict]` | shards back as `{modality -> array}` plus `_present` when enabled; checksum/size verification defaults on |
| `shard_entries` | `(dataset, *, split=None, container=None) -> (entries, container)` | manifest entries (paths rebased if the dataset moved) |
| `read_shard_arrays` | `(entry, container, *, modalities=None, trim=True, registry=None, verify=True) -> {name: array}` | one shard: verify integrity, trim pad, reshape, restore dtype |
| `kuration.core.torch.ShardDataset` | `(dataset, *, split=None, modalities=None, trim=True, container=None, transform=None, include_meta=False, verify=True)` | map-style PyTorch `Dataset` over the above (`[torch]`) |

`dataset` is a dataset directory, a `dataset_manifest.json` path, or a loaded manifest dict.

## Manifests & I/O helpers

| name | signature | does |
|---|---|---|
| `build_config` | `(schema, strategy, *, results, seed, started=None, ended=None) -> dict` | build the reproducibility record written as `config.json` |
| `write_config` | `(path, config) -> None` | serialize a reproducibility configuration as indented JSON |
| `dump_segment_manifest` | `(path, segments) -> None` | JSON manifest of a scored/selected pool |
| `read_text` | `(path) -> str` | read path-agnostic text |
| `write_text` | `(path, text) -> None` | write path-agnostic text |
| `write_bytes` | `(path, data) -> None` | write path-agnostic bytes |
| `exists` | `(path) -> bool` | test a local or remote path |
| `local_copy` | context manager: `(path) -> local file path` | stage a remote object for libs needing a real path |

## Artifact versions and migration

| name | signature | does |
|---|---|---|
| `ARTIFACT_FORMAT_VERSION` | `str` | current `config.json` / dataset artifact contract version |
| `JOB_SCHEMA_VERSION` | `str` | current declarative job contract version |
| `migrate` | `(value, out=None) -> dict` | migrate a job/config dict or JSON path; optionally write formatted JSON |
| `migrate_job` | `(job) -> dict` | deep-copy and migrate a job dictionary to `JOB_SCHEMA_VERSION` |
| `migrate_config` | `(config) -> dict` | deep-copy and migrate an artifact config to `ARTIFACT_FORMAT_VERSION` |

Migration is non-mutating for dictionary inputs. A malformed version, non-object input, or a
future unsupported major version raises `ValueError`. The matching command-line interface is
`kur-migrate`; see [jobs.md](jobs.md) for the serialized job contract.
