# API reference — the public Python surface

Every name importable from `kuration` (plus the public `kuration.core`, `kuration.core.torch`,
`kuration.gameplay`, and `kuration.selection` domains),
grouped by layer, with its real
signature. Optional-extra requirements are tagged (`[methods]`, `[parquet]`, `[array]`,
`[video]`, `[ray]`,
`[cloud]`, `[torch]`, `[hf]`, `[zarr]`, `[scale]`, `[stats]`, `[gameplay]`, `[pixel-shift]`, `[collision]`,
`[action-state]`, `[slam]`, `[slam-viz]`). For concepts and recipes see the per-layer docs
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
| `write_dataset` | `(out_dir, *, schema, adapter, sources, strategy, segmenter, spec, encoder=None, train_hours, test_hours=0.0, sample_hz=None, seed=0, group_of=None, tags_of=None, clip_aggregate="max", span_steps=None, span_seconds=None, workers=1, on_error="skip", resume=False, force=False, streaming=False, progress=None, transforms=(), registry=None, executor="local", memory_budget=None, batch_size=1024, checkpoint_dir=None, spill_dir=None, approximation="exact", max_in_flight=2) -> DatasetWriteResult` | score/transform, fill a train/test **hour budget**, write shards + manifests + config; a memory budget selects the exact bounded checkpoint path |
| `run_experiment` | `(*, schema, adapter, sources, arms, segmenter, target_size, seed=0, baseline=None, group_of=None, tags_of=None, workers=1, on_error="skip", streaming=False, progress=None, executor="local") -> ExperimentResult` | score once over the arms' union, select per arm, JS-divergence vs `baseline` |
| `Arm` | `(name, strategy, transforms=(), registry=None)` | one experiment arm; `union_define(arms)` merges arm definitions |
| `dataset_report` | `(segments, *, metrics=None, hist_bins=10) -> DatasetReport` | per-metric distribution profile (+ tag/group counts) of a scored pool |
| `stats` | `(subject, *, hist_bins=10, max_sources=3, max_ticks_per_source=10000, examples=5, max_values=100000, content=True, verify=False, progress=None, registry=None) -> StatsReport` | unified statistics for a pipeline, count selection, write result, typed report, or existing artifact path |
| `progress_callback` | `(progress="auto") -> ProgressCallback \| None` | resolve `"auto"`, `"notebook"`, `"terminal"`, `"cli"`, `"off"`, a boolean, or an existing callback; pipeline users normally select this through `.options(progress=...)` |
| `kuration.core` | module | explicit import facade for the domain-agnostic top-level API |
| `kuration.gameplay.slam_reconstruct` | `(source, output=None, *, depth_model=None, embedding_model=None, embeddings=None, depth_model_id=None, embedding_model_id=None, runtime_factory=None, join_video=False, tolerance=None, mismatch="error", provider=None, run_if_needed=True, decoder_options=None, export_options=None, resume=False, cache_key=None, progress=None, preflight=True, require_gpu=False, **runtime_options) -> Pipeline` | convenience function—not a class—that reconstructs raw input when needed or opens an existing SLAM manifest, then returns the sole curation pipeline type |
| `kuration.gameplay.action_state_checker` | `(*, baselines=None, anomaly_type="action_state", calibration=None) -> Analyzer` | construct the action-state consistency analyzer without a component-name string; omitting `baselines` self-calibrates (leave-one-out consensus outlier detection) |
| `kuration.gameplay.load_baselines_from_reference` | `(reference, *, controls=None) -> dict` | validate an action-state reference report and derive checker baselines, optionally for a control subset |
| `kuration.gameplay.compare_videos` | `(left, right, *, labels=None, provider=None) -> VideoComparison` | create a synchronized notebook/HTML gameplay review without eagerly importing PyAV or Matplotlib |
| `kuration.selection.BalancedSelector` | `(feature, of, kind="std", bins=8, count=100)` | frozen dataclass; `.apply(pipeline) -> Pipeline` and `.describe() -> str` compile to the `kuration/balanced-selector` chain |
| `kuration.selection.RareEventCoverage` | `(of, feature="event_rate", count=100, weight=1.0)` | frozen dataclass; `.apply(pipeline) -> Pipeline` and `.describe() -> str` compile to the `kuration/rare-event-coverage` chain |
| `kuration.selection.DiverseSubset` | `(features, count=100, n_clusters=32, dedup_threshold=None)` | frozen dataclass; `.apply(pipeline) -> Pipeline` and `.describe() -> str` compile to the `kuration/diverse-subset` chain (`[methods]`) |
| `kuration.selection.run` | `(pipeline, operations) -> Pipeline` | chain `Operation` instances or `{"type": "ClassName", ...}` config dicts over a pipeline |
| `kuration.selection.plan` | `(operations) -> list` | resolve config dicts into `Operation` instances without applying them |
| `kuration.selection.describe` | `(operations) -> str` | multi-line human summary of a resolved plan, naming each operation's compiled SDK chain |
| `kuration.selection.register_operation_type` | `(name, operation_class) -> None` | register a user `Operation` class for `{"type": name, ...}` config-dict use |

Shared kwargs: `group_of(source_id) -> key` sets each segment/track's `group` (for the group
cap and stratified splits); `tags_of(segment) -> iterable` overrides declarative tags;
`sources` contains non-empty string identifiers/URIs (use `str(path)` for `pathlib.Path` values);
`executor` selects ordered local or optional Ray source work; `workers` caps concurrent local
workers or submitted Ray tasks (deterministic); `on_error` is `"skip"` (log + collect, with the
source removed as a unit across feature, transform, and scorer-buffer reads) or `"raise"`;
`streaming=True` is the out-of-core path; `progress(event_dict)` receives phase events without
coupling execution to a particular UI.

- `CurateResult` — dataclass: `selected`, `all_segments`, `schema`, `strategy`, `splits`, `errors`,
  scored `source_tracks`, optional bounded-execution metadata in `execution`, and `seed`.
- `DatasetWriteResult` is mapping-compatible and contains `dataset`, `shards`, `split`,
  `admitted`, `results`, and `errors`; `.stats()` summarizes its written artifact.
- `Arm(name, strategy, transforms=(), registry=None)` — one experiment arm;
  `union_define(arms) -> dict` merges their `define`s.

## Pipeline

The higher-level facade over the entry points above — no explicit schema/adapter/strategy
required; see [pipeline.md](pipeline.md) for the full operator reference.

| name | signature | does |
|---|---|---|
| `input` | `(src=None, *, schema=None, adapter=None, sources=None, fields=None, time=None, entity=None, hz=None, kinds=None, sample=200, globals=None, registry=None, **adapter_kw) -> Pipeline` | sniff `src`, infer a schema (unless `schema=`), build the adapter — eagerly, loudly; an explicit registry also resolves named adapters in isolation |
| `Pipeline` | `(*, schema, adapter, sources, in_memory, ops, registry=None)` | immutable pipeline type; ordinary users construct it through `input(...)`, while extensions receive and return it |
| `infer_schema` | `(sample, *, name="inferred", time=None, entity=None, hz=None, kinds=None) -> (SchemaSpec, hints)` | classify a sample (rows/dict-of-arrays/ticks) into a schema + adapter `fields=` hints |
| `Pipeline.window` | `(window, stride=None, unit="steps") -> Pipeline` | set the `Segmenter` |
| `Pipeline.join_video` | `(videos, *, column="frame", on="timestamp", tolerance=None, mismatch="error", provider=None) -> Pipeline` | attach a strictly aligned lazy video media column |
| `Pipeline.pixel_shift` | `(*, controls=None, boundary="exclude") -> Pipeline` | registered by `kuration.gameplay` as `aintropy/pixel-shift`; add the DIAMOND action/pixel-motion features (`[pixel-shift]` when frames are measured) |
| `Pipeline.action_state_consistency` | `(*, baselines=None, anomaly_type="action_state", controls=None, calibration=None, boundary="exclude", frames=None, actions=None, color_space=None, verify=None, metadata_key="action_state_consistency", replace=False) -> Pipeline` | registered by `kuration.gameplay` as `aintropy/action-state-consistency`; dataset-mode only (audits an already-written dataset), runs the analyzer once and attaches its per-clip verdict under `metadata["gameplay"][metadata_key]`. For a live pipeline, use the `aintropy/action-state-consistency` scorer via `.score(...)` instead. `baselines=` checks against external calibration; `controls=` alone self-calibrates (leave-one-out consensus, knobs via `calibration=`) |
| `Pipeline.collision` | `(*, map_for_source, map_registry, dist_path, disk_cache_dir=".kuration_collision_cache", min_score=2.0, proximity=40.0, min_approach=250.0, max_residual=20.0, slide_angle=75.0, max_slide_speed=400.0, min_release_accel=100.0, max_release_vn=100.0) -> Pipeline` | registered by `kuration.gameplay` as `aintropy/collision`; add per-window CS2 wall-collision count/anomaly features raycast against map geometry (`[collision]` for trimesh/rtree) |
| `Pipeline.collision_directional` | `(*, map_registry, ramp_models_path, map_for_source=None, default_map=None, fallback_ramp=None, disk_cache_dir=None, proximity=None, min_missing_speed=None, concentration_ratio=None, inputs=None, batch_size=None, metadata_key="directional_collision", replace=False) -> Pipeline` | registered by `kuration.gameplay` as `aintropy/collision-directional`; run the tick-stream analyzer once and attach its verdict (including `flagged_ticks` in source coordinates) under `metadata["gameplay"][metadata_key]` |
| `Pipeline.slam_motion_diversity` | `(*, axis="slam_path_length", bins=8) -> Pipeline` | registered by `kuration.gameplay` as `aintropy/slam-motion-diversity`; add SLAM motion features and bin one diversity axis |
| `Pipeline.slam_quality` | `(*, min_score=None, min_tracking_confidence=0.5, translation_ratio=10.0, translation_mad_multiplier=8.0, max_translation_step=None, max_turn_degrees=90.0) -> Pipeline` | registered by `kuration.gameplay` as `aintropy/slam-quality`; add SLAM quality features and optionally filter weak windows |
| `Pipeline.balanced_selector` | `(feature, of, kind="std", bins=8, count=100) -> Pipeline` | registered by `kuration.selection` as `kuration/balanced-selector`; measure a feature, quantile-bin it, and sample equally across bins |
| `Pipeline.rare_event_coverage` | `(of, feature="event_rate", count=100, weight=1.0) -> Pipeline` | registered by `kuration.selection` as `kuration/rare-event-coverage`; measure a boolean/0-1 field's per-window rate and weighted-sample toward it |
| `Pipeline.diverse_subset` | `(features, count=100, n_clusters=32, dedup_threshold=None) -> Pipeline` | registered by `kuration.selection` as `kuration/diverse-subset`; optional dedup then cluster-balanced sampling over pre-measured features (`[methods]`) |
| `Pipeline.transform` | `(name, *, using=None, inputs=None, batch_size=None, device="cpu", **params) -> Pipeline` | add a registered bounded scalar-feature transform |
| `Pipeline.using` | `(registry) -> Pipeline` | resolve adapters, analyzers, containers, encoders, filters, methods, metrics, operators, presets, scorers, and transforms through one explicit `Registry` overlay |
| `Pipeline.metadata` | read-only `dict` property | defensive deep copy of namespaced caller/enrichment metadata carried by the pipeline |
| `Pipeline.with_metadata` | `(namespace, key, value, *, replace=False) -> Pipeline` | attach caller-owned strict-JSON metadata without mutating the input pipeline |
| `Pipeline.enrich` | `(name, parameters=None, *, namespace=None, key=None, replace=False, mode=None, inputs=None, batch_size=None, analyzer=None, **options) -> Pipeline` | drive a registered analyzer once and attach its JSON result, descriptor, invocation, mode, and input-basis fingerprint |
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
| `Pipeline.options` | `(**kw) -> Pipeline` | closed options: `mode, executor, workers, on_error, seed, streaming, resume, sample_hz, progress, memory_budget, batch_size, checkpoint_dir, spill_dir, approximation, max_in_flight`; progress accepts a callback or `"auto"`, `"notebook"`, `"terminal"`, `"cli"`, `"off"` |
| `Pipeline.expect` | `(**expectations) -> Pipeline` | attach closed-grammar quality gates evaluated by `preview()` or `review()` |
| `Pipeline.sample` | `(method=None, *, n=None, hours=None, test_hours=None, weights=None, target=None, normalize=None, group_cap=None, **method_params) -> Pipeline` | configure selection; exactly one of `n=`/`hours=` (`curate` remains as a compatibility alias) |
| `Pipeline.curate` | `(method=None, *, n=None, hours=None, test_hours=None, weights=None, target=None, normalize=None, group_cap=None, **method_params) -> Pipeline` | compatibility alias for `Pipeline.sample` |
| `Pipeline.collect` | `(n=...) -> CurateResult` | terminal — count-path selection. `n` defaults to whatever `.sample(n=...)` configured; pass `n=None` explicitly for every scored and filtered segment, with no selection method/weights/bins required |
| `Pipeline.output` | `(path, *, preset=None, spec=None, container=None, shard_length=64, short_policy="keep", modalities=None, include=None, exclude=None, encoder=None, span_steps=None, span_seconds=None, categorical=None, preset_params=None, force=False, dry_run=False, inspect_sources=False) -> DatasetWriteResult \| ExecutionPlan` | terminal — write shards + manifests + `config.json`; existing artifacts require `force=True`, while `dry_run=True` validates and estimates without writing |
| `Pipeline.report` | `(**kw) -> DatasetReport` | terminal — feature pass only → `dataset_report` |
| `Pipeline.stats` | `(**kwargs) -> StatsReport` | terminal — thin sugar for `stats(pipeline, **kwargs)`: input/candidate statistics before selection, candidate-versus-selected statistics after `sample(n=...)`, or a planned-hours report before output |
| `Pipeline.analysis_mode` | `(mode=None) -> str` | resolve the analyzer execution mode (`None` → `"dataset"` for an artifact-backed pipeline, else `"stream"`) — the rule `analyze`/`enrich` apply, exposed so domain operators can reject mode-specific options up front |
| `Pipeline.analyze` | `(name, parameters=None, *, mode=None, inputs=None, batch_size=None, analyzer=None, **options) -> dict` | terminal — run a registered stateful analyzer; `mode="stream"` (default) drives `add_batch` over the segmented tick stream with lazy media, `mode="dataset"` (default for an artifact-backed pipeline) drives `check_dataset` and returns what `run_dataset_analyzer` returns |
| `Pipeline.check` | `(policy=None, *, verify=False, **expectations) -> QualityGateResult` | terminal — evaluate quality gates without raising; artifact-backed pipelines are gated exactly as `kur-check --policy` gates them, others on a bounded input preview |
| `Pipeline.experiment` | `(arms: dict, n: int, baseline=None) -> ExperimentResult` | terminal — multi-arm comparison (`arms` maps name → `Pipeline` or raw strategy dict) |
| `Pipeline.head` | `(k=5) -> list[Tick]` | inspector — first `k` ticks of the first source |
| `Pipeline.describe` | `() -> dict` | inspector — resolved `{schema, strategy, output, segmenter, options, sources, expectations, metadata}`, no data touched |
| `Pipeline.preview` | `(*, max_sources=3, max_ticks_per_source=10000, examples=5) -> DatasetPreview` | inspector — bounded field quality, source shape, and representative tick sample |
| `Pipeline.review` | `(n=..., *, path=None, examples=10, hist_bins=10, **output_options) -> SelectionReview \| HoursSelectionReview` | terminal — compare a complete count-selection pool, or write/review an hours selection when `path=` is supplied. `n` defaults to whatever `.sample(n=...)` configured; pass `n=None` explicitly to review every scored and filtered segment |
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
| `QualityGateError` | exception carrying `.result` | raised by `check_quality` and failing quality expectations |
| `DatasetComparison` | dict-compatible; `.render()`, `.to_dict()`, and attribute access | structured differences between two curated artifacts |
| `ValidationResult` | dict-compatible; `.to_dict()` and attribute access | programmatic validation result used by the CLI layer |
| `ExecutionPlan` | `(source, schema, window, strategy, output, options, stages=(), warnings=(), physical_plan=None)` | immutable mapping returned by `Pipeline.explain`; `.to_dict()` serializes and `.render(detailed=False)` formats text |
| `inspect_dataset` | `(dataset, *, verify=False, container=None, registry=None) -> DatasetInspection` | inspect manifests/config without original sources; `verify=True` decodes every shard |
| `preview_dataset` | `(*, schema, adapter, sources, max_sources=3, max_ticks_per_source=10000, examples=5) -> DatasetPreview` | classic-API form of the bounded data preview |
| `selection_review` | `(result, *, seed=0, examples=10, hist_bins=10) -> SelectionReview` | review an existing `CurateResult` |
| `hours_selection_review` | `(result) -> HoursSelectionReview` | review an hours-path `DatasetWriteResult` without rescoring |
| `validate_expectations` | `(expectations) -> dict` | validate and copy the closed quality-policy grammar |
| `evaluate_quality` | `(report, expectations=None, **kwargs) -> QualityGateResult` | evaluate every applicable gate without raising |
| `check_quality` | `(report, expectations=None, **kwargs) -> QualityGateResult` | evaluate gates and raise `QualityGateError` on failure |
| `compare_datasets` | `(left, right, *, verify=False) -> DatasetComparison` | compare artifact schema, output, pipeline, provenance, size, and integrity |
| `render_html_report` | `(*reports, title="kuration review") -> str` | render typed/mapping reports as dependency-free static HTML |
| `write_html_report` | `(path, *reports, title="kuration review") -> str` | write the static HTML report locally or through configured path I/O |
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
| `FeatureEngine` | `(schema, *, define=None, registered=None, tags_spec=None, registry=None)`; `.compute(ticks, segments)`, `.score_window(seg, window_ticks)` | runs all metrics over one buffer per segment; `registry` isolates registered metric resolution |
| `score_sources` | `(*, schema, adapter, sources, define, registered, segmenter, tags_spec=None, group_of=None, tags_of=None, workers=1, on_error="skip", streaming=False, progress=None, executor="local", registry=None, required_fields=()) -> (segments, errors, source_tracks)` | the shared ordered feature pass (raises on an empty pool) |

## Input adapters

The tabular and array adapters take `schema` first and expose `.ticks(source_uri)`, plus a
metadata-only `.check_source(source_uri)` preflight (used by
`kur-validate --check-sources`). Curated and video adapters derive their schema from an existing
artifact or media source; see [adapters.md](adapters.md).

| name | constructor | reads |
|---|---|---|
| `TimeseriesAdapter` | `(schema, *, fields, time_column=None, entity_column=None, globals=None)` | CSV / single Parquet (`[parquet]`) / `from_rows(rows)` |
| `JsonlAdapter` | `(schema, *, fields, time_column=None, entity_column=None, globals=None)` | `.jsonl` (one row-object per line) |
| `HDF5Adapter` | `(schema, *, fields, globals=None)` | `.h5`, one file = one source/entity (`[array]`) |
| `NpzAdapter` | `(schema, *, fields, globals=None)` | `.npz` (dep-free) |
| `ParquetDatasetAdapter` | `(schema, *, fields, path, source_per="part", time_column=None, entity_column=None, globals=None)`; `.sources()` | a directory/glob of Parquet parts (`[parquet]`) |
| `HuggingFaceAdapter` | `(schema, *, fields, dataset, split="train", config=None, time_column=None, entity_column=None, globals=None, source_column=None, load_kwargs=None)`; `.sources()` | a HF dataset (`[hf]`) |
| `ZarrAdapter` | `(schema, *, fields, globals=None)` | a Zarr store (`[zarr]`; remote via `[gcs]`, `[s3]`, or `[azure]`) |
| `CuratedDatasetAdapter` | `(dataset_dir, schema=None, registry=None)`; `.sources()`, `.dataset_of` | one dataset directory or a list of directories, re-curatable; `registry` resolves their shard containers |
| `schema_from_output` | `(output_spec, *, name="curated", sample_hz=None, entity_axis="e") -> SchemaSpec` | derive a schema from a dataset's `config.json` output block |
| `register_adapter` | `(name, factory, **metadata) -> None` | register an input-adapter factory and component metadata |
| `get_adapter` | `(name, *, registry=None, **kwargs) -> InputAdapter` | construct a registered adapter, optionally through an isolated registry |
| `InputAdapter` | Protocol: `schema` + `ticks(uri)`; optional lazy `iter_ticks(uri)`, `check_source(uri)`, and `sources_are_paths` | the seam to implement for bespoke formats |
| `FrameRef` | `(uri, source_id, frame_index, timestamp, stream_index=0, pts=None, time_base=None)` | immutable lazy reference to one source video frame |
| `VideoMetadata` | `(uri, width, height, frame_count, duration, average_rate, stream_index=0, variable_rate=False, codec=None)` | probed video dimensions, timing, stream, and codec metadata |
| `MediaProvider` | Protocol: `probe(uri)`, `iter_refs(uri, *, source_id=None)`, `decode_batch(refs)` | injectable media metadata/reference/decoding seam |
| `PyAVMediaProvider` | `(*, remote_cache_size=4)` | bounded CPU video provider (`[video]`); PyAV loads only when probing or decoding |
| `VideoAdapter` | `(sources, *, provider=None, column="frame", rate_tolerance=1e-09)` | turn video-only sources into ticks with one lazy media column |
| `JoinedMediaAdapter` | `(adapter, videos, *, provider=None, column="frame", on="timestamp", tolerance=None, mismatch="error")` | strictly align one video to every source from an existing adapter |

## Batch transforms and execution

| name | signature | does |
|---|---|---|
| `BatchTransform` | Protocol: `transform_batch(batch, ctx) -> mapping`; `output_features` | bounded window/media computation seam |
| `register_batch_transform` | `(name, factory=None, *, outputs=(), inputs=(), media_requirements=None, resources=None, lookbehind=0, lookahead=0, ownership=None, **metadata)` | register a transform factory directly or as a decorator |
| `register_transform` | `(name, factory=None, *, outputs=(), inputs=(), media_requirements=None, resources=None, lookbehind=0, lookahead=0, ownership=None, **metadata)` | compatibility alias for `register_batch_transform` |
| `get_transform` | `(name, *, registry=None) -> BatchTransform factory` | resolve a transform factory from the selected registry |
| `apply_transforms` | `(calls, segments, *, adapter, registry, progress=None, on_error="raise", errors=None)` | execute resolved transforms over canonical segments; source-level skip removes the complete failed source |
| `ExecutionBackend` | Protocol: `map_sources(items, fn, resources)`, `map_batches(...)`, `iter_batches(...)` | ordered execution seam |
| `LocalExecutionBackend` | `()` | in-process ordered backend with bounded producer backpressure |
| `RayExecutionBackend` | `(**init_kwargs)` | optional ordered Ray backend (`[ray]`) |
| `ResourceSpec` | `(num_cpus=1.0, num_gpus=0.0, memory=None)` | portable non-negative task resource request |
| `get_execution_backend` | `(name)` | construct the `"local"` or optional `"ray"` backend |

## Selection

| name | signature | does |
|---|---|---|
| `load_strategy` | `(name_or_spec, schema, *, registry=None, available_metrics=()) -> Strategy` | parse + validate a declarative strategy ([strategies.md](strategies.md)) |
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
| `register_scorer` | `(name, factory, **metadata) -> None` | register a whole-pool scorer factory (`action_residual` ships with `[methods]`) |
| `get_scorer` | `(name, *, registry=None, **kwargs) -> SegmentScorer` | construct a registered scorer, optionally through an isolated registry |
| `apply_scorers` | `(entries, segments, *, schema, adapter, seed=0, progress=None, registry=None, on_error="raise", errors=None) -> None` | run parsed scorers and write outputs into `seg.features`; optional source-level skip applies to second-read buffer failures |

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
| `ProgressSpec` | `ProgressCallback \| str \| bool` | user selection accepted by `progress_callback` and pipeline options |
| `notebook_progress` | `() -> ProgressCallback` | explicit rich notebook renderer |
| `cli_progress` | `(*, file=None) -> ProgressCallback` | throttled terminal renderer, optionally writing to a selected stream |
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
| `StreamAnalyzer` | protocol requiring `summary()` and `add_batch(batch, ctx)` | optional analyzer contract for tick-stream execution; `add_batch` receives the same batch/`TransformContext` a batch transform receives, accumulates state, and returns `None` |
| `SplitLeakageAnalyzer` | `(*, check_groups=True)` | built-in source/group split auditor; `check_groups=False` suppresses group leakage and conflict checks, but never source checks |
| `SplitLeakageAnalyzer.add_assignment` | `(source_id, split, *, group=None) -> None` | record one observation; repeating the same assignment is allowed |
| `SplitLeakageAnalyzer.add_assignments` | `(source_splits, *, source_groups=None) -> None` | record a `{source_id: split}` mapping and optional `{source_id: group}` mapping |
| `SplitLeakageAnalyzer.summary` | `() -> dict` | return the deterministic, JSON-safe leakage report described below |
| `register_analyzer` | `(name, factory=None, *, inputs=(), media_requirements=None, resources=None, lookbehind=0, lookahead=0, ownership=None, **metadata)` | register a discoverable analyzer factory directly or as a decorator |
| `get_analyzer` | `(name, *, registry=None, **params) -> Analyzer` | validate parameters and construct a user-owned analyzer instance |
| `run_dataset_analyzer` | `(name, dataset, *, parameters=None, options=None, registry=None) -> dict` | construct a dataset-capable analyzer, run its hook, and validate the boolean `passed` result |
| `run_stream_analyzer` | `(name, *, schema, adapter, sources, segmenter, parameters=None, options=None, inputs=None, batch_size=None, device="cpu", on_error="raise", progress=None, registry=None, analyzer=None) -> dict` | drive a stream-capable analyzer's `add_batch` hook over a segmented tick stream in one bounded pass (what `Pipeline.analyze()` calls) |
| `import_dataset` | `(sources, out, *, adapter=None, schema=None, fields=None, time=None, entity=None, hz=None, kinds=None, adapter_options=None, source_id=None, split="train", shard_length=None, container="npz", force=False, registry=None) -> dict` | snapshot local sources, detect/construct a registered adapter, and import the complete array or canonical-tick dataset into a committed artifact; generic remote sources fail closed because fsspec cannot bind the read to one generation |

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
| `PartitionCommit` | `(partition_id, identity, data_file, rows, sha256, columns, size_bytes=0, key_range=None, reducer_state=None, errors=())` | checksum-protected committed partition metadata |
| `StageCheckpoint` | `(root, stage_id, identity)` | validate, reuse, publish, and finalize independent partition commits |
| `canonical_hash` | `(value) -> str` | deterministic SHA-256 over canonical JSON |
| `FeatureStageResult` | `(commits, errors, source_tracks, telemetry={})` | bounded feature-island result without a total Python segment pool |
| `checkpoint_feature_stage` | `(*, schema, adapter, sources, define, registered, segmenter, checkpoint, batch_size=1024, tags_spec=None, predicates=(), transforms=(), registry=None, group_of=None, tags_of=None, on_error="skip", progress=None, max_in_flight=2, memory_budget=None, retain_predicate_failures=False) -> FeatureStageResult` | stream segmentation, metrics, predicates, and transforms into checkpoint partitions |
| `external_quantile_edges` | `(partitions, feature, n_bins, *, spill_dir=None) -> ndarray` | exact NumPy-compatible quantiles with disk-proportional state |
| `external_top_k` | `(partitions, *, weights, target_size, normalize="quantile", spill_dir=None, predicates=()) -> list[Segment]` | exact stable top-k that materializes only selected segments |
| `external_uniform_bins` | `(partitions, *, axes, target_size, seed=0, spill_dir=None, predicates=()) -> list[Segment]` | exact multipass quantile organization and deterministic disk-backed equal-bin selection |
| `iter_external_uniform_priorities` | `(partitions, *, axes, seed=0, spill_dir=None) -> Iterator[tuple[Segment, float]]` | exact disk-backed uniform-bin priorities for hours filling |
| `iter_external_top_k_priorities` | `(partitions, *, weights, normalize="quantile", spill_dir=None)` | stream every canonical segment with its exact disk-backed normalized priority |
| `iter_external_span_items` | `(prioritized_segments, source_tracks, *, sample_hz, span_len=None, clip_aggregate="max", group_of=None, spill_dir=None)` | aggregate overlapping segment priorities/features into a bounded span-item stream |
| `iter_external_weighted_priorities` | `(partitions, *, weights, normalize="quantile", seed=0, spill_dir=None)` | stream exact canonical Gumbel priorities from one disk-backed score index and one partition-at-a-time materialization pass |
| `external_weighted` | `(partitions, *, weights, target_size, normalize="quantile", seed=0, spill_dir=None) -> list[Segment]` | select exact hours-path Gumbel priorities while retaining only the selected heap; count-weighted sampling has a distinct NumPy contract |
| `iter_external_hash_dedup` | `(partitions, *, feature_names=None, per_group=False, spill_dir=None)` | exact disk-backed feature-vector dedup with full-payload collision resolution and canonical retention |
| `TransformContext` | mapping-compatible typed context; `.before(token)`, `.after(token)`, `.owned_lengths`, `.inputs`, `.input_token(role)`, `.valid_steps` | typed per-batch execution metadata, boundary access, declared-role→resolved-token inputs, and per-row valid timestep counts; ordinary `ctx["schema"]` access remains compatible and owned lengths/valid steps are read-only |
| `apply_transforms_to_windows` | `(calls, segments, windows, *, adapter, registry, contexts=None, progress=None)` | bounded transform execution over extracted windows; optional contexts carry lookbehind/lookahead ticks and owned lengths |

## Output

Media input contracts are listed under [Input adapters](#input-adapters), and transform/backend
contracts are listed under [Batch transforms and execution](#batch-transforms-and-execution).
`TransformContext` and the bounded physical-planning helpers remain in the preceding section
because they define checkpoint and batch execution records.

| name | signature | does |
|---|---|---|
| `OutputSpec` | `.from_dict({shard_length, short_policy?, container?, presence_mask?, modalities})` | declared modalities + chunking; `_present` enabled by default ([output.md](output.md)) |
| `Modality` | `(name, dtype, shape, src_kind, src_arg)` | one declared modality |
| `OutputEncoder` | type alias: `TimestepOutputEncoder \| BatchOutputEncoder` | derived-tensor seam implementing `timestep(record, ctx)` or `encode_batch(records, ctx)` |
| `PassthroughEncoder` | `()`; `.timestep(record, ctx) -> {}` | default encoder when every modality comes from a field or segment |
| `register_encoder` | `(name, factory, **metadata) -> None` | register an output-encoder factory |
| `get_encoder` | `(name, *, registry=None, **kwargs) -> OutputEncoder` | construct a registered encoder |
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

The facade itself also exports four registration modules and its idempotent registration hook:

| name | signature | does |
|---|---|---|
| `kuration.gameplay.pixel_shift` | module | pixel-shift transform, operator, and report facade |
| `kuration.gameplay.action_state_consistency` | module | action-state scorer (live pipeline), dataset-mode analyzer, and pipeline-operator facade |
| `kuration.gameplay.collision` | module | CS2 wall-collision transform/operator, directional tick-stream analyzer, adapters, and ramp-fit helpers |
| `kuration.gameplay.slam` | module | SLAM reconstruction, trajectory, and point-cloud facade |
| `kuration.gameplay.register` | `() -> None` | idempotently register every maintained gameplay component and stats enricher |

| name | signature | does |
|---|---|---|
| `kuration.gameplay.pixel_shift.pixel_shift` | `(pipeline, *, controls=None, boundary="exclude") -> Pipeline` | compositional `aintropy/pixel-shift` operator backed by the transform of the same name (`[pixel-shift]` for OpenCV execution) |
| `PixelShift` | `(*, controls=None, boundary="exclude")` | bounded batch transform producing per-control shift min/max/mean/measured/count/requested features |
| `PixelShiftReport` | `()`; `.add_segment(...)`, `.print_report()` | user-owned cross-segment instance report |
| `kuration.gameplay.action_state_consistency.ActionStateAnomalyScorer` | `(*, baselines=None, controls=None, calibration=None)`; `.score(segments, ctx) -> ndarray` | `SegmentScorer` registered as `aintropy/action-state-consistency`, reached through `.score(...)` on a live pipeline; per-segment count of `Segment.records["aintropy/pixel-shift"]` instances inconsistent with the baseline |
| `kuration.gameplay.action_state_consistency.action_state_consistency` | `(*, anomaly_type, baselines=None, calibration=None) -> ActionStateConsistencyChecker` | factory registered as the dataset-mode-only analyzer `aintropy/action-state-consistency`; `baselines=None` self-calibrates |
| `kuration.gameplay.action_state_consistency.attach_action_state_consistency` | `(pipeline, *, baselines=None, anomaly_type="action_state", controls=None, calibration=None, boundary="exclude", frames=None, actions=None, color_space=None, verify=None, metadata_key="action_state_consistency", replace=False) -> Pipeline` | dataset-mode-only handler registered as the direct pipeline operator of the same stable name; requires `baselines=` or `controls=` |
| `ActionStateConsistencyChecker` | `(config)`; `.add_segment(...)`, `.clip_verdict(source_id)`, `.summary()`, `.print_report()`, `.resolved_baselines_for_clip(source_id)`, `.is_instance_consistent(source_id, control, cell_key, shift)` | classify clips against pixel-shift baselines (`[action-state]` for frame measurement) |
| `AnomalyChecker` | `(validated_config)` | measurement-agnostic tuple aggregation used by the action-state adapter |
| `kuration.gameplay.collision.CollisionMechanics` | `(*, map_for_source, map_registry, dist_path, **thresholds)` | bounded batch transform registered as `aintropy/collision`: per-window collision counts/anomaly score raycast against map geometry (`[collision]`) |
| `kuration.gameplay.collision.attach_directional_collision` | `(pipeline, *, map_registry, ramp_models_path, map_for_source=None, default_map=None, fallback_ramp=None, disk_cache_dir=None, proximity=None, min_missing_speed=None, concentration_ratio=None, inputs=None, batch_size=None, metadata_key="directional_collision", replace=False) -> Pipeline` | handler registered as the direct pipeline operator `aintropy/collision-directional`; runs the tick-stream analyzer once and attaches its verdict |
| `DirectionalCollisionAnalyzer` | `(*, map_registry, ramp_models_path, map_for_source=None, default_map=None, fallback_ramp=(3.0, 215.0), ...)`; `.process_source(...)`, `.to_records()`, `.summary()` | acceleration-ramp-residual collision detector, drivable over the tick stream or directly per source |
| `kuration.gameplay.collision.fit_ramp_models` | `(sources, *, min_ramps=5) -> dict` | fit the `{(combo_bits, weapon): (tau, cap)}` acceleration-ramp table the directional analyzer consumes |
| `kuration.gameplay.collision.save_ramp_models` | `(models, path) -> None` | write a fitted ramp table to JSON |
| `kuration.gameplay.collision.load_ramp_models` | `(path) -> dict` | read a ramp table written by `save_ramp_models` back into lookup shape |
| `AsgCollisionAdapter` | `()`; `.ticks(source_uri)`, `.check_source(source_uri)`, `.discover_map_for_source(paths)` | InputAdapter registered as `aintropy/collision-asg`: ASG `data.json` -> canonical Ticks, one source per `(match, player)` |
| `TeaPearceDust2Adapter` | `()`; `.ticks(source_uri)`, `.check_source(source_uri)` | InputAdapter registered as `aintropy/collision-teapearce`: TeaPearce dust2 `.npy` -> canonical Ticks, one source per recording |
| `kuration.gameplay.collision.load_directional_arrays` | `(source_uri) -> dict` | flatten one TeaPearce recording's Tick stream into the plain per-source arrays `process_source()`/`fit_ramp_models()` take |
| `VideoComparison` | `(left, right, *, labels=None, provider=None)`; `.render()`, `.save()`, `.probe()`, `.plot(frames=4)`, rich HTML display | synchronized side-by-side video review; `[video]` for default metadata/frame decoding and `[stats]` for contact sheets |
| `kuration.gameplay.compare_videos` | `(left, right, *, labels=None, provider=None) -> VideoComparison` | ergonomic constructor for `VideoComparison` |

`StatsReport.available()` returns `plots`, `default_plot`, `example_modes`, `comparisons`,
`media_examples`, `html`, and `save`. Count-selection evidence modes are `selected`, `rejected`,
`strongest`, and `weakest`; comparisons are `selected-vs-rejected` and
`strongest-vs-weakest`. `n` is bounded to 1–100 and gameplay `frames` to 1–12.

The associated validation and decoding helpers are documented in
[pixel-shift.md](pixel-shift.md), [collision.md](collision.md), and
[action-state-consistency.md](action-state-consistency.md).

## Built-in selection recipes

The `kuration.selection` domain ships in the SDK wheel: three general-purpose curation recipes
with no gameplay/video assumptions, plus a config-driven `run`/`plan`/`describe` runner. See
[selection.md](selection.md).

| name | signature | does |
|---|---|---|
| `kuration.selection.operations` | module | `BalancedSelector`, `RareEventCoverage`, `DiverseSubset` dataclasses |
| `kuration.selection.operators` | module | operator wrappers (`balanced_selector`, `rare_event_coverage`, `diverse_subset`) and their JSON-Schema parameter descriptors |
| `kuration.selection.plan` | module | `Operation` protocol and the `run`/`plan`/`describe`/`register_operation_type` config runner |
| `kuration.selection.register` | `() -> None` | idempotently register the three selection operators |

## SLAM reconstruction and curation

`kuration.gameplay.slam` is a public built-in module. Import and precomputed trajectory curation remain
NumPy-only while live model dependencies stay optional. See [slam.md](slam.md) for the end-to-end
workflow.

| name | signature | does |
|---|---|---|
| `kuration.gameplay.slam.SLAMReconstructor` | `(depth_model=None, *, embedding_model=None, embeddings=None, depth_model_id=None, embedding_model_id=None, runtime_factory=None, **runtime_options)` | global reconstruction coordinator; raw execution needs `[slam]` unless a compatible runtime is injected |
| `SLAMReconstructor.run` / `SLAMReconstructor.__call__` | `(data, *, progress=None) -> SLAMOutput` | notebook-style in-memory reconstruction from `{clip: (T,H,W,3)}` frames |
| `SLAMReconstructor.reconstruct` | `(source, out_dir=None, *, run_if_needed=True, decoder_options=None, export_options=None, resume=False, cache_key=None, progress=None, preflight=True, require_gpu=False) -> SLAMReconstruction` | reuse/export/run as needed and return a portable manifest artifact |
| `SLAMReconstruction` | `(manifest: Path)`; `.to_pipeline(join_video=False, tolerance=None, mismatch="error", provider=None)` | typed result that opens the reconstruction as a normal `Pipeline` |
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
| `load_point_cloud` | `(source) -> PointCloudArtifact` | load an exported scene |
| `curate_point_cloud` | `(source, *, method="voxel", max_points=None, voxel_size=None, min_confidence=None, seed=0, output=None) -> PointCloudArtifact` | deterministically reduce an exported scene and optionally save it |
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
| `kuration.core.torch.ShardDataset` | `(dataset, *, split=None, modalities=None, trim=True, container=None, transform=None, include_meta=False, verify=True, registry=None)` | map-style PyTorch `Dataset` over the above (`[torch]`) |

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

Migration is non-mutating for dictionary inputs. A malformed version, non-object input, or any
unsupported future version (major or minor) raises `ValueError`. The matching command-line interface is
`kur-migrate`; see [jobs.md](jobs.md) for the serialized job contract.
