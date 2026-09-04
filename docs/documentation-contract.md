# Documentation contract

This document defines who the kuration SDK documentation serves, how readers enter and move
through it, and what counts as the supported public interface. It is the source of truth for
documentation structure and coverage; individual guides remain the source of truth for product
behavior.

The SDK is currently a beta (`0.1.x`). Everything identified below as public must be documented
and tested, but breaking changes may still be made deliberately before `1.0` when they are called
out in release notes and reflected in the documentation.

## Reader outcomes

The documentation serves four readers. A page may help more than one, but it should have one
primary reader and outcome.

| reader | question | successful outcome | primary entry |
|---|---|---|---|
| Evaluator | “Will this help me curate my data?” | Produces and inspects a curated dataset in ten minutes | README → quick start |
| Practitioner | “How do I solve a real curation task?” | Adapts a worked example to a local or public dataset | worked examples → cookbook |
| Extender | “How do I support a new format, operator, or method?” | Implements, tests, and discovers one supported extension seam | operator, customization, and adapter guides |
| Operator | “How do I run and reproduce this reliably?” | Validates, runs, resumes, and inspects a job or artifact | CLI/job and output guides |

Reference readers are a behavior rather than a separate audience: any of the four may need an
exact signature, accepted value, return type, exception, or artifact field.

## Entry paths

The README is the SDK landing page. It should help a reader choose a path, not attempt to contain
the complete manual.

1. **Try it** — a copy-and-paste quick start using the pipeline API and a tiny local dataset. It
   writes a real artifact, reads one shard back, and links to the next concepts.
2. **See it on real data** — bounded worked examples for public datasets, beginning with
   TeaPearce Counter-Strike and Open P2P.
3. **Build a pipeline** — task-oriented recipes and the pipeline guide.
4. **Package a decision** — operator and batch-transform examples for reusable workflows.
5. **Customize it** — the classic API, schemas, adapters, metric DSL, strategies, output seams,
   and registries.
6. **Look something up** — the Python API, CLI/job-spec, and artifact references.

The pipeline API is the default teaching interface. The classic API is the explicit path for
custom adapters, encoders, strategies, and lower-level control. JSON jobs and the `kur-*` CLIs
are the reproducible operational path. Documentation should name the path being used whenever a
reader could reasonably confuse them.

## Information architecture

The target structure below mostly reuses the existing pages. New pages are marked **new**.

| layer | canonical content | purpose |
|---|---|---|
| Landing | `README.md` | outcome-oriented product page, minimal quick-start excerpt, proof points, and routes |
| Documentation home | `docs/index.md` | journey-based navigation across guides, extensions, operations, and reference |
| Getting started | `docs/quickstart.md` | complete tested ten-minute path |
| Worked examples | `examples/README.md`, `examples/public_data/` | runnable synthetic and public-data examples |
| Task guides | `docs/usage.md`, `docs/pipeline.md` | recipes and the operator workflow |
| Concepts | `docs/overview.md`, `docs/schema.md`, `docs/dsl.md`, `docs/strategies.md` | mental model and core data contracts |
| Inputs and outputs | `docs/adapters.md`, `docs/output.md` | supported storage, mapping, encoding, and artifacts |
| Extension guides | `docs/operators.md`, `docs/customization.md`, relevant sections of `docs/architecture.md` | reusable operators, supported seams, registries, plugins, and design constraints |
| Design thesis | `DESIGN.md` (repo root) | the cross-cutting progressive-disclosure thesis and decision ladder every layer instantiates; read before adding a seam |
| Operations | `docs/jobs.md`, scale/reliability recipes in `docs/usage.md` | validation, CLI runs, streaming, parallelism, resume, cloud I/O |
| Reference | `docs/api.md`, shipped `job.schema.json` | exact Python and job-spec contracts |
| Roadmap | `ROADMAP.md`, linked detailed plans | release gates and sequenced future direction |
| Maintainer contract | this page | documentation scope, ownership, and coverage rules |

Content has one canonical home. Other pages summarize and link instead of maintaining a second
full explanation. In particular:

- `docs/quickstart.md` owns the complete first run; the README contains a short tested version.
- `docs/pipeline.md` owns operator behavior; the API reference owns exact signatures.
- Layer guides own concepts and examples; `docs/api.md` links to them rather than repeating them.
- `docs/jobs.md` explains the job format; `src/kuration/core/schemas/job.schema.json` is the
  machine-readable key and validation authority.
- `docs/architecture.md` explains design decisions, not ordinary usage.
- `DESIGN.md` states the one cross-cutting thesis (progressive disclosure) and its decision
  ladder; it links to `docs/architecture.md`, `docs/customization.md`, and `docs/operators.md`
  for the per-layer explanations rather than repeating them.

## Supported public interface

“Full API reference” means complete coverage of the boundary in this section. Being public means
the interface is intentionally supported and must not disappear silently. It does not mean the
pre-`1.0` API is frozen.

### Top-level Python API

`kuration.__all__` is the machine-readable inventory for `from kuration import ...`. Its names
are grouped below by user task.

- Entry and results: `input`, `Pipeline`, `infer_schema`, `curate`, `CurateResult`,
  `write_dataset`, `DatasetWriteResult`, `run_experiment`, `Arm`, `union_define`,
  `ExperimentResult`, `dataset_report`, `DatasetReport`, `stats`, `StatsReport`,
  `StatsExamples`,
  `preview_dataset`, `DatasetPreview`,
  `selection_review`, `SelectionReview`, `inspect_dataset`, `DatasetInspection`,
  `compare_datasets`, `DatasetComparison`, `render_html_report`, `write_html_report`,
  `hours_selection_review`, `HoursSelectionReview`, `evaluate_quality`, `check_quality`,
  `validate_expectations`, `QualityGateResult`, `QualityGateError`, `ExecutionPlan`,
  `ValidationResult`.
- Schema and canonical data: `SchemaSpec`, `FieldSpec`, `MediaSpec`, `Tick`, `WindowBuffer`,
  `build_window`, `empty_like`, `Segment`, `Segmenter`, `FrameRef`, `VideoMetadata`.
- Metrics and scoring: `compile_metric`, `register_metric`, `register_compiled`, `FeatureEngine`,
  `score_sources`, `SegmentScorer`, `register_scorer`, `get_scorer`, `apply_scorers`.
- Inputs: `InputAdapter`, `TimeseriesAdapter`, `JsonlAdapter`, `HDF5Adapter`, `NpzAdapter`,
  `ParquetDatasetAdapter`, `HuggingFaceAdapter`, `ZarrAdapter`, `CuratedDatasetAdapter`,
  `VideoAdapter`, `JoinedMediaAdapter`, `MediaProvider`, `PyAVMediaProvider`,
  `schema_from_output`, `register_adapter`, `get_adapter`.
- Batch transforms and execution: `BatchTransform`, `TransformContext`,
  `register_batch_transform`, `register_transform`, `get_transform`, `apply_transforms`,
  `apply_transforms_to_windows`, `ExecutionBackend`, `ResourceSpec`, `LocalExecutionBackend`,
  `RayExecutionBackend`, `get_execution_backend`.
- Bounded execution and physical planning: `SegmentBatch`, `IDENTITY_COLUMNS`,
  `iter_segment_batches`, `CheckpointIdentity`, `PartitionCommit`, `StageCheckpoint`,
  `canonical_hash`, `FeatureStageResult`, `checkpoint_feature_stage`, `external_quantile_edges`,
  `external_top_k`, `external_uniform_bins`, `iter_external_top_k_priorities`,
  `iter_external_span_items`, `iter_external_uniform_priorities`, `external_weighted`,
  `iter_external_weighted_priorities`, `iter_external_hash_dedup`, `OperatorResources`,
  `OperatorTraits`, `PhysicalOperator`, `PhysicalStage`, `PhysicalPlan`,
  `compile_physical_plan`.
- Strategies and selection: `Strategy`, `load_strategy`, `bin_segments`, `quantile_bins`,
  `select`, `segment_priorities`, `assign_splits`, `apply_filters`, `method_priorities`,
  `register_method`, `register_filter`.
- Output and read side: `OutputSpec`, `Modality`, `OutputEncoder`, `PassthroughEncoder`,
  `Track`, `tracks_from_ticks`, `write_track`, `fill_to_hours`, `Container`,
  `register_encoder`, `get_encoder`, `register_container`, `output_preset`, `register_preset`,
  `iter_shards`, `read_shard_arrays`, `shard_entries`.
- Registries, plugins, and progress: `Registry`, `default_registry`, `ComponentDescriptor`,
  `list_components`, `describe_component`, `load_plugins`, `loaded_plugin_modules`,
  `plugin_failures`, `plugin_provenance`, `check_plugins`, `PluginCheckResult`, `ProgressCallback`,
  `ProgressSpec`, `progress_callback`, `notebook_progress`, `cli_progress`,
  `register_operator`, `detect_adapter`, `Analyzer`, `DatasetAnalyzer`, `StreamAnalyzer`,
  `SplitLeakageAnalyzer`, `register_analyzer`, `get_analyzer`, `run_dataset_analyzer`,
  `run_stream_analyzer`.
- Manifests and I/O: `build_config`, `dump_segment_manifest`, `write_config`, `read_text`,
  `write_text`, `write_bytes`, `exists`, `local_copy`, `import_dataset`.
- Versioning and migration: `__version__`, `ARTIFACT_FORMAT_VERSION`, `JOB_SCHEMA_VERSION`, `migrate`,
  `migrate_job`, `migrate_config`.

A documentation coverage check should compare the reference inventory to `kuration.__all__` so
an added or removed export cannot drift unnoticed.

### Pipeline

All non-underscore methods intentionally exposed by `Pipeline` are public:

- Schema and resolution: `using`, `semantic`, `as_position`, `as_angular`, `as_bitmask`,
  `at_rate`.
- Pipeline construction and enrichment: `window`, `join_video`, `transform`, `measure`, `filter`, `bin`,
  `score`, `tag`, `split`, `group`, `options`, `expect`, `sample`, plus direct methods generated
  for registered operators; `with_metadata` and `enrich` retain input-backed state on a
  pipeline; `apply`, `organize`, and `curate` remain compatibility aliases.
- Inspection: the defensive `metadata` property, `head`, `describe`, `preview`, `review`,
  `explain`, and `analysis_mode` (the resolved analyzer execution mode).
- Terminals: `collect`, `report`, `stats`, `analyze`, `check`, `experiment`, `output`.

Registered operator methods normalize the local component name (`acme/signal-diversity` becomes
`signal_diversity`) and expose a namespace-qualified spelling (`acme__signal_diversity`) for
collisions. The reference must cover this generated method contract and the static methods above
even though they are not individual members of `kuration.__all__`.

### Framework integration

`kuration.core.torch.ShardDataset` is public and supported. The built-in
`kuration.gameplay.pixel_shift`, `kuration.gameplay.action_state_consistency`,
`kuration.gameplay.slam`, `kuration.gameplay.compare_videos`, and
`kuration.gameplay.VideoComparison` are also
public. Their direct APIs are documented in `docs/api.md` and their focused guides. Importing
`kuration.gameplay` registers these capabilities but must not import OpenCV, PyTorch, FAISS,
or another optional runtime. Importing the core `kuration` package does not import gameplay.

`kuration.selection.BalancedSelector`, `RareEventCoverage`, `DiverseSubset`, `run`, `plan`,
`describe`, and `register_operation_type` are public, documented in `docs/api.md` and
`docs/selection.md`. Importing `kuration.selection` registers these as `Pipeline`
methods but must not import scikit-learn; that only loads when `DiverseSubset`'s
`cluster-balanced` sampling path actually executes.

Other module members that are absent from `kuration.__all__` are internal unless a future
contract explicitly adds them. Underscore-prefixed classes, functions, methods, and CLI helpers
are always internal.

### Command-line and declarative interfaces

The installed commands are public end-user interfaces:

- Dispatch and discover commands: `kur` (`kur <command>` == `kur-<command>`, plus
  `kur completion`).
- Author a job: `kur-infer`, `kur-init`, `kur-preview`.
- Run and analyze: `kur-curate`, `kur-experiment`, `kur-report`.
- Validate and inspect: `kur-validate`, `kur-explain`, `kur-inspect`, `kur-stats`, `kur-diff`,
  `kur-check`.
- Import and export tensor data: `kur-import`, `kur-export`.
- Discover extensions: `kur-list`, `kur-describe`, `kur-plugin-check`.
- Reconstruct and curate SLAM data: `kur-slam`.
- Maintain execution state: `kur-checkpoint`.
- Upgrade documents: `kur-migrate`.

Their command options and exit behavior belong in `docs/jobs.md` or a linked CLI reference. The
JSON job contract is the shipped `src/kuration/core/schemas/job.schema.json`, including supported version
and migration behavior. Internal functions under `kuration.core.cli` are not a supported Python API.

### Artifact interface

Files intended for downstream consumption are also public contracts: `config.json`,
`dataset_manifest.json`, shard contents, presence masks, paths recorded in manifests, and their
format-version fields. `docs/output.md` owns their human-readable reference. Their serialized
shape should be covered by tests independently of the Python API reference.

## Reference requirements

Each public Python object should document, where applicable:

- import path and exact signature;
- purpose and when to choose it;
- parameter meaning, accepted values, and defaults;
- return type and stable result fields;
- raised validation or execution errors;
- required optional extra;
- one minimal example or a link to a canonical example;
- relevant version or deprecation information.

The reference may be generated from source signatures and docstrings, but grouping, navigation,
and “when to use this” guidance remain curated. Generated output must not expose undocumented
internal members.

## Documentation definition of done

A documentation change is complete when all relevant conditions hold:

- Commands and code blocks run from a clean supported Python environment.
- The quick start is network-independent and contains no placeholders.
- Public-data examples have bounded sample paths, provenance, licensing notes, expected resource
  use, and offline structural fixtures for CI.
- Every public interface defined above appears in its canonical reference.
- Public API additions include docstrings and reference coverage in the same change.
- Internal links, anchors, imports, and public signatures are checked in CI.
- Optional dependencies are named at the point of use.
- Examples state their expected output and fail loudly when an assumption is violated.
- A page links to the next likely task instead of ending at a dead end.

## Out of scope

This contract does not make every implementation module public, promise `1.0` compatibility,
require large public datasets in CI, or turn architecture notes and generated signatures into
substitutes for task-oriented guidance.
