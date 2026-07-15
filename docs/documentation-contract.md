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
| Extender | “How do I support a new format or method?” | Implements and registers one supported extension seam | customization and adapter guides |
| Operator | “How do I run and reproduce this reliably?” | Validates, runs, resumes, and inspects a job or artifact | CLI/job and output guides |

Reference readers are a behavior rather than a separate audience: any of the four may need an
exact signature, accepted value, return type, exception, or artifact field.

## Entry paths

The README is the SDK landing page. It should help a reader choose a path, not attempt to contain
the complete manual.

1. **Try it** — a copy-and-paste quick start using the fluent API and a tiny local dataset. It
   writes a real artifact, reads one shard back, and links to the next concepts.
2. **See it on real data** — bounded worked examples for public datasets, beginning with
   TeaPearce Counter-Strike and Open P2P.
3. **Build a pipeline** — task-oriented recipes and the fluent-pipeline guide.
4. **Customize it** — the classic API, schemas, adapters, metric DSL, strategies, output seams,
   and registries.
5. **Look something up** — the Python API, CLI/job-spec, and artifact references.

The fluent API is the default teaching interface. The classic API is the explicit path for
custom adapters, encoders, strategies, and lower-level control. JSON jobs and the `kur-*` CLIs
are the reproducible operational path. Documentation should name the path being used whenever a
reader could reasonably confuse them.

## Information architecture

The target structure below mostly reuses the existing pages. New pages are marked **new**.

| layer | canonical content | purpose |
|---|---|---|
| Landing | `README.md` | value proposition, minimal quick-start excerpt, route to all entry paths |
| Getting started | `docs/quickstart.md` (**new**) | complete tested ten-minute path |
| Worked examples | `examples/README.md`, `examples/public_data/` (**new**) | runnable synthetic and public-data examples |
| Task guides | `docs/usage.md`, `docs/pipeline.md` | recipes and the fluent operator workflow |
| Concepts | `docs/overview.md`, `docs/schema.md`, `docs/dsl.md`, `docs/strategies.md` | mental model and core data contracts |
| Inputs and outputs | `docs/adapters.md`, `docs/output.md` | supported storage, mapping, encoding, and artifacts |
| Extension guides | `docs/customization.md`, relevant sections of `docs/architecture.md` | supported seams, registries, plugins, and design constraints |
| Operations | `docs/jobs.md`, scale/reliability recipes in `docs/usage.md` | validation, CLI runs, streaming, parallelism, resume, cloud I/O |
| Reference | `docs/api.md`, shipped `job.schema.json` | exact Python and job-spec contracts |
| Maintainer contract | this page | documentation scope, ownership, and coverage rules |

Content has one canonical home. Other pages summarize and link instead of maintaining a second
full explanation. In particular:

- `docs/quickstart.md` owns the complete first run; the README contains a short tested version.
- `docs/pipeline.md` owns fluent operator behavior; the API reference owns exact signatures.
- Layer guides own concepts and examples; `docs/api.md` links to them rather than repeating them.
- `docs/jobs.md` explains the job format; `kuration/schemas/job.schema.json` is the
  machine-readable key and validation authority.
- `docs/architecture.md` explains design decisions, not ordinary usage.

## Supported public interface

“Full API reference” means complete coverage of the boundary in this section. Being public means
the interface is intentionally supported and must not disappear silently. It does not mean the
pre-`1.0` API is frozen.

### Top-level Python API

`kuration.__all__` is the machine-readable inventory for `from kuration import ...`. Its current
95 names are grouped below by user task.

- Entry and results: `input`, `Pipeline`, `infer_schema`, `curate`, `CurateResult`,
  `write_dataset`, `DatasetWriteResult`, `run_experiment`, `Arm`, `union_define`,
  `ExperimentResult`, `dataset_report`, `DatasetReport`, `ExecutionPlan`, `ValidationResult`.
- Schema and canonical data: `SchemaSpec`, `FieldSpec`, `Tick`, `WindowBuffer`, `build_window`,
  `empty_like`, `Segment`, `Segmenter`.
- Metrics and scoring: `compile_metric`, `register_metric`, `register_compiled`, `FeatureEngine`,
  `score_sources`, `SegmentScorer`, `register_scorer`, `get_scorer`, `apply_scorers`.
- Inputs: `InputAdapter`, `TimeseriesAdapter`, `JsonlAdapter`, `HDF5Adapter`, `NpzAdapter`,
  `ParquetDatasetAdapter`, `HuggingFaceAdapter`, `ZarrAdapter`, `CuratedDatasetAdapter`,
  `schema_from_output`, `register_adapter`, `get_adapter`.
- Strategies and selection: `Strategy`, `load_strategy`, `bin_segments`, `quantile_bins`,
  `select`, `segment_priorities`, `assign_splits`, `apply_filters`, `method_priorities`,
  `register_method`, `register_filter`.
- Output and read side: `OutputSpec`, `Modality`, `OutputEncoder`, `PassthroughEncoder`,
  `Track`, `tracks_from_ticks`, `write_track`, `fill_to_hours`, `Container`,
  `register_encoder`, `get_encoder`, `register_container`, `output_preset`, `register_preset`,
  `iter_shards`, `read_shard_arrays`, `shard_entries`.
- Registries, plugins, and progress: `Registry`, `default_registry`, `ComponentDescriptor`,
  `list_components`, `describe_component`, `load_plugins`, `loaded_plugin_modules`,
  `plugin_provenance`, `ProgressCallback`, `register_operator`.
- Manifests and I/O: `build_config`, `dump_segment_manifest`, `write_config`, `read_text`,
  `write_text`, `write_bytes`, `exists`, `local_copy`.
- Versioning and migration: `ARTIFACT_FORMAT_VERSION`, `JOB_SCHEMA_VERSION`, `migrate`,
  `migrate_job`, `migrate_config`.

A documentation coverage check should compare the reference inventory to `kuration.__all__` so
an added or removed export cannot drift unnoticed.

### Fluent pipeline

All non-underscore methods intentionally exposed by `Pipeline` are public:

- Schema and resolution: `using`, `semantic`, `as_position`, `as_angular`, `as_bitmask`,
  `at_rate`.
- Pipeline construction: `apply`, `window`, `measure`, `filter`, `organize`, `score`, `tag`,
  `split`, `group`, `options`, `curate`.
- Inspection: `head`, `describe`, `explain`.
- Terminals: `collect`, `report`, `experiment`, `output`.

The reference must cover these methods even though they are not individual members of
`kuration.__all__`.

### Framework integration

`kuration.torch.ShardDataset` is public and supported. It is kept out of the top-level namespace
so importing `kuration` does not imply a PyTorch dependency.

Other module members that are absent from `kuration.__all__` are internal unless a future
contract explicitly adds them. Underscore-prefixed classes, functions, methods, and CLI helpers
are always internal.

### Command-line and declarative interfaces

The installed commands are public end-user interfaces:

- Run and analyze: `kur-curate`, `kur-experiment`, `kur-report`.
- Validate and inspect: `kur-validate`, `kur-explain`.
- Discover extensions: `kur-list`, `kur-describe`.
- Upgrade documents: `kur-migrate`.

Their command options and exit behavior belong in `docs/jobs.md` or a linked CLI reference. The
JSON job contract is the shipped `kuration/schemas/job.schema.json`, including supported version
and migration behavior. Internal functions under `kuration.cli` are not a supported Python API.

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
