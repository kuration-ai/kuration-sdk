# Changelog

This project follows semantic versioning after `1.0`. During the `0.x` beta, minor releases may
change documented APIs with the notice and deprecation process in `docs/compatibility.md`.

## 0.1.0rc5

Target release: `0.1.0rc5` (current development identity: `0.1.0rc5.dev0`).

- Every built-in operator is now labeled Selection, Annotation, or Aggregation in
  `docs/operators.md` and `docs/operator-inventory.md`, by its primary effect on the segment
  pool. No built-in operator is Aggregation yet — a confirmed gap (reducing a per-window
  feature to a per-group rollup), not an oversight; see `action-state-consistency.md`.
- Renamed `gameplay.slam_pipeline` to `gameplay.slam_reconstruct` (groups with
  `slam_quality`/`slam_motion_diversity` under one `slam_` prefix for discoverability). Clean
  rename, no compatibility alias. Extracted `CollisionMechanics`'s per-tick velocity/turn-rate
  derivation from a private, untested function into a public, directly tested
  `kuration.gameplay.collision.kinematics.extract_kinematics` — the one part of its fused
  per-tick scan/classify loop that is independently reusable; the gating and raycast
  classification stages remain fused, as does `DirectionalCollisionAnalyzer`'s unrelated
  expected-velocity-from-ramp-model computation.
- Added `Segment.records: dict[str, list]`, a non-scalar counterpart to `Segment.features` for
  transforms measuring several discrete per-instance events within a window. `PixelShift` now
  emits its per-press instances there instead of discarding them after collapsing to scalars.
  `action_state_consistency` is rehosted as `ActionStateAnomalyScorer`, a `SegmentScorer`
  reading pixel-shift's own records instead of redundantly re-deriving them from raw frames —
  it scores per window, not per whole clip, since no built-in aggregation operator exists yet
  to reconstruct the old per-clip verdict (see above). Dataset-mode auditing (`check_dataset`)
  never shared computation with a live pixel-shift transform and is unaffected; only the
  tick-stream-driven analyzer path (the actual redundant-computation bug) is retired.
- Eliminated the standalone `kuration-ai` package. Its three curation recipes
  (`BalancedSelector`, `RareEventCoverage`, `DiverseSubset`) and config-driven
  `run`/`plan`/`describe`/`register_operation_type` runner are now first-class SDK
  components under `kuration.selection`, each also registered as a `Pipeline`
  operator (`kuration/balanced-selector`, `kuration/rare-event-coverage`,
  `kuration/diverse-subset`). `kuration-ai` is preserved, unmodified, on the
  `feature/kuration-ai-legacy` branch.
- Development builds now use the distinct `0.1.0rc5.dev0` identity instead of reusing the
  prior `0.1.0rc4` candidate identity. `ARTIFACT_FORMAT_VERSION` is `1.2` for the corrected
  bounded/materialized uniform-bin ordering, experiment-arm tag isolation, and zero
  tag-coverage-floor semantics described below.
- Raised the minimum versions of scikit-learn, PyArrow, protobuf, PyTorch/TorchVision, Ray,
  tqdm, and Pillow past known-vulnerable historical floors; the minimum-dependency CI lane now
  exercises those security-supported baselines.
- Dataset publication is no-clobber by default across pipeline, classic, job/CLI, and import
  paths. `force=True` (or `kur-curate --force`) explicitly replaces a destination; `resume=True`
  remains a separate, fingerprint-matched hours-run intent. Selection, source-contract,
  container-dependency, and output-memory preflights finish before invalidating an old commit;
  one writer lease prevents concurrent publishers from interleaving. Resume markers are now
  commit-checksummed, non-resumable replacements remove stale markers, and bounded checkpoints
  are namespaced by run fingerprint.
- Fixed selection correctness across execution modes: bounded uniform-bin RNG order now matches
  canonical materialized order, predicates remain hard exclusions through tag coverage,
  fractional tag floors round up, experiment arms no longer inherit unrelated tag vocabularies,
  `min_tag_frac=0` is a true no-op, and Python-equal group/tag values have the same meaning in
  memory and SQLite-backed fills.
- Persisted reuse now fails closed unless inputs have adapter, local-content, or immutable remote
  version identities. Readers pin one exact commit generation for each operation, validate
  manifest modality contracts against the committed output config, and reject local shard
  symlinks and non-regular files. Source references and persisted skip diagnostics no longer
  publish credentials or host-local paths.
- Smart imports now validate and consume one private local source snapshot: POSIX ancestry is
  traversed through no-follow directory descriptors, array/tick publication never reopens the
  caller pathname, and directory stores are copied descriptor-relatively. Generic remote imports
  fail closed because an unconditioned fsspec read cannot be bound to a separately reported
  object generation; download the exact immutable generation locally first. Smart import also
  fails before source inspection on platforms without POSIX descriptor-relative open support.
- `on_error="skip"` now applies consistently to transform and scorer second-read failures as
  well as the feature pass; failed sources are removed as a unit, mutable transform state is
  reset before continuing, and total failure remains loud.
- Tightened validation at the boundary: `speed` requires a declared sample rate, numeric filter
  literals must be finite, bitmask values must be positive exact integers, and mixed-case
  `.parquet` suffixes follow one detection/preflight/read path.
- **Analyzers are pipeline operations.** New `Pipeline.analyze(name, parameters=None, *,
  mode=None, inputs=None, batch_size=None, analyzer=None, **options)` terminal drives a
  registered stateful analyzer with no job spec and no CLI. On any `input(...)` pipeline it
  drives a new optional `add_batch(batch, ctx)` hook (`StreamAnalyzer`) over the segmented tick
  stream through the *same* bounded batch/media-decode machinery batch transforms use
  (`transforms.build_batch`, one decode path, per-input lookbehind/lookahead, `ctx.input_token`,
  `ctx.valid_steps`) — a single pass with one batch resident, so it works under
  `streaming=True`. On an artifact-backed pipeline it drives `check_dataset` and returns exactly
  what `run_dataset_analyzer()` returns. Analyzers declare `field:`/`media:` inputs (plus
  optional per-input boundary counts) at registration, validated there and by
  `kur-plugin-check`; unknown analyzers, a missing hook for the resolved mode, and a pool that
  measured zero segments all raise. `run_stream_analyzer()` and `StreamAnalyzer` are exported for
  callers driving an analyzer outside the pipeline API.
- The built-in `aintropy/action-state-consistency` analyzer now runs on both modes. Its
  tick-stream path declares the same input contract as `aintropy/pixel-shift` and measures in
  **real source frames**, so the documented `(source_id, control, cell_key, frame)`
  de-duplication of one press seen by overlapping windows holds (`cross_window_dedup: true`) —
  the guarantee dataset mode has to give up (`frame_space: "shard-concatenation"`).
- New `Pipeline.check(policy=None, *, verify=False, **expectations)` — the pipeline-API
  equivalent of `kur-check --policy`: it merges `.expect(...)`, a policy mapping, and keywords, then evaluates
  them without raising against an artifact inspection (artifact-backed pipelines) or a bounded
  input preview. `CuratedDatasetAdapter.datasets()` is the new adapter seam that lets a pipeline
  reach artifact-level hooks at all.
- `Pipeline.stats(**kwargs)` now forwards every keyword `kuration.stats()` accepts (including
  `progress`) instead of re-declaring a narrower subset that could drift.
- Batch-transform inputs are now roles rather than fixed batch keys. `TransformContext` carries
  `inputs` (declared role → resolved token, in declaration order) and `input_token(role)`, so a
  transform used with a per-use `inputs=` remapping addresses the batch it was actually given
  instead of failing mid-run on a hardcoded registration token. A remapping must supply exactly
  one token per role, of the same kind, and per-input `media_requirements` are matched by role;
  all three are validated while the pipeline is built. The bundled `aintropy/pixel-shift`
  transform now resolves its frames/actions inputs (and its media `color_space`) this way.
- `lookbehind`/`lookahead` may be declared per input (`lookbehind={"field:action": 1}`) instead
  of as one count shared by every input, so a transform no longer pays to extract — or, for
  media, decode — boundary context it never reads. `aintropy/pixel-shift` declares its single
  lookbehind row on the action input only, removing one video-frame decode per segment.
- Every batch transform now receives per-row valid timestep counts (`ctx.valid_steps` and the
  `valid_steps` batch entry). `field:` inputs are `(B,T,E,D)` arrays NaN-padded to the widest
  extent in the batch — a contract change from the earlier `(B,T)` stacking, which raised on a
  ragged batch instead of padding it — and the counts make that padding observable, so a
  transform cannot silently average pad cells; previously only `ownership="rightmost-start"`
  transforms received any length information.
- `pixel_shift` gained a `{control}_shift_measured` (0/1) feature. The transform contract admits
  only finite scalars, so an untested control's `shift_min`/`max`/`mean` must report some number;
  the flag makes "no instance was measured" filterable
  (`.filter("w_shift_measured", ">=", 1)`) instead of indistinguishable from a real zero-shift
  measurement. `pixel_shift.iter_instances()` now validates its own arrays (pass
  `validated=True` for arrays already returned by `validate_segment_inputs`).
- The action-state analyzer accepts NumPy integer segment indexes (normalizing them to `int`),
  and its `check_dataset` no longer presents synthetic offsets as source frames: results report
  `frame_space: "shard-concatenation"`, per-shard `shard_ranges`, and `cross_shard_dedup: false`,
  with the same caveat printed in the report. Manifest entries carry no source start offset, so
  cross-shard de-duplication of overlapping curated spans is documented as unavailable in
  dataset mode instead of silently lost.
- Gameplay video contact sheets label a frame index as `frame N` instead of formatting it as
  `"{:.2f}s"` when a source exposes neither a duration nor an average frame rate.
- Bumped `ARTIFACT_FORMAT_VERSION` to `1.1` for two DSL semantic changes and recorded them in a
  compat table (`ARTIFACT_SEMANTIC_CHANGES` in `kuration.core.versioning`): `dispersion` and
  `min_pair_dist` now use every declared position dimension (`1.0` used only the first two), and
  `scope: per_entity_mean` is now honored on categorical fields (`1.0` silently pooled them).
  Both change *which segments get selected* on 3D-position or per-entity-categorical data, so
  `kur-migrate` now prints the notes for the versions a document has not seen while it migrates
  it. `migration_notes()`/`document_migration_notes()` expose the same table programmatically.
- Registering a compositional operator whose method name is already a real `Pipeline`
  attribute (`report`, `window`, `filter`, `schema`, ...) now fails loudly at registration and in
  `kur-plugin-check`, naming the collision and asking for a rename. Python resolves the attribute
  before `__getattr__`, so such an operator was silently shadowed and unreachable. The reserved
  set is derived from the `Pipeline` class itself.
- An un-namespaced operator now keeps its own method spelling when a plugin registers the same
  local name: a spelling that is a component's complete name outranks the same spelling used as a
  namespaced component's shorthand (`smooth` vs `acme/smooth`, still `acme__smooth()`). An
  ambiguity error now only suggests spellings that actually resolve, falling back to
  `apply("<name>")` when a component has no unique method spelling left.
- Added two pipeline-only public-data walkthroughs. `examples/public_data/ett.py` curates
  real hourly electricity-transformer telemetry, and `person_activity.py` curates 25 real
  wearable-sensor recordings with four tags as the entity axis, a tag-coverage floor that keeps
  rare activities, and source-level splits stratified by person. Both take a local path, fetch
  the data with `--download`, or run offline against synthetic stand-ins with `--demo`; each
  ships a provenance/licensing note and the equivalent command-line session.
- Simplified the numbered examples. `01` is now the genuine five-minute start — `input()` over a
  folder of CSVs with everything inferred — and `02`/`03` drop their hand-rolled adapter classes
  in favour of passing `{source_id: ticks}` straight to `input()`.
- Fixed `dataset_report` crashing on a metric whose values differ only at float-noise scale.
  NumPy widens the histogram range for exactly-constant data but raises "Too many bins for data
  range" just below that, which took down `.review()` and `kur-report` for a near-flat field;
  such a metric now reports as the flat distribution it is.
- `kur-init` now scaffolds `span_steps` alongside its hours budget. Without a span the budget
  admits whole tracks, so a single long recording was emitted in full regardless of
  `--train-hours`.
- Added a `kur` dispatcher so every command is reachable as `kur <command>` alongside its own
  `kur-<command>` script, with `kur --help`/`--list` and `kur completion bash|zsh|fish`.
  Subcommands contributed by installed plugin packages are discovered through their console
  script entry points, so no domain is named in the core dispatcher.
- Added `kur-infer`, `kur-init`, and `kur-preview`, closing the gap before a job spec exists.
  `kur-infer` emits the inferred schema (referenceable from a job by path); `kur-init`
  scaffolds a whole job and resolves it before writing, so what it emits is a job
  `kur-validate` accepts; `kur-preview` decodes a bounded prefix of the sources and reports
  field ranges, missingness, and example ticks.
- Added `kur-stats` and `kur-export` on the artifact side. `kur-stats` exposes the existing
  unified `stats()` report — including plugin-contributed sections — to the shell, and
  `kur-export` writes shards as npz/npy/parquet with an `export_index.json` preserving the
  manifest metadata that bare arrays lose.
- Added `kur-checkpoint list|verify|clear` for stage checkpoint directories. Verification reads
  committed metadata instead of reconstructing the run's `CheckpointIdentity`, so a damaged
  commit is detectable from the checkpoint tree alone; `clear` requires `--yes` and supports
  `--attempts-only`.
- Added `--review`, `--review-json`, and `--review-html` to `kur-curate`, surfacing the
  hours-budget admission review from the run's recorded decisions without a second scoring pass.
- Added `kur-import` and `import_dataset()` with confidence-based registered-adapter detection.
  NPZ, HDF5, and Zarr use lossless tensor hooks; CSV, JSONL, and single/partitioned Parquet use
  ordinary schema inference and canonical ticks. Plugin adapters can advertise detection and
  raw-array import hooks. Imports default to source-aligned shards and publish the same
  checksummed transactional artifacts consumed by inspection, analyzers, and loaders.
- Added artifact-level analyzer execution to `kur-check`, with separate JSON construction
  parameters and dataset options, composable quality-gate output, and exit-code-2 verdicts.
  Action-state consistency can now read configurable frame/action modalities from curated
  shards, while split leakage can audit assignments directly from a dataset manifest.
- Added unified lifecycle stats for pipelines, curated selections, write results, and existing
  dataset artifacts. Reports render directly in notebooks, can be saved as standalone HTML, and
  offer optional Matplotlib plots through the `stats` extra; artifact stats inspect the actual
  emitted, padding-trimmed arrays. Importing the gameplay domain now enriches the same report
  with video, action, and SLAM signals and a gameplay plot; synchronized video comparison,
  automatic Jupyter/terminal progress selection, and a model-free JupyterLab example complete
  the end-user workflow. Reports discover their available plots and evidence modes, choose an
  applicable domain plot automatically, and expose bounded selected/rejected/strongest/weakest
  examples; live gameplay reports add frame strips, actions, and SLAM diagnostics without
  placing media in serialized report data.
- Added `kuration.core` and `kuration.gameplay` domain surfaces. Pixel shift, action-state
  consistency, and SLAM now ship together in the SDK wheel instead of separate first-party
  plugin distributions; OpenCV, Torch, FAISS, and related runtimes remain extras.
- Physically moved the generic SDK implementation under `kuration.core` and made
  `kuration.gameplay` the only maintained domain package. Direct implementation imports such as
  `kuration.pipeline` now use `kuration.core.pipeline`; package-level imports such as
  `from kuration import Pipeline` remain available. Removed the former gameplay compatibility
  packages, and nested Depth Anything privately under gameplay SLAM.
- Integrated the native `kuration.gameplay.slam.SLAMReconstructor` and `slam_pipeline()`
  workflow for optional video/frame reconstruction,
  stable model/config provenance, resumable portable exports, trajectory quality and diversity
  curation, point-cloud reduction, preflight, and `kur-slam`. Precomputed manifests remain a
  lightweight NumPy-only SDK input; global model execution is intentionally separate from
  bounded compositional operators.

## 0.1.0rc4

- Added a discoverable `analyzer` registry kind for user-owned capabilities that aggregate
  across segments, with `register_analyzer()`, `get_analyzer()`, and CLI discovery. The
  action-state consistency plugin now registers its real clip-level analyzer instead of an
  unrelated stateless compliance component. A dependency-free `kuration/split-leakage`
  analyzer audits existing source and group assignments across dataset splits.
- Corrected pipeline and bounded execution around duplicate sources, exact vector widths,
  registered-metric transform dependencies, cross-kind feature collisions, scorer parameters and
  registry lookup, callable metric overlays, test-split validation, and checkpoint partition
  identity. The same validation now applies consistently at direct public execution boundaries.
- Artifact readers now verify recorded shard size and SHA-256 by default, including
  `iter_shards()`, `read_shard_arrays()`, `ShardDataset`, and curated-dataset adapters. Pass
  `verify=False` only for explicitly trusted artifacts. Commit sidecars, rebased legacy paths,
  split names, and shard paths are validated before reads, preventing traversal and stale or
  corrupted shard reuse.
- Made shard filenames and manifest entity IDs type-preserving so values such as integer `1` and
  string `"1"` cannot overwrite each other. Shard writing now rejects non-finite or extension
  object IDs that have no canonical value identity instead of relying on unsafe string or pickle
  representations.
- Fixed adapter and media correctness for large integer timestamps, missing categorical values,
  Parquet dataset glob preflight and source fingerprints, Arrow chunk/batch ingestion, duplicate
  columnar bin names, joined-media fingerprints, multi-video frame rate and shape validation, and
  finite ordered timestamp alignment.
- Tightened registry and plugin contracts: malformed targets and true no-op entry points fail
  transactionally with persistent diagnostics; pre-imported package-owned registrations remain
  valid; nested parameter schemas are checked at registration; and `enum`, `const`, and
  `uniqueItems` use JSON-type-sensitive equality rather than Python's boolean/integer aliasing.
- Corrected job-versus-artifact migration classification, strict-mode `kur-explain` policy,
  vector event-reduction preflight, nested non-finite review JSON conversion, group-report identity
  collisions, actionable video inference errors, and feature-schema drift detection.
- Hardened the SLAM plugin with exact timestamp coverage and explicit point-cloud format/version
  validation in both manifests and NPZ artifacts.
- Added focused regression suites for the comprehensive correctness review and its independent
  cross-subsystem follow-up, alongside full core, plugin, typing, lint, example, and distribution
  build verification.

## 0.1.0rc3

- Added transactional plugin loading: failed initializers roll back registry mutations, explicit
  modules fail fast, and broken ambient entry points no longer hide healthy plugins; ambient
  failures remain visible through discovery CLIs, `plugin_failures()`, and run provenance.
- Added `kur-plugin-check`, the public `check_plugins()` API, and a one-command isolated plugin
  validation workflow covering dependencies, tests, lint, wheels, entry points, and descriptors.
- Added typed `TransformContext` boundary accessors while retaining the rc2 batch-key
  compatibility form, plus deterministic lookbehind/lookahead and overlapping-window ownership.
- Bounded source scheduling, Ray submission, transform context memory, Hugging Face partition
  behavior, and the pixel-shift radial cache; kept Arrow publication on the caller thread and
  removed process-global reporting assumptions. Ray's `workers` now caps submitted tasks even
  when `max_in_flight` is larger.
- Hardened source-ID normalization, normalized timestamp grouping, count-budget tag coverage,
  remote replacement failure behavior, and unknown-total progress reporting. Direct low-level
  curation calls now require non-empty string source IDs; convert `pathlib.Path` values with `str`.
- Binning now fails explicitly when any candidate lacks a requested axis feature instead of
  silently treating the missing metric as zero.
- Added cross-mode determinism conformance tests spanning serial/parallel, streaming/materialized,
  bounded/reference, and optional Ray execution.
- Isolated every repository plugin in CI and added exact-CI-artifact plugin publication with
  metadata, hash, wheel-content, dependency, and installed-discovery checks.
- Corrected affected documentation claims and added regression coverage for realistic failure,
  boundary, overlap, retry, packaging, and discovery scenarios.
- Pixel-shift radial geometry is now computed and cached as float32 (rather than float64), so
  measurements may differ by normal float32 rounding; the two-entry cache deliberately bounds
  global memory and may recompute when a process cycles through more than two resolutions.

## 0.1.0rc2

- Added integrity-aware shard resume, structured execution diagnostics, Python 3.14 and
  minimum-dependency CI, coverage artifacts, and real optional-backend integration tests.
- Hardened transform dispatch so implementation `TypeError`s are not mistaken for signature
  mismatches, and fixed transform discovery through `kur-list`.
- Standardized CLI failures as concise one-line errors by default while retaining full
  tracebacks behind `--debug`.
- Hardened release automation with pinned build tools, artifact validation, CI-gated artifact
  promotion, platform and minimal-install smoke coverage, and trusted PyPI publishing.
- Added a runtime `kuration.__version__`, a security policy, and an explicit compatibility
  policy.
- Reworked the README and documentation around product outcomes and user journeys, added a
  searchable documentation site, editable diagrams, and compositional/transform-backed operator
  examples.
- Added support and governance guidance, contribution templates, documentation link checks, and
  deployment configuration for the published documentation site.
- Fixed bounded columnar execution against the declared PyArrow 15 minimum.

## 0.1.0rc1

- Initial release candidate for the schema-driven curation SDK, pipeline, job/CLI layer,
  versioned artifacts, extension registries, bounded execution, and optional integrations.
