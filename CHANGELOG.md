# Changelog

This project follows semantic versioning after `1.0`. During the `0.x` beta, minor releases may
change documented APIs with the notice and deprecation process in `docs/compatibility.md`.

## Unreleased

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

- Initial release candidate for the schema-driven curation SDK, fluent pipeline, job/CLI layer,
  versioned artifacts, extension registries, bounded execution, and optional integrations.
