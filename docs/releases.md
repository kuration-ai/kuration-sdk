# Release notes

These notes highlight user-visible changes in the current release candidate. The repository
[changelog](changelog.md) is the authoritative
history for every release.

## Current `0.1.0rc5` candidate (`0.1.0rc5.dev0` development build)

The changes on this branch target `0.1.0rc5`. The candidate:

- separates the dataset-agnostic implementation into `kuration.core` and ships pixel shift,
  action-state consistency, and SLAM together under the maintained `kuration.gameplay` domain;
- adds direct operator methods, lifecycle statistics and evidence, synchronized video
  review, smart dataset import, artifact export, checkpoint maintenance, and the unified `kur`
  command;
- makes analyzers first-class pipeline terminals, including bounded tick-stream execution and
  artifact-backed checks;
- integrates native SLAM reconstruction, portable exports, trajectory quality/diversity
  curation, point-cloud reduction, preflight, and `kur-slam`; and
- strengthens fail-closed publication and reuse, exact cross-executor selection semantics,
  pinned artifact reads, source-level error handling, and boundary validation.

Dataset writes are no-clobber by default, persisted reuse requires a strong source identity,
and unsupported future artifact or job versions fail loudly. See the
[0.1.0rc5 changelog](changelog.md#010rc5)
for the complete evolving list.

## Prior `0.1.0rc4` candidate

The `0.1.0rc4` candidate focused on correctness, artifact integrity, and explicit extension
contracts. It was not tagged or published to PyPI from this repository.

### New analyzer capabilities

- A discoverable `analyzer` registry kind now supports user-owned capabilities that aggregate
  evidence across multiple segments. Register and resolve analyzers with `register_analyzer()`
  and `get_analyzer()`, or inspect them with `kur-list analyzers` and
  `kur-describe analyzer ...`.
- The gameplay domain's action-state consistency capability now exposes its clip-level checker
  as an analyzer.
- The dependency-free `kuration/split-leakage` analyzer audits existing source and group
  assignments across dataset splits.

### Correctness and validation

- Pipeline and bounded execution now reject duplicate sources, feature-name collisions, invalid
  vector widths, scorer-parameter mismatches, and test-split errors consistently at public
  execution boundaries.
- Adapter and media handling is stricter for integer timestamps, missing categorical values,
  Parquet inputs, Arrow batches, joined media, multi-video metadata, and timestamp alignment.
- Plugin loading is transactional: malformed targets and true no-op entry points remain visible
  as failures, while valid registrations from pre-imported packages continue to work.
- JSON Schema `enum`, `const`, and `uniqueItems` validation now distinguishes JSON types, so
  booleans no longer alias integers.

### Artifact integrity and identity

- `iter_shards()`, `read_shard_arrays()`, `ShardDataset`, and curated-dataset adapters verify
  recorded shard sizes and SHA-256 digests by default. Set `verify=False` only for artifacts you
  explicitly trust.
- Manifest paths, commit sidecars, split names, and rebased legacy paths are validated before
  reads, preventing path traversal and reuse of stale or corrupted shards.
- Shard filenames and manifest entity IDs preserve value types. Values such as integer `1` and
  string `"1"` no longer collide, and unsupported extension objects fail before writing.

### Built-in SLAM

- Exported timestamp maps must cover every trajectory exactly; each timestamp array must match
  the trajectory length, contain finite values, and increase strictly.
- Point-cloud manifests and NPZ files now require supported Kuration format and version markers.

### Upgrade notes

- Artifact verification is now enabled by default. Callers that intentionally skip integrity
  checks must opt out explicitly with `verify=False`.
- Custom entity-ID objects without a canonical value representation are no longer writable. Use
  `None`, booleans, integers, finite floats, strings, bytes, tuples, frozensets, or compatible
  NumPy scalars.
- A plugin entry point must register at least one owned component; a successful import with no
  registration is reported as a plugin failure.
- Action-state consistency now ships as a built-in analyzer using the analyzer registry API.

See the [compatibility policy](compatibility.md) for beta-version guarantees and the
[releasing guide](releasing.md) for the maintainer workflow.
