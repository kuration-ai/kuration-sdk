# kuration-sdk roadmap

This roadmap communicates direction, sequencing, and release gates. It is not a promise of dates.
Detailed behavior belongs in `docs/`; completed implementation history remains in `PLAN.md`.

Status labels:

- **Shipped** — implemented and covered by the current test/documentation contract.
- **Next** — required before or immediately after the first public SDK release.
- **Planned** — agreed direction with a design, but not yet scheduled for implementation.
- **Deferred** — useful work intentionally postponed until scale or demand justifies it.

## Shipped — first SDK foundation

- Schema-driven time-series ingestion and closed metric grammar.
- Immutable pipelines with schema inference, inspection, and loud validation.
- Count- and hours-based selection, diversity binning, filters, learned scorers, tags, splits,
  reports, and multi-arm experiments.
- Local, cloud, streaming, parallel, resumable, and deterministic execution paths.
- NPZ, JSON, HDF5, and Parquet outputs plus NumPy and PyTorch readers.
- Discoverable extension registries for adapters, metrics, scorers, encoders, methods, filters,
  containers, presets, and compositional operators.
- Versioned jobs/artifacts, migrations, CLIs, runnable examples, CI, wheel builds, and PyPI
  trusted-publishing automation.
- Python 3.11–3.14 and declared-minimum dependency testing, real optional integration tests,
  coverage artifacts, checksummed resume repair, and structured worker diagnostics.
- Release notes, a beta compatibility/deprecation policy, and a security reporting policy.
- Preferred pipeline names `.bin()` and `.sample()` with compatibility aliases retained.
- Transactional plugin discovery, installed-plugin contract checking, isolated repository-plugin
  validation, and independent exact-CI-artifact plugin releases.
- Typed transform boundary context, deterministic overlapping-window ownership, and execution
  conformance coverage across worker, streaming, bounded, and optional Ray modes.

## Next — public `0.1` release readiness

- Resolve corporate license and author metadata and add the matching `LICENSE` file.
- Confirm the public `kuration-sdk` project name and configure the PyPI trusted publisher.
- Perform the final release-candidate smoke test and tag the accepted build.
These items gate public publishing. Pipeline execution optimization does not gate `0.1`.

## Delivered — product usability and artifact handoff

- Conventional `src/kuration/` repository layout; the import package remains `kuration`.
- Output-aware dry runs with optional bounded source, candidate-window, and shard estimates.
- Post-run Python/CLI artifact inspection with full shard integrity verification.
- Actionable CLI remedies, execution-path parity coverage, and a pipeline-first documentation path.
- Bounded input previews, candidate-vs-selected review, artifact diffing, and dependency-free
  static HTML reports.
- Hours-budget admission review and reusable Python/CLI quality gates.
- A CLI covering the whole lifecycle, not just execution: schema inference, job scaffolding,
  and bounded previews before a run; distribution stats, array export, and checkpoint
  maintenance after one; all under a single `kur` dispatcher with shell completion, which
  discovers plugin-contributed commands through their entry points.

## Delivered — synchronized video ingestion

- Let `input("*.mp4")` create a video-only pipeline whose `frame` media column is semantically
  typed for compatible video transforms.
- Pair each ordinary data source with an MP4 by frame index or presentation timestamp.
- Add schema-level media metadata while representing frame values as lazy auxiliary references,
  outside ordinary metric buffers.
- Decode only selected or explicitly transformed frames in bounded batches.
- Emit raw `uint8` frame tensors as their own output modality/column.
- Add strict alignment validation and streaming behavior.

The first decoder is an optional `[video]` module. Installing the core SDK must not install a
video or GPU framework.

## Delivered — optional device-aware transforms

- Add a dependency-free batched-transform protocol and registry to the core.
- Allow registered transforms to derive scalar selection features from fields or media batches.
- Support explicit CPU/GPU resources and bounded batches with resolved calls in pipeline
  provenance.
- Keep Torch, JAX, CUDA, model libraries, and hardware codecs in optional plugin packages.
- Fail with an actionable installation hint when a requested implementation is absent; never
  silently change model, device, dtype, or precision.

GPU is an execution resource, not a required SDK dependency or a separate selection semantic.

## Delivered — optional Ray execution backend

- Extract the current process-pool work behind a backend-neutral ordered executor.
- Add an optional `[ray]` task backend for source work while transforms retain bounded local
  worker semantics.
- Preserve canonical ordering, error semantics, progress, and provenance across backends.

## Deferred — scale and optimization work

- Total-segment-pool spill or compact external feature tables for pools larger than driver RAM.
- Count-path resume.
- Decode-once caches shared by visual transforms and selected output.
- Hardware video decoding, dynamic batching, and prefetching after representative benchmarks.
- Ray Data-native video ingestion after its API and performance justify making it a backend
  implementation detail.

These optimizations should be driven by measured bottlenecks and must preserve the simple local
path.

## Toward `1.0`

A stable `1.0` should require:

- a documented and tested public Python, CLI, job, and artifact compatibility policy;
- a deprecation window and migrations for renamed or serialized interfaces;
- stable extension descriptors and plugin-loading behavior;
- reproducibility and deterministic-selection guarantees across supported executors;
- security, support, and release processes suitable for public use;
- evidence from multiple real datasets and at least one external extension package.

Roadmap proposals should include motivation, contract impact, optional dependencies, compute and
memory implications, reproducibility risks, tests, documentation, and migration needs.
