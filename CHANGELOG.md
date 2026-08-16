# Changelog

All notable changes to Kuration AI live here. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [v0.0.1] — 2026-08-11

First release.

### Added

- Three built-in curation operations: `BalancedSelector` (activity balancing across a signal), `RareEventCoverage` (weighted sampling for rare events), `DiverseSubset` (cluster-based deduplication in feature space).
- Extension pattern via `register(name, cls)` for user-defined operations. Any class with `.apply(pipeline) -> pipeline` qualifies.
- `kai` CLI with `list` and `describe` subcommands, including runtime-extras annotation.
- Three call styles (direct apply, list of instances, config dict) with a config-parity guarantee enforced by test.
- Gallery of three self-contained examples under `examples/`, each with a runnable script and a README with the exact expected output.
- 52-test smoke suite covering happy paths, config parity, validation errors, config-resolution errors, CLI, and README examples.

### Notes

- `kuration-sdk` (the underlying pipeline engine) is not on PyPI yet, so `pip install kuration-ai` currently requires source access to `aintropy-ai/kuration-sdk`. See [CONTRIBUTING.md](./CONTRIBUTING.md) for the dev install.
- GitHub Actions CI (`.github/workflows/tests.yml`) requires the `KURATION_SDK_TOKEN` repo secret to be set to run.
