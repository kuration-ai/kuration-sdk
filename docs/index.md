# Kuration documentation

Kuration turns sequential source data into selected, inspectable, training-ready dataset
artifacts. This documentation is organized around what you are trying to accomplish rather than
the package's module layout.

## Start here

| If you are… | Read |
|---|---|
| Evaluating kuration | [Core concepts and product model](overview.md) |
| Building a first dataset | [Guided quickstart](quickstart.md) |
| Looking for a specific operation | [Pipeline guide](pipeline.md) |
| Integrating an internal workflow | [Task cookbook](usage.md) |

## Build datasets

- [Schemas and semantics](schema.md) — declare fields, dimensions, media, rates, and meaning.
- [Adapters and inputs](adapters.md) — CSV, JSONL, Parquet, arrays, Hugging Face, video, and
  custom sources.
- [Metrics and the closed DSL](dsl.md) — derive validated per-window signals.
- [Selection strategies](strategies.md) — filter, rank, bin, cover, split, and budget.
- [Outputs and artifact integrity](output.md) — modalities, containers, manifests, checksums,
  resume, and reading data back.

## Extend kuration

- [Building operators](operators.md) — package a reusable curation decision, connect bounded
  computation, test its contract, and distribute it as a plugin.
- [Customization and discovery](customization.md) — registries, descriptors, metrics, scorers,
  analyzers, adapters, encoders, methods, filters, containers, and presets.
- [Architecture](architecture.md) — internal layers, execution boundaries, and design decisions.

## Operate and audit

- [Jobs and CLI reference](jobs.md) — validate, explain, run, inspect, compare, and gate JSON jobs.
- [Cookbook: scale and reliability](usage.md#11-scale-robustness) — streaming, parallelism,
  bounded checkpoints, Ray, cloud paths, failure behavior, and resume.
- [Compatibility policy](compatibility.md) — supported surfaces, beta evolution, migrations, and
  deprecation.
- [Release notes](releases.md) — what changed in the current release candidate and any migration
  considerations.
- [Security policy](security.md) — private reporting and runtime trust boundaries.
- [Releasing](releasing.md) — build, smoke-test, tag, and trusted-publishing process.

## Reference

- [Python API](api.md) — supported public names and exact signatures.
- [Pipeline operators](pipeline.md#operator-reference) — method forms and validation behavior.
- [Job specification](jobs.md) — declarative keys and command behavior.
- [Documentation contract](documentation-contract.md) — audience and public-boundary rules.

## Runnable examples

The [`examples/`](https://github.com/aintropy-solutions/kuration-sdk/tree/main/examples) directory contains download-free demos plus bounded
public-data workflows. Start with:

```bash
python examples/quickstart.py --out quickstart-out
python examples/05_custom_operators.py
python examples/06_gameplay_operator_methods.py
```

Every example used as a primary documentation path is exercised by CI.
