# Changelog

This project follows semantic versioning after `1.0`. During the `0.x` beta, minor releases may
change documented APIs with the notice and deprecation process in `docs/compatibility.md`.

## Unreleased

No changes yet.

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
