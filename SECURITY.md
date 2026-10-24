# Security policy

## Supported versions

Before the first public release, security fixes land on the current development branch; no
published package version is supported yet. After publication and until `1.0`, fixes are made on
the latest published `0.x` release only. Users should upgrade to that newest release before
validating a suspected issue.

## Reporting a vulnerability

Use GitHub's [private vulnerability reporting
form](https://github.com/kuration-ai/kuration-sdk/security/advisories/new). Please do not open a
public issue containing vulnerability details. Include the affected version, a minimal
reproduction, impact, and any suggested mitigation.

If GitHub's private form is unavailable, open a public issue that asks the maintainers to provide
a private reporting channel, but include no exploit, reproduction, sensitive log, credential, or
other vulnerability detail in that issue. Wait for a private channel before sharing the report.

The security owner is Nirmit Desai (`@nirmit-desai-aintropy`). Maintainers will triage reports,
coordinate remediation and disclosure, and credit reporters who want attribution. The private
GitHub form above is the supported reporting channel. No response-time SLA is offered at this
time; one will be published here if that changes.

## Trust boundaries

- Plugin modules, entry points, registered callbacks, and job `plugins` are executable Python
  code. Load only code you trust; kuration does not sandbox extensions.
- Job files and artifact manifests are data, but they may select plugins, paths, and cloud
  backends. Treat untrusted jobs as untrusted code until those references are reviewed.
- Cloud credentials are supplied by the runtime and are never meant to be embedded in jobs or
  artifacts. Scope credentials to the required storage paths.
- Shard checksums detect accidental or post-write modification; they are integrity metadata, not
  signatures or proof of publisher identity.
- Spill directories, checkpoints, and output directories may contain source-derived data. Apply
  the same access controls and retention policy as the source dataset.
