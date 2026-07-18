# Security policy

## Supported versions

Until the first stable release, security fixes are made on the latest published `0.x` release
only. Users should upgrade to the newest release before reporting or validating a suspected
issue.

## Reporting a vulnerability

Use GitHub's **Report a vulnerability** form in the repository Security tab. Please do not open
a public issue for an undisclosed vulnerability. Include the affected version, a minimal
reproduction, impact, and any suggested mitigation. If private vulnerability reporting is not
available, contact the repository owner privately through their GitHub profile.

The maintainer will acknowledge a report, assess impact, coordinate a fix and disclosure, and
credit reporters who want attribution. No response-time SLA is promised while the project is
maintained by one person.

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
