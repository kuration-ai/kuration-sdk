# Governance

## Maintainers

Repository maintainers merge changes, manage releases, and steward the documented compatibility
boundary. Maintainer access is granted or removed by the sponsoring organization through its
normal repository-administration process. A named ownership roster and CODEOWNERS team should be
added once the organization designates those roles.

## Decisions

Ordinary changes are proposed through pull requests and accepted after tests, documentation, and
compatibility impact have been reviewed. Significant public-API, artifact-format, security,
dependency, or roadmap changes should begin with an issue or design proposal so tradeoffs are
recorded before implementation.

Maintainers seek technical consensus. When consensus is not possible, the maintainers responsible
for the affected area decide and record the rationale in the issue or pull request. Decisions must
follow the compatibility policy, security policy, and published deprecation process.

## Releases and security

Only maintainers authorized for the protected PyPI environment may publish. Releases use the
documented CI and trusted-publishing workflow; contributors should never upload artifacts directly.
Undisclosed vulnerability handling stays private under [SECURITY.md](https://aintropy-solutions.github.io/kuration-sdk/security/).

## Changes to governance

Governance changes use the same public pull-request process. They should identify affected roles,
transition steps, and any new private contact or approval requirements without exposing sensitive
information.
