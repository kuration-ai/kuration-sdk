# Releasing to PyPI

The CI workflow tests Python 3.11 through 3.14, declared minimum dependencies, core-only installs,
macOS and Windows smoke paths, and each optional integration at both current and minimum versions.
It also resolves the pinned minimum set and audits its complete dependency graph against the
Python vulnerability database. It builds the wheel and source distribution once with the pinned
tools in `.github/constraints/release.txt`, runs `twine check` and `check-wheel-contents`, installs
the wheel in a clean environment, and records SHA-256 hashes.

A published final GitHub release runs `.github/workflows/publish.yml`. That workflow resolves the
release tag to a commit, requires a successful `CI` workflow for that exact commit, downloads its
tested distribution artifact, verifies it again, and promotes those same bytes to PyPI with OpenID
Connect. It never rebuilds a release artifact. No long-lived PyPI API token is stored in GitHub.

## One-time setup

1. Confirm the public package name `kuration-sdk` in PyPI.
2. Add a PyPI trusted publisher for this GitHub repository, workflow `publish.yml`, and
   environment `pypi`.
3. Protect the GitHub `pypi` environment if releases require approval.
4. Enable GitHub Pages with **GitHub Actions** as its source; `.github/workflows/docs.yml` deploys
   the strict MkDocs build from `main`.
5. Resolve the pending corporate license and author metadata in `pyproject.toml`, and add the
   matching `LICENSE` file before the first public release.
6. Designate security and maintenance owners so the repository can publish a private contact,
   response targets, and a `CODEOWNERS` file.

## Release

1. Set `project.version` to a release candidate such as `0.1.0rc5` and run the complete test and
   static-check suite.
2. Open the release pull request and wait for the `Required CI` check. Do not publish a release
   from a commit whose CI run is pending, failed, or expired.
3. Tag the accepted commit and publish a GitHub prerelease to validate the release workflow without
   uploading to PyPI. Manual workflow runs also build and validate artifacts but never publish.
4. Once the candidate is accepted, set the final version, update the changelog, and wait for
   `Required CI` on that exact commit.
5. Tag that commit and publish the final GitHub release. The workflow promotes the CI-built wheel
   and sdist, publishes them to PyPI, and attaches the distributions plus `SHA256SUMS` to the
   GitHub release.
6. Immediately after the release, advance `project.version` to the next planned `.devN` identity
   before merging new Unreleased work. CI rejects a non-empty Unreleased section that still builds
   with an already released changelog version.

The publishing job runs only for a non-prerelease GitHub release and verifies that its tag matches
`project.version`; a PEP 440 pre-release version also fails if the GitHub release is accidentally
marked final. Release events fail closed until both approved `project.license` metadata and a root
`LICENSE` file exist; manual dry runs remain available while that corporate decision is pending. A
release created before exact-commit CI succeeds will fail safely; re-run the workflow after CI
completes. PyPI rejects a filename/version that was already uploaded, so a retry after a partial
release must use a new version if the original upload succeeded.

Documentation deployment is independent of package publication. Merges to `main` build the site
strictly and deploy the `main` documentation. Release tags remain the authoritative source for
version-specific documentation until a multi-version documentation policy is adopted.
