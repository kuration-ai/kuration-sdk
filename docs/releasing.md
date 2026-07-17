# Releasing to PyPI

The CI workflow builds a wheel and source distribution on every supported Python version, then
installs the wheel in a clean environment and verifies imports, package data, typing metadata,
and every `kur-*` entry point. Separate jobs smoke-test each optional integration not installed
by the main test matrix. A published GitHub release runs
`.github/workflows/publish.yml`, which builds both artifacts and publishes them to PyPI with
OpenID Connect; no long-lived API token is stored in GitHub.

## One-time setup

1. Confirm the public package name `kuration-sdk` in PyPI.
2. Add a PyPI trusted publisher for this GitHub repository, workflow `publish.yml`, and
   environment `pypi`.
3. Protect the GitHub `pypi` environment if releases require approval.
4. Resolve the pending corporate license and author metadata in `pyproject.toml`, and add the
   matching `LICENSE` file before the first public release.

## Release

1. Set `project.version` to a release candidate such as `0.1.0rc1` and run the complete test and
   static-check suite.
2. Build the wheel and sdist, install the wheel in a clean environment outside the checkout, and
   smoke-test imports, package data, and every `kur-*` command.
3. Tag the release candidate and validate it through CI without publishing a final release.
4. Once the candidate is accepted, set the final version, repeat the build and clean smoke test,
   and tag that exact version.
5. Publish the final GitHub release. Publishing the release triggers the PyPI workflow.

The workflow can also be started manually after the trusted publisher is configured. PyPI will
reject a filename/version that has already been uploaded, so every retry after a partial release
must use the same still-unpublished artifacts or a new version.
