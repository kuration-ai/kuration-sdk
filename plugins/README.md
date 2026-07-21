# Kuration plugins

This directory is the repository home for domain-specific extensions. Each child directory is an
independent Python distribution: it owns its dependencies, release metadata, entry point, tests,
and documentation. Plugin packages are intentionally not included in the `kuration-sdk` wheel.

Start by copying [`_template`](./_template/), then rename the distribution, import package,
entry-point key, and registered component namespace. Keep the following contract:

```text
plugins/<plugin-name>/
├── pyproject.toml
├── README.md
├── src/<import_package>/
│   └── __init__.py
└── tests/
    └── test_plugin.py
```

The plugin's `pyproject.toml` must expose a module or zero-argument initializer:

```toml
[project.entry-points."kuration.plugins"]
my_plugin = "my_plugin:register"
```

Registration imports must stay lightweight. Import optional frameworks inside factories or
execution methods and include an actionable installation hint in every component descriptor.
Use namespaced component names, declare a closed parameter schema, and ensure execution is
deterministic and free of process-global result state.

From the repository root, install and test one plugin with:

```bash
python scripts/check_plugin.py plugins/<plugin-name>
```

The script and CI discover every child containing `pyproject.toml`, create a fresh virtual
environment, install only core plus that plugin, run `pip check`, tests, Ruff, build,
wheel-content checks, reinstall the wheel, and run `kur-plugin-check` against every declared entry
point. The shared SDK suite separately verifies discovery, worker reconstruction, provenance,
registry isolation, bounded transform execution, and cross-mode determinism.

Repository-maintained plugins release independently from core. Use a tag of the form
`plugin-<directory>-v<project.version>` (for example `plugin-pixel_shift-v0.1.0`) and publish a
GitHub release. `.github/workflows/publish-plugin.yml` validates, smoke-tests, and publishes that
plugin's wheel and sdist through the `pypi-plugins` trusted-publisher environment. A manual run is
a non-publishing dry run. License metadata and the repository `LICENSE` remain release gates.
