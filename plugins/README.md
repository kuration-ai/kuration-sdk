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
python -m pip install -e . -e 'plugins/<plugin-name>[test]'
python -m pytest plugins/<plugin-name>/tests -q
```

CI discovers every child containing `pyproject.toml`, installs its `test` extra, and runs its
tests, Ruff, build, and wheel-content checks. The shared SDK suite separately verifies discovery,
worker reconstruction, provenance, registry isolation, and bounded transform execution.
