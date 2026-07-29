# Kuration plugin template

Copy this directory to `plugins/<plugin-name>` and replace every `template` identifier. The
example registers a scalar window metric and a directly callable pipeline operator through a
zero-argument entry-point initializer. Once installed, use it like any built-in:

```python
pipeline = pipeline.balanced_window_lengths(bins=4)
```

Before opening a pull request, test both direct registration and installed discovery:

```bash
python scripts/check_plugin.py plugins/<plugin-name>
kur-describe metric your-namespace/your-component --json
kur-plugin-check your-entry-point --json
```

Every descriptor should include the distribution's `package`, an actionable `install_hint`, a
closed parameter schema, version, and stability. Repository releases use the tag
`plugin-<directory>-v<project.version>` after the plugin passes CI in its isolated environment.
