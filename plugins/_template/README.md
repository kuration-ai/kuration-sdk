# Kuration plugin template

Copy this directory to `plugins/<plugin-name>` and replace every `template` identifier. The
example registers one scalar window metric through a zero-argument entry-point initializer.

Before opening a pull request, test both direct registration and installed discovery:

```bash
python scripts/check_plugin.py plugins/<plugin-name>
kur-describe metric your-namespace/your-component --json
kur-plugin-check your-entry-point --json
```

Every descriptor should include the distribution's `package`, an actionable `install_hint`, a
closed parameter schema, version, and stability. Repository releases use the tag
`plugin-<directory>-v<project.version>` after the plugin passes CI in its isolated environment.
