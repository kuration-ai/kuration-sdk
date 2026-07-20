# Kuration plugin template

Copy this directory to `plugins/<plugin-name>` and replace every `template` identifier. The
example registers one scalar window metric through a zero-argument entry-point initializer.

Before opening a pull request, test both direct registration and installed discovery:

```bash
python -m pip install -e . -e 'plugins/<plugin-name>[test]'
python -m pytest plugins/<plugin-name>/tests -q
kur-describe metric your-namespace/your-component --json
```
