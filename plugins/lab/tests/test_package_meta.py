"""Distribution invariants: version mirrors, extension hooks, import purity."""
from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def test_versions_stay_mirrored():
    """publish-plugin.yml reads pyproject statically; package.json ships in the bundle;
    __init__.__version__ is the API handshake — all three must agree."""
    pyproject = tomllib.loads((PLUGIN_ROOT / "pyproject.toml").read_text())
    package = json.loads((PLUGIN_ROOT / "package.json").read_text())
    import kuration_sdk_lab

    assert pyproject["project"]["version"] == package["version"] == kuration_sdk_lab.__version__


def test_no_kuration_plugins_entry_point():
    """kuration-sdk-lab is a UI distribution, not a pipeline component: it must NOT register in
    the kuration.plugins group (scripts/check_plugin.py deliberately skips it in CI)."""
    pyproject = tomllib.loads((PLUGIN_ROOT / "pyproject.toml").read_text())
    entry_points = pyproject["project"].get("entry-points", {})
    assert "kuration.plugins" not in entry_points


def test_extension_discovery_hooks():
    import kuration_sdk_lab

    points = kuration_sdk_lab._jupyter_server_extension_points()
    assert points[0]["module"] == "kuration_sdk_lab.app"
    paths = kuration_sdk_lab._jupyter_labextension_paths()
    assert paths[0]["dest"] == "kuration-sdk-lab"


def test_import_is_pure():
    """`import kuration_sdk_lab` must not drag in jupyter_server, tornado, or the SDK — the same
    import discipline the main package pins in tests/test_gameplay.py."""
    code = (
        "import sys\n"
        "import kuration_sdk_lab\n"
        "banned = {'jupyter_server', 'tornado', 'kuration', 'IPython', 'numpy'}\n"
        "loaded = banned & set(sys.modules)\n"
        "assert not loaded, f'import kuration_sdk_lab loaded {sorted(loaded)}'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=PLUGIN_ROOT)
