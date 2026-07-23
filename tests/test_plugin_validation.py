from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from kuration import check_plugins, default_registry
from kuration.core.plugin_validation import PluginCheckResult


def _plugin_module(tmp_path: Path, name: str, *, complete: bool) -> str:
    metadata = (
        "description='A checked test metric', "
        "parameters={'type': 'object', 'properties': {}, 'additionalProperties': False}, "
        "package='kuration-plugin-check-test', "
        "install_hint='pip install kuration-plugin-check-test', version='1'"
        if complete else (
            "description='', parameters={'type': 'object', 'properties': {}, "
            "'additionalProperties': True}")
    )
    (tmp_path / f"{name}.py").write_text(
        "from kuration import default_registry\n"
        "def metric(window): return 1.0\n"
        f"default_registry.register_metric({'acme/checked' if complete else 'unchecked'!r}, "
        f"metric, {metadata})\n"
    )
    return name


def test_plugin_checker_validates_and_transactionally_restores_module(tmp_path, monkeypatch):
    module = _plugin_module(tmp_path, "valid_check_plugin", complete=True)
    monkeypatch.syspath_prepend(str(tmp_path))

    [result] = check_plugins(module)

    assert result.passed
    assert [(item["kind"], item["name"]) for item in result.descriptors] == [
        ("metric", "acme/checked")]
    assert "acme/checked" not in default_registry.names("metric")
    assert module not in sys.modules


def test_plugin_checker_accepts_a_discoverable_direct_api_analyzer(tmp_path, monkeypatch):
    module = "valid_analyzer_plugin"
    (tmp_path / f"{module}.py").write_text(
        "from kuration import register_analyzer\n"
        "class Analyzer:\n"
        "    def summary(self): return {}\n"
        "def build(): return Analyzer()\n"
        "register_analyzer('acme/stateful-check', build, "
        "description='Cross-segment check', "
        "parameters={'type': 'object', 'properties': {}, 'additionalProperties': False}, "
        "package='kuration-plugin-check-test', "
        "install_hint='pip install kuration-plugin-check-test', version='1')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    [result] = check_plugins(module)

    assert result.passed
    assert [(item["kind"], item["name"]) for item in result.descriptors] == [
        ("analyzer", "acme/stateful-check")
    ]
    assert "acme/stateful-check" not in default_registry.names("analyzer")


def test_plugin_checker_reports_actionable_descriptor_contract_errors(tmp_path, monkeypatch):
    module = _plugin_module(tmp_path, "invalid_check_plugin", complete=False)
    monkeypatch.syspath_prepend(str(tmp_path))

    [result] = check_plugins(module)

    assert not result.passed
    assert any("namespaced" in error for error in result.errors)
    assert any("description" in error for error in result.errors)
    assert any("package metadata" in error for error in result.errors)
    assert any("install_hint" in error for error in result.errors)
    assert any("component version" in error for error in result.errors)
    assert any("additionalProperties=false" in error for error in result.errors)
    assert "unchecked" not in default_registry.names("metric")


def test_concurrent_plugin_checks_are_isolated(tmp_path, monkeypatch):
    modules = []
    for index in range(2):
        name = f"concurrent_check_plugin_{index}"
        module = _plugin_module(tmp_path, name, complete=True)
        path = tmp_path / f"{name}.py"
        path.write_text(path.read_text() + "import time\ntime.sleep(0.05)\n")
        modules.append(module)
    monkeypatch.syspath_prepend(str(tmp_path))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(check_plugins, modules))

    assert all(result[0].passed for result in results)
    assert "acme/checked" not in default_registry.names("metric")


def test_plugin_check_result_is_json_safe_and_human_readable():
    result = PluginCheckResult(
        "demo", "demo:register", ({"kind": "metric", "name": "demo/metric"},),
        warnings=("consider docs",))
    assert json.loads(json.dumps(result.to_dict()))["passed"] is True
    assert result.render().startswith("PASS demo (1 component(s))")
    assert "warning: consider docs" in result.render()


def test_plugin_check_cli_reports_explicit_module(tmp_path):
    module = _plugin_module(tmp_path, "cli_check_plugin", complete=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), environment.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-m", "kuration.core.cli.plugin_check", module, "--json"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload[0]["passed"] is True
    assert payload[0]["descriptors"][0]["name"] == "acme/checked"
