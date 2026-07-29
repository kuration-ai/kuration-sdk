"""Discovery and migration commands share the public CLI error contract."""
from __future__ import annotations

import subprocess
import sys


def _run(function: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", f"from kuration.core.cli.{function.rsplit('.', 1)[0]} "
         f"import {function.rsplit('.', 1)[1]}; {function.rsplit('.', 1)[1]}()", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_discovery_cli_accepts_transform_kind():
    result = _run("discovery.list_main", "transforms", "--json")
    assert result.returncode == 0, result.stderr
    assert result.stdout.lstrip().startswith("[")


def test_describe_failure_is_one_line_unless_debug_requested():
    result = _run("discovery.describe_main", "transform", "does-not-exist")
    assert result.returncode == 1
    assert result.stderr.startswith("error: ValueError:")
    assert "Traceback" not in result.stderr

    debug = _run("discovery.describe_main", "transform", "does-not-exist", "--debug")
    assert debug.returncode == 1
    assert "Traceback" in debug.stderr


def test_migrate_failure_is_one_line_unless_debug_requested(tmp_path):
    missing = str(tmp_path / "missing.json")
    result = _run("migrate.main", missing)
    assert result.returncode == 1
    assert result.stderr.startswith("error: FileNotFoundError:")
    assert "Traceback" not in result.stderr

    debug = _run("migrate.main", missing, "--debug")
    assert debug.returncode == 1
    assert "Traceback" in debug.stderr
