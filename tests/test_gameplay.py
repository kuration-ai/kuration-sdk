"""Packaging and import contracts for the grouped gameplay facade."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tomllib

import kuration as kur
from kuration import gameplay


ROOT = Path(__file__).parents[1]


def test_gameplay_facade_groups_the_public_capabilities():
    assert gameplay.action_state_checker is kur.action_state_checker
    assert gameplay.slam_pipeline is kur.slam_pipeline
    assert gameplay.PixelShiftReport is kur.pixel_shift.PixelShiftReport
    assert (
        gameplay.ActionStateConsistencyChecker
        is kur.action_state_consistency.ActionStateConsistencyChecker
    )
    assert gameplay.SLAMPipeline is kur.SLAMPipeline
    assert gameplay.input_slam is kur.input_slam


def test_gameplay_import_does_not_load_heavy_runtimes():
    code = """
import sys
from kuration import gameplay
assert gameplay.SLAMPipeline
assert gameplay.PixelShiftReport
assert gameplay.ActionStateConsistencyChecker
assert "torch" not in sys.modules
assert "cv2" not in sys.modules
assert "faiss" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_gameplay_extra_and_compatibility_alias_cover_the_domain_runtimes():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = config["project"]["optional-dependencies"]

    domain = set().union(
        *(extras[name] for name in ("pixel-shift", "action-state", "slam", "slam-viz"))
    )
    assert set(extras["gameplay"]) == domain
    assert extras["standard-operators"] == extras["gameplay"]


def test_all_extra_is_the_union_of_every_runtime_extra():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = config["project"]["optional-dependencies"]
    runtime_names = set(extras) - {"all", "dev", "docs", "standard-operators", "gameplay"}
    runtime_dependencies = set().union(*(extras[name] for name in runtime_names))

    assert set(extras["all"]) == runtime_dependencies
