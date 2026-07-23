"""Packaging and import contracts for the grouped gameplay facade."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tomllib

from kuration import core, gameplay
from kuration.pipeline import Pipeline


ROOT = Path(__file__).parents[1]


def test_gameplay_facade_groups_the_public_capabilities():
    assert gameplay.PixelShiftReport is gameplay.pixel_shift.PixelShiftReport
    assert (
        gameplay.ActionStateConsistencyChecker
        is gameplay.action_state_consistency.ActionStateConsistencyChecker
    )
    assert gameplay.SLAMPipeline is gameplay.slam.SLAMPipeline
    assert gameplay.input_slam is gameplay.slam.input_slam
    assert core.Pipeline is Pipeline


def test_gameplay_pipeline_preserves_domain_methods_across_core_operations():
    pipeline = gameplay.input(
        [{"timestamp": 0.0, "signal": 1.0}, {"timestamp": 1.0, "signal": 2.0}],
        time="timestamp",
    )
    derived = pipeline.window(1).measure("signal")

    assert isinstance(derived, gameplay.GameplayPipeline)
    assert callable(derived.pixel_shift)
    assert callable(derived.slam_motion)


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


def test_core_import_does_not_import_or_register_gameplay():
    code = """
import sys
import kuration
assert "kuration.gameplay" not in sys.modules
assert "aintropy/pixel-shift" not in kuration.default_registry.names("operator")
assert "aintropy/slam-quality" not in kuration.default_registry.names("operator")
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_core_plugin_discovery_can_load_bundled_gameplay():
    code = """
from kuration import default_registry, load_plugins
assert "aintropy/pixel-shift" not in default_registry.names("operator")
load_plugins()
assert "aintropy/pixel-shift" in default_registry.names("operator")
assert "aintropy/action-state-consistency" in default_registry.names("analyzer")
assert "aintropy/slam-motion-diversity" in default_registry.names("operator")
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
