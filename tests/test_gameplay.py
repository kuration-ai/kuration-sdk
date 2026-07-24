"""Packaging and import contracts for the gameplay operator domain."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tomllib

from kuration import core, gameplay, input
from kuration.core.pipeline import Pipeline


ROOT = Path(__file__).parents[1]


def test_gameplay_domain_groups_the_public_capabilities():
    assert gameplay.PixelShiftReport is gameplay.pixel_shift.PixelShiftReport
    assert (
        gameplay.ActionStateConsistencyChecker
        is gameplay.action_state_consistency.ActionStateConsistencyChecker
    )
    assert gameplay.SLAMReconstructor is gameplay.slam.SLAMReconstructor
    assert gameplay.input_slam is gameplay.slam.input_slam
    assert core.Pipeline is Pipeline


def test_gameplay_operators_extend_the_plain_core_pipeline():
    pipeline = input(
        [{"timestamp": 0.0, "signal": 1.0}, {"timestamp": 1.0, "signal": 2.0}],
        time="timestamp",
    )
    derived = pipeline.window(1).measure("signal")

    assert type(derived) is Pipeline
    assert callable(derived.pixel_shift)
    assert callable(derived.slam_motion_diversity)
    assert callable(derived.slam_quality)
    assert not hasattr(gameplay, "GameplayPipeline")
    assert not hasattr(gameplay, "as_gameplay")
    assert not hasattr(gameplay, "input")


def test_action_state_convenience_uses_the_registered_analyzer():
    baselines = {
        "W": {"1": {"expected_sign": 1, "min": 0.0, "max": 1.0}},
    }

    checker = gameplay.action_state_checker(baselines=baselines)

    assert isinstance(checker, gameplay.ActionStateConsistencyChecker)
    assert checker.baselines == baselines


def test_gameplay_import_does_not_load_heavy_runtimes():
    code = """
import sys
from kuration import gameplay
assert gameplay.SLAMReconstructor
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


def test_minimum_constraints_cover_every_slam_runtime_dependency():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = config["project"]["optional-dependencies"]
    requirements = [*extras["slam"], *extras["slam-viz"]]
    constrained = {
        line.split("==", 1)[0].strip().lower().replace("_", "-"): line.split("==", 1)[1]
        for line in (ROOT / ".github/constraints/minimum.txt").read_text().splitlines()
        if line and not line.startswith("#") and "==" in line
    }

    def numeric_version(value: str) -> tuple[int, ...]:
        return tuple(int(part) for part in value.split("."))

    for requirement in requirements:
        name, minimum = requirement.split(">=", 1)
        normalized_name = name.strip().lower().replace("_", "-")
        assert normalized_name in constrained
        assert numeric_version(constrained[normalized_name]) >= numeric_version(minimum)


def test_all_extra_is_the_union_of_every_runtime_extra():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = config["project"]["optional-dependencies"]
    runtime_names = set(extras) - {"all", "dev", "docs", "standard-operators", "gameplay"}
    runtime_dependencies = set().union(*(extras[name] for name in runtime_names))

    assert set(extras["all"]) == runtime_dependencies
