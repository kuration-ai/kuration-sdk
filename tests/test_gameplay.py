"""Packaging and import contracts for the gameplay operator domain."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from kuration import Registry, core, gameplay, input
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
    assert callable(derived.action_state_consistency)
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


def test_gameplay_registration_is_idempotent_but_rejects_name_hijacking():
    from kuration.core._registration import needs_registration

    registry = Registry()

    def expected():
        return None

    def conflicting():
        return None

    assert needs_registration(registry, "operator", "demo/gameplay", expected) is True
    registry.register_operator("demo/gameplay", expected, package="expected")
    assert needs_registration(registry, "operator", "demo/gameplay", expected) is False
    with pytest.raises(ValueError, match="already owned"):
        needs_registration(registry, "operator", "demo/gameplay", conflicting)


def test_video_contact_sheet_never_labels_a_frame_index_as_seconds():
    """Regression test for a review-caught finding: with neither a duration nor an average
    rate, `plot()` fell back to the frame index for its reference value but still rendered
    it as "{:.2f}s" -- labelling an index as seconds. A frame index is now labelled as one,
    while a real timebase still renders seconds."""
    from kuration import FrameRef
    from kuration.gameplay.video import _frame_caption

    ref = FrameRef("clip.mp4", "clip.mp4", 3, 3.0)
    assert _frame_caption(ref, timed=False) == "frame 3"
    assert _frame_caption(ref, timed=True) == "3.00s"


def test_video_contact_sheet_labels_index_fallback_as_a_frame_number(tmp_path):
    """The same finding through the real contact sheet: an untimed provider's titles carry
    frame numbers, a provider with a duration carries seconds."""
    import numpy as np
    import pytest

    pytest.importorskip("matplotlib")
    from kuration import FrameRef, VideoMetadata

    class UntimedProvider:
        capabilities = {"streaming": True, "variable_frame_rate": True}

        def __init__(self, *, duration=None, average_rate=None):
            self.duration = duration
            self.average_rate = average_rate

        def probe(self, uri):
            return VideoMetadata(uri, 4, 4, 4, self.duration, self.average_rate)

        def iter_refs(self, uri, *, source_id=None):
            for index in range(4):
                yield FrameRef(uri, source_id or uri, index, float(index))

        def decode_batch(self, refs):
            return np.stack([np.full((4, 4, 3), ref.frame_index, np.uint8) for ref in refs])

    left = tmp_path / "left.mp4"
    right = tmp_path / "right.mp4"
    left.write_bytes(b"left")
    right.write_bytes(b"right")

    untimed = gameplay.compare_videos(left, right).plot(
        frames=2, provider=UntimedProvider())
    titles = [axis.get_title() for axis in untimed.axes]
    assert all("s" not in title.split("\n")[1] for title in titles)
    assert {title.split("\n")[1] for title in titles} == {"frame 0", "frame 3"}

    timed = gameplay.compare_videos(left, right).plot(
        frames=2, provider=UntimedProvider(duration=1.5))
    assert {axis.get_title().split("\n")[1] for axis in timed.axes} == {"0.00s", "1.50s"}


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


def test_slam_star_import_stays_torch_free_and_dir_lists_lazy_runtime_names():
    code = """
import sys
namespace = {}
exec("from kuration.gameplay.slam import *", namespace)
assert "SLAMReconstructor" in namespace
assert "SLAMOperator" not in namespace
import kuration.gameplay.slam as slam
assert {"CameraPose", "SLAMOperator", "SLAMOutput"} <= set(dir(slam))
assert "torch" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_core_import_does_not_import_or_register_gameplay():
    code = """
import sys
import kuration
assert "kuration.gameplay" not in sys.modules
assert "aintropy/pixel-shift" not in kuration.default_registry.names("operator")
assert "aintropy/action-state-consistency" not in kuration.default_registry.names("operator")
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
assert "aintropy/action-state-consistency" in default_registry.names("operator")
assert "aintropy/slam-motion-diversity" in default_registry.names("operator")
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_gameplay_extra_and_compatibility_alias_cover_the_domain_runtimes():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = config["project"]["optional-dependencies"]

    domain = set().union(
        *(extras[name] for name in ("pixel-shift", "action-state", "collision", "slam",
                                    "slam-viz"))
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
