"""Runs the README's `Write your own operation` example verbatim.

If this test breaks, the README's claim that the extension seam works is
broken: GTM tenet 6 (every claim executed) demands we catch it in CI.
"""
from dataclasses import dataclass

from kuration import input
from kuration_sdk_wrapper import BalancedSelector, run


@dataclass(frozen=True)
class OnlyActiveWindows:
    threshold: float = 0.1

    def apply(self, pipeline):
        # Uses a distinct feature name so BalancedSelector's default 'activity'
        # column doesn't collide: SDK measure names must be unique per pipeline.
        return (pipeline
                .measure("signal_activity", "std", of="signal")
                .filter("signal_activity", ">", self.threshold))


def _base_pipeline():
    sessions = {
        f"session-{i}": [
            {"t": float(step), "signal": float(i * step % 7)}
            for step in range(200)
        ]
        for i in range(3)
    }
    return input(sessions, time="t", hz=10.0).window(8, stride=4)


class TestReadmeExtensionSeamExample:
    def test_user_op_and_builtin_compose_in_run(self):
        pipeline = _base_pipeline()
        result = run(pipeline, [
            OnlyActiveWindows(threshold=0.15),
            BalancedSelector(feature="activity", of="signal", bins=4, count=20),
        ]).options(mode="strict", seed=1).collect()
        assert len(result.selected) > 0
        assert len(result.selected) <= 20

    def test_hero_snippet_still_prints_expected_count(self):
        # Mirrors the README hero block exactly.
        sessions = {
            "device-a": [{"t": float(i), "signal": float(i * 3 % 7)} for i in range(100)]
        }
        pipeline = input(sessions, time="t", hz=10.0).window(8, stride=4)
        result = (BalancedSelector(feature="activity", of="signal", bins=4, count=20)
                  .apply(pipeline).options(seed=1).collect())
        assert len(result.selected) == 20
