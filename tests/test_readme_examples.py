"""Execute the README's selection and SLAM code samples, and the SLAM sample in docs/domains.md,
verbatim on synthetic data, so a sample that stops matching the API fails here."""
from __future__ import annotations

import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]


def _python_blocks(path: str) -> list[str]:
    text = (ROOT / path).read_text(encoding="utf-8")
    return [m.group(1) for m in re.finditer(r"^```python\n(.*?)^```", text, re.S | re.M)]


def _block(path: str, marker: str) -> str:
    matches = [b for b in _python_blocks(path) if marker in b]
    assert len(matches) == 1, f"{path}: expected exactly one python block containing {marker!r}"
    return matches[0]


def _selection_pipeline():
    from kuration import input

    rng = np.random.default_rng(0)

    def rows(n=400):
        return [
            {"t": float(i), "turn_rate": float(rng.normal(0, 1 + (i // 50) % 4)),
             "collision": int(rng.random() < 0.03)}
            for i in range(n)
        ]

    sessions = {f"s{k}": rows() for k in range(6)}
    return input(sessions, time="t", hz=1).window(10, stride=5)


def test_readme_invocation_styles_select_the_same_windows():
    from kuration.selection import BalancedSelector, RareEventCoverage, run

    namespace = {
        "pipeline": _selection_pipeline().options(seed=1),
        "BalancedSelector": BalancedSelector,
        "RareEventCoverage": RareEventCoverage,
        "run": run,
    }
    exec(_block("README.md", "# 1. Direct"), namespace)

    def key(result):
        return sorted(
            (s.source_id, s.start_idx, s.end_idx, round(s.features["motion"], 9))
            for s in result.selected
        )

    # The sample rebinds `result` three times; re-run each style to compare them.
    pipeline = namespace["pipeline"]
    direct = BalancedSelector(feature="motion", of="turn_rate").apply(
        RareEventCoverage(of="collision").apply(pipeline)).collect()
    chained = namespace["run"](pipeline, [
        RareEventCoverage(of="collision"),
        BalancedSelector(feature="motion", of="turn_rate"),
    ]).collect()
    configured = namespace["run"](pipeline, namespace["config"]).collect()
    assert key(direct) == key(chained) == key(configured)
    assert key(direct)


def test_readme_custom_operation_samples_run_and_register():
    namespace = {"pipeline": _selection_pipeline()}
    exec(_block("README.md", "class OnlyActiveWindows"), namespace)
    exec(_block("README.md", "register_operation_type("), namespace)
    from kuration.selection import run

    # The registered type now resolves from a config dict, like a shipped operation.
    result = run(namespace["pipeline"], [
        {"type": "OnlyActiveWindows", "threshold": 0.15},
        {"type": "BalancedSelector", "feature": "motion", "of": "turn_rate",
         "bins": 4, "count": 20},
    ]).options(seed=1).collect()
    assert result.selected


def _write_slam_export(directory: Path) -> None:
    from kuration.gameplay.slam import export_slam_output

    def pose(position, forward):
        position = np.asarray(position, dtype=np.float64)
        frustum = np.stack([position, position + [-1, -1, 1], position + [1, -1, 1],
                            position + [-1, 1, 1], position + [1, 1, 1]])
        return SimpleNamespace(frustum_points=frustum, cx=320.0, cy=240.0, f=500.0,
                               x_axis=np.array([1.0, 0.0, 0.0]),
                               y_axis=np.array([0.0, 1.0, 0.0]),
                               z_axis=np.asarray(forward, dtype=np.float64))

    trajectories, poses = {}, {}
    for name, speed, turn in [("slow", 0.03, 0.02), ("fast", 0.12, 0.10),
                              ("arc", 0.07, 0.06), ("wide", 0.2, 0.01)]:
        path = np.array([[speed * i * math.cos(turn * i), speed * i * math.sin(turn * i), 0.0]
                         for i in range(96)])
        forwards = [(math.cos(turn * i), math.sin(turn * i), 0.0) for i in range(96)]
        trajectories[name] = path
        poses[name] = [pose(p, f) for p, f in zip(path, forwards, strict=True)]
    output = SimpleNamespace(trajectories=trajectories, poses=poses,
                             points=np.zeros((2, 3)), colors=np.zeros((2, 3), np.uint8))
    export_slam_output(output, directory, sample_hz=10.0)


@pytest.mark.parametrize("path", ["README.md", "docs/domains.md"])
def test_slam_stats_and_evidence_samples_run(path, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    monkeypatch.chdir(tmp_path)
    _write_slam_export(tmp_path / "slam-export")
    namespace: dict = {}
    exec(_block(path, "gameplay.slam_reconstruct("), namespace)
    assert type(namespace["report"]).__name__ == "StatsReport"
