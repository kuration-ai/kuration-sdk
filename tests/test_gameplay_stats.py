"""Gameplay enrichments over the ordinary stats and visualization surface."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import kuration.gameplay as gameplay
from kuration import FrameRef, SchemaSpec, Tick, input


ROOT = Path(__file__).resolve().parents[1]


SLAM_SCHEMA = SchemaSpec.from_dict(
    {
        "name": "gameplay-trajectory",
        "entity_axis": "camera",
        "sample_hz": 10.0,
        "fields": {
            "camera_position": {
                "kind": "vector",
                "dims": ["x", "y", "z"],
                "semantics": "position",
            },
            "camera_z_axis": {
                "kind": "vector",
                "dims": ["x", "y", "z"],
            },
            "tracking_confidence": {"kind": "scalar"},
            "action": {"kind": "categorical"},
        },
    }
)

MEDIA_SCHEMA = SchemaSpec.from_dict(
    {
        "name": "gameplay-media",
        "entity_axis": "player",
        "sample_hz": 10.0,
        "fields": {
            "score": {"kind": "scalar"},
            "action": {"kind": "categorical"},
        },
        "media": {
            "frame": {
                "kind": "video-frame",
                "dtype": "uint8",
                "shape": [3, 4, 3],
                "layout": "HWC",
                "color_space": "RGB",
                "timing": "frame_index",
            }
        },
    }
)


def _trajectory(speed: float, confidence: float):
    return [
        Tick(
            time=step / 10,
            entities={
                0: {
                    "camera_position": (speed * step, 0.0, 0.0),
                    "camera_z_axis": (
                        math.cos(step / 20),
                        math.sin(step / 20),
                        0.0,
                    ),
                    "tracking_confidence": confidence,
                    "action": "move" if step % 2 else "look",
                }
            },
        )
        for step in range(16)
    ]


def test_gameplay_import_enriches_the_same_stats_report():
    report = (
        input(
            {
                "steady": _trajectory(0.05, 0.98),
                "fast": _trajectory(0.15, 0.82),
            },
            schema=SLAM_SCHEMA,
        )
        .window(8, stride=4)
        .slam_motion_diversity(axis="slam_path_length", bins=3)
        .slam_quality()
        .sample("top-k", n=2, weights={"slam_quality_score": 1.0})
        .stats()
    )

    domain = report.sections["gameplay"]
    assert domain["kind"] == "gameplay"
    assert domain["actions"][0]["name"] == "action"
    assert "slam_quality_score" in domain["slam_metrics"]
    assert any(item["label"] == "SLAM quality" for item in domain["summary"])
    assert "gameplay SLAM quality:" in report.render()
    assert "Domain insights" in report._repr_html_()


def test_gameplay_stats_plot_uses_registered_plot_kind():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    report = (
        input({"steady": _trajectory(0.05, 0.98)}, schema=SLAM_SCHEMA)
        .window(8, stride=4)
        .slam_quality()
        .report()
    )
    from kuration import stats

    figure = stats(report).plot(kind="gameplay")
    assert figure.axes[0].get_title() == "Gameplay signals"


def test_gameplay_plot_is_automatic_and_evidence_renders_live_frames():
    ticks = [
        Tick(
            time=index / 10,
            entities={
                0: {
                    "score": float(index),
                    "action": "move" if index % 2 else "look",
                }
            },
            media={
                "frame": np.full((3, 4, 3), index * 20, dtype=np.uint8)
            },
        )
        for index in range(8)
    ]
    report = (
        input({"match": ticks}, schema=MEDIA_SCHEMA)
        .window(4, stride=4)
        .measure("action_value", "mean", of="score")
        .measure("w_shift_mean", "mean", of="score")
        .measure("w_n_instances", "count", of="score")
        .sample("top-k", n=1, weights={"action_value": 1.0})
        .stats()
    )

    assert report.available()["default_plot"] == "gameplay"
    assert report.available()["media_examples"] is True
    assert "w_shift_mean" in report.sections["gameplay"]["action_metrics"]
    evidence = report.examples(compare="selected-vs-rejected", n=1, frames=2)
    html = evidence._repr_html_()
    assert html.count("data:image/png;base64,") == 4
    assert "action:" in html
    assert "W shift mean=" in html
    assert "W n instances=" in html
    assert "Gameplay evidence" in html

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    assert report.plot().axes[0].get_title() == "Gameplay signals"


class _VideoProvider:
    def probe(self, source):
        from kuration import VideoMetadata

        return VideoMetadata(source, 4, 3, 5, 0.4, 10.0, codec="test")

    def iter_refs(self, source, *, source_id=None):
        for index in range(5):
            yield FrameRef(source, source_id or source, index, index / 10)

    def decode_batch(self, refs):
        return np.stack(
            [
                np.full((3, 4, 3), ref.frame_index * 40, dtype=np.uint8)
                for ref in refs
            ]
        )


def test_video_comparison_renders_saves_probes_and_plots(tmp_path):
    left = tmp_path / "before.mp4"
    right = tmp_path / "after.mp4"
    left.write_bytes(b"before")
    right.write_bytes(b"after")
    comparison = gameplay.compare_videos(
        left,
        right,
        labels=("Raw <capture>", "Curated"),
        provider=_VideoProvider(),
    )

    html = comparison.render(embed=True)
    assert "data:video/mp4;base64" in html
    assert "Raw &lt;capture&gt;" in html
    assert "Synchronize playback" in html
    assert comparison.probe()["left"]["frame_count"] == 5

    output = tmp_path / "comparison.html"
    assert comparison.save(output, embed=True) == str(output)
    assert "<!doctype html>" in output.read_text()

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    figure = comparison.plot(frames=3)
    assert len(figure.axes) == 6


def test_gameplay_stats_notebook_is_valid_and_all_code_cells_compile():
    path = ROOT / "examples" / "notebooks" / "gameplay_stats.ipynb"
    notebook = json.loads(path.read_text())

    assert notebook["nbformat"] == 4
    assert any("progress=\"auto\"" in "".join(cell.get("source") or ())
               for cell in notebook["cells"])
    assert any("compare_videos" in "".join(cell.get("source") or ())
               for cell in notebook["cells"])
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(path), "exec")
