"""Automatic bounded top-k hours execution, provenance, and resume."""
import json

import pytest

from kuration import input

pytest.importorskip("pyarrow")


def _pipeline(resume=False):
    return (input([{"x": float(value)} for value in range(24)], hz=2)
            .window(4, stride=2)
            .measure("level", "mean", of="x")
            .sample("top-k", hours=0.002, weights={"level": 1.0})
            .options(resume=resume, batch_size=3, on_error="raise"))


def test_hours_output_automatically_uses_bounded_feature_and_span_plan(tmp_path):
    out = tmp_path / "dataset"
    result = _pipeline().output(str(out), span_steps=4, categorical="omit")
    assert result["admitted"]
    config = json.loads((out / "config.json").read_text())
    assert config["execution"]["bounded"] is True
    assert config["execution"]["physical_plan"]["mode"] == "bounded-hours-top-k"
    assert config["execution"]["telemetry"]["checkpoint_misses"] == 4
    assert config["results"]["segments_total"] == 11
    commits = list(
        (out / ".kuration" / "checkpoints").glob("*/stage000/commits/*.json"))
    assert len(commits) == 4


def test_hours_resume_reuses_feature_commits_and_preserves_admission(tmp_path):
    out = tmp_path / "resume"
    first = _pipeline().output(str(out), span_steps=4, categorical="omit")
    mtimes = {path.name: path.stat().st_mtime_ns for path in
              (out / ".kuration" / "checkpoints").glob("*/stage000/data/*.arrow")}
    second = _pipeline(resume=True).output(str(out), span_steps=4, categorical="omit")
    again = {path.name: path.stat().st_mtime_ns for path in
             (out / ".kuration" / "checkpoints").glob("*/stage000/data/*.arrow")}
    assert second["admitted"] == first["admitted"]
    assert again == mtimes


def test_weighted_hours_uses_exact_bounded_gumbel_priority_plan(tmp_path):
    out = tmp_path / "weighted"
    pipeline = (input([{"x": float(value), "y": float(30 - value)}
                       for value in range(30)], hz=2)
                .window(4, stride=2).measure("x_mean", "mean", of="x")
                .measure("y_mean", "mean", of="y")
                .sample("weighted", hours=0.002,
                        weights={"x_mean": 0.8, "y_mean": -0.1})
                .options(seed=29, batch_size=4, on_error="raise"))
    result = pipeline.output(str(out), span_steps=5, categorical="omit")
    assert result["admitted"]
    config = json.loads((out / "config.json").read_text())
    assert config["execution"]["physical_plan"]["mode"] == "bounded-hours-weighted"


def test_uniform_hours_uses_exact_bounded_bin_priority_plan(tmp_path):
    out = tmp_path / "uniform"
    pipeline = (input([{"x": float(value)} for value in range(30)], hz=2)
                .window(4, stride=2).measure("level", "mean", of="x")
                .bin("level", bins=4).sample("uniform-bins", hours=0.002)
                .options(seed=19, batch_size=4, on_error="raise"))
    result = pipeline.output(str(out), span_steps=5, categorical="omit")
    assert result["admitted"]
    config = json.loads((out / "config.json").read_text())
    assert config["execution"]["physical_plan"]["mode"] == "bounded-hours-uniform-bins"


def test_hours_group_callback_is_reused_across_selection_fill_and_writing(tmp_path):
    calls = []

    def group_of(source):
        calls.append(source)
        return "one-group"

    pipeline = (
        input([{"x": float(value)} for value in range(12)], hz=2)
        .window(4, stride=2)
        .measure("level", "mean", of="x")
        .group(group_of)
        .sample("top-k", hours=0.001, weights={"level": 1.0})
        .options(on_error="raise")
    )

    pipeline.output(str(tmp_path / "group-once"), span_steps=4, categorical="omit")

    assert len(calls) == 1


def test_memory_budget_rejects_unbounded_hours_method_before_execution(tmp_path):
    pipeline = (input([{"x": float(value)} for value in range(12)], hz=2)
                .window(4, stride=2).measure("level", "mean", of="x")
                .bin("level", bins=3)
                .sample("target-distribution", hours=0.001, target="flat")
                .options(memory_budget=1024, on_error="raise"))
    with pytest.raises(ValueError, match="bounded exact hours plan.*target-distribution"):
        pipeline.output(str(tmp_path / "unsafe"), categorical="omit")
