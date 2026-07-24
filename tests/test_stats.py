"""Unified statistics across input, selection, and emitted artifact stages."""
from __future__ import annotations

import json

import pytest

from kuration import StatsExamples, StatsReport, input, stats


def _pipeline(*, selected: int | None = None):
    sessions = {
        "alpha": [
            {"t": i, "entity": "a", "x": float(i), "kind": "early" if i < 4 else "late"}
            for i in range(8)
        ],
        "beta": [
            {"t": i, "entity": "b", "x": float(i + 10),
             "kind": "early" if i < 4 else "late"}
            for i in range(8)
        ],
    }
    pipeline = (
        input(sessions, time="t", entity="entity", hz=2)
        .window(4, stride=2)
        .measure("level", "mean", of="x")
        .options(mode="strict", seed=5)
    )
    if selected is not None:
        pipeline = pipeline.sample(
            "top-k", n=selected, weights={"level": 1.0}
        )
    return pipeline


def test_pipeline_stats_profiles_uncurated_input_and_candidates():
    report = _pipeline().stats()

    assert isinstance(report, StatsReport)
    assert report.stage == "uncurated"
    assert report.sections["input"]["fields"]["x"]["max"] == 17.0
    assert report.sections["input"]["fields"]["kind"]["unique"] == 2
    assert report.sections["candidates"]["n_segments"] == 6
    assert report.sections["candidates"]["metrics"]["level"]["max"] == 15.5
    assert "candidates: 6 segments" in report.render()
    assert "<table" in report._repr_html_()


def test_pipeline_and_curate_result_stats_compare_selection_without_registry_views():
    pipeline = _pipeline(selected=2)
    fluent = pipeline.stats(examples=1)
    result = pipeline.collect()
    direct = result.stats(examples=1)

    assert fluent.stage == direct.stage == "curated"
    assert fluent.sections["selection"]["selected"] == 2
    assert direct.sections["selection"]["candidates"] == 6
    assert (
        direct.sections["selection"]["metrics"]["level"]["selected"]["mean"]
        > direct.sections["selection"]["metrics"]["level"]["candidate"]["mean"]
    )
    assert (
        direct.sections["selection"]["examples"][0]["priority"]
        == fluent.sections["selection"]["examples"][0]["priority"]
    )


def test_stats_discovers_actions_and_selects_bounded_evidence():
    report = _pipeline(selected=2).stats(examples=2)

    available = report.available()
    assert available["default_plot"] == "distribution"
    assert available["example_modes"] == [
        "selected", "rejected", "strongest", "weakest", "input"
    ]
    assert available["comparisons"] == [
        "selected-vs-rejected", "strongest-vs-weakest"
    ]

    evidence = report.examples(
        metric="level", compare="selected-vs-rejected", n=2
    )
    assert isinstance(evidence, StatsExamples)
    assert len(evidence.groups["selected"]) == 2
    assert len(evidence.groups["rejected"]) == 2
    assert evidence.groups["selected"][0]["rank_value"] >= \
        evidence.groups["selected"][1]["rank_value"]
    assert "Stats evidence" in evidence._repr_html_()

    detached = StatsReport(report.to_dict()).examples(mode="weakest", n=1)
    assert detached.bounded is True
    assert len(detached.groups["weakest"]) == 1


def test_output_result_and_path_stats_scan_actual_emitted_modalities(tmp_path):
    output = tmp_path / "dataset"
    write_result = _pipeline(selected=2).output(
        output, categorical="omit", shard_length=4
    )

    assert write_result.dataset == str(output)
    from_result = write_result.stats(max_values=50)
    from_path = stats(output, max_values=50)

    for report in (from_result, from_path):
        assert report.stage == "output"
        assert report.sections["artifact"]["shards"] == 2
        assert report.sections["contents"]["shards_scanned"] == 2
        assert report.sections["contents"]["modalities"]["x"]["rows"] == 8
        assert report.sections["contents"]["modalities"]["x"]["min"] == 12.0
        assert report.sections["contents"]["modalities"]["x"]["max"] == 17.0

    html_path = tmp_path / "stats.html"
    assert from_result.save(html_path, title="Output stats") == str(html_path)
    html = html_path.read_text()
    assert "<!doctype html>" in html and "Output stats" in html
    assert "Output modalities" in html


def test_hours_pipeline_stats_is_explicitly_pending_output():
    report = (
        _pipeline()
        .sample("top-k", hours=0.001, weights={"level": 1.0})
        .stats()
    )

    assert report.stage == "planned"
    assert report.sections["budget_plan"]["train_hours"] == 0.001
    assert any(
        warning["code"] == "selection-pending-output"
        for warning in report.warnings
    )


def test_hours_output_stats_reports_actual_admission(tmp_path):
    write_result = (
        _pipeline()
        .sample("top-k", hours=4 / 3600, weights={"level": 1.0})
        .output(tmp_path / "hours", categorical="omit", shard_length=4)
    )
    report = write_result.stats(max_values=50)

    assert report.stage == "output"
    assert report.sections["hours_selection"]["candidate_items"] == 2
    assert report.sections["hours_selection"]["admitted_items"] == 1
    assert report.sections["artifact"]["splits"]["train"]["steps"] == 8


def test_stats_validates_subject_and_sampling_arguments():
    with pytest.raises(TypeError, match="stats\\(\\) accepts"):
        stats(object())
    with pytest.raises(ValueError, match="hist_bins"):
        _pipeline().stats(hist_bins=0)


def test_stats_plot_returns_figure_without_showing():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    report = _pipeline().stats()
    figure = report.plot("level")
    assert figure.axes[0].get_title() == "level distribution"


def test_stats_report_remains_json_safe():
    report = _pipeline(selected=2).stats()
    encoded = json.dumps(report.to_dict())
    assert '"stage": "curated"' in encoded
