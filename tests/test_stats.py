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
        input(sessions, time="t", entity="entity", hz=1)
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
    via_pipeline = pipeline.stats(examples=1)
    result = pipeline.collect()
    direct = result.stats(examples=1)

    assert via_pipeline.stage == direct.stage == "curated"
    assert via_pipeline.sections["selection"]["selected"] == 2
    assert direct.sections["selection"]["candidates"] == 6
    assert (
        direct.sections["selection"]["metrics"]["level"]["selected"]["mean"]
        > direct.sections["selection"]["metrics"]["level"]["candidate"]["mean"]
    )
    assert (
        direct.sections["selection"]["examples"][0]["priority"]
        == via_pipeline.sections["selection"]["examples"][0]["priority"]
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


# --- content-scan honesty (2026-07 stats review) -------------------------------------------

def _sparse_entity_dataset(path, *, presence_mask: bool = True, shard_length: int = 8):
    """Write an artifact where entity 'a' exists only every other tick.

    The writer keeps a row for every tick of every track and flags the absent ones in
    `_present`, zero-filling their values — the exact shape of the bug this pins.
    """
    rows = []
    for step in range(8):
        rows.append({"t": step, "entity": "b", "x": 5.0, "kind": "keep"})
        if step % 2 == 0:
            rows.append({"t": step, "entity": "a", "x": 100.0, "kind": "keep"})
    modalities = {
        "x": {"dtype": "float32", "shape": [], "source": "field:x"},
        "kind": {"dtype": "U8", "shape": [], "source": "field:kind"},
    }
    pipeline = (
        input({"s": rows}, time="t", entity="entity", hz=1)
        .window(8, stride=8)
        .measure("level", "mean", of="x")
        .options(mode="strict", seed=1)
        .sample("top-k", n=1, weights={"level": 1.0})
    )
    if presence_mask:
        return pipeline.output(path, modalities=modalities, shard_length=shard_length)
    from kuration import OutputSpec

    spec = OutputSpec.from_dict(
        {
            "modalities": modalities,
            "shard_length": shard_length,
            "container": "npz",
            "presence_mask": False,
        }
    )
    return pipeline.output(path, spec=spec)


def test_content_stats_exclude_absent_entity_rows(tmp_path):
    """Finding 1: `_present=False` rows are zero-fill, not data."""
    _sparse_entity_dataset(tmp_path / "sparse")
    contents = stats(tmp_path / "sparse").sections["contents"]
    x = contents["modalities"]["x"]

    # 4 of 16 rows belong to the absent half of entity 'a' — x never actually held 0.0.
    assert contents["presence_masked"] is True
    assert x["rows"] == 16 and x["absent_rows"] == 4 and x["values"] == 12
    assert x["present_fraction"] == 0.75
    assert x["presence_masked"] is True
    assert x["min"] == 5.0 and x["max"] == 100.0
    assert x["mean"] == pytest.approx((4 * 100.0 + 8 * 5.0) / 12)
    # `_present` itself stays unmasked — it is the record of what was absent.
    assert contents["modalities"]["_present"]["presence_masked"] is False
    assert contents["modalities"]["_present"]["mean"] == pytest.approx(0.75)


def test_content_stats_without_presence_mask_record_that_they_are_unmasked(tmp_path):
    """Finding 1: no `_present` modality is the one honest reason to accumulate every row."""
    _sparse_entity_dataset(tmp_path / "plain", presence_mask=False)
    contents = stats(tmp_path / "plain").sections["contents"]
    x = contents["modalities"]["x"]

    assert "_present" not in contents["modalities"]
    assert contents["presence_masked"] is False
    assert x["presence_masked"] is False
    assert x["absent_rows"] == 0 and x["values"] == 16
    assert x["min"] == 0.0  # zero-fill is visible instead of silently masked


def test_categorical_modality_reports_sampled_top_values(tmp_path):
    """Coverage: the text/categorical accumulator (`_text_chunk` + top_values)."""
    _sparse_entity_dataset(tmp_path / "text")
    kind = stats(tmp_path / "text").sections["contents"]["modalities"]["kind"]

    assert kind["dtype"].endswith("U8")
    assert kind["unique_sampled"] == 1
    assert kind["top_values"] == {"keep": 12}  # the 4 absent rows are excluded
    assert kind["sampled_values"] == 12
    assert "min" not in kind  # non-numeric modalities get no numeric statistics


def test_dataset_report_describes_finite_values_and_counts_non_finite():
    """Finding 2: summary statistics and the histogram now agree on non-finite values."""
    from kuration import Segment, dataset_report

    segments = [
        Segment("s", i, i + 1, (i, i + 1), {"m": float("nan") if i % 2 else float(i)})
        for i in range(8)
    ]
    report = dataset_report(segments)
    metric = report["metrics"]["m"]

    assert metric["n_nonfinite"] == 4 and metric["n_finite"] == 4
    assert metric["mean"] == pytest.approx((0 + 2 + 4 + 6) / 4)
    assert metric["min"] == 0.0 and metric["max"] == 6.0
    assert sum(metric["histogram"]["counts"]) == 4
    assert any(item["code"] == "non-finite-metric" for item in report["warnings"])
    json.dumps(report.to_dict(), allow_nan=False)  # no NaN/Infinity tokens

    empty = dataset_report(
        [Segment("s", 0, 1, (0, 1), {"m": float("inf")})]
    )["metrics"]["m"]
    assert empty["mean"] is None and empty["p95"] is None and empty["n_finite"] == 0
    json.dumps(empty, allow_nan=False)


def test_none_statistics_render_as_dashes_not_typeerrors():
    """All-non-finite metrics report None statistics; text renders must tolerate that."""
    from kuration.core.results import SelectionReview

    review = SelectionReview({
        "selected": 1, "candidates": 2, "selection_rate": 0.5,
        "rationale": {"method": "top_k"},
        "coverage": {"selected": {"sources": 1, "bins": None}},
        "metrics": {"m": {"candidate": {"mean": None}, "selected": {"mean": None},
                          "mean_delta": None}},
        "warnings": [],
    })
    rendered = review.render()
    assert "—" in rendered and "nan" not in rendered.lower()
    """A constant metric near 2**53 must report as flat, not crash the histogram."""
    from kuration import Segment, dataset_report

    value = float(2**53 + 1)
    segments = [Segment("s", i, i + 1, (i, i + 1), {"m": value}) for i in range(4)]
    histogram = dataset_report(segments)["metrics"]["m"]["histogram"]

    assert sum(histogram["counts"]) == 4
    assert histogram["edges"][0] < histogram["edges"][-1]


def _nonfinite_dataset(path):
    rows = [
        {"t": step, "entity": "a", "x": float("nan") if step >= 4 else 1.0,
         "y": float(step)}
        for step in range(8)
    ]
    return (
        input({"s": rows}, time="t", entity="entity", hz=1)
        .window(8, stride=8)
        .measure("level", "mean", of="y")
        .options(mode="strict", seed=1)
        .sample("top-k", n=1, weights={"level": 1.0})
        .output(
            path,
            modalities={"x": {"dtype": "float32", "shape": [], "source": "field:x"}},
            shard_length=8,
        )
    )


def test_stats_reports_non_finite_modality_values_as_strict_json(tmp_path):
    """Finding 2: NaN payload values are counted and never serialized as `NaN`."""
    _nonfinite_dataset(tmp_path / "nan")
    report = stats(tmp_path / "nan")
    x = report.sections["contents"]["modalities"]["x"]

    assert x["non_finite"] == 4 and x["finite"] == 4
    assert x["mean"] == 1.0  # the finite half only
    assert any(item["code"] == "non-finite-values" for item in report.warnings)
    json.dumps(report.to_dict(), allow_nan=False)
    assert "non-finite" in report._repr_html_()


def test_verify_without_a_content_scan_still_checks_shard_integrity(tmp_path):
    """Finding 3: `--verify --no-content` used to verify nothing at all."""
    _sparse_entity_dataset(tmp_path / "corrupt")
    shard = sorted((tmp_path / "corrupt").rglob("*.npz"))[0]
    shard.write_bytes(shard.read_bytes() + b"tamper")

    verified = stats(tmp_path / "corrupt", content=False, verify=True)
    artifact = verified.sections["artifact"]["verification"]
    assert artifact["requested"] is True and artifact["valid"] is False
    assert any(item["code"] == "shard-verification" for item in verified.warnings)

    unverified = stats(tmp_path / "corrupt", content=False, verify=False)
    assert unverified.sections["artifact"]["verification"]["requested"] is False
    assert unverified.warnings == []


def test_content_scan_verification_is_loud_about_a_corrupted_shard(tmp_path):
    """Coverage: the `verify=True` content path checks manifest checksums while decoding."""
    _sparse_entity_dataset(tmp_path / "scan")
    intact = stats(tmp_path / "scan", verify=True)
    assert intact.sections["contents"]["verified"] is True

    shard = sorted((tmp_path / "scan").rglob("*.npz"))[0]
    shard.write_bytes(shard.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum mismatch"):
        stats(tmp_path / "scan", verify=True)


def _hours_write_result(path):
    sessions = {
        f"s{index}": [
            {"t": step, "entity": "e", "x": float(step + index)} for step in range(8)
        ]
        for index in range(4)
    }
    return (
        input(sessions, time="t", entity="entity", hz=1)
        .window(8, stride=8)
        .measure("level", "mean", of="x")
        .options(mode="strict", seed=3)
        .sample("top-k", hours=12 / 3600, weights={"level": 1.0})
        .output(path, shard_length=8)
    )


def test_examples_bounds_embedded_evidence_and_rejects_unusable_values(tmp_path):
    """Finding 4: `--examples` is honored where evidence exists and refused where it can't be."""
    write_result = _hours_write_result(tmp_path / "hours")
    unbounded = stats(write_result).sections["hours_selection"]["examples"]
    bounded = stats(write_result, examples=1).sections["hours_selection"]["examples"]

    assert max(len(items) for items in unbounded.values()) > 1
    assert all(len(items) <= 1 for items in bounded.values())

    # A plain artifact embeds no examples: say so rather than ignoring the flag.
    with pytest.raises(ValueError, match="embeds none"):
        stats(tmp_path / "hours", examples=1)
    assert stats(tmp_path / "hours").stage == "output"


def _two_shard_dataset(path):
    sessions = {
        "low": [{"t": step, "entity": "e", "x": float(step)} for step in range(8)],
        "high": [{"t": step, "entity": "e", "x": 100.0 + step} for step in range(8)],
    }
    return (
        input(sessions, time="t", entity="entity", hz=1)
        .window(8, stride=8)
        .measure("level", "mean", of="x")
        .options(mode="strict", seed=1)
        .sample("top-k", n=2, weights={"level": 1.0})
        .output(path, shard_length=8)
    )


def _four_shard_dataset(path):
    sessions = {
        f"s{index}": [
            {"t": step, "entity": "e", "x": 100.0 * index + step} for step in range(8)
        ]
        for index in range(4)
    }
    return (
        input(sessions, time="t", entity="entity", hz=1)
        .window(8, stride=8)
        .measure("level", "mean", of="x")
        .options(mode="strict", seed=1)
        .sample("top-k", n=4, weights={"level": 1.0})
        .output(path, shard_length=8)
    )


def test_sampled_values_span_every_shard(tmp_path):
    """Finding 5: subsample the pool evenly instead of truncating to the earliest shards."""
    _four_shard_dataset(tmp_path / "shards")
    # 4 shards × ceil(5/4)=2 collected = a pool of 8 thinned to 5: truncation would drop the
    # last shards entirely and contradict the recorded even-sampling method.
    x = stats(tmp_path / "shards", max_values=5).sections["contents"]["modalities"]["x"]

    assert x["percentiles_sampled"] is True
    assert x["histogram"]["sampled_values"] == 5
    # truncation would keep [0, 7, 100, 107, 200] — p95 180, top edge 200.
    assert x["p95"] > 200.0
    assert x["histogram"]["edges"][-1] >= 307.0
    assert x["max"] == 307.0  # exact statistics still see every value


def test_percentiles_are_marked_when_sampled(tmp_path):
    """Finding 6: sampled percentiles are labelled next to the exact mean/std/min/max."""
    _two_shard_dataset(tmp_path / "marked")
    sampled = stats(tmp_path / "marked", max_values=4)
    exact = stats(tmp_path / "marked", max_values=1000)

    assert sampled.sections["contents"]["modalities"]["x"]["percentiles_sampled"] is True
    assert exact.sections["contents"]["modalities"]["x"]["percentiles_sampled"] is False
    assert "p50/p95 sampled" in sampled._repr_html_()
    assert "p50/p95 exact" in exact._repr_html_()
    # the text render carries the marker through the report's warnings
    assert "[sampled-percentiles]" in sampled.render()
    assert "sampled-percentiles" not in exact.render()


def test_modality_missing_from_some_shards_is_reported(tmp_path, monkeypatch):
    """Finding 7: a modality absent from a shard undercounts rows — say so."""
    from kuration.core import loaders

    _two_shard_dataset(tmp_path / "partial")
    real = loaders.read_shard_arrays
    seen: list[str] = []

    def patched(entry, container, **kwargs):
        arrays = real(entry, container, **kwargs)
        seen.append(entry["path"])
        if len(seen) > 1:
            arrays = {name: value for name, value in arrays.items() if name != "x"}
        return arrays

    monkeypatch.setattr(loaders, "read_shard_arrays", patched)
    report = stats(tmp_path / "partial")
    x = report.sections["contents"]["modalities"]["x"]

    assert x["shards_present"] == 1
    assert report.sections["contents"]["shards_scanned"] == 2
    assert any(
        item["code"] == "modality-missing-from-shards" and item["modality"] == "x"
        for item in report.warnings
    )


def test_cross_shard_dtype_drift_raises(tmp_path, monkeypatch):
    """Coverage: one modality with two dtypes across shards is a corrupt artifact."""
    from kuration.core import loaders

    _two_shard_dataset(tmp_path / "drift")
    real = loaders.read_shard_arrays
    seen: list[str] = []

    def patched(entry, container, **kwargs):
        arrays = dict(real(entry, container, **kwargs))
        seen.append(entry["path"])
        if len(seen) > 1:
            arrays["x"] = arrays["x"].astype("float64")
        return arrays

    monkeypatch.setattr(loaders, "read_shard_arrays", patched)
    with pytest.raises(ValueError, match="changes dtype or shape across shards"):
        stats(tmp_path / "drift")


def test_integer_extrema_are_exact_beyond_the_float64_mantissa():
    """Finding 13: int64 magnitudes above 2**53 must not round through float64."""
    import numpy as np

    from kuration.core.stats import _finalize_modality, _numeric_chunk

    accumulator = {
        "dtype": "int64", "shape": (), "numeric": True, "rows": 2, "values": 0,
        "finite": 0, "non_finite": 0, "mean": 0.0, "m2": 0.0,
        "min": float("inf"), "max": float("-inf"), "int_min": None, "int_max": None,
        "samples": [], "shards_present": 1, "masked_shards": 0, "absent_rows": 0,
    }
    exact = 2**53 + 1
    _numeric_chunk(accumulator, np.array([exact, -exact], dtype=np.int64), 10)
    out = _finalize_modality(accumulator, hist_bins=4, max_values=10)

    assert out["max"] == exact and out["min"] == -exact
    assert isinstance(out["max"], int) and not isinstance(out["max"], bool)
    assert out["mean"] == pytest.approx(0.0)  # mean/std stay float by design


def test_embedded_examples_reject_a_metric_missing_from_some_items():
    """Finding 8: the availability check is per item, like the live selection path."""
    report = StatsReport(
        stage="curated",
        subject="selection-review",
        sections={
            "selection": {
                "metrics": {"level": {}},
                "evidence": {
                    "selected": [
                        {"source": "a", "features": {"level": 1.0}},
                        {"source": "b", "features": {}},
                    ],
                    "rejected": [],
                },
            }
        },
        warnings=[],
    )

    with pytest.raises(ValueError, match="unknown evidence metric 'level'"):
        report.examples(mode="selected", metric="level", n=2)
    # an empty group cannot vouch for a name either — validate against the metric namespace
    with pytest.raises(ValueError, match="unknown evidence metric 'nope'"):
        report.examples(mode="rejected", metric="nope", n=1)
    assert report.examples(mode="rejected", metric="level", n=1).groups["rejected"] == []


def test_selection_examples_reject_an_unknown_metric_on_an_empty_pool():
    """Finding 9: an empty pool must not silently accept a metric name."""
    from kuration.core.review import selection_examples

    result = (
        _pipeline(selected=2)
        .filter("level", ">", 1000)  # excludes every candidate
        .collect()
    )
    assert result.selected == []

    with pytest.raises(ValueError, match="unknown evidence metric 'nope'"):
        selection_examples(result, mode="selected", metric="nope", n=1)
    assert selection_examples(result, mode="selected", metric="level", n=1) == []


def test_plot_rejects_a_malformed_histogram():
    """Finding 10: mismatched counts/edges used to plot nothing at all."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    report = StatsReport(
        stage="output",
        subject="artifact",
        sections={
            "contents": {
                "modalities": {
                    "x": {"histogram": {"counts": [1, 2, 3], "edges": [0.0, 1.0]}}
                }
            }
        },
        warnings=[],
    )

    with pytest.raises(ValueError, match="malformed histogram"):
        report.plot("x")


def test_kur_stats_verify_without_content_exits_non_zero(tmp_path):
    """Finding 3, at the CLI: `--verify --no-content` reports and exits like kur-inspect."""
    from conftest import run_cli

    _sparse_entity_dataset(tmp_path / "cli")
    ok = run_cli(
        "kuration.core.cli.stats", tmp_path / "cli", "--verify", "--no-content", "--quiet"
    )
    assert "artifact:" in ok.stdout

    shard = sorted((tmp_path / "cli").rglob("*.npz"))[0]
    shard.write_bytes(shard.read_bytes() + b"tamper")
    failed = run_cli(
        "kuration.core.cli.stats", tmp_path / "cli", "--verify", "--no-content", "--quiet",
        expect=2,
    )
    assert "shard-verification" in failed.stdout

    rejected = run_cli(
        "kuration.core.cli.stats", tmp_path / "cli", "--examples", "2", "--no-content",
        "--quiet", expect=1,
    )
    assert "embeds none" in rejected.stderr
