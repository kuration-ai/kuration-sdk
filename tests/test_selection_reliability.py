"""Regression coverage for selection parity and enforceable reliability contracts."""
from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pytest

from kuration import (SchemaSpec, Segment, SegmentBatch, evaluate_quality, input,
                       load_strategy, segment_priorities)
from kuration.core.binning import bin_segments
from kuration.core.external import (iter_external_top_k_priorities,
                               iter_external_uniform_priorities,
                               iter_external_weighted_priorities)
from kuration.core.methods import feature_matrix
from kuration.core.output.external_fill import external_fill_to_hours
from kuration.core.output.fill import fill_to_hours
from kuration.core.strategy import _normalize


HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None


def test_binning_rejects_a_segment_missing_the_axis_feature():
    segments = [
        Segment("complete", 0, 1, (0, 0), {"score": 1.0}),
        Segment("missing", 1, 2, (1, 1), {}),
    ]

    with pytest.raises(ValueError, match=r"binning axis 'score'.*'missing'.*has no such feature"):
        bin_segments(segments, [("score", 3)])


def test_quantile_normalization_is_tie_aware_and_constant_neutral():
    values = np.asarray([1.0, 1.0, 3.0, 3.0, 5.0])
    assert _normalize(values, "quantile") == pytest.approx([0.125, 0.125, 0.625, 0.625, 1.0])
    assert np.array_equal(_normalize(np.ones(20), "quantile"), np.zeros(20))
    with pytest.raises(ValueError, match="finite"):
        _normalize(np.asarray([1.0, np.nan]), "quantile")

    permutation = np.asarray([4, 1, 3, 0, 2])
    permuted = _normalize(values[permutation], "quantile")
    assert permuted[np.argsort(permutation)] == pytest.approx(_normalize(values, "quantile"))

    segments = [Segment("s", index, index + 1, (index, index), {"x": value, "c": 7.0})
                for index, value in enumerate(values)]
    matrix = feature_matrix(segments, ["x", "c"], normalize="quantile")
    assert matrix[:, 0] == pytest.approx(_normalize(values, "quantile"))
    assert np.array_equal(matrix[:, 1], np.zeros(len(values)))


@pytest.mark.skipif(not HAS_PYARROW, reason="bounded reducers need PyArrow")
def test_external_quantile_priorities_match_tie_aware_local_reference(tmp_path):
    values = [1.0, 1.0, 3.0, 3.0, 5.0]
    segments = [Segment("source", index, index + 1, (index, index), {"x": value})
                for index, value in enumerate(values)]
    path = tmp_path / "ties.arrow"
    SegmentBatch.from_segments(segments).write_ipc(str(path))
    actual = [priority for _segment, priority in iter_external_top_k_priorities(
        [str(path)], weights={"x": 1.0}, normalize="quantile",
        spill_dir=str(tmp_path / "spill"))]
    assert actual == pytest.approx(_normalize(np.asarray(values), "quantile"))

    constant = tmp_path / "constant.arrow"
    SegmentBatch.from_segments([
        Segment("source", index, index + 1, (index, index), {"x": 7.0})
        for index in range(5)
    ]).write_ipc(str(constant))
    assert [priority for _segment, priority in iter_external_top_k_priorities(
        [str(constant)], weights={"x": 1.0}, normalize="quantile")] == [0.0] * 5


@pytest.mark.skipif(not HAS_PYARROW, reason="bounded reducers need PyArrow")
@pytest.mark.parametrize("method,normalize", [
    ("top-k", "quantile"), ("top-k", "minmax"), ("top-k", "zscore"),
    ("weighted", "quantile"), ("uniform-bins", "quantile"),
])
def test_external_predicate_priorities_match_legacy_pool_semantics(
        tmp_path, method, normalize):
    values = [
        (0.0, 10.0, 1.0), (10.0, 0.0, 1.0),
        *[(float(index + 1), float(20 + index), 0.0) for index in range(9)],
    ]
    segments = [Segment("source", index, index + 1, (index, index),
                        {"x": x, "y": y, "gate": gate})
                for index, (x, y, gate) in enumerate(values)]
    schema = SchemaSpec.from_dict({
        "name": "selection-parity", "entity_axis": "entity",
        "fields": {name: {"kind": "scalar"} for name in ("x", "y", "gate")},
    })
    selection = {"method": method, "normalize": normalize,
                 "predicates": [{"metric": "gate", "op": ">=", "value": 1.0}]}
    spec = {"selection": selection}
    if method in {"top-k", "weighted"}:
        selection["weights"] = {"x": 0.4, "y": 0.6}
    else:
        spec["binning"] = {"axes": [{"metric": "x", "bins": 3}]}
        bin_segments(segments, [("x", 3)])
    strategy = load_strategy(spec, schema, available_metrics=("x", "y", "gate"))
    expected = segment_priorities(segments, strategy, seed=17)

    path = tmp_path / f"{method}-{normalize}.arrow"
    SegmentBatch.from_segments(segments).write_ipc(str(path))
    if method == "top-k":
        stream = iter_external_top_k_priorities(
            [str(path)], weights=strategy.weights, normalize=normalize)
    elif method == "weighted":
        stream = iter_external_weighted_priorities(
            [str(path)], weights=strategy.weights, normalize=normalize, seed=17)
    else:
        stream = iter_external_uniform_priorities(
            [str(path)], axes=strategy.axes, seed=17)
    actual = [priority if segment.features["gate"] >= 1 else float("-inf")
              for segment, priority in stream]
    assert actual == pytest.approx(expected)


def _predicate_pipeline():
    sessions = {
        "pass-a": [{"t": 0, "entity": "e", "x": 0.0, "y": 10.0, "gate": 1.0}],
        "pass-b": [{"t": 0, "entity": "e", "x": 10.0, "y": 0.0, "gate": 1.0}],
        **{f"fail-{index:02d}": [{"t": 0, "entity": "e", "x": float(index + 1),
                                   "y": float(20 + index), "gate": 0.0}]
           for index in range(9)},
    }
    return (input(sessions, time="t", entity="entity", hz=1).window(1)
            .measure("mx", "mean", of="x").measure("my", "mean", of="y")
            .measure("gate_ok", "mean", of="gate").filter("gate_ok", ">=", 1)
            .sample("top-k", hours=1 / 3600, weights={"mx": 0.4, "my": 0.6})
            .options(batch_size=3, on_error="raise"))


@pytest.mark.skipif(not HAS_PYARROW, reason="bounded reducers need PyArrow")
def test_bounded_predicates_preserve_legacy_normalization_semantics(tmp_path, monkeypatch):
    pipeline = _predicate_pipeline()
    bounded = pipeline.output(
        tmp_path / "bounded", shard_length=1, span_steps=1, categorical="omit")

    # The bounded-plan feature probe treats an unavailable PyArrow as a supported legacy
    # fallback.  No legacy-path code imports PyArrow after that probe.
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    legacy = pipeline.output(
        tmp_path / "legacy", shard_length=1, span_steps=1, categorical="omit")
    assert bounded.admitted == legacy.admitted == {("pass-b", "e", 0)}


@pytest.mark.skipif(not HAS_PYARROW, reason="bounded execution needs PyArrow")
def test_memory_budget_derives_and_enforces_live_batch_limits(tmp_path):
    pipeline = (input([{"x": float(value)} for value in range(24)], hz=2)
                .window(4, stride=2).measure("level", "mean", of="x")
                .sample("top-k", hours=0.002, weights={"level": 1.0})
                .options(memory_budget=20_000, batch_size=100, max_in_flight=8,
                         on_error="raise"))
    destination = tmp_path / "budgeted"
    pipeline.output(destination, span_steps=4, categorical="omit")
    telemetry = json.loads((destination / "config.json").read_text())["execution"]["telemetry"]
    assert telemetry["memory_budget"] == 20_000
    assert telemetry["batch_size"] < telemetry["requested_batch_size"]
    assert telemetry["max_in_flight"] == 1
    assert telemetry["estimated_peak_live_bytes"] <= telemetry["memory_budget"]
    assert (telemetry["output_accumulator_bytes"] * telemetry["output_jobs_per_pass"]
            <= telemetry["memory_budget"])

    impossible = tmp_path / "impossible"
    with pytest.raises(ValueError, match="cannot hold one feature window"):
        pipeline.options(memory_budget=100).output(
            impossible, span_steps=4, categorical="omit")
    assert not (impossible / ".kuration_output_spec.json").exists()


def test_fill_accepts_numeric_priority_strings_and_rejects_duplicate_ids(tmp_path):
    items = [
        {"id": "a", "source_id": "s", "priority": "2.5", "seconds": 1.0,
         "group": None},
        {"id": "b", "source_id": "s", "priority": "1.5", "seconds": 1.0,
         "group": None},
    ]
    local = fill_to_hours(items, train_hours=1 / 3600, test_hours=0, return_stats=True)
    external = external_fill_to_hours(
        iter(items), train_hours=1 / 3600, test_hours=0,
        spill_dir=str(tmp_path / "spill"))
    assert local == external[:3]
    assert local[0] == {"a"}

    duplicate = [items[0], dict(items[0])]
    with pytest.raises(ValueError, match="duplicate fill item id"):
        fill_to_hours(duplicate, train_hours=1 / 3600, test_hours=0)
    with pytest.raises(ValueError, match="duplicate fill item id"):
        external_fill_to_hours(
            iter(duplicate), train_hours=1 / 3600, test_hours=0,
            spill_dir=str(tmp_path / "spill"))


@pytest.mark.parametrize("expectation,check_name", [
    ({"max_missing": {"x": 0.0}}, "max_missing.x"),
    ({"min_tag_coverage": {"rare": 1}}, "min_tag_coverage.rare"),
    ({"max_group_fraction": 1.0}, "max_group_fraction"),
    ({"min_budget_utilization": 0.9}, "min_budget_utilization.train"),
    ({"max_budget_underfill_seconds": 0}, "max_budget_underfill_seconds.train"),
    ({"max_skipped_sources": 0}, "max_skipped_sources"),
    ({"require_modalities": ["x"]}, "require_modalities"),
    ({"require_verified": True}, "require_verified"),
    ({"min_shards": 1}, "min_shards"),
])
def test_requested_quality_gate_fails_when_evidence_is_unavailable(expectation, check_name):
    result = evaluate_quality({}, expectation)
    assert result.passed is False
    assert check_name in {check["name"] for check in result.checks}
    assert all("unavailable" in check["message"] for check in result.checks)
