"""Regression tests for correctness issues found by the core SDK audit."""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from kuration import Segment, SegmentBatch, Tick, default_registry, external_top_k
from kuration.core.adapters.base import get_adapter, validate_adapter_availability
from kuration.core.bounded import _passes as bounded_passes
from kuration.core.schema import SchemaSpec
from kuration.core.select import enforce_group_cap
from kuration.core.strategy import _passes as in_memory_passes
from kuration.core.transforms import _field_window, _stack_field_windows


def test_external_zscore_retains_low_variance_feature_with_large_offset(tmp_path):
    pytest.importorskip("pyarrow")
    values = 1e12 + np.arange(40, dtype=np.float64) / 100
    segments = [
        Segment("source", index, index + 1, (index, index), {"x": float(value)})
        for index, value in enumerate(values)
    ]
    paths = []
    for offset in range(0, len(segments), 7):
        path = tmp_path / f"part-{offset}.arrow"
        SegmentBatch.from_segments(segments[offset:offset + 7]).write_ipc(str(path))
        paths.append(str(path))

    selected = external_top_k(
        paths, weights={"x": 1.0}, target_size=6, normalize="zscore",
        spill_dir=str(tmp_path / "spill"))

    assert [segment.start_idx for segment in selected] == list(range(39, 33, -1))


def test_bounded_and_in_memory_predicates_share_missing_feature_policy():
    segment = Segment("source", 0, 1, (0, 0), {})
    predicates = [{"metric": "absent", "op": ">", "value": -1}]

    for evaluator in (in_memory_passes, bounded_passes):
        with pytest.raises(ValueError, match="has no feature 'absent'"):
            evaluator(segment, predicates)


def test_transform_field_windows_are_nan_padded_across_changing_rosters():
    schema = SchemaSpec.from_dict({
        "name": "transform-roster", "entity_axis": "entity",
        "fields": {"position": {"kind": "vector", "dims": ["x", "y"]}},
    })
    ticks = [
        Tick(0, {"a": {"position": [1, 2]}}),
        Tick(1, {"a": {"position": [3, 4]}, "b": {"position": [5, 6]}}),
        Tick(2, {"b": {"position": [7, 8]}}),
    ]

    values = _field_window(ticks, "position", schema)

    assert values.shape == (3, 2, 2)
    assert values[0, 0].tolist() == [1, 2]
    assert np.isnan(values[0, 1]).all()
    assert np.isnan(values[2, 0]).all()
    assert values[2, 1].tolist() == [7, 8]

    other = _field_window(ticks[:1], "position", schema)
    stacked = _stack_field_windows([values, other])
    assert stacked.shape == (2, 3, 2, 2)
    assert np.isnan(stacked[1, 1:]).all()
    assert np.isnan(stacked[1, :, 1]).all()


def test_curate_forwards_workers_to_filters_and_selection(monkeypatch):
    curate_module = importlib.import_module("kuration.core.curate")
    methods_module = importlib.import_module("kuration.core.methods")
    segment = Segment("source", 0, 1, (0, 0), {"x": 1.0})
    observed = {}

    monkeypatch.setattr(
        curate_module, "score_sources",
        lambda **kwargs: ([segment], [], {"source": ((0, 1),)}))

    def apply_filters(segments, strategy, *, seed, workers):
        observed["filter_workers"] = workers
        return segments

    def select(segments, strategy, *, target_size, tag_vocab, seed, workers):
        observed["select_workers"] = workers
        return segments[:target_size]

    monkeypatch.setattr(methods_module, "apply_filters", apply_filters)
    monkeypatch.setattr(curate_module, "select", select)
    strategy = SimpleNamespace(
        define={}, filters=[{"type": "test"}], axes=[], scorers=[],
        tags=None, registry=None,
        referenced_metrics=lambda: set(),
        scorer_outputs=lambda: set(),
    )
    schema = SchemaSpec.from_dict({
        "name": "workers", "fields": {"x": {"kind": "scalar"}},
    })

    result = curate_module.curate(
        schema=schema, adapter=SimpleNamespace(), sources=["source"],
        strategy=strategy, segmenter=SimpleNamespace(), target_size=1, workers=7)

    assert result.selected == [segment]
    assert observed == {"filter_workers": 7, "select_workers": 7}


def test_default_registry_is_authoritative_for_adapter_lookup():
    name = "tests/audit-direct-adapter"
    factory = lambda **kwargs: ("adapter", kwargs)  # noqa: E731
    default_registry.register_adapter(name, factory)
    try:
        assert get_adapter(name, answer=42) == ("adapter", {"answer": 42})
    finally:
        default_registry._entries["adapter"].pop(name, None)


def test_adapter_preflight_parses_filter_values_as_values_not_field_names():
    schema = SchemaSpec.from_dict({
        "name": "preflight-parser", "entity_axis": "entity",
        "fields": {
            "category": {"kind": "categorical"},
            "value_named_like_a_field": {"kind": "scalar"},
        },
    })
    adapter = SimpleNamespace(
        available_fields={"category"}, available_globals=set())

    validate_adapter_availability(
        adapter, schema,
        define={"metric": {
            "source": "entity.category", "reduce": "count",
            "filter": "category==value_named_like_a_field",
        }})


def test_group_cap_does_not_floor_below_a_fractional_fair_share():
    segments = [
        Segment("source", index, index + 1, (index, index), group=group)
        for index, group in enumerate(("a", "a", "a", "b", "b"))
    ]

    capped = enforce_group_cap(segments, max_group_overshoot=1.0)

    assert len(capped) == 5
