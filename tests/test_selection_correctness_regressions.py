"""Focused regressions for selection semantics shared across execution paths."""
from __future__ import annotations

import importlib.util
import random

import numpy as np
import pytest

import kuration.core.experiment as experiment_module
from kuration import (
    Arm,
    SchemaSpec,
    Segment,
    SegmentBatch,
    external_uniform_bins,
    iter_external_uniform_priorities,
    load_strategy,
    run_experiment,
)
from kuration.core.binning import bin_segments
from kuration.core.output.external_fill import external_fill_to_hours
from kuration.core.output.fill import fill_to_hours
from kuration.core.select import (
    enforce_group_cap,
    enforce_tag_coverage,
    sample_uniform_bins,
)
from kuration.core.strategy import _weighted_score, segment_priorities, select


HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None


def test_weighted_scores_ignore_mapping_insertion_order_under_cancellation():
    segments = [
        Segment("zero", 0, 1, (0, 0), {"a": 0.0, "b": 0.0, "c": 0.0}),
        Segment("one", 0, 1, (0, 0), {"a": 1.0, "b": 1.0, "c": 1.0}),
    ]
    first = {"a": 1e16, "b": -1e16, "c": 1.0}
    reordered = {"a": 1e16, "c": 1.0, "b": -1e16}

    np.testing.assert_array_equal(
        _weighted_score(segments, first, "minmax"),
        _weighted_score(segments, reordered, "minmax"),
    )


def _tag_strategy(schema: SchemaSpec, name: str, source: str):
    return load_strategy(
        {
            "name": name,
            "define": {"score": {"source": "entity.value", "reduce": "mean"}},
            "tags": {"from": [source]},
            "selection": {
                "method": "top-k",
                "weights": {"score": 1.0},
                "tag_coverage": {"min_tag_frac": 0.5},
            },
        },
        schema,
    )


def test_experiment_passes_each_arm_only_its_own_tag_vocabulary(monkeypatch):
    schema = SchemaSpec.from_dict(
        {
            "name": "experiment-tags",
            "entity_axis": "entity",
            "fields": {
                "value": {"kind": "scalar"},
                "kind": {"kind": "categorical"},
                "region": {"kind": "categorical"},
            },
        }
    )
    arms = [
        Arm("kind", _tag_strategy(schema, "kind", "kind")),
        Arm("region", _tag_strategy(schema, "region", "region")),
    ]
    segments = [
        Segment(
            "source",
            0,
            1,
            (0, 0),
            {"score": 2.0},
            tags=frozenset({"kind:common", "region:north"}),
        ),
        Segment(
            "source",
            1,
            2,
            (1, 1),
            {"score": 1.0},
            tags=frozenset({"kind:rare", "region:south"}),
        ),
    ]

    def fake_score_sources(**kwargs):
        assert kwargs["tags_spec"] == {"from": ["kind", "region"]}
        return segments, [], {}

    vocabularies = {}

    def fake_select(pool, strategy, *, target_size, tag_vocab, seed, workers):
        vocabularies[strategy.name] = tuple(tag_vocab)
        return pool[:target_size]

    monkeypatch.setattr(experiment_module, "score_sources", fake_score_sources)
    monkeypatch.setattr(experiment_module, "select", fake_select)

    run_experiment(
        schema=schema,
        adapter=object(),
        sources=["source"],
        arms=arms,
        segmenter=object(),
        target_size=1,
    )

    assert vocabularies == {
        "kind": ("kind:common", "kind:rare"),
        "region": ("region:north", "region:south"),
    }


def test_zero_tag_fraction_disables_count_coverage():
    common = [
        Segment(
            f"common-{index}",
            0,
            1,
            (0, 0),
            bin=(index,),
            tags=frozenset({"common"}),
        )
        for index in range(2)
    ]
    rare = Segment("rare", 0, 1, (0, 0), bin=(2,), tags=frozenset({"rare"}))

    selected = enforce_tag_coverage(
        common,
        [*common, rare],
        tag_vocab=("common", "rare"),
        target_size=2,
        min_tag_frac=0,
    )

    assert selected == common


def test_filter_tag_coverage_cannot_restore_a_predicate_failure():
    schema = SchemaSpec.from_dict(
        {
            "name": "filter-coverage",
            "entity_axis": "entity",
            "fields": {
                "score": {"kind": "scalar"},
                "kind": {"kind": "categorical"},
            },
        }
    )
    strategy = load_strategy(
        {
            "tags": {"from": ["kind"]},
            "selection": {
                "method": "filter",
                "predicates": [{"metric": "score", "op": ">=", "value": 1}],
                "tag_coverage": {"min_tag_frac": 1.0},
            },
        },
        schema,
        available_metrics=("score",),
    )
    passing = Segment(
        "passing", 0, 1, (0, 0), {"score": 1.0}, tags=frozenset({"common"})
    )
    failing_rare = Segment(
        "failing", 0, 1, (0, 0), {"score": 0.0}, tags=frozenset({"rare"})
    )

    selected = select(
        [passing, failing_rare],
        strategy,
        target_size=1,
        tag_vocab=("common", "rare"),
    )

    assert selected == [passing]


def test_count_tag_fraction_rounds_up_as_a_minimum_floor():
    common = [
        Segment(
            f"common-{index}",
            0,
            1,
            (0, 0),
            bin=(index,),
            tags=frozenset({"common"}),
        )
        for index in range(5)
    ]
    rare = [
        Segment(
            f"rare-{index}",
            0,
            1,
            (0, 0),
            bin=(index + 5,),
            tags=frozenset({"rare"}),
        )
        for index in range(3)
    ]

    selected = enforce_tag_coverage(
        common,
        [*common, *rare],
        tag_vocab=("rare",),
        target_size=5,
        min_tag_frac=0.5,
    )

    assert sum("rare" in segment.tags for segment in selected) == 3


def test_zero_tag_fraction_disables_hours_reserve(tmp_path):
    items = [
        {
            "id": "common",
            "source_id": "common",
            "priority": 2.0,
            "seconds": 1.0,
            "group": None,
            "tags": {"common"},
        },
        {
            "id": "rare",
            "source_id": "rare",
            "priority": 1.0,
            "seconds": 1.0,
            "group": None,
            "tags": {"rare"},
        },
    ]
    kwargs = {
        "train_hours": 1 / 3600,
        "test_hours": 0,
        "tag_coverage": {"min_tag_frac": 0},
    }

    local = fill_to_hours(items, **kwargs)
    external = external_fill_to_hours(
        iter(items), spill_dir=str(tmp_path / "spill-zero-tags"), **kwargs
    )

    assert local[0] == external[0] == {"common"}


def test_count_group_cap_distinguishes_none_from_literal_sentinel():
    groups = [None, None, "\x00", "\x00", "other", "other"]
    selected = [
        Segment(f"source-{index}", 0, 1, (0, 0), bin=(index,), group=group)
        for index, group in enumerate(groups)
    ]

    assert enforce_group_cap(selected, max_group_overshoot=1.0) == selected


def test_hours_group_cap_distinguishes_none_from_literal_sentinel(tmp_path):
    items = [
        {
            "id": "none",
            "source_id": "none",
            "priority": 2.0,
            "seconds": 1.0,
            "group": None,
        },
        {
            "id": "literal",
            "source_id": "literal",
            "priority": 1.0,
            "seconds": 1.0,
            "group": "\x00",
        },
    ]
    kwargs = {
        "train_hours": 2 / 3600,
        "test_hours": 0,
        "max_group_overshoot": 1.0,
    }

    local = fill_to_hours(items, return_stats=True, **kwargs)
    external = external_fill_to_hours(
        iter(items), spill_dir=str(tmp_path / "spill-groups"), **kwargs
    )

    assert local[0] == external[0] == {"none", "literal"}
    assert local[2]["items_rejected_group_cap"] == 0
    assert external[2]["items_rejected_group_cap"] == 0


def test_hours_group_equality_matches_python_locally_and_externally(tmp_path):
    items = [
        {"id": "integer", "source_id": "a", "priority": 3.0, "seconds": 1.0, "group": 1},
        {"id": "boolean", "source_id": "b", "priority": 2.0, "seconds": 1.0, "group": True},
        {"id": "string", "source_id": "c", "priority": 1.0, "seconds": 1.0, "group": "1"},
    ]
    kwargs = {
        "train_hours": 3 / 3600,
        "test_hours": 0,
        "max_group_overshoot": 1.0,
    }

    local = fill_to_hours(items, return_stats=True, **kwargs)
    external = external_fill_to_hours(
        iter(items), spill_dir=str(tmp_path / "spill-equal-groups"), **kwargs
    )

    assert local[0] == external[0] == {"integer", "string"}
    assert local[2]["items_rejected_group_cap"] == 1
    assert external[2]["items_rejected_group_cap"] == 1


def test_hours_tag_equality_and_mixed_types_match_external_fill(tmp_path):
    items = [
        {
            "id": "integer",
            "source_id": "a",
            "priority": 3.0,
            "seconds": 1.0,
            "group": None,
            "tags": (1,),
        },
        {
            "id": "boolean",
            "source_id": "b",
            "priority": 1.0,
            "seconds": 1.0,
            "group": None,
            "tags": (True,),
        },
        {
            "id": "string",
            "source_id": "c",
            "priority": 2.0,
            "seconds": 1.0,
            "group": None,
            "tags": ("1",),
        },
    ]
    kwargs = {
        "train_hours": 2 / 3600,
        "test_hours": 0,
        "tag_coverage": {"min_tag_frac": 1.0},
    }

    local = fill_to_hours(items, **kwargs)
    external = external_fill_to_hours(
        iter(items), spill_dir=str(tmp_path / "spill-equal-tags"), **kwargs
    )

    assert local[0] == external[0] == {"integer", "string"}


def test_declarative_strategy_rejects_duplicate_bin_metrics():
    schema = SchemaSpec.from_dict(
        {
            "name": "duplicate-axes",
            "entity_axis": "entity",
            "fields": {"x": {"kind": "scalar"}},
        }
    )
    with pytest.raises(ValueError, match="duplicate bin axis"):
        load_strategy(
            {
                "binning": {
                    "axes": [
                        {"metric": "x", "bins": 2},
                        {"metric": "x", "bins": 4},
                    ]
                },
                "selection": {"method": "uniform-bins"},
            },
            schema,
            available_metrics=("x",),
        )


def _noncanonical_uniform_partitions(tmp_path):
    segments = [
        Segment("source", index, index + 1, (index, index), {"x": float(index)})
        for index in range(12)
    ]
    paths = []
    for offset in range(0, len(segments), 4):
        path = tmp_path / f"part-{offset // 4}.arrow"
        SegmentBatch.from_segments(segments[offset : offset + 4]).write_ipc(str(path))
        paths.append(str(path))
    return segments, list(reversed(paths))


@pytest.mark.skipif(not HAS_PYARROW, reason="external reducers need PyArrow")
def test_external_uniform_count_uses_canonical_bin_order_for_rng_parity(tmp_path):
    segments, paths = _noncanonical_uniform_partitions(tmp_path)
    reference = [
        Segment(
            segment.source_id,
            segment.start_idx,
            segment.end_idx,
            segment.time_range,
            dict(segment.features),
        )
        for segment in segments
    ]
    axes = [("x", 4)]
    bin_segments(reference, axes)
    expected = sample_uniform_bins(
        reference, target_size=8, rng=random.Random(37)
    )

    actual = external_uniform_bins(
        paths,
        axes=axes,
        target_size=8,
        seed=37,
        spill_dir=str(tmp_path / "spill-count"),
    )

    assert [(segment.start_idx, segment.bin) for segment in actual] == [
        (segment.start_idx, segment.bin) for segment in expected
    ]


@pytest.mark.skipif(not HAS_PYARROW, reason="external reducers need PyArrow")
def test_external_uniform_priorities_use_canonical_bin_order_for_rng_parity(tmp_path):
    segments, paths = _noncanonical_uniform_partitions(tmp_path)
    reference = [
        Segment(
            segment.source_id,
            segment.start_idx,
            segment.end_idx,
            segment.time_range,
            dict(segment.features),
        )
        for segment in segments
    ]
    schema = SchemaSpec.from_dict(
        {
            "name": "uniform-priorities",
            "entity_axis": "entity",
            "fields": {"x": {"kind": "scalar"}},
        }
    )
    strategy = load_strategy(
        {
            "binning": {"axes": [{"metric": "x", "bins": 4}]},
            "selection": {"method": "uniform-bins"},
        },
        schema,
        available_metrics=("x",),
    )
    bin_segments(reference, strategy.axes)
    expected = segment_priorities(reference, strategy, seed=37)

    actual = list(
        iter_external_uniform_priorities(
            paths,
            axes=strategy.axes,
            seed=37,
            spill_dir=str(tmp_path / "spill-priorities"),
        )
    )

    assert [segment.start_idx for segment, _priority in actual] == list(range(12))
    assert [priority for _segment, priority in actual] == pytest.approx(expected)
