"""Stateful analyzer discovery, construction, validation, and CLI coverage."""
from __future__ import annotations

import json
import sys

import pytest

from kuration import (
    DatasetAnalyzer,
    Registry,
    SplitLeakageAnalyzer,
    describe_component,
    get_analyzer,
    list_components,
    register_analyzer,
    run_dataset_analyzer,
)


PARAMETERS = {
    "type": "object",
    "properties": {"threshold": {"type": "number", "minimum": 0}},
    "required": ["threshold"],
    "additionalProperties": False,
}


class _CounterAnalyzer:
    def __init__(self, threshold):
        self.threshold = threshold
        self.values = []

    def add(self, value):
        self.values.append(value)

    def summary(self):
        return {"above": sum(value > self.threshold for value in self.values)}


@register_analyzer(
    "test/counter-analyzer",
    description="Accumulate values across calls",
    parameters=PARAMETERS,
    version="1",
)
def _counter_analyzer(*, threshold):
    return _CounterAnalyzer(threshold)


def test_analyzer_is_discoverable_and_constructed_as_user_owned_state():
    rows = list_components("analyzer")
    assert "test/counter-analyzer" in {row["name"] for row in rows}
    assert describe_component("analyzer", "test/counter-analyzer")["version"] == "1"

    analyzer = get_analyzer("test/counter-analyzer", threshold=1.0)
    analyzer.add(0.5)
    analyzer.add(2.0)
    assert analyzer.summary() == {"above": 1}


def test_analyzer_parameters_are_eagerly_validated():
    with pytest.raises(ValueError, match="missing parameter"):
        get_analyzer("test/counter-analyzer")
    with pytest.raises(ValueError, match="must be >= 0"):
        get_analyzer("test/counter-analyzer", threshold=-1)
    with pytest.raises(ValueError, match="unknown parameter"):
        get_analyzer("test/counter-analyzer", threshold=1, typo=True)


def test_analyzer_factory_contract_is_validated():
    registry = Registry()
    with pytest.raises(ValueError, match="callable factory"):
        registry.register_analyzer("test/not-callable", object())
    registry.register_analyzer(
        "test/no-summary",
        lambda: object(),
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )
    with pytest.raises(ValueError, match="callable summary"):
        get_analyzer("test/no-summary", registry=registry)


def test_analyzer_is_available_through_discovery_cli(monkeypatch, capsys):
    from kuration.core.cli.discovery import list_main

    monkeypatch.setattr(sys, "argv", ["kur-list", "analyzers", "--json"])
    list_main()
    rows = json.loads(capsys.readouterr().out)
    assert "test/counter-analyzer" in {row["name"] for row in rows}


def test_split_leakage_analyzer_is_a_discoverable_builtin():
    descriptor = describe_component("analyzer", "kuration/split-leakage")
    assert descriptor["capabilities"] == {"cross_segment": True, "stateful": True}

    analyzer = get_analyzer("kuration/split-leakage")
    assert isinstance(analyzer, SplitLeakageAnalyzer)
    assert analyzer.summary() == {
        "passed": True,
        "check_groups": True,
        "observations": 0,
        "unique_assignments": 0,
        "sources": 0,
        "groups": 0,
        "splits": [],
        "source_leaks": [],
        "group_leaks": [],
        "source_group_conflicts": [],
    }


def test_split_leakage_allows_repeated_segments_from_one_assignment():
    analyzer = SplitLeakageAnalyzer()
    analyzer.add_assignment("clip-1", "train", group="session-1")
    analyzer.add_assignment("clip-1", "train", group="session-1")

    summary = analyzer.summary()
    assert summary["passed"] is True
    assert summary["observations"] == 2
    assert summary["unique_assignments"] == 1


def test_split_leakage_reports_sources_and_groups_crossing_splits():
    analyzer = SplitLeakageAnalyzer()
    analyzer.add_assignment("clip-1", "train", group="session-1")
    analyzer.add_assignment("clip-1", "test", group="session-1")
    analyzer.add_assignment("clip-2", "test", group="session-2")
    analyzer.add_assignment("clip-3", "train", group="session-2")

    summary = analyzer.summary()
    assert summary["passed"] is False
    assert summary["source_leaks"] == [
        {"source_id": "clip-1", "splits": ["test", "train"]},
    ]
    assert summary["group_leaks"] == [
        {"group": "session-1", "splits": ["test", "train"], "sources": ["clip-1"]},
        {
            "group": "session-2",
            "splits": ["test", "train"],
            "sources": ["clip-2", "clip-3"],
        },
    ]
    json.dumps(summary)


def test_split_leakage_reports_one_source_claiming_multiple_groups():
    analyzer = SplitLeakageAnalyzer()
    analyzer.add_assignment("clip-1", "train", group="session-2")
    analyzer.add_assignment("clip-1", "train", group="session-1")

    assert analyzer.summary()["source_group_conflicts"] == [
        {"source_id": "clip-1", "groups": ["session-1", "session-2"]},
    ]


def test_split_leakage_can_disable_group_checks_but_not_source_checks():
    analyzer = get_analyzer("kuration/split-leakage", check_groups=False)
    analyzer.add_assignment("clip-1", "train", group="session-1")
    analyzer.add_assignment("clip-2", "test", group="session-1")
    assert analyzer.summary()["passed"] is True
    assert analyzer.summary()["group_leaks"] == []

    analyzer.add_assignment("clip-1", "test", group="session-1")
    assert analyzer.summary()["passed"] is False
    assert analyzer.summary()["source_leaks"]


def test_split_leakage_accepts_sdk_style_assignment_mappings():
    analyzer = SplitLeakageAnalyzer()
    analyzer.add_assignments(
        {"clip-2": "test", "clip-1": "train"},
        source_groups={"clip-1": "session-1", "clip-2": "session-2"},
    )
    assert analyzer.summary()["splits"] == ["test", "train"]
    assert analyzer.summary()["passed"] is True

    with pytest.raises(ValueError, match="absent from source_splits"):
        analyzer.add_assignments(
            {"clip-1": "train"}, source_groups={"unknown": "session-3"}
        )


def test_dataset_analyzer_contract_runs_split_leakage_manifest(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "config.json").write_text(json.dumps({"output": {"container": "npz"}}))
    (dataset / "dataset_manifest.json").write_text(json.dumps({
        "shards": [
            {"path": "train/a.npz", "source_id": "clip-1", "split": "train"},
            {"path": "test/b.npz", "source_id": "clip-1", "split": "test"},
        ]
    }))

    analyzer = SplitLeakageAnalyzer()
    assert isinstance(analyzer, DatasetAnalyzer)
    result = run_dataset_analyzer("kuration/split-leakage", dataset)

    assert result["passed"] is False
    assert result["name"] == "kuration/split-leakage"
    assert result["summary"]["source_leaks"] == [
        {"source_id": "clip-1", "splits": ["test", "train"]},
    ]


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        (("", "train"), {}, "source_id"),
        (("clip-1", ""), {}, "split"),
        (("clip-1", "train"), {"group": ""}, "group"),
        (("clip-1", "train"), {"group": 1}, "group"),
    ],
)
def test_split_leakage_rejects_invalid_assignments(args, kwargs, message):
    with pytest.raises(ValueError, match=message):
        SplitLeakageAnalyzer().add_assignment(*args, **kwargs)
