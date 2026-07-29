"""Cross-subsystem regressions found while reviewing the combined remediation patch."""
from __future__ import annotations

from kuration import Segment, default_registry, input
from kuration.core.report import dataset_report
from kuration.core.versioning import ARTIFACT_FORMAT_VERSION, migrate


class _SameTextGroup:
    def __init__(self, value: int):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, _SameTextGroup) and self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return "same"


def test_callable_metric_overlay_still_discovers_ambient_plugin_metrics(monkeypatch):
    name = "tests/lazy-plugin-after-callable"
    default_registry._entries["metric"].pop(name, None)

    def load_plugin():
        if name not in default_registry.names("metric"):
            default_registry.register_metric(name, lambda window: float(window.T))

    monkeypatch.setattr("kuration.core.plugins.load_plugins", load_plugin)
    try:
        pipeline = (input([{"x": 1.0}, {"x": 2.0}])
                    .measure("local", lambda window: 1.0)
                    .measure(name))
        assert pipeline._registered_metrics == [name]
    finally:
        default_registry._entries["metric"].pop(name, None)


def test_unversioned_resolved_pipeline_config_is_migrated_as_an_artifact():
    config = {
        "schema": {"name": "demo", "fields": {"x": {"kind": "scalar"}}},
        "strategy": {"selection": {"method": "top-k", "weights": {"x": 1}}},
        "output": {"container": "npz"},
        "results": {"selected": 1},
        "pipeline": [{"op": "sample"}],
        "output_request": {"preset": "training"},
        "output_preset": {"name": "training"},
        "mode": {"name": "strict"},
    }
    migrated = migrate(config)
    assert migrated["artifact_format_version"] == ARTIFACT_FORMAT_VERSION
    assert "job_schema_version" not in migrated


def test_dataset_report_separates_unequal_same_type_groups_with_the_same_text():
    segments = [
        Segment("a", 0, 1, (0, 1), group=_SameTextGroup(1)),
        Segment("b", 0, 1, (0, 1), group=_SameTextGroup(2)),
        Segment("c", 0, 1, (0, 1), group=_SameTextGroup(1)),
    ]
    groups = dataset_report(segments)["groups"]
    assert sorted(groups.values()) == [1, 2]
    assert len(groups) == 2
