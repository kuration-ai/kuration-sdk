from __future__ import annotations

import json

import numpy as np
import pytest

from kuration import ComponentDescriptor, Registry, SchemaSpec, Segment, default_registry
from kuration.core.cli.explain import explain_job
from kuration.core.dsl import compile_metric
from kuration.core.infer import sample_source
from kuration.core.plugin_validation import check_plugins
from kuration.core.registry import validate_parameters
from kuration.core.report import dataset_report
from kuration.core.review import _json_value
from kuration.core.strategy import load_strategy
from kuration.core.versioning import migrate


def test_migrate_recognizes_an_unversioned_job_and_rejects_dual_versions():
    job = {
        "schema": {"name": "demo", "fields": {"x": {"kind": "scalar"}}},
        "adapter": {"type": "timeseries", "fields": {"x": "x"}},
        "segmenter": {"window": 2, "stride": 2},
        "strategy": {"selection": {"method": "top-k", "weights": {"x": 1}}},
    }
    migrated = migrate(job)
    assert migrated["job_schema_version"] == "1.0"
    assert "artifact_format_version" not in migrated

    with pytest.raises(ValueError, match="both job_schema_version and artifact_format_version"):
        migrate({"job_schema_version": "1.0", "artifact_format_version": "1.0"})


def test_action_residual_descriptor_and_strategy_validate_parameters_eagerly():
    descriptor = default_registry.descriptor("scorer", "action_residual")
    schema = descriptor.to_dict()["parameters"]
    assert schema["required"] == ["obs", "action"]
    assert schema["properties"]["k"]["minimum"] == 1

    data_schema = SchemaSpec.from_dict({
        "name": "scorer", "entity_axis": "entity",
        "fields": {"obs": {"kind": "scalar"}, "action": {"kind": "scalar"}},
    })
    registry = default_registry.snapshot()

    def strategy(scorer):
        return {"scorers": [scorer],
                "selection": {"method": "top-k", "weights": {"residual": 1.0}}}

    with pytest.raises(ValueError, match="missing parameter"):
        load_strategy(strategy({"name": "action_residual", "as": "residual"}),
                      data_schema, registry=registry)
    with pytest.raises(ValueError, match="unknown parameter.*typo"):
        load_strategy(strategy({"name": "action_residual", "as": "residual",
                                "obs": "obs", "action": "action", "typo": 1}),
                      data_schema, registry=registry)
    loaded = load_strategy(strategy({"name": "action_residual", "as": "residual",
                                     "obs": "obs", "action": "action", "k": 2}),
                           data_schema, registry=registry)
    assert loaded.scorers == [("action_residual", "residual",
                               {"obs": "obs", "action": "action", "k": 2})]


def test_event_vector_unique_reductions_fail_at_compile_time():
    schema = SchemaSpec.from_dict({
        "name": "events", "fields": {"x": {"kind": "scalar"}},
        "event_streams": {"events": {"fields": {
            "direction": {"kind": "vector", "dims": ["x", "y"]},
        }}},
    })
    compile_metric({"source": "events.direction", "reduce": "count"}, schema)
    compile_metric({"source": "events.direction", "reduce": "rate"}, schema)
    for reduction in ("nunique", "entropy"):
        with pytest.raises(ValueError, match="vector"):
            compile_metric({"source": "events.direction", "reduce": reduction}, schema)


def test_registry_preserves_open_schema_and_rejects_invalid_nested_rules():
    registry = Registry()
    registry.register_metric("review/open", lambda _window: 1.0, parameters={})
    descriptor = registry.descriptor("metric", "review/open")
    assert descriptor.to_dict()["parameters"] == {}
    validate_parameters(descriptor, {"arbitrary": {"nested": True}})

    with pytest.raises(ValueError, match="invalid regex pattern"):
        ComponentDescriptor("review/regex", "metric", parameters={
            "type": "object",
            "properties": {"value": {"type": "string", "pattern": "["}},
        })
    with pytest.raises(ValueError, match="unsupported keyword"):
        ComponentDescriptor("review/unknown-schema", "metric", parameters={
            "type": "object", "properties": {"value": {"format": "email"}},
        })


def test_parameter_enum_const_and_uniqueness_use_json_type_equality():
    descriptor = ComponentDescriptor("review/json-equality", "metric", parameters={
        "type": "object",
        "properties": {
            "enum_value": {"enum": [True]},
            "const_value": {"const": False},
            "unique_values": {"type": "array", "uniqueItems": True},
        },
        "required": ["enum_value", "const_value", "unique_values"],
        "additionalProperties": False,
    })

    # JSON booleans and numbers are distinct, while integer and real representations of the
    # same JSON number compare equal.
    validate_parameters(descriptor, {
        "enum_value": True, "const_value": False, "unique_values": [True, 1],
    })
    with pytest.raises(ValueError, match="must be one of"):
        validate_parameters(descriptor, {
            "enum_value": 1, "const_value": False, "unique_values": [],
        })
    with pytest.raises(ValueError, match="must equal False"):
        validate_parameters(descriptor, {
            "enum_value": True, "const_value": 0, "unique_values": [],
        })
    with pytest.raises(ValueError, match="must contain unique items"):
        validate_parameters(descriptor, {
            "enum_value": True, "const_value": False, "unique_values": [1, 1.0],
        })


def test_plugin_loader_rejects_non_module_non_callable_entry_point(monkeypatch):
    import kuration.core.plugins as plugin_state

    class FakeEntryPoint:
        name = "review-invalid"
        value = "review_invalid:target"

        def load(self):
            return 42

    class FakeEntryPoints(list):
        def select(self, **kwargs):
            return self if kwargs == {"group": "kuration.plugins"} else []

    snapshot = plugin_state._plugin_state_snapshot()
    scanned = plugin_state._ENTRY_POINTS_SCANNED
    plugin_state._ENTRY_POINTS_SCANNED = False
    monkeypatch.setattr(plugin_state, "entry_points",
                        lambda: FakeEntryPoints([FakeEntryPoint()]))
    try:
        with pytest.warns(RuntimeWarning, match="module or zero-argument initializer"):
            [result] = plugin_state.load_plugins()
        assert result["kind"] == "error"
        assert result["error_type"] == "builtins.TypeError"
        assert plugin_state.plugin_failures()[0]["name"] == "review-invalid"
    finally:
        plugin_state._restore_plugin_state(snapshot, None)
        plugin_state._ENTRY_POINTS_SCANNED = scanned


def test_plugin_loader_rejects_noop_entry_point_with_persistent_diagnostic(monkeypatch):
    import kuration.core.plugins as plugin_state

    def initialize():
        pass

    initialize.__module__ = "review_noop_plugin"

    class FakeEntryPoint:
        name = "review-noop"
        value = "review_noop_plugin:initialize"

        def load(self):
            return initialize

    class FakeEntryPoints(list):
        def select(self, **kwargs):
            return self if kwargs == {"group": "kuration.plugins"} else []

    snapshot = plugin_state._plugin_state_snapshot()
    scanned = plugin_state._ENTRY_POINTS_SCANNED
    plugin_state._ENTRY_POINTS_SCANNED = False
    monkeypatch.setattr(plugin_state, "entry_points",
                        lambda: FakeEntryPoints([FakeEntryPoint()]))
    try:
        with pytest.warns(RuntimeWarning, match="registered no components"):
            [result] = plugin_state.load_plugins()
        assert result["kind"] == "error"
        assert result["error_type"] == "builtins.ValueError"
        assert plugin_state.plugin_failures()[0]["name"] == "review-noop"
    finally:
        plugin_state._restore_plugin_state(snapshot, None)
        plugin_state._ENTRY_POINTS_SCANNED = scanned


def test_plugin_loader_accepts_preimported_entry_point_with_owned_component(
        tmp_path, monkeypatch):
    import importlib
    import kuration.core.plugins as plugin_state

    module_name = "review_preimported_plugin"
    package = tmp_path / module_name
    package.mkdir()
    (package / "implementation.py").write_text(
        "def metric(window): return 1.0\n"
    )
    (package / "__init__.py").write_text(
        "from kuration import default_registry\n"
        "from .implementation import metric\n"
        "default_registry.register_metric('review/preimported', metric)\n"
        "def initialize(): pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    class FakeEntryPoint:
        name = "review-preimported"
        value = f"{module_name}:initialize"

        def load(self):
            return importlib.import_module(module_name).initialize

    class FakeEntryPoints(list):
        def select(self, **kwargs):
            return self if kwargs == {"group": "kuration.plugins"} else []

    snapshot = plugin_state._plugin_state_snapshot()
    scanned = plugin_state._ENTRY_POINTS_SCANNED
    try:
        # Importing before discovery means the initializer makes no registry change. The
        # implementation lives in a package submodule, but ownership still makes this a valid
        # plugin rather than a no-op.
        importlib.import_module(module_name)
        plugin_state._ENTRY_POINTS_SCANNED = False
        monkeypatch.setattr(plugin_state, "entry_points",
                            lambda: FakeEntryPoints([FakeEntryPoint()]))
        assert plugin_state.load_plugins() == [{
            "kind": "entry_point", "name": "review-preimported",
            "value": f"{module_name}:initialize",
        }]
    finally:
        plugin_state._restore_plugin_state(snapshot, module_name)
        plugin_state._ENTRY_POINTS_SCANNED = scanned


def test_plugin_checker_rejects_a_malformed_nested_parameter_schema(tmp_path, monkeypatch):
    module = "review_bad_parameter_schema"
    (tmp_path / f"{module}.py").write_text(
        "from kuration import default_registry\n"
        "default_registry.register_metric(\n"
        "    'review/bad-schema', lambda window: 1.0,\n"
        "    description='bad nested schema',\n"
        "    parameters={'type': 'object', 'properties': {\n"
        "        'value': {'type': 'string', 'pattern': '['}},\n"
        "        'additionalProperties': False},\n"
        "    package='review-plugin', install_hint='pip install review-plugin', version='1')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    [result] = check_plugins(module)
    assert not result.passed
    assert any("invalid regex pattern" in error for error in result.errors)


def test_video_sample_source_has_an_actionable_error():
    with pytest.raises(ValueError, match=r"does not decode video.*VideoAdapter"):
        sample_source("missing.mp4")


def test_dataset_report_keeps_group_types_distinct():
    segments = [
        Segment("a", 0, 1, (0, 1), group=1),
        Segment("b", 0, 1, (0, 1), group="1"),
    ]
    assert dataset_report(segments)["groups"] == {"int:1": 1, "str:1": 1}


def test_strict_explain_uses_resolved_raise_policy():
    job = {
        "mode": "strict",
        "schema": {"name": "strict", "fields": {"x": {"kind": "scalar"}}},
        "adapter": {"type": "timeseries", "fields": {"x": "x"}},
        "sources": ["not-read.csv"],
        "segmenter": {"window": 2, "stride": 2},
        "strategy": {
            "define": {"mean_x": {"source": "x", "reduce": "mean"}},
            "selection": {"method": "top-k", "weights": {"mean_x": 1.0}},
        },
    }
    plan = explain_job(job)
    assert plan["execution"]["on_error"] == "raise"
    assert not any(item["code"] == "source-errors-skipped" for item in plan["warnings"])


def test_review_json_conversion_replaces_nested_nonfinite_numbers():
    converted = _json_value({
        "array": np.asarray([1.0, np.nan, np.inf, -np.inf]),
        "scalar": np.float64(np.nan),
    })
    assert converted == {"array": [1.0, None, None, None], "scalar": None}
    json.dumps(converted, allow_nan=False)
