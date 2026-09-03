"""Regression coverage for plugin usability, isolation, provenance, and bounded transforms."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from kuration import Registry, SchemaSpec, Segment, Tick, default_registry, input, load_plugins
from kuration.core.registry import validate_parameters
from kuration.core.transforms import apply_transforms


SCHEMA = SchemaSpec.from_dict({
    "name": "review-findings", "entity_axis": "entity", "sample_hz": 1.0,
    "fields": {"x": {"kind": "scalar"}},
})


def test_transform_reread_is_interleaved_with_bounded_execution():
    events = []

    class Adapter:
        schema = SCHEMA

        def ticks(self, source):
            events.append(f"read:{source}")
            return [Tick(index, {0: {"x": float(index)}}) for index in range(4)]

    class Transform:
        def transform_batch(self, batch, ctx):
            events.append("transform")
            return {"out": np.arange(len(batch["segment_ids"]), dtype=np.float64)}

    registry = Registry()
    registry.register_transform("test/bounded", Transform)
    calls = [{"name": "test/bounded", "implementation": "test/bounded", "outputs": ["out"],
              "inputs": ["field:x"], "batch_size": 1, "device": None, "params": {}}]
    segments = [Segment(source, 0, 4, (0, 3)) for source in ("a", "b")]

    apply_transforms(calls, segments, adapter=Adapter(), registry=registry)

    assert events == ["read:a", "transform", "read:b", "transform"]
    assert all(segment.features["out"] == 0.0 for segment in segments)


def test_plugin_provenance_tracks_registered_implementation_modules(tmp_path, monkeypatch):
    package = tmp_path / "review_plugin"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "implementation.py").write_text(
        "def metric(window):\n    return float(len(window))\n")
    (package / "plugin.py").write_text(
        "from kuration import register_metric\n"
        "from .implementation import metric\n"
        "register_metric('review/provenance')(metric)\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    from kuration.core import plugins as plugin_state
    try:
        load_plugins(["review_plugin.plugin"], include_entry_points=False)
        assert "review_plugin.implementation" in plugin_state.loaded_plugin_modules()
        provenance = {entry["module"]: entry for entry in plugin_state.plugin_provenance()}
        original_hash = provenance["review_plugin.implementation"]["sha256"]
        (package / "implementation.py").write_text(
            "def metric(window):\n    return float(len(window)) + 1.0\n")
        changed = {entry["module"]: entry for entry in plugin_state.plugin_provenance()}
        assert changed["review_plugin.implementation"]["sha256"] != original_hash
    finally:
        from kuration.core.dsl.registry import METRICS
        METRICS.pop("review/provenance", None)
        default_registry._entries["metric"].pop("review/provenance", None)
        for name in ("review_plugin.plugin", "review_plugin.implementation", "review_plugin"):
            sys.modules.pop(name, None)
            plugin_state._LOADED_MODULES.discard(name)


def test_job_can_disable_ambient_entry_point_discovery(monkeypatch):
    import kuration.core.plugins as plugin_state

    class BrokenEntryPoint:
        name, value = "broken", "broken_plugin:register"

        def load(self):
            raise AssertionError("ambient entry point must not be loaded")

    class EntryPoints(list):
        def select(self, **kwargs):
            return self if kwargs == {"group": "kuration.plugins"} else []

    old_scanned = plugin_state._ENTRY_POINTS_SCANNED
    plugin_state._ENTRY_POINTS_SCANNED = False
    monkeypatch.setattr(plugin_state, "entry_points", lambda: EntryPoints([BrokenEntryPoint()]))
    job = {
        "include_plugin_entry_points": False,
        "schema": SCHEMA.to_dict(),
        "adapter": {"type": "timeseries", "fields": {"x": "x"}},
        "sources": ["not-read.csv"],
        "segmenter": {"window": 2, "stride": 2},
        "strategy": {
            "define": {"xm": {"source": "entity.x", "reduce": "mean"}},
            "selection": {"method": "top-k", "weights": {"xm": 1.0}},
        },
    }
    try:
        from kuration.core.cli.validate import validate_job
        assert validate_job(job)["kind"] == "report"
    finally:
        plugin_state._ENTRY_POINTS_SCANNED = old_scanned


def test_job_excludes_entry_points_discovered_earlier(monkeypatch):
    import kuration.core.plugins as plugin_state

    def ambient_metric(window):
        return float(len(window))

    def initialize():
        default_registry.register_metric("review/ambient", ambient_metric)

    class AmbientEntryPoint:
        name, value = "ambient", "review_ambient:initialize"

        def load(self):
            return initialize

    class EntryPoints(list):
        def select(self, **kwargs):
            return self if kwargs == {"group": "kuration.plugins"} else []

    key = "ambient=review_ambient:initialize"
    component_key = ("metric", "review/ambient")
    old_scanned = plugin_state._ENTRY_POINTS_SCANNED
    old_baseline = plugin_state._ENTRY_POINT_BASELINES.get(component_key, ...)
    old_components = plugin_state._MODULE_COMPONENTS.get(initialize.__module__)
    module_was_loaded = initialize.__module__ in plugin_state._LOADED_MODULES
    plugin_state._ENTRY_POINTS_SCANNED = False
    plugin_state._LOADED_ENTRY_POINTS.discard(key)
    monkeypatch.setattr(plugin_state, "entry_points",
                        lambda: EntryPoints([AmbientEntryPoint()]))
    try:
        load_plugins()
        assert "review/ambient" in default_registry.names("metric")
        from kuration.core.cli.job import resolve_job
        resolved = resolve_job({
            "include_plugin_entry_points": False,
            "schema": SCHEMA.to_dict(),
            "adapter": {"type": "timeseries", "fields": {"x": "x"}},
            "sources": ["not-read.csv"],
            "segmenter": {"window": 2, "stride": 2},
            "strategy": {
                "define": {"xm": {"source": "entity.x", "reduce": "mean"}},
                "selection": {"method": "top-k", "weights": {"xm": 1.0}},
            },
        }, enumerate_sources=False)
        assert "review/ambient" not in resolved.registry.names("metric")
    finally:
        default_registry._entries["metric"].pop("review/ambient", None)
        plugin_state._LOADED_ENTRY_POINTS.discard(key)
        plugin_state._ENTRY_POINTS_SCANNED = old_scanned
        if old_baseline is ...:
            plugin_state._ENTRY_POINT_BASELINES.pop(component_key, None)
        else:
            plugin_state._ENTRY_POINT_BASELINES[component_key] = old_baseline
        if old_components is None:
            plugin_state._MODULE_COMPONENTS.pop(initialize.__module__, None)
        else:
            plugin_state._MODULE_COMPONENTS[initialize.__module__] = old_components
        if not module_was_loaded:
            plugin_state._LOADED_MODULES.discard(initialize.__module__)


def test_input_resolves_named_adapter_from_isolated_registry():
    class Adapter:
        in_memory = True

        def __init__(self, schema):
            self.schema = schema

        def sources(self):
            return ["memory"]

        def ticks(self, source):
            return [Tick(0, {0: {"x": 1.0}})]

    registry = Registry(parent=default_registry)
    registry.register_adapter("review/adapter", Adapter)
    pipeline = input(adapter="review/adapter", schema=SCHEMA, registry=registry)

    assert type(pipeline.adapter) is Adapter
    assert pipeline._registry is not registry
    assert pipeline._registry.get("adapter", "review/adapter") is Adapter


def test_component_parameter_schema_validates_nested_arrays_and_objects():
    registry = Registry()
    registry.register_operator("review/schema-shapes", lambda pipeline: pipeline, parameters={
        "type": "object",
        "properties": {
            "controls": {"type": "array", "minItems": 1,
                         "items": {"type": "string", "pattern": "^[A-Z]+$"}},
            "config": {"type": "object", "required": ["threshold"],
                       "properties": {"threshold": {"type": "number", "exclusiveMinimum": 0}},
                       "additionalProperties": False},
        },
        "required": ["controls"],
        "additionalProperties": False,
    })
    descriptor = registry.descriptor("operator", "review/schema-shapes")

    validate_parameters(descriptor, {"controls": ["W"], "config": {"threshold": 0.5}})
    with pytest.raises(ValueError, match=r"controls.*\[0\].*JSON type string"):
        validate_parameters(descriptor, {"controls": [1]})
    with pytest.raises(ValueError, match="must match pattern"):
        validate_parameters(descriptor, {"controls": ["mouse"]})
    with pytest.raises(ValueError, match="missing parameter"):
        validate_parameters(descriptor, {"controls": ["W"], "config": {}})
