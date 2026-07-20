from __future__ import annotations

from kuration import Registry, default_registry, describe_component
from kuration_plugin_template import register, window_length


def test_initializer_registers_a_described_metric():
    register()
    descriptor = describe_component(
        "metric", "template/window-length", registry=default_registry.snapshot())
    assert descriptor["version"] == "1"
    assert descriptor["stability"] == "experimental"


def test_metric_can_be_used_in_an_isolated_registry():
    registry = Registry()
    registry.register_metric("template/window-length", window_length)
    assert registry.get("metric", "template/window-length") is window_length
