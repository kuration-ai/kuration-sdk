from __future__ import annotations

from kuration import Registry, default_registry, describe_component, input
from kuration_plugin_template import balanced_window_lengths, register, window_length


def test_initializer_registers_a_described_metric():
    register()
    descriptor = describe_component(
        "metric", "template/window-length", registry=default_registry.snapshot())
    assert descriptor["version"] == "1"
    assert descriptor["stability"] == "experimental"
    assert descriptor["package"] == "kuration-plugin-template"
    assert descriptor["install_hint"] == "pip install kuration-plugin-template"
    assert descriptor["parameters"]["additionalProperties"] is False


def test_metric_can_be_used_in_an_isolated_registry():
    registry = Registry()
    registry.register_metric("template/window-length", window_length)
    assert registry.get("metric", "template/window-length") is window_length


def test_plugin_operator_is_a_direct_pipeline_method():
    register()
    pipeline = (
        input([{"x": float(i)} for i in range(8)])
        .window(4)
        .balanced_window_lengths(bins=2)
    )
    assert pipeline._ops[-1]["args"] == ["template/balanced-window-lengths"]


def test_operator_can_be_used_in_an_isolated_registry():
    registry = Registry()
    registry.register_metric("template/window-length", window_length)
    registry.register_operator("template/balanced-window-lengths", balanced_window_lengths)
    pipeline = input([{"x": float(i)} for i in range(8)], registry=registry).window(4)
    assert pipeline.balanced_window_lengths()._ops[-1]["op"] == "apply"
