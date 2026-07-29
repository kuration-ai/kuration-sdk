"""Minimal separately packaged Kuration plugin."""
from __future__ import annotations


def window_length(window) -> float:
    """Return the number of ticks in a segment window."""
    return float(len(window))


def balanced_window_lengths(pipeline, *, bins: int = 4):
    """Measure window lengths and balance selection across them."""
    return (
        pipeline.measure("template/window-length")
        .bin("template/window-length", bins=bins)
    )


def register() -> None:
    """Register this distribution's components with Kuration's default registry."""
    from kuration import default_registry

    name = "template/window-length"
    if name not in default_registry.names("metric"):
        default_registry.register_metric(
            name,
            window_length,
            description="Return the number of ticks in a segment window",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            package="kuration-plugin-template",
            install_hint="pip install kuration-plugin-template",
            version="1",
            stability="experimental",
        )
    operator_name = "template/balanced-window-lengths"
    if operator_name not in default_registry.names("operator"):
        default_registry.register_operator(
            operator_name,
            balanced_window_lengths,
            description="Balance selected segments across observed window lengths",
            parameters={
                "type": "object",
                "properties": {"bins": {"type": "integer", "minimum": 2}},
                "additionalProperties": False,
            },
            package="kuration-plugin-template",
            install_hint="pip install kuration-plugin-template",
            version="1",
            stability="experimental",
        )
