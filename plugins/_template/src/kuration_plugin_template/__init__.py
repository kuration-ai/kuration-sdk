"""Minimal separately packaged Kuration plugin."""
from __future__ import annotations


def window_length(window) -> float:
    """Return the number of ticks in a segment window."""
    return float(len(window))


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
