"""Named-metric registry — the seam that makes curation malleable.

Every per-segment scalar metric is registered under a stable name; a strategy then
references metrics *by name* (which to bin on, which to weight). Adding a curation
hypothesis is therefore: define a metric (DSL spec, or a Python escape-hatch fn), then
reference it in a JSON strategy — with no edits to the selection machinery.

Unlike the CS2-specific original, there is a single metric kind here: a function over a
`WindowBuffer`. DSL-compiled metrics and hand-written Python metrics share that signature,
so the registry treats them interchangeably.
"""
from __future__ import annotations

from typing import Callable

# A metric maps one segment's extracted window to a scalar. (Typed as `object` to avoid a
# buffer<->registry import cycle; it is always a kuration.core.buffer.WindowBuffer.)
WindowMetric = Callable[[object], float]

METRICS: dict[str, WindowMetric] = {}


def register_metric(name: str, **metadata) -> Callable[[WindowMetric], WindowMetric]:
    """Decorator: register a hand-written Python metric (the escape hatch for anything the
    closed grammar can't express). Duplicate names are a programmer error — fail loudly."""
    def deco(fn: WindowMetric) -> WindowMetric:
        if name in METRICS:
            raise ValueError(f"metric {name!r} already registered")
        METRICS[name] = fn
        from ..registry import default_registry
        default_registry.register("metric", name, fn, replace=True, **metadata)
        return fn
    return deco


def register_compiled(name: str, fn: WindowMetric) -> None:
    """Register a DSL-compiled metric. Overwrites on re-registration: a strategy's `define`
    block recompiles to an equivalent fn and workers re-register per run, so last-wins is
    correct (not a collision)."""
    METRICS[name] = fn
    from ..registry import default_registry
    default_registry.register("metric", name, fn, replace=True)
