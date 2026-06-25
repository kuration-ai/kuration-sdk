"""FeatureEngine — populate each segment's `features` dict, amortizing one buffer per segment.

A profile's metrics come from two places: inline DSL specs (`define`, compiled against the
schema via the closed grammar) and registered Python metrics (the escape hatch, by name).
Both are `fn(WindowBuffer) -> float`, so the engine builds one `WindowBuffer` per segment and
runs every metric against it — the extraction cost is paid once per segment, not once per
metric.
"""
from __future__ import annotations

from typing import Optional

from .buffer import Tick, build_window
from .dsl.grammar import compile_metric
from .dsl.registry import METRICS
from .schema import SchemaSpec
from .segment import Segment


class FeatureEngine:
    def __init__(self, schema: SchemaSpec, *, define: Optional[dict] = None,
                 registered: Optional[list] = None):
        self.schema = schema
        self.fns = {}
        for name, spec in (define or {}).items():
            self.fns[name] = compile_metric(spec, schema)   # validates at construction (loud)
        for name in (registered or []):
            if name not in METRICS:
                raise ValueError(f"unknown registered metric {name!r}; known: {sorted(METRICS)}")
            self.fns[name] = METRICS[name]

    def compute(self, ticks: list[Tick], segments: list[Segment]) -> list[Segment]:
        """Fill `seg.features` for each segment in place; returns the segments."""
        for seg in segments:
            w = build_window(self.schema, ticks[seg.start_idx:seg.end_idx])
            for name, fn in self.fns.items():
                seg.features[name] = fn(w)
        return segments
