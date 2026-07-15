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
from .tags import compile_tagger


class FeatureEngine:
    """Compile metrics and optional tag rules, then populate ``Segment`` objects.

    ``define`` contains closed-grammar metric specs and ``registered`` contains registered
    metric names. Unknown metrics, invalid specs, and duplicate names raise ``ValueError``.
    """

    def __init__(self, schema: SchemaSpec, *, define: Optional[dict] = None,
                 registered: Optional[list] = None, tags_spec: Optional[dict] = None):
        self.schema = schema
        self.fns = {}
        for name, spec in (define or {}).items():
            self.fns[name] = compile_metric(spec, schema)   # validates at construction (loud)
        for name in (registered or []):
            if name in self.fns:
                raise ValueError(f"metric {name!r} is both DSL-defined and registered — the "
                                 f"registered fn would silently shadow the define; rename one")
            if name not in METRICS:
                raise ValueError(f"unknown registered metric {name!r}; known: {sorted(METRICS)}")
            self.fns[name] = METRICS[name]
        self.tagger = compile_tagger(tags_spec, schema) if tags_spec else None

    def score_window(self, seg: Segment, window_ticks: list[Tick]) -> Segment:
        """Score one already-extracted window (the streaming entry point). Builds the buffer for
        just this window — no source-wide tick list needed."""
        w = build_window(self.schema, window_ticks)
        for name, fn in self.fns.items():
            seg.features[name] = fn(w)
        if self.tagger is not None:
            seg.tags = self.tagger(w)
        return seg

    def compute(self, ticks: list[Tick], segments: list[Segment]) -> list[Segment]:
        """Fill `seg.features` (and `seg.tags` if a tags spec was given) in place."""
        for seg in segments:
            self.score_window(seg, ticks[seg.start_idx:seg.end_idx])
        return segments
