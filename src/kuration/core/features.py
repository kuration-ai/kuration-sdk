"""FeatureEngine — populate each segment's `features` dict, amortizing one buffer per segment.

A profile's metrics come from two places: inline DSL specs (`define`, compiled against the
schema via the closed grammar) and registered Python metrics (the escape hatch, by name).
Both are `fn(WindowBuffer) -> float`, so the engine builds one `WindowBuffer` per segment and
runs every metric against it — the extraction cost is paid once per segment, not once per
metric.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .buffer import Tick, build_window
from .dsl.grammar import compile_metric
from .schema import SchemaSpec
from .segment import Segment
from .tags import compile_tagger


class FeatureEngine:
    """Compile metrics and optional tag rules, then populate ``Segment`` objects.

    ``define`` contains closed-grammar metric specs and ``registered`` contains registered
    metric names. Unknown metrics, invalid specs, and duplicate names raise ``ValueError``.
    """

    def __init__(self, schema: SchemaSpec, *, define: Optional[dict] = None,
                 registered: Optional[list] = None, tags_spec: Optional[dict] = None,
                 registry=None):
        self.schema = schema
        self.fns = {}
        for name, spec in (define or {}).items():
            self.fns[name] = compile_metric(spec, schema)   # validates at construction (loud)
        from .plugins import load_plugins
        from .registry import default_registry
        registry = registry or default_registry
        missing = [name for name in (registered or [])
                   if name not in registry.names("metric")]
        if missing and registry is default_registry:
            load_plugins()
        for name in (registered or []):
            if name in self.fns:
                raise ValueError(f"metric {name!r} is both DSL-defined and registered — the "
                                 f"registered fn would silently shadow the define; rename one")
            if name not in registry.names("metric"):
                raise ValueError(f"unknown registered metric {name!r}; known: "
                                 f"{registry.names('metric')}")
            self.fns[name] = registry.get("metric", name)
        self.tagger = compile_tagger(tags_spec, schema) if tags_spec else None

    def score_window(self, seg: Segment, window_ticks: list[Tick]) -> Segment:
        """Score one already-extracted window (the streaming entry point). Builds the buffer for
        just this window — no source-wide tick list needed."""
        w = build_window(self.schema, window_ticks)
        for name, fn in self.fns.items():
            value = fn(w)
            array = np.asarray(value)
            if array.shape != ():
                raise ValueError(
                    f"metric {name!r} must return one finite scalar for source "
                    f"{seg.source_id!r} window [{seg.start_idx}, {seg.end_idx}), got shape "
                    f"{array.shape}")
            try:
                scalar = float(array)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"metric {name!r} must return one finite numeric scalar for source "
                    f"{seg.source_id!r} window [{seg.start_idx}, {seg.end_idx}), got "
                    f"{value!r}") from error
            if not math.isfinite(scalar):
                raise ValueError(
                    f"metric {name!r} returned non-finite {scalar!r} for source "
                    f"{seg.source_id!r} window [{seg.start_idx}, {seg.end_idx})")
            seg.features[name] = scalar
        if self.tagger is not None:
            seg.tags = self.tagger(w)
        return seg

    def compute(self, ticks: list[Tick], segments: list[Segment]) -> list[Segment]:
        """Fill `seg.features` (and `seg.tags` if a tags spec was given) in place."""
        for seg in segments:
            self.score_window(seg, ticks[seg.start_idx:seg.end_idx])
        return segments
