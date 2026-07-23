"""Declarative tag derivation — populate `Segment.tags` from data, no callback needed.

`tag_coverage` (a selection post-pass) guarantees a floor count per categorical label, but it
only works if the labels are populated. A strategy's `tags` block derives them declaratively:
a segment's tags are the distinct categorical values observed in its window for the listed
fields / event streams. Tags are strings `"field:value"` (or `"stream.field:value"`), so they
serialize cleanly into manifests. A `tags_of` callback remains available as an escape hatch and
overrides the declarative tags when both are present.
"""
from __future__ import annotations

import numpy as np

from .schema import SchemaSpec


def compile_tagger(spec, schema: SchemaSpec):
    """Compile a `tags` spec ({"from": [field | "stream.field", ...]} or a bare list) to
    `fn(WindowBuffer) -> frozenset[str]`. Validates sources at compile time (loud)."""
    froms = spec.get("from") if isinstance(spec, dict) else spec
    if not froms:
        raise ValueError("tags spec needs a non-empty 'from' list")
    plan: list[tuple[str, ...]] = []
    for src in froms:
        if "." in src:
            stream, field = src.split(".", 1)
            sd = schema.event_streams.get(stream)
            if not sd or field not in sd["fields"]:
                raise ValueError(f"tag source {src!r}: no such event field")
            plan.append(("event", stream, field))
        else:
            fs = schema.fields.get(src)
            if fs is None or fs.kind != "categorical":
                raise ValueError(f"tag source {src!r} must be a categorical field")
            plan.append(("field", src))

    def fn(w) -> frozenset:
        tags: set = set()
        for p in plan:
            if p[0] == "field":
                field = p[1]
                codes, vocab = w.categorical_field(field)
                for c in np.unique(codes):
                    if c >= 0:
                        tags.add(f"{field}:{vocab[int(c)]}")
            else:
                _, stream, field = p
                for evs in w.events.get(stream, []):
                    for e in evs:
                        v = e.get(field)
                        if v is not None:
                            tags.add(f"{stream}.{field}:{v}")
        return frozenset(tags)
    return fn
