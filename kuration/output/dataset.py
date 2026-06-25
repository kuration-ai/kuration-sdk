"""`write_dataset` — the end-to-end output pipeline.

sources -> features -> segment priorities -> per-source priority -> hours-budget track fill
(train/test) -> write each admitted track's shards -> dataset_manifest + reproducible config
(embedding schema + strategy + output spec). The hours budget is what makes the emitted set
the strategy's *best* N hours rather than a random draw.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Optional

from ..adapters.base import InputAdapter
from ..binning import bin_segments
from ..manifest import build_config, write_config
from ..schema import SchemaSpec
from ..scoring import score_sources
from ..segment import Segmenter
from ..strategy import Strategy, segment_priorities
from .base import OutputEncoder
from .fill import fill_to_hours
from .shardwriter import write_track
from .spec import OutputSpec
from .track import Track, tracks_from_ticks


def write_dataset(out_dir: str, *, schema: SchemaSpec, adapter: InputAdapter,
                  sources: Iterable[str], strategy: Strategy, segmenter: Segmenter,
                  spec: OutputSpec, encoder: Optional[OutputEncoder] = None,
                  train_hours: float, test_hours: float = 0.0,
                  sample_hz: Optional[float] = None, seed: int = 0,
                  group_of: Optional[Callable[[str], object]] = None,
                  tags_of: Optional[Callable] = None,
                  clip_aggregate: str = "max", span_steps: Optional[int] = None,
                  span_seconds: Optional[float] = None, workers: int = 1,
                  on_error: str = "skip", resume: bool = False) -> dict:
    """`workers>1` parallelizes the feature pass; `on_error='skip'` logs + skips bad sources;
    `resume=True` skips re-writing shards whose files already exist.

    `span_steps` / `span_seconds` subdivide each track into independently-budgeted spans, so
    the fill can keep the *best parts* of a long source rather than the whole source or nothing.
    Unset → one span per whole track (the default)."""
    sample_hz = sample_hz or schema.sample_hz
    if not sample_hz:
        raise ValueError("write_dataset needs sample_hz (schema.sample_hz or explicit) for the "
                         "hours budget")
    span_len = span_steps if span_steps else (round(span_seconds * sample_hz) if span_seconds else None)
    registered = [r for r in strategy.referenced_metrics() if r not in strategy.define]

    # Parallel score pass: returns features + per-source track lengths (so the fill needs no
    # second read of the data). The data itself is re-read only for admitted sources, below.
    all_segments, errors, source_tracks = score_sources(
        schema=schema, adapter=adapter, sources=sources, define=strategy.define,
        registered=registered, segmenter=segmenter, tags_spec=strategy.tags,
        group_of=group_of, tags_of=tags_of, workers=workers, on_error=on_error)

    pool = all_segments
    if strategy.filters:
        from ..methods import apply_filters
        pool = apply_filters(all_segments, strategy, seed=seed)
    bin_segments(pool, strategy.axes)
    prio = segment_priorities(pool, strategy, seed=seed)

    # Per-source segment (start, end, priority, tags) so a span's priority/tags aggregate only
    # the segments overlapping THAT span (not the whole source).
    segs_by_src: dict = {}
    for i, s in enumerate(pool):
        segs_by_src.setdefault(s.source_id, []).append((s.start_idx, s.end_idx, float(prio[i]), s.tags))
    _reduce = {"max": max, "min": min, "mean": lambda x: sum(x) / len(x)}[clip_aggregate]

    def _span_prio_tags(src, start, end):
        prios, tags = [], set()
        for st, en, p, tg in segs_by_src.get(src, ()):
            if st < end and en > start:
                prios.append(p)
                tags |= tg
        return (_reduce(prios) if prios else float("-inf")), frozenset(tags)

    # Build span items: (source, entity, span_index) -> {priority, seconds, group, tags}.
    items, span_index = [], {}
    for src, tlist in source_tracks.items():
        g = group_of(src) if group_of else None
        for (eid, n) in tlist:
            bounds = ([(s, min(s + span_len, n)) for s in range(0, n, span_len)]
                      if span_len else [(0, n)])
            for k, (start, end) in enumerate(bounds):
                p, tg = _span_prio_tags(src, start, end)
                items.append({"id": (src, eid, k), "source_id": src, "priority": p,
                              "seconds": (end - start) / sample_hz, "group": g, "tags": tg})
                span_index[(src, eid, k)] = (start, end)
    n_tracks = sum(len(t) for t in source_tracks.values())
    admitted, split = fill_to_hours(items, train_hours=train_hours, test_hours=test_hours,
                                    tag_coverage=strategy.tag_coverage, seed=seed)

    # Re-read each admitted source once; write each admitted span as its own (sub-)track.
    entries = []
    by_source: dict = {}
    for (src, eid, k) in admitted:
        by_source.setdefault(src, []).append((eid, k))
    for src in sorted(by_source):
        tracks = {t.entity_id: t for t in tracks_from_ticks(schema, list(adapter.ticks(src)), src)}
        for (eid, k) in by_source[src]:
            tr = tracks.get(eid)
            if tr is None:
                continue
            start, end = span_index[(src, eid, k)]
            sub = tr if (start, end) == (0, tr.n_steps) else Track(
                tr.source_id, tr.entity_id, tr.records[start:end], dict(tr.meta))
            sub.meta["group"] = group_of(src) if group_of else None
            suffix = "" if span_len is None else f"_span{k:03d}"
            entries += write_track(out_dir, sub, spec, encoder=encoder, schema=schema,
                                   split=split.get(src, "train"), resume=resume,
                                   name_suffix=suffix)

    from ..io import write_text
    write_text(f"{out_dir}/dataset_manifest.json",
               json.dumps({"count": len(entries), "shards": entries}, indent=2))
    results = {"shards": len(entries), "tracks_admitted": len(admitted),
               "tracks_total": n_tracks, "sources_skipped": len(errors),
               "n_train": sum(1 for v in split.values() if v == "train"),
               "n_test": sum(1 for v in split.values() if v == "test")}
    cfg = build_config(schema, strategy, results=results, seed=seed)
    cfg["output"] = spec.to_dict()
    write_config(f"{out_dir}/config.json", cfg)
    return {"shards": entries, "split": split, "admitted": admitted, "results": results,
            "errors": errors}
