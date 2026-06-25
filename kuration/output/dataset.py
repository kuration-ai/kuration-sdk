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
from .fill import aggregate_source_priorities, fill_to_hours
from .shardwriter import write_track
from .spec import OutputSpec
from .track import tracks_from_ticks


def write_dataset(out_dir: str, *, schema: SchemaSpec, adapter: InputAdapter,
                  sources: Iterable[str], strategy: Strategy, segmenter: Segmenter,
                  spec: OutputSpec, encoder: Optional[OutputEncoder] = None,
                  train_hours: float, test_hours: float = 0.0,
                  sample_hz: Optional[float] = None, seed: int = 0,
                  group_of: Optional[Callable[[str], object]] = None,
                  tags_of: Optional[Callable] = None,
                  clip_aggregate: str = "max", workers: int = 1,
                  on_error: str = "skip", resume: bool = False) -> dict:
    """`workers>1` parallelizes the feature pass; `on_error='skip'` logs + skips bad sources;
    `resume=True` skips re-writing shards whose files already exist."""
    sample_hz = sample_hz or schema.sample_hz
    if not sample_hz:
        raise ValueError("write_dataset needs sample_hz (schema.sample_hz or explicit) for the "
                         "hours budget")
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
    src_prio = aggregate_source_priorities(pool, prio, clip_aggregate)

    # per-source tags (union over the source's segments) so the fill's tag reserve can see them
    src_tags: dict = {}
    for s in all_segments:
        if s.tags:
            src_tags.setdefault(s.source_id, set()).update(s.tags)

    n_tracks = sum(len(t) for t in source_tracks.values())
    items = [{"id": (src, eid), "source_id": src,
              "priority": src_prio.get(src, float("-inf")), "seconds": n / sample_hz,
              "group": (group_of(src) if group_of else None),
              "tags": src_tags.get(src, frozenset())}
             for src, tlist in source_tracks.items() for (eid, n) in tlist]
    admitted, split = fill_to_hours(items, train_hours=train_hours, test_hours=test_hours,
                                    tag_coverage=strategy.tag_coverage, seed=seed)

    # Re-read each admitted source once (only admitted sources touch the data again).
    entries = []
    by_source: dict = {}
    for (src, eid) in admitted:
        by_source.setdefault(src, []).append(eid)
    for src in sorted(by_source):
        tracks = {t.entity_id: t for t in tracks_from_ticks(schema, list(adapter.ticks(src)), src)}
        for eid in by_source[src]:
            tr = tracks.get(eid)
            if tr is None:
                continue
            tr.meta["group"] = group_of(src) if group_of else None
            entries += write_track(out_dir, tr, spec, encoder=encoder, schema=schema,
                                   split=split.get(src, "train"), resume=resume)

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
