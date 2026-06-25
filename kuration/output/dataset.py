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
from ..features import FeatureEngine
from ..manifest import build_config, write_config
from ..schema import SchemaSpec
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
                  clip_aggregate: str = "max") -> dict:
    sample_hz = sample_hz or schema.sample_hz
    if not sample_hz:
        raise ValueError("write_dataset needs sample_hz (schema.sample_hz or explicit) for the "
                         "hours budget")
    registered = [r for r in strategy.referenced_metrics() if r not in strategy.define]
    engine = FeatureEngine(schema, define=strategy.define, registered=registered)

    all_segments = []
    ticks_by_source: dict = {}
    for src in sources:
        ticks = list(adapter.ticks(src))
        ticks_by_source[src] = ticks
        segs = segmenter.segments(src, ticks)
        g = group_of(src) if group_of else None
        for s in segs:
            s.group = g
        engine.compute(ticks, segs)
        all_segments.extend(segs)

    all_segments.sort(key=lambda s: (s.source_id, s.start_idx))   # determinism
    pool = all_segments
    if strategy.filters:
        from ..methods import apply_filters
        pool = apply_filters(all_segments, strategy, seed=seed)
    bin_segments(pool, strategy.axes)
    prio = segment_priorities(pool, strategy, seed=seed)
    src_prio = aggregate_source_priorities(pool, prio, clip_aggregate)

    tracks = []
    for src, ticks in ticks_by_source.items():
        for tr in tracks_from_ticks(schema, ticks, src):
            tr.meta["group"] = group_of(src) if group_of else None
            tracks.append(tr)

    items = [{"id": (t.source_id, t.entity_id), "source_id": t.source_id,
              "priority": src_prio.get(t.source_id, float("-inf")),
              "seconds": t.n_steps / sample_hz, "group": t.meta.get("group")} for t in tracks]
    admitted, split = fill_to_hours(items, train_hours=train_hours, test_hours=test_hours,
                                    seed=seed)

    entries = []
    for t in tracks:
        if (t.source_id, t.entity_id) in admitted:
            sp = split.get(t.source_id, "train")
            entries += write_track(out_dir, t, spec, encoder=encoder, schema=schema, split=sp)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{out_dir}/dataset_manifest.json").write_text(
        json.dumps({"count": len(entries), "shards": entries}, indent=2))
    results = {"shards": len(entries), "tracks_admitted": len(admitted),
               "tracks_total": len(tracks),
               "n_train": sum(1 for v in split.values() if v == "train"),
               "n_test": sum(1 for v in split.values() if v == "test")}
    cfg = build_config(schema, strategy, results=results, seed=seed)
    cfg["output"] = spec.to_dict()
    write_config(f"{out_dir}/config.json", cfg)
    return {"shards": entries, "split": split, "admitted": admitted, "results": results}
