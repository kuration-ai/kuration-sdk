"""Top-level curation pipeline: sources -> ticks -> windows -> features -> bin -> select.

Ties the Phase A/B engine to the Phase C selection. Deterministic: segments are sorted by
`(source_id, start_idx)` before binning/selection so the seeded RNGs give the same result
regardless of the order sources were processed (the invariant the reference pipeline pins).
The hours-over-output-spans budget fill is deferred to the output phase (you can't budget
"hours of output" before the output unit exists); Phase C selects a target *count*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from .adapters.base import InputAdapter
from .binning import bin_segments
from .features import FeatureEngine
from .schema import SchemaSpec
from .segment import Segment, Segmenter
from .select import assign_splits
from .strategy import Strategy, select


@dataclass
class CurateResult:
    selected: list[Segment]
    all_segments: list[Segment]
    schema: SchemaSpec
    strategy: Strategy
    splits: dict = field(default_factory=dict)   # source_id -> "train"/"test" (if test_frac>0)


def curate(*, schema: SchemaSpec, adapter: InputAdapter, sources: Iterable[str],
           strategy: Strategy, segmenter: Segmenter, target_size: int, seed: int = 0,
           group_of: Optional[Callable[[str], object]] = None,
           tags_of: Optional[Callable[[Segment], Iterable]] = None,
           test_frac: float = 0.0) -> CurateResult:
    """Curate a target-count segment set from `sources`.

    group_of(source_id) -> group key (sets Segment.group for the group cap + split strata);
    tags_of(segment) -> tags (sets Segment.tags for tag coverage). Both optional.
    """
    referenced = strategy.referenced_metrics()
    registered = [r for r in referenced if r not in strategy.define]  # load_strategy verified these exist
    engine = FeatureEngine(schema, define=strategy.define, registered=registered)

    all_segments: list[Segment] = []
    source_groups: dict = {}
    for src in sources:
        ticks = list(adapter.ticks(src))
        segs = segmenter.segments(src, ticks)
        g = group_of(src) if group_of else None
        source_groups[src] = g
        for s in segs:
            s.group = g
        engine.compute(ticks, segs)
        if tags_of:
            for s in segs:
                s.tags = frozenset(tags_of(s))
        all_segments.extend(segs)

    all_segments.sort(key=lambda s: (s.source_id, s.start_idx))   # determinism
    pool = all_segments
    if strategy.filters:
        from .methods import apply_filters
        pool = apply_filters(all_segments, strategy, seed=seed)
    bin_segments(pool, strategy.axes)
    tag_vocab = sorted({t for s in pool for t in s.tags}, key=str)
    selected = select(pool, strategy, target_size=target_size, tag_vocab=tag_vocab, seed=seed)

    splits: dict = {}
    if test_frac > 0:
        splits = assign_splits(source_groups, test_frac, seed)
    return CurateResult(selected=selected, all_segments=all_segments, schema=schema,
                        strategy=strategy, splits=splits)
