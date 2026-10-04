"""Top-level curation pipeline: sources -> ticks -> windows -> features -> bin -> select.

Ties the Phase A/B engine to the Phase C selection. Deterministic: segments are sorted by
`(source_id, start_idx)` before binning/selection so the seeded RNGs give the same result
regardless of the order sources were processed (the invariant the reference pipeline pins).
The hours-over-output-spans budget fill is deferred to the output phase (you can't budget
"hours of output" before the output unit exists); Phase C selects a target *count*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterable, Optional

from .adapters.base import InputAdapter
from .binning import bin_segments
from .schema import SchemaSpec
from .scoring import score_sources
from .segment import Segment, Segmenter
from .select import assign_splits
from .strategy import Strategy, _passes, select


@dataclass
class CurateResult:
    selected: list[Segment]
    all_segments: Any
    schema: SchemaSpec
    strategy: Strategy
    splits: dict = field(default_factory=dict)   # source_id -> "train"/"test" (if test_frac>0)
    errors: list = field(default_factory=list)   # [{source, error}] for skipped sources
    source_tracks: dict = field(default_factory=dict)  # source -> [(entity_id, n_steps)]
    execution: Optional[dict] = None
    seed: int = 0

    def stats(self, **kwargs):
        """Summarize candidate-versus-selected statistics without rescoring."""
        from .stats import stats

        return stats(self, **kwargs)


def curate(*, schema: SchemaSpec, adapter: InputAdapter, sources: Iterable[str],
           strategy: Strategy, segmenter: Segmenter, target_size: Optional[int], seed: int = 0,
           group_of: Optional[Callable[[str], object]] = None,
           tags_of: Optional[Callable[[Segment], Iterable]] = None,
           test_frac: float = 0.0, workers: int = 1, on_error: str = "skip",
           streaming: bool = False, progress=None, transforms=(), registry=None,
           executor: str = "local") -> CurateResult:
    """Curate a target-count segment set from `sources`.

    Sources are non-empty string identifiers/URIs. Convert ``pathlib.Path`` values with ``str``;
    this keeps segment IDs, worker transport, sorting, manifests, and resume identity canonical.
    group_of(source_id) -> group key (sets Segment.group for the group cap + split strata);
    tags_of(segment) -> tags (sets Segment.tags for tag coverage). Both optional.
    `workers>1` parallelizes the feature pass; `on_error='skip'` logs + skips a bad source;
    `streaming=True` scores each source out-of-core with a streaming-capable adapter.
    ``target_size=None`` skips selection entirely: ``selected`` becomes every segment that
    passed the strategy's predicates/filters, in canonical sorted order — no binning method,
    weights, or axes required.
    """
    from .strategy import validate_target_size
    target_size = validate_target_size(target_size)
    sources = list(sources)
    if not sources:
        raise ValueError("curate() needs at least one source")
    if (not isinstance(test_frac, (int, float)) or isinstance(test_frac, bool)
            or not math.isfinite(test_frac) or not 0 <= test_frac <= 1):
        raise ValueError(f"test_frac must be a finite number in [0, 1], got {test_frac!r}")
    referenced = strategy.referenced_metrics()
    referenced |= {
        token.split(":", 1)[1]
        for call in transforms for token in call["inputs"] if token.startswith("feature:")
    }
    # scorer outputs are produced by the scorer stage, not the feature pass — exclude them.
    transform_outputs = {o for t in transforms for o in t["outputs"]}
    registered = sorted(
        r for r in referenced if r not in strategy.define
        and r not in strategy.scorer_outputs() and r not in transform_outputs
    )
    transform_fields = {
        token.split(":", 1)[1]
        for call in transforms for token in call["inputs"] if token.startswith("field:")}
    all_segments, errors, source_tracks = score_sources(
        schema=schema, adapter=adapter, sources=sources, define=strategy.define,
        registered=registered, segmenter=segmenter, tags_spec=strategy.tags,
        group_of=group_of, tags_of=tags_of, workers=workers, on_error=on_error,
        streaming=streaming, progress=progress, executor=executor,
        registry=registry or strategy.registry, required_fields=transform_fields)

    if transforms:
        from .transforms import apply_transforms
        all_segments = apply_transforms(
            transforms, all_segments, adapter=adapter,
            registry=registry or strategy.registry, progress=progress,
            on_error=on_error, errors=errors)

    if strategy.scorers:   # whole-pool learned/action-aware scoring -> features
        from .scorers import apply_scorers
        apply_scorers(strategy.scorers, all_segments, schema=schema, adapter=adapter, seed=seed,
                      progress=progress, registry=strategy.registry,
                      on_error=on_error, errors=errors)

    from .progress import emit
    emit(progress, "selection", "start", total=len(all_segments))
    pool = all_segments
    if strategy.filters:
        from .methods import apply_filters
        pool = apply_filters(all_segments, strategy, seed=seed, workers=workers)
    bin_segments(pool, strategy.axes)
    if target_size is None:
        selected = [s for s in pool if _passes(s, strategy.predicates)] \
            if strategy.predicates else list(pool)
    else:
        tag_vocab = sorted({t for s in pool for t in s.tags}, key=str)
        selected = select(pool, strategy, target_size=target_size, tag_vocab=tag_vocab,
                          seed=seed, workers=workers)
    emit(progress, "selection", "complete", current=len(selected), total=len(pool),
         detail=f"{len(selected)} selected")

    splits: dict = {}
    if test_frac > 0:
        source_groups = {s.source_id: s.group for s in all_segments}
        splits = assign_splits(source_groups, test_frac, seed)
    return CurateResult(selected=selected, all_segments=all_segments, schema=schema,
                        strategy=strategy, splits=splits, errors=errors,
                        source_tracks=source_tracks, seed=seed)
