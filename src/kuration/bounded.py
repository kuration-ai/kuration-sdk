"""Bounded feature-island execution into checkpointed Arrow partitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .checkpoints import PartitionCommit, StageCheckpoint
from .columnar import SegmentBatch
from .features import FeatureEngine


_OPS = {
    ">": lambda left, right: left > right,
    ">=": lambda left, right: left >= right,
    "<": lambda left, right: left < right,
    "<=": lambda left, right: left <= right,
    "==": lambda left, right: left == right,
    "!=": lambda left, right: left != right,
}


def _passes(segment, predicates) -> bool:
    return all(_OPS[predicate["op"]](segment.features[predicate["metric"]],
                                     predicate["value"])
               for predicate in predicates or ())


@dataclass(frozen=True)
class FeatureStageResult:
    commits: tuple[PartitionCommit, ...]
    errors: tuple[dict, ...]
    source_tracks: dict[Any, tuple[tuple[Any, int], ...]]
    telemetry: dict[str, Any] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return sum(commit.rows for commit in self.commits)

    @property
    def candidate_rows(self) -> int:
        return sum(int((commit.reducer_state or {}).get("candidate_rows", commit.rows))
                   for commit in self.commits)


class SizedSegmentPool:
    """Non-materialized candidate-pool accounting used by bounded terminal results."""

    def __init__(self, count: int):
        self.count = int(count)

    def __len__(self) -> int:
        return self.count


def checkpoint_feature_stage(*, schema, adapter, sources: Iterable, define: dict,
                             registered, segmenter, checkpoint: StageCheckpoint,
                             batch_size: int = 1024, tags_spec=None,
                             predicates=(), transforms=(), registry=None,
                             group_of: Callable | None = None,
                             tags_of: Callable | None = None,
                             on_error: str = "skip", progress=None,
                             max_in_flight: int = 2) -> FeatureStageResult:
    """Run a fused feature island with bounded batches and producer backpressure.

    Matching partition commits skip feature and transform computation on restart. Sources are
    still scanned to recover stable window boundaries and track lengths. At most
    ``max_in_flight * batch_size`` windows are live across producer and worker.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if on_error not in {"skip", "raise"}:
        raise ValueError("on_error must be 'skip' or 'raise'")
    if (not isinstance(max_in_flight, int) or isinstance(max_in_flight, bool)
            or max_in_flight < 1):
        raise ValueError("max_in_flight must be a positive integer")
    if transforms and registry is None:
        from .registry import default_registry
        registry = default_registry
    sources = list(sources)
    registered = tuple(registered)
    feature_names = tuple(dict.fromkeys([*define, *registered,
                                        *(output for call in transforms
                                          for output in call["outputs"])]))
    commits: list[PartitionCommit] = []
    errors: list[dict] = []
    tracks: dict[Any, tuple[tuple[Any, int], ...]] = {}
    from .progress import emit
    emit(progress, "scoring", "start", total=len(sources), detail="bounded feature island")
    transform_runner = None
    checkpoint_hits = 0
    checkpoint_misses = 0

    for source_index, source in enumerate(sources, 1):
        source_commits: list[PartitionCommit] = []
        try:
            engine = FeatureEngine(schema, define=define, registered=list(registered),
                                   tags_spec=tags_spec)
            seen, entity_order, tick_count = set(), [], 0

            def tracked(ticks):
                nonlocal tick_count
                for tick in ticks:
                    tick_count += 1
                    for entity_id in tick.entities:
                        if entity_id not in seen:
                            seen.add(entity_id)
                            entity_order.append(entity_id)
                    yield tick

            ticks = (adapter.iter_ticks(source) if hasattr(adapter, "iter_ticks")
                     and not getattr(adapter, "in_memory", False)
                     else iter(adapter.ticks(source)))
            group = group_of(source) if group_of else None

            def batches():
                pending_segments, pending_windows = [], []
                partition_index = 0
                for segment, window in segmenter.stream(source, tracked(ticks)):
                    segment.group = group
                    pending_segments.append(segment)
                    pending_windows.append(window)
                    if len(pending_segments) == batch_size:
                        yield (f"{source!s}:{partition_index:08d}",
                               pending_segments, pending_windows)
                        partition_index += 1
                        pending_segments, pending_windows = [], []
                if pending_segments:
                    yield (f"{source!s}:{partition_index:08d}",
                           pending_segments, pending_windows)

            def process(item):
                nonlocal transform_runner, checkpoint_hits, checkpoint_misses
                partition_id, pending_segments, pending_windows = item
                reusable = checkpoint.reusable(partition_id)
                if reusable is not None:
                    checkpoint_hits += 1
                    return reusable
                checkpoint_misses += 1
                for segment, window in zip(pending_segments, pending_windows):
                    engine.score_window(segment, window)
                if transforms:
                    if transform_runner is None:
                        from .transforms import _TransformRunner
                        transform_runner = _TransformRunner(
                            transforms, adapter=adapter, registry=registry)
                    transform_runner.apply(pending_segments, pending_windows, progress=None)
                selected = [segment for segment in pending_segments
                            if _passes(segment, predicates)]
                if tags_of:
                    for segment in selected:
                        segment.tags = frozenset(tags_of(segment))
                batch = SegmentBatch.from_segments(selected, feature_names=feature_names)
                key_range = None
                if selected:
                    keys = [(str(segment.source_id), segment.start_idx, segment.end_idx)
                            for segment in selected]
                    key_range = (keys[0], keys[-1])
                return checkpoint.write_partition(
                    partition_id, batch, key_range=key_range,
                    reducer_state={"candidate_rows": len(pending_segments),
                                   "admitted_rows": len(selected)})

            from .execution import LocalExecutionBackend
            source_commits.extend(LocalExecutionBackend().iter_batches(
                batches(), process, max_in_flight=max_in_flight, workers=1))
            # A source is the error-isolation unit.  Do not expose partitions from a source
            # that failed after one or more batches had already committed; those files may
            # remain reusable on restart, but they are not candidates in this run.
            commits.extend(source_commits)
            tracks[source] = tuple((entity_id, tick_count) for entity_id in entity_order)
            emit(progress, "scoring", "advance", current=source_index,
                 total=len(sources), source=source, status="ok")
        except Exception as error:
            if on_error == "raise":
                raise RuntimeError(f"bounded feature source {source!r} failed: {error!r}") from error
            errors.append({"source": source, "error": repr(error)})
            emit(progress, "scoring", "advance", current=source_index,
                 total=len(sources), source=source, status="err")
    if sources and not commits:
        raise RuntimeError(f"segment pool is EMPTY: {len(sources)} source(s) yielded no "
                           f"segments ({len(errors)} skipped; bounded execution)")
    emit(progress, "scoring", "complete", current=len(sources), total=len(sources),
         detail=f"{sum(commit.rows for commit in commits)} rows (bounded)")
    telemetry = {"partitions": len(commits), "checkpoint_hits": checkpoint_hits,
                 "checkpoint_misses": checkpoint_misses,
                 "checkpoint_bytes": sum(commit.size_bytes for commit in commits),
                 "candidate_rows": sum(int((commit.reducer_state or {}).get(
                     "candidate_rows", commit.rows)) for commit in commits),
                 "output_rows": sum(commit.rows for commit in commits),
                 "batch_size": batch_size, "max_in_flight": max_in_flight}
    return FeatureStageResult(commits=tuple(commits), errors=tuple(errors),
                              source_tracks=tracks, telemetry=telemetry)
