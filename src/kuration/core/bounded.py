"""Bounded feature-island execution into checkpointed Arrow partitions."""
from __future__ import annotations

from dataclasses import dataclass, field
import sys
from typing import Any, Callable, Iterable

import numpy as np

from .checkpoints import PartitionCommit, StageCheckpoint
from .columnar import SegmentBatch
from .features import FeatureEngine
from .strategy import _passes


def _deep_size(value, seen: set[int] | None = None) -> int:
    """Conservative size of one live window's Python/NumPy payload."""
    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        return max(size, int(value.nbytes))
    if isinstance(value, dict):
        return size + sum(_deep_size(key, seen) + _deep_size(item, seen)
                          for key, item in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return size + sum(_deep_size(item, seen) for item in value)
    values = getattr(value, "__dict__", None)
    return size + (_deep_size(values, seen) if values is not None else 0)


def _buffer_bytes(schema, ticks: int, entities: int) -> int:
    entities = max(1, entities)
    numeric_width = sum(field.width for field in schema.fields.values()
                        if field.kind != "categorical")
    categorical = sum(field.kind == "categorical" for field in schema.fields.values())
    # WindowBuffer uses float64 numeric/global arrays, int64 categorical arrays, and a bool
    # validity mask.  A factor of three covers the buffer plus reducer/transform staging.
    arrays = ticks * (entities * (8 * numeric_width + 8 * categorical + 1)
                      + 8 * len(schema.globals))
    return 3 * arrays


def _minimum_window_bytes(schema, window: int) -> int:
    # Python Tick/entity containers have a non-zero cost even for an otherwise empty schema.
    return 4096 + window * 256 + _buffer_bytes(schema, window, 1)


def _window_working_bytes(schema, window) -> int:
    entities = len({entity for tick in window for entity in tick.entities})
    return 4096 + _deep_size(window) + _buffer_bytes(schema, len(window), entities)


def resolve_bounded_memory_limits(*, schema, segmenter, memory_budget: int | None,
                                  batch_size: int, max_in_flight: int,
                                  transforms=()) -> dict[str, int | None]:
    """Resolve enforceable feature-stage batching limits from a byte budget.

    A budgeted run deliberately uses one in-flight batch: dynamic entity rosters make a safe
    multi-batch reservation unknowable before windows are read.  Batches are then shortened
    at runtime using measured Python payload plus schema-derived array working space.
    """
    if memory_budget is None:
        return {"memory_budget": None, "batch_size": batch_size,
                "max_in_flight": max_in_flight, "batch_memory_limit": None,
                "declared_transform_memory": 0}
    declared = max((int((call.get("resources") or {}).get("memory_bytes") or 0)
                    for call in transforms), default=0)
    halo = max((int(call.get("lookbehind", 0)) + int(call.get("lookahead", 0))
                for call in transforms), default=0)
    minimum = _minimum_window_bytes(schema, segmenter.window + halo)
    available = memory_budget - declared
    if available < minimum:
        raise ValueError(
            f"memory_budget={memory_budget} bytes cannot hold one feature window "
            f"(estimated minimum {minimum + declared} bytes, including "
            f"{declared} bytes declared by transforms)")
    return {"memory_budget": memory_budget,
            "batch_size": min(batch_size, max(1, available // minimum)),
            "max_in_flight": 1, "batch_memory_limit": available,
            "declared_transform_memory": declared}

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
                             max_in_flight: int = 2, memory_budget: int | None = None,
                             retain_predicate_failures: bool = False) -> FeatureStageResult:
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
    if memory_budget is not None and (not isinstance(memory_budget, int)
                                      or isinstance(memory_budget, bool) or memory_budget < 1):
        raise ValueError("memory_budget must be a positive integer or None")
    if not isinstance(retain_predicate_failures, bool):
        raise ValueError("retain_predicate_failures must be true or false")
    from .adapters.base import validate_adapter_availability
    validate_adapter_availability(adapter, schema, define=define, tags_spec=tags_spec)
    limits = resolve_bounded_memory_limits(
        schema=schema, segmenter=segmenter, memory_budget=memory_budget,
        batch_size=batch_size, max_in_flight=max_in_flight, transforms=transforms)
    resolved_batch_size = limits["batch_size"]
    resolved_max_in_flight = limits["max_in_flight"]
    resolved_transform_memory = limits["declared_transform_memory"]
    assert isinstance(resolved_batch_size, int)
    assert isinstance(resolved_max_in_flight, int)
    assert isinstance(resolved_transform_memory, int)
    effective_batch_size = resolved_batch_size
    effective_max_in_flight = resolved_max_in_flight
    batch_memory_limit = limits["batch_memory_limit"]
    declared_transform_memory = resolved_transform_memory
    transform_lookbehind = max((call.get("lookbehind", 0) for call in transforms), default=0)
    transform_lookahead = max((call.get("lookahead", 0) for call in transforms), default=0)
    if transforms and registry is None:
        from .registry import default_registry
        registry = default_registry
    sources = list(sources)
    seen_sources: set[str] = set()
    for source in sources:
        if not isinstance(source, str) or not source:
            raise ValueError("sources must contain only non-empty strings")
        if source in seen_sources:
            raise ValueError(f"sources must be unique; duplicate source ID {source!r}")
        seen_sources.add(source)
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
    estimated_peak_live_bytes = 0
    largest_window_bytes = 0

    for source_index, source in enumerate(sources, 1):
        source_commits: list[PartitionCommit] = []
        try:
            engine = FeatureEngine(schema, define=define, registered=list(registered),
                                   tags_spec=tags_spec, registry=registry)
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
                nonlocal estimated_peak_live_bytes, largest_window_bytes
                pending_segments: list[Any] = []
                pending_windows: list[Any] = []
                pending_contexts: list[Any] = []
                pending_bytes = 0
                partition_index = 0

                def owned_windows():
                    pending = None
                    for entry in segmenter.stream_with_context(
                            source, tracked(ticks), lookbehind=transform_lookbehind,
                            lookahead=transform_lookahead):
                        if pending is not None:
                            prior_segment, prior_window, before, after = pending
                            owned = min(prior_segment.end_idx, entry[0].start_idx) \
                                - prior_segment.start_idx
                            yield prior_segment, prior_window, {
                                "lookbehind": before, "lookahead": after,
                                "owned_length": owned}
                        pending = entry
                    if pending is not None:
                        segment, window, before, after = pending
                        yield segment, window, {
                            "lookbehind": before, "lookahead": after,
                            "owned_length": segment.end_idx - segment.start_idx}

                for segment, window, context in owned_windows():
                    live_ticks = [*context["lookbehind"], *window, *context["lookahead"]]
                    window_bytes = (_window_working_bytes(schema, live_ticks)
                                    if batch_memory_limit is not None else 0)
                    largest_window_bytes = max(largest_window_bytes, window_bytes)
                    if batch_memory_limit is not None and window_bytes > batch_memory_limit:
                        raise MemoryError(
                            f"one feature window needs an estimated {window_bytes} bytes, "
                            f"exceeding memory_budget={memory_budget} bytes after reserved "
                            f"transform memory; reduce window width/entity density or raise "
                            "memory_budget")
                    if (pending_segments and batch_memory_limit is not None
                            and pending_bytes + window_bytes > batch_memory_limit):
                        estimated_peak_live_bytes = max(
                            estimated_peak_live_bytes, pending_bytes + declared_transform_memory)
                        yield (f"{source!s}:{partition_index:08d}",
                               pending_segments, pending_windows, pending_contexts)
                        partition_index += 1
                        pending_segments, pending_windows, pending_contexts = [], [], []
                        pending_bytes = 0
                    segment.group = group
                    pending_segments.append(segment)
                    pending_windows.append(window)
                    pending_contexts.append(context)
                    pending_bytes += window_bytes
                    if len(pending_segments) == effective_batch_size:
                        estimated_peak_live_bytes = max(
                            estimated_peak_live_bytes, pending_bytes + declared_transform_memory)
                        yield (f"{source!s}:{partition_index:08d}",
                               pending_segments, pending_windows, pending_contexts)
                        partition_index += 1
                        pending_segments, pending_windows, pending_contexts = [], [], []
                        pending_bytes = 0
                if pending_segments:
                    estimated_peak_live_bytes = max(
                        estimated_peak_live_bytes, pending_bytes + declared_transform_memory)
                    yield (f"{source!s}:{partition_index:08d}",
                           pending_segments, pending_windows, pending_contexts)

            def process(item):
                nonlocal transform_runner, checkpoint_hits, checkpoint_misses
                partition_id, pending_segments, pending_windows, pending_contexts = item
                reusable = checkpoint.reusable(partition_id)
                if reusable is not None:
                    checkpoint_hits += 1
                    return ("commit", reusable)
                checkpoint_misses += 1
                for segment, window in zip(pending_segments, pending_windows):
                    engine.score_window(segment, window)
                if transforms:
                    if transform_runner is None:
                        from .transforms import _TransformRunner
                        transform_runner = _TransformRunner(
                            transforms, adapter=adapter, registry=registry)
                    transform_runner.apply(pending_segments, pending_windows,
                                           contexts=pending_contexts, progress=None)
                selected = [segment for segment in pending_segments
                            if _passes(segment, predicates)]
                written = pending_segments if retain_predicate_failures else selected
                if tags_of:
                    for segment in written:
                        segment.tags = frozenset(tags_of(segment))
                key_range = None
                if written:
                    keys = [(str(segment.source_id), segment.start_idx, segment.end_idx)
                            for segment in written]
                    key_range = (keys[0], keys[-1])
                return ("write", partition_id, written, key_range,
                        {"candidate_rows": len(pending_segments),
                         "admitted_rows": len(selected)})

            from .execution import LocalExecutionBackend
            for result in LocalExecutionBackend().iter_batches(
                    batches(), process, max_in_flight=effective_max_in_flight, workers=1):
                if result[0] == "commit":
                    source_commits.append(result[1])
                    continue
                _, partition_id, written, key_range, reducer_state = result
                # Keep Arrow construction and checkpoint publication on the caller thread.
                # Some supported Arrow/Python combinations initialize native allocators lazily
                # and are not safe when their first conversion occurs on a background worker.
                batch = SegmentBatch.from_segments(written, feature_names=feature_names)
                source_commits.append(checkpoint.write_partition(
                    partition_id, batch, key_range=key_range,
                    reducer_state=reducer_state))
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
            # A transform may retain mutable/model state. Reconstruct it after any failed
            # source so source-level skip isolation also isolates component state.
            transform_runner = None
            from .scoring import _redact_diagnostic
            errors.append({
                "source": _redact_diagnostic(source) if isinstance(source, str) else source,
                "error": _redact_diagnostic(repr(error)),
                "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
                "message": _redact_diagnostic(str(error)),
                "phase": "bounded-feature-stage",
            })
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
                 "batch_size": effective_batch_size,
                 "max_in_flight": effective_max_in_flight,
                 "requested_batch_size": batch_size,
                 "requested_max_in_flight": max_in_flight,
                 "memory_budget": memory_budget,
                 "estimated_peak_live_bytes": (estimated_peak_live_bytes
                                               if memory_budget is not None else None),
                 "largest_window_bytes": (largest_window_bytes
                                          if memory_budget is not None else None),
                 "declared_transform_memory": declared_transform_memory}
    return FeatureStageResult(commits=tuple(commits), errors=tuple(errors),
                              source_tracks=tracks, telemetry=telemetry)
