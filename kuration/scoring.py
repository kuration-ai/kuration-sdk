"""Parallel, fault-tolerant feature pass — shared by curate / experiment / write_dataset.

The single place sources are read, segmented and scored. Parallelism is per-source over a
process pool: each worker rebuilds the `FeatureEngine` in-process (the compiled DSL metrics
are closures and can't be pickled, so they're recompiled from the JSON `define` + schema —
both picklable; the adapter and segmenter pickle as-is). A source that raises is
**skipped-and-logged** (loud: stderr line per skip + a summary + collected in the returned
errors), never silently dropped. Determinism holds: the parent sorts the pooled segments by
`(source_id, start_idx)`, so the result is identical regardless of worker count or completion
order.

The pool explicitly uses `spawn` to avoid unsafe forks from multithreaded runtimes and to behave
consistently across Python versions. Custom registered metrics must therefore live in an
importable module; a metric registered only in `__main__` is not visible to workers and fails
loudly.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from .features import FeatureEngine


def _entities_in_order(ticks) -> list:
    seen, out = set(), []
    for tk in ticks:
        for eid in tk.entities:
            if eid not in seen:
                seen.add(eid)
                out.append(eid)
    return out


def _score_one(args):
    """Worker: read one source, segment, compute features. Returns a status tuple so a
    failure is data (skipped in the parent), not a pool-killing exception."""
    (schema, define, registered, tags_spec, adapter, segmenter, source, group, streaming,
     plugin_modules) = args
    try:
        from .plugins import load_plugins
        load_plugins(list(plugin_modules))
        engine = FeatureEngine(schema, define=define, registered=registered, tags_spec=tags_spec)
        if streaming:
            # Stream windows from a lazy tick iterator: never hold more than one window. A
            # wrapper observes every tick as it passes the deque, so total length + entity
            # roster (for the output track fill) come from the SAME single pass.
            n, order, seen = 0, [], set()

            def tracked(it):
                nonlocal n
                for tk in it:
                    n += 1
                    for eid in tk.entities:
                        if eid not in seen:
                            seen.add(eid)
                            order.append(eid)
                    yield tk

            segs = []
            tick_iter = (adapter.iter_ticks(source) if hasattr(adapter, "iter_ticks")
                         else adapter.ticks(source))
            for seg, win in segmenter.stream(source, tracked(tick_iter)):
                seg.group = group
                engine.score_window(seg, win)
                segs.append(seg)
            return ("ok", source, segs, [(eid, n) for eid in order])

        ticks = list(adapter.ticks(source))
        segs = segmenter.segments(source, ticks)
        for s in segs:
            s.group = group
        engine.compute(ticks, segs)
        track_lens = [(eid, len(ticks)) for eid in _entities_in_order(ticks)]
        return ("ok", source, segs, track_lens)
    except Exception as e:
        return ("err", source, repr(e), None)


def score_sources(*, schema, adapter, sources, define, registered, segmenter,
                  tags_spec=None, group_of=None, tags_of=None, workers: int = 1,
                  on_error: str = "skip", streaming: bool = False, progress=None):
    """Return (all_segments_sorted, errors, source_tracks).

      errors:        [{"source", "error"}] for skipped sources (on_error='skip')
      source_tracks: {source_id -> [(entity_id, n_steps)]} — the output unit lengths, so the
                     track fill needs no second read of the data.
    `on_error`: 'skip' (log + continue) or 'raise'. `group_of(source)` is evaluated in the
    parent (callables needn't be picklable); `tags_of` is applied in the parent post-hoc.
    """
    if streaming and getattr(adapter, "supports_streaming", True) is False:
        raise ValueError(f"{type(adapter).__name__} cannot provide bounded-memory streaming; "
                         "use an uncompressed array store such as HDF5/Zarr or streaming=False")
    if on_error not in ("skip", "raise"):
        raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError(f"workers must be a positive integer, got {workers!r}")
    if not isinstance(streaming, bool):
        raise ValueError("streaming must be true or false")
    from .dsl.registry import METRICS
    from .plugins import loaded_plugin_modules
    metric_modules = {METRICS[r].__module__ for r in registered if r in METRICS}
    plugin_modules = tuple(sorted(set(loaded_plugin_modules()) | metric_modules))
    worker_modules = plugin_modules if workers and workers > 1 else ()
    items = [(schema, define, registered, tags_spec, adapter, segmenter, src,
              (group_of(src) if group_of else None), streaming, worker_modules) for src in sources]
    from .progress import emit
    emit(progress, "scoring", "start", total=len(items))
    if workers and workers > 1:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as ex:
            results: list[Any] = [None] * len(items)
            futures = {ex.submit(_score_one, item): i for i, item in enumerate(items)}
            done = 0
            for future in as_completed(futures):
                i = futures[future]
                results[i] = future.result()
                done += 1
                emit(progress, "scoring", "advance", current=done, total=len(items),
                     source=results[i][1], status=results[i][0])
    else:
        results = []
        for i, item in enumerate(items, 1):
            result = _score_one(item)
            results.append(result)
            emit(progress, "scoring", "advance", current=i, total=len(items),
                 source=result[1], status=result[0])

    all_segments, errors, source_tracks = [], [], {}
    for status, src, payload, track_lens in results:
        if status == "ok":
            all_segments.extend(payload)
            source_tracks[src] = track_lens
        elif on_error == "raise":
            raise RuntimeError(f"scoring source {src!r} failed: {payload}")
        else:
            print(f"  [skip] source {src!r}: {payload}", file=sys.stderr)
            errors.append({"source": src, "error": payload})
    if errors:
        print(f"  [skip] {len(errors)} source(s) skipped due to errors", file=sys.stderr)
    if items and not all_segments:
        # An empty pool silently propagates into an empty-but-valid-looking dataset (the
        # vacuous-success failure mode) — total failure must raise even under on_error='skip'.
        raise RuntimeError(
            f"segment pool is EMPTY: {len(items)} source(s) yielded no segments"
            + (f" ({len(errors)} skipped; first error: {errors[0]['error']})" if errors
               else " (no errors — is every source shorter than the segment window?)"))

    if tags_of:
        for s in all_segments:
            s.tags = frozenset(tags_of(s))
    all_segments.sort(key=lambda s: (s.source_id, s.start_idx))   # determinism
    emit(progress, "scoring", "complete", current=len(items), total=len(items),
         detail=f"{len(all_segments)} segments, {len(errors)} skipped")
    return all_segments, errors, source_tracks
