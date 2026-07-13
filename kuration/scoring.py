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
from concurrent.futures import ProcessPoolExecutor

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
    schema, define, registered, tags_spec, adapter, segmenter, source, group, streaming = args
    try:
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
                  on_error: str = "skip", streaming: bool = False):
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
    items = [(schema, define, registered, tags_spec, adapter, segmenter, src,
              (group_of(src) if group_of else None), streaming) for src in sources]
    if workers and workers > 1:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as ex:
            results = list(ex.map(_score_one, items))
    else:
        results = [_score_one(it) for it in items]

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
    return all_segments, errors, source_tracks
