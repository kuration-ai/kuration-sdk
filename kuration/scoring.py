"""Parallel, fault-tolerant feature pass — shared by curate / experiment / write_dataset.

The single place sources are read, segmented and scored. Parallelism is per-source over a
process pool: each worker rebuilds the `FeatureEngine` in-process (the compiled DSL metrics
are closures and can't be pickled, so they're recompiled from the JSON `define` + schema —
both picklable; the adapter and segmenter pickle as-is). A source that raises is
**skipped-and-logged** (loud: stderr line per skip + a summary + collected in the returned
errors), never silently dropped. Determinism holds: the parent sorts the pooled segments by
`(source_id, start_idx)`, so the result is identical regardless of worker count or completion
order.

Note: the default process start method on Linux is `fork`, so the metric registry (built-ins
and any `@register_metric` defined before the call) is inherited by workers. Under `spawn`,
custom metrics must be importable at module scope.
"""
from __future__ import annotations

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
            for seg, win in segmenter.stream(source, tracked(adapter.ticks(source))):
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
    items = [(schema, define, registered, tags_spec, adapter, segmenter, src,
              (group_of(src) if group_of else None), streaming) for src in sources]
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
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

    if tags_of:
        for s in all_segments:
            s.tags = frozenset(tags_of(s))
    all_segments.sort(key=lambda s: (s.source_id, s.start_idx))   # determinism
    return all_segments, errors, source_tracks
