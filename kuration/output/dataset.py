"""`write_dataset` — the end-to-end output pipeline.

sources -> features -> segment priorities -> per-source priority -> hours-budget track fill
(train/test) -> write each admitted track's shards -> dataset_manifest + reproducible config
(embedding schema + strategy + output spec). The hours budget is what makes the emitted set
the strategy's *best* N hours rather than a random draw.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional

from ..adapters.base import InputAdapter
from ..binning import bin_segments
from ..manifest import build_config, write_config
from ..schema import SchemaSpec
from ..scoring import score_sources
from ..segment import Segmenter
from ..strategy import Strategy, segment_priorities
from .base import OutputEncoder, PassthroughEncoder
from .fill import fill_to_hours
from .shardwriter import write_track
from .spec import OutputSpec
from .track import Track, tracks_from_ticks
from ..versioning import ARTIFACT_FORMAT_VERSION


def _check_resume_run(out_dir: str, signature: dict, resume: bool) -> None:
    """Pin the complete effective run because resume skips existing files by name."""
    from ..io import exists, read_text, write_text
    marker = f"{out_dir}/.kuration_output_spec.json"
    if resume and exists(marker):
        old = json.loads(read_text(marker))
        old_output = old.get("output")
        new_output = signature["payload"]["output"]
        if old_output != new_output:
            raise ValueError(
                f"resume=True but {out_dir!r} was started with a different output spec "
                f"(fingerprint mismatch); existing shards would not match the "
                f"new manifest — use a fresh out_dir or restore the original spec")
        old_fp = old.get("run_fingerprint")
        if old_fp is None:
            raise ValueError("resume=True found a legacy output-only marker; complete run "
                             "provenance is unavailable, so reusing its shards is unsafe")
        if old_fp != signature["fingerprint"]:
            raise ValueError(
                f"resume=True but {out_dir!r} was started with different inputs or run "
                f"configuration ({old_fp[:12]} != {signature['fingerprint'][:12]}); use a "
                f"fresh out_dir or restore the original job")
    if not (resume and exists(marker)):
        write_text(marker, json.dumps({"artifact_format_version": ARTIFACT_FORMAT_VERSION,
                                       "run_fingerprint": signature["fingerprint"],
                                       "output": signature["payload"]["output"]}, indent=2))


def write_dataset(out_dir: str, *, schema: SchemaSpec, adapter: InputAdapter,
                  sources: Iterable[str], strategy: Strategy, segmenter: Segmenter,
                  spec: OutputSpec, encoder: Optional[OutputEncoder] = None,
                  train_hours: float, test_hours: float = 0.0,
                  sample_hz: Optional[float] = None, seed: int = 0,
                  group_of: Optional[Callable[[str], object]] = None,
                  tags_of: Optional[Callable] = None,
                  clip_aggregate: str = "max", span_steps: Optional[int] = None,
                  span_seconds: Optional[float] = None, workers: int = 1,
                  on_error: str = "skip", resume: bool = False, streaming: bool = False,
                  progress=None, transforms=(), registry=None, executor: str = "local") -> dict:
    """`workers>1` parallelizes the feature pass; `on_error='skip'` logs + skips bad sources;
    `resume=True` skips re-writing shards whose files already exist.

    `span_steps` / `span_seconds` subdivide each track into independently-budgeted spans, so
    the fill can keep the *best parts* of a long source rather than the whole source or nothing.
    Unset → one span per whole track (the default).

    `streaming=True` scores and writes out-of-core when the adapter provides a bounded-memory
    tick iterator. Adapters that cannot honor that contract reject the option loudly."""
    started = datetime.now(timezone.utc).isoformat()
    sources = list(sources)
    sample_hz = schema.sample_hz if sample_hz is None else sample_hz
    if sample_hz is None:
        raise ValueError("write_dataset needs sample_hz (schema.sample_hz or explicit) for the "
                         "hours budget")
    if (not isinstance(sample_hz, (int, float)) or isinstance(sample_hz, bool)
            or not math.isfinite(sample_hz) or sample_hz <= 0):
        raise ValueError(f"sample_hz must be a finite positive number, got {sample_hz!r}")
    for name, value in (("train_hours", train_hours), ("test_hours", test_hours)):
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    if clip_aggregate not in {"max", "min", "mean"}:
        raise ValueError("clip_aggregate must be 'max', 'min', or 'mean'")
    if span_steps is not None and span_seconds is not None:
        raise ValueError("pass only one of span_steps or span_seconds")
    if span_steps is not None and (not isinstance(span_steps, int)
                                   or isinstance(span_steps, bool) or span_steps < 1):
        raise ValueError("span_steps must be a positive integer")
    if span_seconds is not None and (
            not isinstance(span_seconds, (int, float)) or isinstance(span_seconds, bool)
            or not math.isfinite(span_seconds) or span_seconds <= 0):
        raise ValueError("span_seconds must be a finite positive number")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not isinstance(resume, bool) or not isinstance(streaming, bool):
        raise ValueError("resume and streaming must be true or false")
    spec.validate_schema(schema)
    span_len = span_steps if span_steps is not None else (
        round(span_seconds * sample_hz) if span_seconds is not None else None)
    if span_len is not None and span_len < 1:
        raise ValueError("span_seconds is shorter than one sample at the effective sample_hz")
    if span_len is not None and spec.short_policy == "drop" and span_len < spec.shard_length:
        raise ValueError(
            f"span_steps={span_len} < shard_length={spec.shard_length} with "
            f"short_policy='drop': every admitted span would chunk to zero shards while "
            f"still consuming the hours budget")
    from ..progress import emit
    emit(progress, "provenance", "start", total=len(sources))
    from ..provenance import build_run_signature
    signature = build_run_signature(
        schema=schema, strategy=strategy, spec=spec, segmenter=segmenter, sources=sources,
        adapter=adapter, encoder=(encoder or PassthroughEncoder()), seed=seed, sample_hz=sample_hz,
        train_hours=train_hours, test_hours=test_hours, span_len=span_len,
        clip_aggregate=clip_aggregate, group_of=group_of, tags_of=tags_of)
    _check_resume_run(out_dir, signature, resume)
    emit(progress, "provenance", "complete", current=len(sources), total=len(sources),
         detail=signature["fingerprint"][:12])
    transform_outputs = {o for t in transforms for o in t["outputs"]}
    registered = [r for r in strategy.referenced_metrics() if r not in strategy.define
                  and r not in strategy.scorer_outputs() and r not in transform_outputs]

    # Parallel score pass: returns features + per-source track lengths (so the fill needs no
    # second read of the data). The data itself is re-read only for admitted sources, below.
    all_segments, errors, source_tracks = score_sources(
        schema=schema, adapter=adapter, sources=sources, define=strategy.define,
        registered=registered, segmenter=segmenter, tags_spec=strategy.tags,
        group_of=group_of, tags_of=tags_of, workers=workers, on_error=on_error,
        streaming=streaming, progress=progress, executor=executor)

    if transforms:
        from ..transforms import apply_transforms
        apply_transforms(transforms, all_segments, adapter=adapter,
                         registry=registry or strategy.registry, progress=progress)

    if strategy.scorers:   # whole-pool learned/action-aware scoring -> features
        from ..scorers import apply_scorers
        apply_scorers(strategy.scorers, all_segments, schema=schema, adapter=adapter, seed=seed,
                      progress=progress, registry=strategy.registry)

    emit(progress, "selection", "start", total=len(all_segments))
    pool = all_segments
    if strategy.filters:
        from ..methods import apply_filters
        pool = apply_filters(all_segments, strategy, seed=seed)
    bin_segments(pool, strategy.axes)
    prio = segment_priorities(pool, strategy, seed=seed)

    # Per-source segment data so a span's priority/tags/features aggregate only the segments
    # overlapping THAT span (not the whole source). Aggregated features populate Track.meta,
    # which makes documented `segment:<feature>` output modalities work end to end.
    segs_by_src: dict = {}
    for i, s in enumerate(pool):
        segs_by_src.setdefault(s.source_id, []).append(
            (s.start_idx, s.end_idx, float(prio[i]), s.tags, dict(s.features)))
    def _reduce(values: list[float]) -> float:
        if clip_aggregate == "max":
            return max(values)
        if clip_aggregate == "min":
            return min(values)
        return sum(values) / len(values)

    def _span_summary(src, start, end):
        prios: list[float] = []
        tags: set = set()
        feature_values: dict[str, list[float]] = {}
        for st, en, p, tg, features in segs_by_src.get(src, ()):
            if st < end and en > start:
                prios.append(p)
                tags |= tg
                for name, value in features.items():
                    feature_values.setdefault(name, []).append(float(value))
        meta = {name: _reduce(values) for name, values in feature_values.items()}
        return ((_reduce(prios) if prios else float("-inf")), frozenset(tags), meta)

    # Build span items: (source, entity, span_index) -> {priority, seconds, group, tags}.
    items, span_index, span_meta = [], {}, {}
    for src, tlist in source_tracks.items():
        g = group_of(src) if group_of else None
        for (eid, n) in tlist:
            bounds = ([(s, min(s + span_len, n)) for s in range(0, n, span_len)]
                      if span_len else [(0, n)])
            for k, (start, end) in enumerate(bounds):
                p, tg, meta = _span_summary(src, start, end)
                items.append({"id": (src, eid, k), "source_id": src, "priority": p,
                              "seconds": (end - start) / sample_hz, "group": g, "tags": tg})
                span_index[(src, eid, k)] = (start, end)
                span_meta[(src, eid, k)] = meta
    n_tracks = sum(len(t) for t in source_tracks.values())
    group_overshoot = (strategy.group_cap or {}).get("max_group_overshoot") \
        if strategy.group_cap else None
    admitted, split, budget = fill_to_hours(
        items, train_hours=train_hours, test_hours=test_hours,
        max_group_overshoot=group_overshoot, tag_coverage=strategy.tag_coverage, seed=seed,
        return_stats=True)
    emit(progress, "selection", "complete", current=len(admitted), total=len(items),
         detail=f"{len(admitted)} items admitted")

    # Re-read each admitted source once; write each admitted span as its own (sub-)track.
    entries = []
    by_source: dict = {}
    # `admitted` is a set — iterate in sorted order so manifest entry order (and thus
    # ShardDataset indexing) is hash-seed-independent.
    for (src, eid, k) in sorted(admitted, key=lambda t: (str(t[0]), str(t[1]), t[2])):
        by_source.setdefault(src, []).append((eid, k))
    emit(progress, "writing", "start", total=len(by_source))
    for source_i, src in enumerate(sorted(by_source), 1):
        suffix = lambda k: "" if span_len is None else f"_span{k:03d}"
        if streaming:
            from .stream import stream_source_shards
            jobs = [{"eid": eid, "span_index": k, "start": span_index[(src, eid, k)][0],
                     "end": span_index[(src, eid, k)][1], "split": split.get(src, "train"),
                     "name_suffix": suffix(k), "group": group_of(src) if group_of else None,
                     "meta": span_meta[(src, eid, k)]}
                    for (eid, k) in by_source[src]]
            tick_iter = (adapter.iter_ticks(src) if hasattr(adapter, "iter_ticks")
                         else adapter.ticks(src))
            entries += stream_source_shards(out_dir, src, tick_iter, jobs,
                                            schema=schema, spec=spec, encoder=encoder, resume=resume,
                                            media_provider=getattr(adapter, "provider", None))
            emit(progress, "writing", "advance", current=source_i, total=len(by_source),
                 source=src, status="ok")
            continue
        tracks = {t.entity_id: t for t in tracks_from_ticks(schema, list(adapter.ticks(src)), src)}
        for (eid, k) in by_source[src]:
            tr = tracks.get(eid)
            if tr is None:
                continue
            start, end = span_index[(src, eid, k)]
            sub = tr if (start, end) == (0, tr.n_steps) else Track(
                tr.source_id, tr.entity_id, tr.records[start:end], dict(tr.meta))
            sub.meta.update(span_meta[(src, eid, k)])
            sub.meta["group"] = group_of(src) if group_of else None
            entries += write_track(out_dir, sub, spec, encoder=encoder, schema=schema,
                                   split=split.get(src, "train"), resume=resume,
                                   name_suffix=suffix(k),
                                   media_provider=getattr(adapter, "provider", None))
        emit(progress, "writing", "advance", current=source_i, total=len(by_source),
             source=src, status="ok")
    emit(progress, "writing", "complete", current=len(by_source), total=len(by_source),
         detail=f"{len(entries)} shards")

    emitted = {sp: sum(e["n_steps"] for e in entries if e["split"] == sp) / sample_hz
               for sp in ("train", "test")}
    budget["emitted_seconds"] = emitted
    budget["emission_loss_seconds"] = {
        sp: max(0.0, budget["admitted_seconds"][sp] - emitted[sp])
        for sp in ("train", "test")}
    budget["emitted_underfill_seconds"] = {
        sp: max(0.0, budget["requested_seconds"][sp] - emitted[sp])
        for sp in ("train", "test")}

    from ..io import write_text
    write_text(f"{out_dir}/dataset_manifest.json",
               json.dumps({"artifact_format_version": ARTIFACT_FORMAT_VERSION,
                           "count": len(entries), "shards": entries, "budget": budget}, indent=2))
    results = {"shards": len(entries), "tracks_admitted": len(admitted),
               "tracks_total": n_tracks, "sources_skipped": len(errors),
               "n_train": sum(1 for v in split.values() if v == "train"),
               "n_test": sum(1 for v in split.values() if v == "test"),
               "budget": budget}
    ended = datetime.now(timezone.utc).isoformat()
    cfg = build_config(schema, strategy, results=results, seed=seed, started=started, ended=ended)
    cfg["output"] = spec.to_dict()
    cfg["provenance"] = signature
    write_config(f"{out_dir}/config.json", cfg)
    from ..results import DatasetWriteResult
    return DatasetWriteResult(shards=entries, split=split, admitted=admitted, results=results,
                              errors=errors)
