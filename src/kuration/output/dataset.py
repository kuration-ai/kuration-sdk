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
from typing import Any, Callable, Iterable, Optional

from ..adapters.base import InputAdapter
from ..binning import bin_segments
from ..manifest import build_config, write_config
from ..schema import SchemaSpec
from ..scoring import score_sources
from ..segment import Segmenter
from ..strategy import Strategy, segment_priorities
from .base import OutputEncoder, PassthroughEncoder
from .external_fill import external_fill_to_hours
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


def _registered_features(strategy, transforms):
    transform_outputs = {output for call in transforms for output in call["outputs"]}
    return [name for name in strategy.referenced_metrics()
            if name not in strategy.define and name not in strategy.scorer_outputs()
            and name not in transform_outputs]


def _span_stream(prioritized, source_tracks, *, sample_hz, span_len, clip_aggregate, group_of,
                 spill_dir):
    from ..external import iter_external_span_items
    return iter_external_span_items(
        prioritized, source_tracks, sample_hz=sample_hz, span_len=span_len,
        clip_aggregate=clip_aggregate, group_of=group_of, spill_dir=spill_dir)


def _legacy_hours_selection(*, out_dir, schema, adapter, sources, strategy, segmenter,
                            sample_hz, span_len, clip_aggregate, group_of, tags_of, workers,
                            on_error, streaming, progress, transforms, registry, executor,
                            train_hours, test_hours, seed):
    all_segments, errors, source_tracks = score_sources(
        schema=schema, adapter=adapter, sources=sources, define=strategy.define,
        registered=_registered_features(strategy, transforms), segmenter=segmenter,
        tags_spec=strategy.tags, group_of=group_of, tags_of=tags_of, workers=workers,
        on_error=on_error, streaming=streaming, progress=progress, executor=executor)
    if transforms:
        from ..transforms import apply_transforms
        apply_transforms(transforms, all_segments, adapter=adapter,
                         registry=registry or strategy.registry, progress=progress)
    if strategy.scorers:
        from ..scorers import apply_scorers
        apply_scorers(strategy.scorers, all_segments, schema=schema, adapter=adapter, seed=seed,
                      progress=progress, registry=strategy.registry)
    from ..progress import emit
    emit(progress, "selection", "start", total=len(all_segments))
    pool = all_segments
    if strategy.filters:
        from ..methods import apply_filters
        pool = apply_filters(all_segments, strategy, seed=seed)
    bin_segments(pool, strategy.axes)
    priorities = segment_priorities(pool, strategy, seed=seed)
    by_source: dict[str, list[tuple[Any, float]]] = {}
    for index, segment in enumerate(pool):
        by_source.setdefault(segment.source_id, []).append(
            (segment, float(priorities[index])))

    def prioritized_segments():
        for source in sorted(by_source, key=str):
            for segment, priority in by_source[source]:
                yield segment, priority

    from ..io import is_remote
    spill_dir = None if is_remote(out_dir) else f"{out_dir}/.kuration/spill"
    spans = _span_stream(
        prioritized_segments(), source_tracks, sample_hz=sample_hz, span_len=span_len,
        clip_aggregate=clip_aggregate, group_of=group_of, spill_dir=spill_dir)
    group_overshoot = (strategy.group_cap or {}).get("max_group_overshoot") \
        if strategy.group_cap else None
    admitted, split, budget, details = external_fill_to_hours(
        spans, train_hours=train_hours, test_hours=test_hours,
        max_group_overshoot=group_overshoot, tag_coverage=strategy.tag_coverage,
        seed=seed, spill_dir=spill_dir)
    emit(progress, "selection", "complete", current=len(admitted),
         total=budget["items_total"], detail=f"{len(admitted)} items admitted")
    return (admitted, split, budget, details, errors, source_tracks, None,
            len(all_segments))


def _bounded_hours_selection(*, out_dir, signature, schema, adapter, sources, strategy,
                             segmenter, sample_hz, span_len, clip_aggregate, group_of,
                             tags_of, on_error, progress, transforms, registry,
                             train_hours, test_hours, batch_size, checkpoint_dir, spill_dir,
                             max_in_flight):
    from ..bounded import checkpoint_feature_stage
    from ..checkpoints import CheckpointIdentity, StageCheckpoint, canonical_hash
    from ..external import (iter_external_top_k_priorities,
                            iter_external_uniform_priorities,
                            iter_external_weighted_priorities)
    from ..io import is_remote
    from .external_fill import external_fill_to_hours

    spill_dir = spill_dir or (None if is_remote(out_dir) else f"{out_dir}/.kuration/spill")
    checkpoint_dir = checkpoint_dir or f"{out_dir}/.kuration/checkpoints"
    physical = {
        "version": 1, "backend": "local", "mode": f"bounded-hours-{strategy.method}",
        "resolved_transforms": list(transforms),
        "stages": [
            {"id": "stage000", "mode": "pipeline_parallel",
             "operators": ["read", "segment", "metrics", "predicates", "transforms"],
             "checkpoint": "partition-reusable"},
            {"id": "stage001", "mode": "data_parallel",
             "operators": ["normalize", "top-k-priority", "span-fill"],
             "checkpoint": "reducer-state", "passes": 2},
            {"id": "stage002", "mode": "pipeline_parallel", "operators": ["output"],
             "checkpoint": "partition-reusable"}],
        "spill_path": spill_dir}
    active_registry = registry or strategy.registry
    components = []
    if active_registry is not None:
        for call in transforms:
            try:
                components.append(active_registry.descriptor(
                    "transform", call["implementation"]).to_dict())
            except ValueError:
                pass
    identity = CheckpointIdentity(
        logical_plan_hash=canonical_hash(signature["payload"]),
        physical_plan_hash=canonical_hash(physical),
        input_fingerprints=tuple(signature["payload"].get("sources", ())),
        schema=schema.to_dict(), components=tuple(components),
        parameters={"seed": signature["payload"]["seed"]})
    checkpoint = StageCheckpoint(checkpoint_dir, "stage000", identity)
    stage = checkpoint_feature_stage(
        schema=schema, adapter=adapter, sources=sources, define=strategy.define,
        registered=_registered_features(strategy, transforms), segmenter=segmenter,
        checkpoint=checkpoint, batch_size=batch_size, tags_spec=strategy.tags,
        predicates=strategy.predicates, transforms=transforms,
        registry=registry or strategy.registry, group_of=group_of, tags_of=tags_of,
        on_error=on_error, progress=progress, max_in_flight=max_in_flight)
    marker = checkpoint.finalize([commit.partition_id for commit in stage.commits])
    normalize = "quantile" if strategy.normalize == "none" else strategy.normalize
    if strategy.method == "top-k":
        prioritized = iter_external_top_k_priorities(
            stage.commits, weights=strategy.weights, normalize=normalize, spill_dir=spill_dir)
    elif strategy.method == "weighted":
        prioritized = iter_external_weighted_priorities(
            stage.commits, weights=strategy.weights, normalize=normalize,
            seed=signature["payload"]["seed"], spill_dir=spill_dir)
    else:
        prioritized = iter_external_uniform_priorities(
            stage.commits, axes=strategy.axes, seed=signature["payload"]["seed"],
            spill_dir=spill_dir)
    spans = _span_stream(
        prioritized, stage.source_tracks, sample_hz=sample_hz, span_len=span_len,
        clip_aggregate=clip_aggregate, group_of=group_of, spill_dir=spill_dir)
    group_overshoot = (strategy.group_cap or {}).get("max_group_overshoot") \
        if strategy.group_cap else None
    admitted, split, budget, details = external_fill_to_hours(
        spans, train_hours=train_hours, test_hours=test_hours,
        max_group_overshoot=group_overshoot, tag_coverage=strategy.tag_coverage,
        seed=signature["payload"]["seed"], spill_dir=spill_dir)
    execution = {"bounded": True, "physical_plan": physical, "checkpoint": marker,
                 "telemetry": stage.telemetry}
    return (admitted, split, budget, details, list(stage.errors), stage.source_tracks,
            execution, stage.candidate_rows)


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
                  progress=None, transforms=(), registry=None, executor: str = "local",
                  memory_budget: int | None = None, batch_size: int = 1024,
                  checkpoint_dir: str | None = None, spill_dir: str | None = None,
                  approximation: str = "exact", max_in_flight: int = 2) -> dict:
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
    if (not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1):
        raise ValueError("batch_size must be a positive integer")
    if (not isinstance(max_in_flight, int) or isinstance(max_in_flight, bool)
            or max_in_flight < 1):
        raise ValueError("max_in_flight must be a positive integer")
    if memory_budget is not None and (not isinstance(memory_budget, int)
                                      or isinstance(memory_budget, bool) or memory_budget < 1):
        raise ValueError("memory_budget must be a positive integer or None")
    if approximation not in {"exact", "approximate"}:
        raise ValueError("approximation must be 'exact' or 'approximate'")
    if approximation == "approximate":
        raise ValueError(f"selection method {strategy.method!r} has no declared approximate "
                         "hours implementation yet")
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
    bounded = (strategy.method in {"top-k", "weighted", "uniform-bins"}
               and not strategy.filters and not strategy.scorers
               and executor == "local")
    if bounded:
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            bounded = False
    if memory_budget is not None and not bounded:
        blockers = []
        if strategy.method not in {"top-k", "weighted", "uniform-bins"}:
            blockers.append(f"method {strategy.method!r}")
        if strategy.filters:
            blockers.append("global filters")
        if strategy.scorers:
            blockers.append("global scorers")
        if executor != "local":
            blockers.append(f"executor {executor!r}")
        if not blockers:
            blockers.append("PyArrow is unavailable")
        raise ValueError("memory_budget requires a bounded exact hours plan; unsupported: "
                         + ", ".join(blockers))
    from ..progress import emit
    emit(progress, "provenance", "start", total=len(sources))
    from ..provenance import build_run_signature
    signature = build_run_signature(
        schema=schema, strategy=strategy, spec=spec, segmenter=segmenter, sources=sources,
        adapter=adapter, encoder=(encoder or PassthroughEncoder()), seed=seed, sample_hz=sample_hz,
        train_hours=train_hours, test_hours=test_hours, span_len=span_len,
        clip_aggregate=clip_aggregate, group_of=group_of, tags_of=tags_of,
        transforms=transforms, registry=registry)
    _check_resume_run(out_dir, signature, resume)
    emit(progress, "provenance", "complete", current=len(sources), total=len(sources),
         detail=signature["fingerprint"][:12])
    if bounded:
        emit(progress, "selection", "start", total=None)
        selection = _bounded_hours_selection(
            out_dir=out_dir, signature=signature, schema=schema, adapter=adapter,
            sources=sources, strategy=strategy, segmenter=segmenter, sample_hz=sample_hz,
            span_len=span_len, clip_aggregate=clip_aggregate, group_of=group_of,
            tags_of=tags_of, on_error=on_error, progress=progress, transforms=transforms,
            registry=registry, train_hours=train_hours, test_hours=test_hours,
            batch_size=batch_size, checkpoint_dir=checkpoint_dir, spill_dir=spill_dir,
            max_in_flight=max_in_flight)
    else:
        selection = _legacy_hours_selection(
            out_dir=out_dir, schema=schema, adapter=adapter, sources=sources,
            strategy=strategy, segmenter=segmenter, sample_hz=sample_hz,
            span_len=span_len, clip_aggregate=clip_aggregate, group_of=group_of,
            tags_of=tags_of, workers=workers, on_error=on_error, streaming=streaming,
            progress=progress, transforms=transforms, registry=registry, executor=executor,
            train_hours=train_hours, test_hours=test_hours, seed=seed)
    (admitted, split, budget, admitted_details, errors, source_tracks, execution,
     segment_count) = selection
    n_tracks = sum(len(tracks) for tracks in source_tracks.values())
    if bounded:
        emit(progress, "selection", "complete", current=len(admitted),
             total=budget["items_total"], detail=f"{len(admitted)} items admitted")

    # Re-read each admitted source once; write each admitted span as its own (sub-)track.
    entries = []
    by_source: dict[str, list[tuple[Any, int]]] = {}
    # `admitted` is a set — iterate in sorted order so manifest entry order (and thus
    # ShardDataset indexing) is hash-seed-independent.
    for (src, eid, k) in sorted(admitted, key=lambda t: (str(t[0]), str(t[1]), t[2])):
        by_source.setdefault(src, []).append((eid, k))
    emit(progress, "writing", "start", total=len(by_source))
    for source_i, src in enumerate(sorted(by_source), 1):
        suffix = lambda k: "" if span_len is None else f"_span{k:03d}"
        if streaming:
            from .stream import stream_source_shards
            jobs = [{"eid": eid, "span_index": k,
                     "start": admitted_details[(src, eid, k)]["start"],
                     "end": admitted_details[(src, eid, k)]["end"],
                     "split": split.get(src, "train"),
                     "name_suffix": suffix(k), "group": group_of(src) if group_of else None,
                     "meta": admitted_details[(src, eid, k)]["meta"]}
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
            detail = admitted_details[(src, eid, k)]
            start, end = detail["start"], detail["end"]
            sub = tr if (start, end) == (0, tr.n_steps) else Track(
                tr.source_id, tr.entity_id, tr.records[start:end], dict(tr.meta))
            sub.meta.update(detail["meta"])
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
               "tracks_total": n_tracks, "segments_total": segment_count,
               "sources_skipped": len(errors),
               "n_train": sum(1 for v in split.values() if v == "train"),
               "n_test": sum(1 for v in split.values() if v == "test"),
               "budget": budget}
    ended = datetime.now(timezone.utc).isoformat()
    cfg = build_config(schema, strategy, results=results, seed=seed, started=started, ended=ended)
    cfg["output"] = spec.to_dict()
    cfg["provenance"] = signature
    cfg["errors"] = errors
    if execution is not None:
        cfg["execution"] = execution
    write_config(f"{out_dir}/config.json", cfg)
    from ..results import DatasetWriteResult
    return DatasetWriteResult(shards=entries, split=split, admitted=admitted, results=results,
                              errors=errors)
