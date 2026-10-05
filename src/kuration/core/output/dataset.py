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
from ..results import DatasetWriteResult
from ..schema import SchemaSpec
from ..scoring import score_sources
from ..segment import Segmenter
from ..strategy import Strategy, segment_priorities
from .base import OutputEncoder, PassthroughEncoder
from .external_fill import external_fill_to_hours
from .shardwriter import entity_id_token, write_track
from .spec import OutputSpec
from .track import Track, tracks_from_ticks
from ..versioning import ARTIFACT_FORMAT_VERSION


def _admitted_order_key(item):
    source_id, entity_id, span_index = item
    return str(source_id), entity_id_token(entity_id), span_index


def _check_resume_run(out_dir: str, signature: dict, resume: bool, *,
                      publish: bool = True) -> None:
    """Validate resume identity and optionally publish the run marker.

    Writers use ``publish=False`` before selection so a failed replacement cannot mutate a
    committed artifact. The marker is published only after ``begin_dataset`` establishes the
    write boundary.
    """
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
    if publish and not (resume and exists(marker)):
        write_text(marker, json.dumps({"artifact_format_version": ARTIFACT_FORMAT_VERSION,
                                       "run_fingerprint": signature["fingerprint"],
                                       "output": signature["payload"]["output"]}, indent=2))


def _resume_entries(out_dir: str, resume: bool) -> dict[str, dict]:
    """Load checksum-bearing entries from the last committed manifest, if any."""
    from ..io import exists, read_text
    path = f"{out_dir}/dataset_manifest.json"
    if not resume or not exists(path):
        return {}
    manifest = json.loads(read_text(path))
    # A sidecar-v1 manifest is reusable only after its final commit record has been
    # published and still matches both pieces of dataset metadata.  An interrupted run
    # is expected here: treat it as an empty resume index so every shard is verified and
    # rewritten instead of trusting partial metadata.
    if manifest.get("commit_protocol") is not None:
        from ..loaders import _verify_commit
        try:
            _verify_commit(out_dir, manifest)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            # A writer may have published ``state=writing`` and then stopped before replacing
            # the old manifest. That manifest remains a safe resume index because every shard
            # is checksum-validated below; accept it only when the writing marker itself pins
            # the exact manifest bytes we just read.
            try:
                from ..io import file_integrity
                commit = json.loads(read_text(f"{out_dir}/dataset.commit.json"))
                expected = (commit.get("files") or {}).get("dataset_manifest.json")
                resumable_write = (commit.get("protocol") == "sidecar-v1"
                                   and commit.get("state") == "writing"
                                   and isinstance(expected, dict)
                                   and file_integrity(path) == expected)
            except (FileNotFoundError, OSError, ValueError, TypeError,
                    json.JSONDecodeError):
                resumable_write = False
            if not resumable_write:
                return {}
    from ..loaders import rebase_entries
    entries = rebase_entries(list(manifest.get("shards", [])), out_dir)
    return {entry["path"]: entry for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("path"), str)}


def _registered_features(strategy, transforms):
    transform_outputs = {output for call in transforms for output in call["outputs"]}
    transform_inputs = {
        token.split(":", 1)[1]
        for call in transforms for token in call["inputs"]
        if token.startswith("feature:")
    }
    return sorted(
        name for name in strategy.referenced_metrics() | transform_inputs
        if name not in strategy.define and name not in strategy.scorer_outputs()
        and name not in transform_outputs
    )


def _minimum_output_working_bytes(spec: OutputSpec, span_len: int | None) -> int:
    """Conservative known-size portion of one streaming output accumulator."""
    import numpy as np
    rows = min(spec.shard_length, span_len) if span_len is not None else spec.shard_length
    row_bytes = sum(np.dtype(modality.dtype).itemsize * math.prod(modality.shape or (1,))
                    for modality in spec.modalities)
    if spec.presence_mask:
        row_bytes += 1
    # Output arrays and encoder/provider staging need at least two payload copies; the fixed
    # per-row allowance covers canonical record/container overhead not represented by dtype.
    return 4096 + rows * (2 * row_bytes + 256)


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
        on_error=on_error, streaming=streaming, progress=progress, executor=executor,
        registry=registry or strategy.registry)
    if transforms:
        from ..transforms import apply_transforms
        all_segments = apply_transforms(
            transforms, all_segments, adapter=adapter,
            registry=registry or strategy.registry, progress=progress,
            on_error=on_error, errors=errors)
    if strategy.scorers:
        from ..scorers import apply_scorers
        apply_scorers(strategy.scorers, all_segments, schema=schema, adapter=adapter, seed=seed,
                      progress=progress, registry=strategy.registry,
                      on_error=on_error, errors=errors)
    from ..progress import emit
    emit(progress, "selection", "start", total=len(all_segments))
    pool = all_segments
    if strategy.filters:
        from ..methods import apply_filters
        pool = apply_filters(all_segments, strategy, seed=seed, workers=workers)
    bin_segments(pool, strategy.axes)
    priorities = segment_priorities(pool, strategy, seed=seed, workers=workers)
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
                             max_in_flight, memory_budget):
    from ..bounded import checkpoint_feature_stage, resolve_bounded_memory_limits
    from ..checkpoints import (
        CheckpointIdentity,
        StageCheckpoint,
        artifact_checkpoint_summary,
        canonical_hash,
    )
    from ..external import (iter_external_top_k_priorities,
                            iter_external_uniform_priorities,
                            iter_external_weighted_priorities)
    from ..io import is_remote
    from .external_fill import external_fill_to_hours

    spill_dir = spill_dir or (None if is_remote(out_dir) else f"{out_dir}/.kuration/spill")
    checkpoint_root = checkpoint_dir or f"{out_dir}/.kuration/checkpoints"
    limits = resolve_bounded_memory_limits(
        schema=schema, segmenter=segmenter, memory_budget=memory_budget,
        batch_size=batch_size, max_in_flight=max_in_flight, transforms=transforms)
    physical = {
        "version": 1, "backend": "local", "mode": f"bounded-hours-{strategy.method}",
        "resolved_transforms": list(transforms),
        "memory_budget": memory_budget,
        "batch_size": limits["batch_size"],
        "max_in_flight": limits["max_in_flight"],
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
    artifact_physical = {
        **physical,
        "spill_path": None,
        "spill_configured": spill_dir is not None,
    }
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
        physical_plan_hash=canonical_hash(artifact_physical),
        input_fingerprints=tuple(signature["payload"].get("sources", ())),
        schema=schema.to_dict(), components=tuple(components),
        parameters={"seed": signature["payload"]["seed"]})
    checkpoint = StageCheckpoint(
        f"{checkpoint_root.rstrip('/')}/{identity.fingerprint}", "stage000", identity)
    stage = checkpoint_feature_stage(
        schema=schema, adapter=adapter, sources=sources, define=strategy.define,
        registered=_registered_features(strategy, transforms), segmenter=segmenter,
        checkpoint=checkpoint, batch_size=batch_size, tags_spec=strategy.tags,
        predicates=strategy.predicates, transforms=transforms,
        registry=registry or strategy.registry, group_of=group_of, tags_of=tags_of,
        on_error=on_error, progress=progress, max_in_flight=max_in_flight,
        memory_budget=memory_budget, retain_predicate_failures=True)
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
    if strategy.predicates:
        # Preserve local semantics: normalize/bin over the complete candidate pool, then mark
        # predicate failures ineligible.  Filtering partitions before the reducer makes an
        # optional PyArrow install change quantiles and therefore the selected dataset.
        from ..strategy import _passes
        prioritized = ((segment, priority if _passes(segment, strategy.predicates)
                        else float("-inf")) for segment, priority in prioritized)
    spans = _span_stream(
        prioritized, stage.source_tracks, sample_hz=sample_hz, span_len=span_len,
        clip_aggregate=clip_aggregate, group_of=group_of, spill_dir=spill_dir)
    group_overshoot = (strategy.group_cap or {}).get("max_group_overshoot") \
        if strategy.group_cap else None
    admitted, split, budget, details = external_fill_to_hours(
        spans, train_hours=train_hours, test_hours=test_hours,
        max_group_overshoot=group_overshoot, tag_coverage=strategy.tag_coverage,
        seed=signature["payload"]["seed"], spill_dir=spill_dir)
    execution = {
        "bounded": True,
        "physical_plan": artifact_physical,
        "checkpoint": artifact_checkpoint_summary(marker),
                 "telemetry": stage.telemetry}
    return (admitted, split, budget, details, list(stage.errors), stage.source_tracks,
            execution, stage.candidate_rows)


def _write_dataset_impl(out_dir: str, *, schema: SchemaSpec, adapter: InputAdapter,
                        sources: Iterable[str], strategy: Strategy, segmenter: Segmenter,
                        spec: OutputSpec, encoder: Optional[OutputEncoder] = None,
                        train_hours: float, test_hours: float = 0.0,
                        sample_hz: Optional[float] = None, seed: int = 0,
                        group_of: Optional[Callable[[str], object]] = None,
                        tags_of: Optional[Callable] = None,
                        clip_aggregate: str = "max", span_steps: Optional[int] = None,
                        span_seconds: Optional[float] = None, workers: int = 1,
                        on_error: str = "skip", resume: bool = False, force: bool = False,
                        streaming: bool = False, progress=None, transforms=(), registry=None,
                        executor: str = "local", memory_budget: int | None = None,
                        batch_size: int = 1024,
                        checkpoint_dir: str | None = None, spill_dir: str | None = None,
                        approximation: str = "exact", max_in_flight: int = 2,
                        _config_updates: Optional[dict] = None) -> DatasetWriteResult:
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
    if (train_hours > 0 or test_hours > 0) and not sources:
        raise ValueError(
            "hours output received no sources for a non-zero budget; nothing can be selected "
            "or written"
        )
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
    if not isinstance(resume, bool) or not isinstance(force, bool) \
            or not isinstance(streaming, bool):
        raise ValueError("resume, force, and streaming must be true or false")
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
    if _config_updates:
        from ..provenance import strict_json_value
        strict_json_value(_config_updates, label="output config metadata")
    spec.validate_schema(schema)
    from .containers import validate_container_availability
    validate_container_availability(spec.container, registry=registry or strategy.registry)
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
    if bounded and memory_budget is not None:
        # Resolve before provenance/output markers are published so an impossible budget is a
        # side-effect-free preflight failure.
        from ..bounded import resolve_bounded_memory_limits
        resolve_bounded_memory_limits(
            schema=schema, segmenter=segmenter, memory_budget=memory_budget,
            batch_size=batch_size, max_in_flight=max_in_flight, transforms=transforms)
        supports_streaming = getattr(adapter, "supports_streaming", True)
        if callable(supports_streaming):
            supports_streaming = supports_streaming()
        if supports_streaming is False:
            raise ValueError("memory_budget requires an adapter with bounded streaming output; "
                             f"{type(adapter).__name__} declares supports_streaming=False")
        output_minimum = _minimum_output_working_bytes(spec, span_len)
        if memory_budget < output_minimum:
            raise ValueError(
                f"memory_budget={memory_budget} bytes cannot hold one output shard accumulator "
                f"(estimated minimum {output_minimum} bytes); reduce shard/span size or raise "
                "memory_budget")
    from ..manifest import (
        validate_dataset_destination,
        validate_source_destination_separation,
    )
    destination_state = validate_dataset_destination(
        out_dir, force=force, resume=resume)
    validate_source_destination_separation(out_dir, sources, adapter=adapter)
    resolved_checkpoint_dir, resolved_spill_dir = checkpoint_dir, spill_dir
    from ..progress import emit
    emit(progress, "provenance", "start", total=len(sources))
    from ..provenance import (
        build_run_signature,
        require_reusable_source_identities,
        source_identities_allow_reuse,
    )
    signature = build_run_signature(
        schema=schema, strategy=strategy, spec=spec, segmenter=segmenter, sources=sources,
        adapter=adapter, encoder=(encoder or PassthroughEncoder()), seed=seed, sample_hz=sample_hz,
        train_hours=train_hours, test_hours=test_hours, span_len=span_len,
        clip_aggregate=clip_aggregate, group_of=group_of, tags_of=tags_of,
        transforms=transforms, registry=registry)
    _check_resume_run(out_dir, signature, resume, publish=False)
    source_identities = signature["payload"]["sources"]
    reusable_sources = source_identities_allow_reuse(source_identities)
    if resume:
        require_reusable_source_identities(
            source_identities, purpose="resume=True shard reuse")
    if bounded and not reusable_sources:
        if memory_budget is not None:
            require_reusable_source_identities(
                source_identities, purpose="memory-bounded checkpoint reuse")
        # Automatic bounded execution persists reusable feature partitions. A weak logical
        # source ID cannot safely key those partitions, so use the non-checkpointed exact
        # implementation when no hard memory bound was requested.
        bounded = False
    if bounded:
        from ..work_paths import resolve_bounded_work_paths
        resolved_checkpoint_dir, resolved_spill_dir = resolve_bounded_work_paths(
            out_dir=out_dir, sources=sources, adapter=adapter,
            checkpoint_dir=checkpoint_dir, spill_dir=spill_dir)
    resume_entries = _resume_entries(out_dir, resume)
    emit(progress, "provenance", "complete", current=len(sources), total=len(sources),
         detail=signature["fingerprint"][:12])
    from ..scoring import _cached_group_callback
    group_of = _cached_group_callback(group_of)
    if bounded:
        emit(progress, "selection", "start", total=None)
        selection = _bounded_hours_selection(
            out_dir=out_dir, signature=signature, schema=schema, adapter=adapter,
            sources=sources, strategy=strategy, segmenter=segmenter, sample_hz=sample_hz,
            span_len=span_len, clip_aggregate=clip_aggregate, group_of=group_of,
            tags_of=tags_of, on_error=on_error, progress=progress, transforms=transforms,
            registry=registry, train_hours=train_hours, test_hours=test_hours,
            batch_size=batch_size, checkpoint_dir=resolved_checkpoint_dir,
            spill_dir=resolved_spill_dir,
            max_in_flight=max_in_flight, memory_budget=memory_budget)
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

    # Validate every path identity and prove that the selected rows can yield at least one
    # shard before crossing the destructive force-replacement boundary.
    ordered_admitted = sorted(admitted, key=_admitted_order_key)
    if (train_hours > 0 or test_hours > 0) and not ordered_admitted:
        raise ValueError(
            "selection is empty for a non-zero hours budget — nothing to write (check sources, "
            "filters/predicates, errors, and the segment window)"
        )
    if ordered_admitted and spec.short_policy == "drop":
        predicted_shards = sum(
            max(0, int(admitted_details[item]["end"]) - int(admitted_details[item]["start"]))
            // spec.shard_length
            for item in ordered_admitted
        )
        if predicted_shards == 0:
            raise ValueError(
                "selection would emit zero shards: every admitted track/span is shorter than "
                f"shard_length={spec.shard_length} with short_policy='drop'; use "
                "short_policy='keep'/'pad' or reduce shard_length"
            )

    # Publication starts only after selection succeeds. Until this point an existing committed
    # destination remains byte-for-byte readable. Forced cleanup happens after the writing
    # sidecar is visible, so readers can never observe a partially replaced generation.
    from ..manifest import begin_dataset, clear_dataset_shards, pin_writing_dataset
    begin_dataset(out_dir)
    if force and destination_state != "absent":
        clear_dataset_shards(out_dir)
    _check_resume_run(out_dir, signature, resume, publish=True)
    pin_writing_dataset(out_dir)

    # Re-read each admitted source once; write each admitted span as its own (sub-)track.
    entries = []
    by_source: dict[str, list[tuple[Any, int]]] = {}
    # `admitted` is a set — iterate in sorted order so manifest entry order (and thus
    # ShardDataset indexing) is hash-seed-independent.
    for (src, eid, k) in ordered_admitted:
        by_source.setdefault(src, []).append((eid, k))
    emit(progress, "writing", "start", total=len(by_source))
    for source_i, src in enumerate(sorted(by_source), 1):
        # Content-derived, not rank-derived (F1): two curations that each admit a different
        # span at the same rank must not alias one filename.
        suffix = lambda start, end: (
            "" if span_len is None else f"_span{start:08d}_{end:08d}")
        expected_track_rows = list(source_tracks.get(src, ()))
        expected_entities = [entity_id for entity_id, _ in expected_track_rows]
        expected_lengths = {int(length) for _, length in expected_track_rows}
        if len(expected_lengths) != 1:
            raise ValueError(f"source {src!r} has inconsistent scored track lengths "
                             f"{sorted(expected_lengths)}")
        expected_ticks = next(iter(expected_lengths))
        if streaming or memory_budget is not None:
            from .stream import stream_source_shards
            jobs = [{"eid": eid, "span_index": k,
                     "start": admitted_details[(src, eid, k)]["start"],
                     "end": admitted_details[(src, eid, k)]["end"],
                     "split": split.get(src, "train"),
                     "name_suffix": suffix(admitted_details[(src, eid, k)]["start"],
                                          admitted_details[(src, eid, k)]["end"]),
                     "group": group_of(src) if group_of else None,
                     "meta": admitted_details[(src, eid, k)]["meta"]}
                    for (eid, k) in by_source[src]]
            # A budgeted output limits concurrent shard accumulators and replays a source for
            # additional batches.  This trades I/O for an enforceable bound instead of quietly
            # materializing a complete admitted source or every entity-span at once.
            jobs_per_pass = len(jobs)
            if memory_budget is not None:
                output_working = _minimum_output_working_bytes(spec, span_len)
                jobs_per_pass = max(1, memory_budget // output_working)
                if execution is not None:
                    execution["telemetry"]["output_accumulator_bytes"] = output_working
                    execution["telemetry"]["output_jobs_per_pass"] = jobs_per_pass
            for start in range(0, len(jobs), jobs_per_pass):
                tick_iter = (adapter.iter_ticks(src) if hasattr(adapter, "iter_ticks")
                             and not getattr(adapter, "in_memory", False)
                             else adapter.ticks(src))
                entries += stream_source_shards(
                    out_dir, src, tick_iter, jobs[start:start + jobs_per_pass],
                    schema=schema, spec=spec, encoder=encoder, resume=resume,
                    media_provider=getattr(adapter, "provider", None),
                    resume_entries=resume_entries, registry=registry or strategy.registry,
                    expected_ticks=expected_ticks, expected_entities=expected_entities)
            emit(progress, "writing", "advance", current=source_i, total=len(by_source),
                 source=src, status="ok")
            continue
        tracks = {t.entity_id: t for t in tracks_from_ticks(schema, list(adapter.ticks(src)), src)}
        actual_lengths = {entity_id: track.n_steps for entity_id, track in tracks.items()}
        expected_length_map = dict(expected_track_rows)
        if actual_lengths != expected_length_map:
            raise ValueError(f"source {src!r} changed after selection: scored tracks "
                             f"{expected_length_map!r}, output tracks {actual_lengths!r}")
        for (eid, k) in by_source[src]:
            tr = tracks.get(eid)
            if tr is None:
                raise ValueError(f"source {src!r} changed after selection: admitted entity "
                                 f"{eid!r} is no longer present")
            detail = admitted_details[(src, eid, k)]
            start, end = detail["start"], detail["end"]
            if tr.n_steps < end:
                raise ValueError(f"source {src!r} changed after selection: admitted span ends "
                                 f"at {end}, but entity {eid!r} now has {tr.n_steps} steps")
            sub = tr if (start, end) == (0, tr.n_steps) else Track(
                tr.source_id, tr.entity_id, tr.records[start:end], dict(tr.meta))
            sub.meta.update(detail["meta"])
            sub.meta["group"] = group_of(src) if group_of else None
            entries += write_track(out_dir, sub, spec, encoder=encoder, schema=schema,
                                   split=split.get(src, "train"), resume=resume,
                                   name_suffix=suffix(start, end),
                                   media_provider=getattr(adapter, "provider", None),
                                   resume_entries=resume_entries,
                                   registry=registry or strategy.registry)
        emit(progress, "writing", "advance", current=source_i, total=len(by_source),
             source=src, status="ok")
    emit(progress, "writing", "complete", current=len(by_source), total=len(by_source),
         detail=f"{len(entries)} shards")

    from ..provenance import assert_sources_unchanged
    assert_sources_unchanged(sources, adapter, signature["payload"]["sources"])

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
    from ..manifest import portable_shard_entries
    manifest_entries = portable_shard_entries(out_dir, entries)
    write_text(f"{out_dir}/dataset_manifest.json",
               json.dumps({"artifact_format_version": ARTIFACT_FORMAT_VERSION,
                           "commit_protocol": "sidecar-v1",
                           "count": len(entries), "shards": manifest_entries,
                           "budget": budget}, indent=2))
    results = {"shards": len(entries), "tracks_admitted": len(admitted),
               "tracks_total": n_tracks, "segments_total": segment_count,
               "sources_skipped": len(errors),
               "n_train_sources": len({entry["source_id"] for entry in entries
                                       if entry["split"] == "train"}),
               "n_test_sources": len({entry["source_id"] for entry in entries
                                      if entry["split"] == "test"}),
               "budget": budget}
    ended = datetime.now(timezone.utc).isoformat()
    cfg = build_config(schema, strategy, results=results, seed=seed, started=started, ended=ended)
    cfg["output"] = spec.to_dict()
    cfg["provenance"] = signature
    cfg["errors"] = errors
    if execution is not None:
        cfg["execution"] = execution
    if _config_updates:
        cfg.update(_config_updates)
    write_config(f"{out_dir}/config.json", cfg)
    from ..manifest import commit_dataset
    commit_dataset(out_dir)
    return DatasetWriteResult(dataset=str(out_dir), shards=entries, split=split,
                              admitted=admitted, results=results, errors=errors)


def write_dataset(out_dir: str, *, schema: SchemaSpec, adapter: InputAdapter,
                  sources: Iterable[str], strategy: Strategy, segmenter: Segmenter,
                  spec: OutputSpec, encoder: Optional[OutputEncoder] = None,
                  train_hours: float, test_hours: float = 0.0,
                  sample_hz: Optional[float] = None, seed: int = 0,
                  group_of: Optional[Callable[[str], object]] = None,
                  tags_of: Optional[Callable] = None,
                  clip_aggregate: str = "max", span_steps: Optional[int] = None,
                  span_seconds: Optional[float] = None, workers: int = 1,
                  on_error: str = "skip", resume: bool = False, force: bool = False,
                  streaming: bool = False, progress=None, transforms=(), registry=None,
                  executor: str = "local", memory_budget: int | None = None,
                  batch_size: int = 1024,
                  checkpoint_dir: str | None = None, spill_dir: str | None = None,
                  approximation: str = "exact",
                  max_in_flight: int = 2) -> DatasetWriteResult:
    """Write the best spans for an hours budget as one committed dataset artifact."""
    from ..io import _normalize_explicit_path
    out_dir = _normalize_explicit_path(out_dir, label="out_dir")
    sources = list(sources)
    from ..manifest import dataset_writer_lease, validate_source_destination_separation
    validate_source_destination_separation(out_dir, sources, adapter=adapter)
    with dataset_writer_lease(out_dir):
        return _write_dataset_impl(
            out_dir, schema=schema, adapter=adapter, sources=sources, strategy=strategy,
            segmenter=segmenter, spec=spec, encoder=encoder, train_hours=train_hours,
            test_hours=test_hours, sample_hz=sample_hz, seed=seed, group_of=group_of,
            tags_of=tags_of, clip_aggregate=clip_aggregate, span_steps=span_steps,
            span_seconds=span_seconds, workers=workers, on_error=on_error, resume=resume,
            force=force, streaming=streaming, progress=progress, transforms=transforms,
            registry=registry, executor=executor, memory_budget=memory_budget,
            batch_size=batch_size, checkpoint_dir=checkpoint_dir, spill_dir=spill_dir,
            approximation=approximation, max_in_flight=max_in_flight)
