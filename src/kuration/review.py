"""Human-facing preview and selection-review helpers."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import islice
import math
from typing import Any

import numpy as np

from .report import dataset_report
from .results import DatasetPreview, HoursSelectionReview, SelectionReview


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def _field_summary(kind: str, values: list, present: int, expected: int) -> dict:
    base = {"kind": kind, "present": present, "missing": max(0, expected - present),
            "missing_fraction": ((expected - present) / expected if expected else 0.0)}
    if kind == "categorical":
        base["top_values"] = dict(Counter(str(value) for value in values).most_common(10))
        base["unique"] = len({str(value) for value in values})
        return base
    numeric = []
    for value in values:
        try:
            numeric.extend(np.asarray(value, dtype=np.float64).reshape(-1).tolist())
        except (TypeError, ValueError):
            continue
    finite = np.asarray([value for value in numeric if math.isfinite(value)], dtype=np.float64)
    if finite.size:
        base.update({"min": float(finite.min()), "max": float(finite.max()),
                     "mean": float(finite.mean()), "p50": float(np.percentile(finite, 50))})
    return base


def preview_dataset(*, schema, adapter, sources, max_sources: int = 3,
                    max_ticks_per_source: int = 10_000,
                    examples: int = 5) -> DatasetPreview:
    """Read a bounded source sample and summarize shape, quality, and representative ticks."""
    for name, value in (("max_sources", max_sources),
                        ("max_ticks_per_source", max_ticks_per_source),
                        ("examples", examples)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    sources = list(sources)
    sampled = sources[:max_sources]
    field_values: dict[str, list] = defaultdict(list)
    field_present: Counter = Counter()
    global_values: dict[str, list] = defaultdict(list)
    global_present: Counter = Counter()
    entity_records = 0
    ticks_total = 0
    source_reports = []
    tick_examples: list[dict] = []
    truncated = False

    for source in sampled:
        iterator = (adapter.iter_ticks(source)
                    if hasattr(adapter, "iter_ticks") and not getattr(adapter, "in_memory", False)
                    else iter(adapter.ticks(source)))
        ticks = list(islice(iterator, max_ticks_per_source + 1))
        source_truncated = len(ticks) > max_ticks_per_source
        truncated = truncated or source_truncated
        ticks = ticks[:max_ticks_per_source]
        entities = {entity for tick in ticks for entity in tick.entities}
        source_reports.append({
            "source": str(source), "ticks": len(ticks), "entities": len(entities),
            "time_start": _json_value(ticks[0].time) if ticks else None,
            "time_end": _json_value(ticks[-1].time) if ticks else None,
            "truncated": source_truncated,
        })
        ticks_total += len(ticks)
        for tick in ticks:
            entity_records += len(tick.entities)
            for values in tick.entities.values():
                for name in schema.fields:
                    if name in values and values[name] is not None:
                        field_present[name] += 1
                        field_values[name].append(values[name])
            for name in schema.globals:
                if name in tick.globals and tick.globals[name] is not None:
                    global_present[name] += 1
                    global_values[name].append(tick.globals[name])
            if len(tick_examples) < examples:
                tick_examples.append({"source": str(source), "time": _json_value(tick.time),
                                      "entities": _json_value(tick.entities),
                                      "globals": _json_value(tick.globals),
                                      "events": _json_value(tick.events)})

    fields = {name: _field_summary(spec.kind, field_values[name], field_present[name],
                                   entity_records)
              for name, spec in schema.fields.items()}
    globals_ = {name: _field_summary(spec.kind, global_values[name], global_present[name],
                                     ticks_total)
                for name, spec in schema.globals.items()}
    warnings = []
    for name, summary in {**fields, **globals_}.items():
        if summary["missing_fraction"] > 0:
            warnings.append({"code": "missing-values", "field": name,
                             "message": f"{name} is missing in "
                                        f"{summary['missing_fraction']:.1%} of sampled records"})
    if truncated or len(sampled) < len(sources):
        warnings.append({"code": "bounded-preview", "message":
                         "Preview is based on a bounded source/tick sample."})
    return DatasetPreview(
        schema=schema.to_dict(), sources={"total": len(sources), "sampled": len(sampled),
                                          "details": source_reports},
        records={"ticks": ticks_total, "entity_records": entity_records},
        fields=fields, globals=globals_, examples=tick_examples, warnings=warnings,
        sampling={"max_sources": max_sources, "max_ticks_per_source": max_ticks_per_source,
                  "truncated": truncated or len(sampled) < len(sources)})


def _segment_example(segment, priority) -> dict:
    return {"source": segment.source_id, "start_idx": segment.start_idx,
            "end_idx": segment.end_idx, "time_range": _json_value(segment.time_range),
            "features": _json_value(segment.features), "bin": _json_value(segment.bin),
            "tags": sorted(map(str, segment.tags)), "group": _json_value(segment.group),
            "priority": (float(priority) if priority is not None and math.isfinite(priority)
                         else None)}


def selection_review(result, *, seed: int = 0, examples: int = 10,
                     hist_bins: int = 10) -> SelectionReview:
    """Compare a ``CurateResult`` candidate pool with its selected segments."""
    if not isinstance(examples, int) or isinstance(examples, bool) or examples < 1:
        raise ValueError("examples must be a positive integer")
    input_candidates = list(result.all_segments)
    selected = list(result.selected)
    review_pool = input_candidates
    if result.strategy.filters:
        from .methods import apply_filters
        review_pool = apply_filters(input_candidates, result.strategy, seed=seed)
    priorities_by_id = {}
    if review_pool:
        from .strategy import segment_priorities
        priorities = segment_priorities(review_pool, result.strategy, seed=seed)
        priorities_by_id = {id(segment): priority for segment, priority
                            in zip(review_pool, priorities)}
        candidates = [segment for segment, priority in zip(review_pool, priorities)
                      if math.isfinite(priority)]
    else:
        candidates = []
    candidate_report = dataset_report(candidates, hist_bins=hist_bins)
    selected_report = dataset_report(selected, hist_bins=hist_bins)
    metrics = {}
    for name in sorted(set(candidate_report["metrics"]) | set(selected_report["metrics"])):
        before = candidate_report["metrics"].get(name) or {}
        after = selected_report["metrics"].get(name) or {}
        metrics[name] = {"candidate": before, "selected": after,
                         "mean_delta": (after.get("mean", 0.0) - before.get("mean", 0.0)),
                         "p50_delta": (after.get("p50", 0.0) - before.get("p50", 0.0))}
    source_counts = Counter(segment.source_id for segment in selected)
    split_counts = Counter(result.splits.get(segment.source_id, "unsplit")
                           for segment in selected)
    rationale = {"method": result.strategy.method, "weights": result.strategy.weights,
                 "normalize": result.strategy.normalize, "target": result.strategy.target,
                 "predicates": result.strategy.predicates or [],
                 "tag_coverage": result.strategy.tag_coverage,
                 "group_cap": result.strategy.group_cap}
    return SelectionReview(
        input_candidates=len(input_candidates), excluded=len(input_candidates) - len(candidates),
        candidates=len(candidates), selected=len(selected), selection_rate=(
            len(selected) / len(candidates) if candidates else 0.0),
        rationale=rationale, metrics=metrics,
        coverage={"candidate": {"sources": candidate_report["n_sources"],
                                 "groups": candidate_report["groups"],
                                 "tags": candidate_report["tags"],
                                 "bins": candidate_report.get("bins")},
                  "selected": {"sources": selected_report["n_sources"],
                                "groups": selected_report["groups"],
                                "tags": selected_report["tags"],
                                "bins": selected_report.get("bins"),
                                "source_contribution": dict(sorted(source_counts.items())),
                                "splits": dict(sorted(split_counts.items()))}},
        examples=[_segment_example(segment, priorities_by_id.get(id(segment)))
                  for segment in selected[:examples]],
        errors=list(result.errors),
        warnings=([{"code": "source-errors", "message":
                    f"{len(result.errors)} source(s) were skipped"}] if result.errors else []))


def hours_selection_review(result) -> HoursSelectionReview:
    """Explain an hours-path ``DatasetWriteResult`` without another scoring pass."""
    results = result.get("results") or {}
    budget = results.get("budget") or {}
    required = {"requested_seconds", "admitted_seconds", "candidate_seconds",
                "items_total", "items_admitted"}
    if not required <= set(budget):
        raise ValueError("hours_selection_review needs an hours-path DatasetWriteResult")
    emitted = budget.get("emitted_seconds") or budget["admitted_seconds"]
    split_budget = {}
    warnings = []
    for split in ("train", "test"):
        requested = float(budget["requested_seconds"].get(split, 0.0))
        admitted = float(budget["admitted_seconds"].get(split, 0.0))
        emitted_seconds = float(emitted.get(split, 0.0))
        utilization = emitted_seconds / requested if requested > 0 else 1.0
        split_budget[split] = {
            "requested_seconds": requested, "candidate_seconds": float(
                budget["candidate_seconds"].get(split, 0.0)),
            "admitted_seconds": admitted, "emitted_seconds": emitted_seconds,
            "underfill_seconds": max(0.0, requested - admitted),
            "emitted_underfill_seconds": max(0.0, requested - emitted_seconds),
            "emission_loss_seconds": max(0.0, admitted - emitted_seconds),
            "utilization": utilization,
        }
        if requested > 0 and emitted_seconds < requested:
            warnings.append({"code": "budget-underfill", "split": split,
                             "message": f"{split} emitted {emitted_seconds:.3f}s of "
                                        f"{requested:.3f}s requested"})
    review = budget.get("review") or {}
    return HoursSelectionReview(
        candidate_items=int(budget["items_total"]),
        admitted_items=int(budget["items_admitted"]),
        rejected_items=int(budget["items_total"] - budget["items_admitted"]),
        admission_rate=(budget["items_admitted"] / budget["items_total"]
                        if budget["items_total"] else 0.0),
        budget=split_budget,
        decision_counts=dict(review.get("decision_counts") or {}),
        priorities=dict(review.get("priority") or {}),
        examples=dict(review.get("examples") or {"admitted": [], "rejected": []}),
        source_splits={"train": sum(1 for value in result.get("split", {}).values()
                                    if value == "train"),
                       "test": sum(1 for value in result.get("split", {}).values()
                                   if value == "test")},
        errors=list(result.get("errors") or ()), warnings=warnings)
