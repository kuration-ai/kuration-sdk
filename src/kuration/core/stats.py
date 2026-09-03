"""Unified, data-first statistics for pipelines, selections, and written artifacts."""
from __future__ import annotations

from collections import Counter
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .io import write_text
from .results import (
    DatasetInspection,
    DatasetPreview,
    DatasetReport,
    DatasetWriteResult,
    HoursSelectionReview,
    SelectionReview,
    StatsExamples,
    StatsReport,
)

_DEFAULT_EXAMPLES = 5
_STATS_ENRICHERS: dict[str, Callable[[Any, StatsReport], dict | None]] = {}
_STATS_PLOTTERS: dict[str, Callable[..., Any]] = {}
_STATS_EXAMPLE_RENDERERS: dict[str, Callable[[StatsExamples], str | None]] = {}


def register_stats_enricher(
    name: str,
    enricher: Callable[[Any, StatsReport], dict | None],
    *,
    plotter: Callable[..., Any] | None = None,
    example_renderer: Callable[[StatsExamples], str | None] | None = None,
) -> None:
    """Register an advanced domain section for the ordinary :func:`stats` result.

    Domain packages use this hook so importing a domain unlocks relevant analysis while the
    baseline core remains domain-agnostic. Most users should call ``stats()`` rather than
    registering enrichers directly.
    """
    if not isinstance(name, str) or not name or not name.replace("_", "").isalnum():
        raise ValueError("stats enricher name must be a non-empty identifier")
    if not callable(enricher):
        raise ValueError("stats enricher must be callable")
    existing = _STATS_ENRICHERS.get(name)
    if existing is not None and existing is not enricher:
        raise ValueError(f"stats enricher {name!r} is already registered")
    if plotter is not None and not callable(plotter):
        raise ValueError("stats plotter must be callable")
    if example_renderer is not None and not callable(example_renderer):
        raise ValueError("stats example renderer must be callable")
    existing_plotter = _STATS_PLOTTERS.get(name)
    if existing_plotter is not None and existing_plotter is not plotter:
        raise ValueError(f"stats plotter {name!r} is already registered")
    existing_renderer = _STATS_EXAMPLE_RENDERERS.get(name)
    if existing_renderer is not None and existing_renderer is not example_renderer:
        raise ValueError(f"stats example renderer {name!r} is already registered")
    _STATS_ENRICHERS[name] = enricher
    if plotter is not None:
        _STATS_PLOTTERS[name] = plotter
    if example_renderer is not None:
        _STATS_EXAMPLE_RENDERERS[name] = example_renderer


def _apply_enrichers(subject: Any, report: StatsReport) -> StatsReport:
    if not _STATS_ENRICHERS:
        return report
    sections = dict(report.get("sections") or {})
    changed = False
    for name, enricher in _STATS_ENRICHERS.items():
        value = enricher(subject, report)
        if value is not None:
            sections[name] = value
            changed = True
    if not changed:
        return report
    document = report.to_dict()
    document["sections"] = sections
    enriched = StatsReport(document)
    context = report.__dict__.get("_stats_context")
    if context is not None:
        enriched._stats_context = context
    return enriched


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _pipeline_stats(
    pipeline,
    *,
    hist_bins: int,
    max_sources: int,
    max_ticks_per_source: int,
    examples: int,
) -> StatsReport:
    preview = pipeline.preview(
        max_sources=max_sources,
        max_ticks_per_source=max_ticks_per_source,
        examples=examples,
    )
    if pipeline._n is not None:
        from .review import selection_review

        result = pipeline.collect()
        review = selection_review(
            result,
            seed=pipeline._options["seed"],
            examples=examples,
            hist_bins=hist_bins,
        )
        if pipeline._expectations:
            from .quality import check_quality

            review["quality"] = check_quality(
                review, pipeline._expectations
            ).to_dict()
        report = StatsReport(
            stage="curated",
            subject="pipeline",
            sections={"input": preview.to_dict(), "selection": review.to_dict()},
            warnings=list(preview.get("warnings") or ())
            + list(review.get("warnings") or ()),
        )
        report._stats_context = {"pipeline": pipeline, "selection": result}
        return report

    report = pipeline.report(hist_bins=hist_bins)
    # candidate warnings (e.g. non-finite metric values) belong in the lifecycle surface too
    warnings = list(preview.get("warnings") or ()) + list(report.get("warnings") or ())
    stage = "uncurated"
    sections: dict[str, Any] = {
        "input": preview.to_dict(),
        "candidates": report.to_dict(),
    }
    if pipeline._hours is not None:
        stage = "planned"
        sections["budget_plan"] = {
            "train_hours": float(pipeline._hours),
            "test_hours": float(pipeline._test_hours),
        }
        warnings.append(
            {
                "code": "selection-pending-output",
                "message": (
                    "Hours admission is resolved over output tracks; call "
                    ".output(path).stats() for curated and emitted statistics."
                ),
            }
        )
    return StatsReport(
        stage=stage,
        subject="pipeline",
        sections=sections,
        warnings=warnings,
    )


def _curate_result_stats(result, *, hist_bins: int, examples: int) -> StatsReport:
    from .review import selection_review

    review = selection_review(
        result, seed=result.seed, examples=examples, hist_bins=hist_bins
    )
    report = StatsReport(
        stage="curated",
        subject="selection",
        sections={"selection": review.to_dict()},
        warnings=list(review.get("warnings") or ()),
    )
    report._stats_context = {"selection": result}
    return report


def _sample_flat(array: np.ndarray, limit: int) -> np.ndarray:
    flat = np.asarray(array).reshape(-1)
    if len(flat) <= limit:
        return flat
    indexes = np.linspace(0, len(flat) - 1, num=limit, dtype=np.int64)
    return flat[indexes]


def _even_subsample(values: list, limit: int) -> list:
    """Thin one collected pool evenly instead of truncating it.

    Per-shard collection over-collects (``ceil(max_values / shards)`` each); keeping the first
    ``max_values`` would retain only the earliest shards and quietly contradict the recorded
    even sampling method, biasing sampled percentiles and histograms toward the first shards.
    """
    if len(values) <= limit:
        return values
    indexes = np.linspace(0, len(values) - 1, num=limit, dtype=np.int64)
    return [values[int(index)] for index in indexes]


def _json_number(value) -> float | None:
    """Return a strict-JSON-safe float: a non-finite statistic becomes ``None``, not ``NaN``."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _numeric_chunk(accumulator: dict, values: np.ndarray, sample_limit: int) -> None:
    raw = np.asarray(values)
    numeric = raw.astype(np.float64).reshape(-1)
    accumulator["values"] += int(numeric.size)
    if raw.dtype.kind in "bui" and raw.size:
        # int64 magnitudes above 2**53 do not survive the float64 cast, so keep integer
        # extrema exactly; mean/std stay float by design.
        chunk_min, chunk_max = int(raw.min()), int(raw.max())
        accumulator["int_min"] = (chunk_min if accumulator["int_min"] is None
                                  else min(accumulator["int_min"], chunk_min))
        accumulator["int_max"] = (chunk_max if accumulator["int_max"] is None
                                  else max(accumulator["int_max"], chunk_max))
    finite = numeric[np.isfinite(numeric)]
    accumulator["non_finite"] += int(numeric.size - finite.size)
    if finite.size:
        prior_count = accumulator["finite"]
        chunk_count = int(finite.size)
        chunk_mean = float(finite.mean(dtype=np.float64))
        chunk_m2 = float(np.square(finite - chunk_mean).sum(dtype=np.float64))
        combined_count = prior_count + chunk_count
        delta = chunk_mean - accumulator["mean"]
        accumulator["mean"] += delta * chunk_count / combined_count
        accumulator["m2"] += (
            chunk_m2
            + delta * delta * prior_count * chunk_count / combined_count
        )
        accumulator["finite"] = combined_count
        accumulator["min"] = min(accumulator["min"], float(finite.min()))
        accumulator["max"] = max(accumulator["max"], float(finite.max()))
    if sample_limit > 0:
        sampled = _sample_flat(finite, sample_limit)
        accumulator["samples"].extend(float(value) for value in sampled)


def _text_chunk(accumulator: dict, values: np.ndarray, sample_limit: int) -> None:
    flat = np.asarray(values).reshape(-1)
    accumulator["values"] += int(flat.size)
    sampled = _sample_flat(flat, sample_limit)
    accumulator["samples"].extend(str(value) for value in sampled)


def _finalize_modality(accumulator: dict, *, hist_bins: int, max_values: int) -> dict:
    rows = accumulator["rows"]
    absent = accumulator["absent_rows"]
    out = {
        "dtype": accumulator["dtype"],
        "shape": list(accumulator["shape"]),
        "rows": rows,
        "values": accumulator["values"],
        "shards_present": accumulator["shards_present"],
        "presence_masked": bool(accumulator["masked_shards"]),
        "absent_rows": absent,
        "present_fraction": ((rows - absent) / rows) if rows else 1.0,
    }
    samples = _even_subsample(accumulator["samples"], max_values)
    if accumulator["numeric"]:
        finite = accumulator["finite"]
        out["finite"] = finite
        out["non_finite"] = accumulator["non_finite"]
        if finite:
            from .report import _histogram

            variance = max(0.0, accumulator["m2"] / finite)
            sample = np.asarray(samples, dtype=np.float64)
            # shared with the metric reports: tolerates constant / near-constant samples
            counts, edges = _histogram(sample, hist_bins)
            out.update(
                {
                    "min": (accumulator["int_min"] if accumulator["int_min"] is not None
                            else _json_number(accumulator["min"])),
                    "max": (accumulator["int_max"] if accumulator["int_max"] is not None
                            else _json_number(accumulator["max"])),
                    "mean": _json_number(accumulator["mean"]),
                    "std": _json_number(math.sqrt(variance)),
                    "p50": (_json_number(np.percentile(sample, 50))
                            if sample.size else None),
                    "p95": (_json_number(np.percentile(sample, 95))
                            if sample.size else None),
                    # exact only when every finite value survived sampling
                    "percentiles_sampled": bool(sample.size < finite),
                    "histogram": {
                        "counts": counts.tolist(),
                        "edges": [float(edge) for edge in edges],
                        "sampled_values": int(sample.size),
                    },
                }
            )
    else:
        counts = Counter(samples)
        out.update(
            {
                "unique_sampled": len(counts),
                "top_values": dict(counts.most_common(10)),
                "sampled_values": len(samples),
            }
        )
    return out


def _artifact_contents(
    dataset,
    *,
    hist_bins: int,
    max_values: int,
    verify: bool,
    progress=None,
    registry=None,
    _snapshot=None,
) -> dict:
    from .loaders import (
        _assert_snapshot_current,
        _resolve_operation_snapshot,
        _snapshot_entries,
        _validate_snapshot_shard,
        read_shard_arrays,
    )
    from .progress import emit

    snapshot = _snapshot or _resolve_operation_snapshot(dataset, None)
    entries = _snapshot_entries(snapshot)
    container = snapshot.container
    per_shard_sample = max(1, math.ceil(max_values / max(1, len(entries))))
    accumulators: dict[str, dict] = {}
    emit(progress, "stats", "start", total=len(entries))
    for index, entry in enumerate(entries, 1):
        _assert_snapshot_current(snapshot, operation="artifact statistics")
        _validate_snapshot_shard(snapshot, entry)
        arrays = read_shard_arrays(
            entry, container, trim=True, verify=verify, registry=registry
        )
        _validate_snapshot_shard(snapshot, entry)
        _assert_snapshot_current(snapshot, operation="artifact statistics")
        # `trim=True` drops trailing pad only; interior rows for an absent entity are
        # zero-filled and flagged by `_present`. Accumulating those zeros reported a min of 0
        # and a halved mean for a field that never held 0 — mask them out per row.
        present = arrays.get("_present")
        if present is not None:
            present = np.asarray(present)
            if present.dtype != np.bool_ or present.ndim != 1:
                raise ValueError(
                    f"shard {entry.get('path')!r} _present must be a 1-D bool mask, got "
                    f"{present.ndim}-D {present.dtype}"
                )
        for name, array in arrays.items():
            array = np.asarray(array)
            numeric = array.dtype.kind in "buif"
            shape = tuple(array.shape[1:])
            accumulator = accumulators.setdefault(
                name,
                {
                    "dtype": str(array.dtype),
                    "shape": shape,
                    "numeric": numeric,
                    "rows": 0,
                    "values": 0,
                    "finite": 0,
                    "non_finite": 0,
                    "mean": 0.0,
                    "m2": 0.0,
                    "min": float("inf"),
                    "max": float("-inf"),
                    "int_min": None,
                    "int_max": None,
                    "samples": [],
                    "shards_present": 0,
                    "masked_shards": 0,
                    "absent_rows": 0,
                },
            )
            if (
                accumulator["dtype"] != str(array.dtype)
                or accumulator["shape"] != shape
                or accumulator["numeric"] != numeric
            ):
                raise ValueError(
                    f"output modality {name!r} changes dtype or shape across shards"
                )
            rows = int(array.shape[0])
            accumulator["rows"] += rows
            accumulator["shards_present"] += 1
            # `_present` itself stays unmasked — it is the record of what was absent.
            mask = None if name == "_present" else present
            if mask is not None:
                if int(mask.shape[0]) != rows:
                    raise ValueError(
                        f"shard {entry.get('path')!r} _present has {int(mask.shape[0])} rows "
                        f"but modality {name!r} has {rows}"
                    )
                array = array[mask]
                accumulator["absent_rows"] += rows - int(array.shape[0])
                accumulator["masked_shards"] += 1
            if numeric:
                _numeric_chunk(accumulator, array, per_shard_sample)
            else:
                _text_chunk(accumulator, array, per_shard_sample)
        emit(
            progress,
            "stats",
            "advance",
            current=index,
            total=len(entries),
            source=entry.get("source_id"),
        )
    _assert_snapshot_current(snapshot, operation="artifact statistics")
    emit(
        progress,
        "stats",
        "complete",
        current=len(entries),
        total=len(entries),
        detail=f"{len(accumulators)} modalities",
    )
    modalities = {
        name: _finalize_modality(
            accumulator, hist_bins=hist_bins, max_values=max_values
        )
        for name, accumulator in sorted(accumulators.items())
    }
    warnings: list[dict] = []
    for name, accumulator in sorted(accumulators.items()):
        if accumulator["shards_present"] != len(entries):
            warnings.append(
                {
                    "code": "modality-missing-from-shards",
                    "modality": name,
                    "message": (
                        f"modality {name!r} appears in {accumulator['shards_present']} of "
                        f"{len(entries)} shard(s); its statistics cover those shards only"
                    ),
                }
            )
        if accumulator["masked_shards"] and (
            accumulator["masked_shards"] != accumulator["shards_present"]
        ):
            warnings.append(
                {
                    "code": "presence-mask-partial",
                    "modality": name,
                    "message": (
                        f"modality {name!r} was presence-masked in "
                        f"{accumulator['masked_shards']} of "
                        f"{accumulator['shards_present']} shard(s); the rest had no _present "
                        "mask and were accumulated whole"
                    ),
                }
            )
        if modalities[name].get("percentiles_sampled"):
            # p50/p95/histogram sit next to an exact mean/std/min/max — say which is which.
            warnings.append(
                {
                    "code": "sampled-percentiles",
                    "modality": name,
                    "message": (
                        f"modality {name!r} p50/p95 and histogram come from "
                        f"{(modalities[name].get('histogram') or {}).get('sampled_values', 0)} "
                        f"of {accumulator['finite']} finite values "
                        f"(max_values={max_values}); mean/std/min/max are exact"
                    ),
                }
            )
        if accumulator["numeric"] and accumulator["non_finite"]:
            warnings.append(
                {
                    "code": "non-finite-values",
                    "modality": name,
                    "message": (
                        f"modality {name!r} has {accumulator['non_finite']} non-finite value(s); "
                        "statistics describe the finite values only"
                    ),
                }
            )
    return {
        "shards_scanned": len(entries),
        "verified": verify,
        "presence_masked": any(
            accumulator["masked_shards"] for accumulator in accumulators.values()
        ),
        "sampling": {
            "max_values_per_modality": max_values,
            "method": "even-within-shard-then-even-across-pool",
        },
        "modalities": modalities,
        "warnings": warnings,
    }


def _artifact_stats(
    dataset,
    *,
    hist_bins: int,
    max_values: int,
    verify: bool,
    content: bool,
    examples: int = _DEFAULT_EXAMPLES,
    progress=None,
    write_result: DatasetWriteResult | None = None,
    registry=None,
) -> StatsReport:
    from .inspection import _inspect_resolved_dataset
    from .loaders import _assert_snapshot_current, _resolve_operation_snapshot

    # A content scan decodes (and, with verify=True, checksums) every shard, so verifying the
    # manifest again would double the work. Without a content scan, `verify` has to reach the
    # inspection or `--verify --no-content` would verify nothing at all.
    snapshot = _resolve_operation_snapshot(dataset, None)
    inspection = _inspect_resolved_dataset(
        snapshot,
        verify=verify and not content,
        registry=registry,
    )
    sections: dict[str, Any] = {"artifact": inspection.to_dict()}
    warnings: list[dict] = []
    for error in (inspection.get("verification") or {}).get("errors") or ():
        warnings.append(
            {
                "code": "shard-verification",
                "message": f"{error.get('path')}: {error.get('error')}",
            }
        )
    if content:
        contents = _artifact_contents(
            dataset,
            hist_bins=hist_bins,
            max_values=max_values,
            verify=verify,
            progress=progress,
            registry=registry,
            _snapshot=snapshot,
        )
        sections["contents"] = contents
        warnings.extend(contents.get("warnings") or ())
    if write_result is not None:
        results = write_result.get("results") or {}
        if results.get("budget"):
            from .review import hours_selection_review

            review = hours_selection_review(write_result)
            document = review.to_dict()
            # `examples` bounds embedded evidence here exactly as it does on the pipeline path.
            document["examples"] = {
                mode: list(items)[:examples]
                for mode, items in (document.get("examples") or {}).items()
            }
            sections["hours_selection"] = document
            warnings.extend(review.get("warnings") or ())
    if examples != _DEFAULT_EXAMPLES and not (
        sections.get("hours_selection") or {}
    ).get("examples"):
        raise ValueError(
            "examples= bounds embedded selection evidence, and this artifact embeds none; "
            "omit it, or take stats from the pipeline, CurateResult, or hours "
            "DatasetWriteResult that owns the selection"
        )
    report = StatsReport(
        stage="output",
        subject=str(dataset),
        sections=sections,
        warnings=warnings,
    )
    _assert_snapshot_current(snapshot, operation="artifact statistics")
    return report


def _base_stats(
    subject,
    *,
    hist_bins: int = 10,
    max_sources: int = 3,
    max_ticks_per_source: int = 10_000,
    examples: int = _DEFAULT_EXAMPLES,
    max_values: int = 100_000,
    content: bool = True,
    verify: bool = False,
    progress=None,
    registry=None,
) -> StatsReport:
    """Return unified statistics for a pipeline, selection result, or written artifact.

    Pipeline statistics infer their stage from the immutable pipeline plan. Written artifacts
    may be passed as a directory, ``dataset_manifest.json`` path, or the
    ``DatasetWriteResult`` returned by ``output()``/``write_dataset()``.
    """
    hist_bins = _positive_int("hist_bins", hist_bins)
    max_sources = _positive_int("max_sources", max_sources)
    max_ticks_per_source = _positive_int(
        "max_ticks_per_source", max_ticks_per_source
    )
    examples = _positive_int("examples", examples)
    max_values = _positive_int("max_values", max_values)
    if not isinstance(content, bool):
        raise ValueError("content must be true or false")
    if not isinstance(verify, bool):
        raise ValueError("verify must be true or false")
    if isinstance(subject, StatsReport):
        return subject

    from .curate import CurateResult
    from .pipeline import Pipeline

    if isinstance(subject, Pipeline):
        if progress is not None:
            pipeline = subject.options(progress=progress)
        else:
            pipeline = subject
        return _pipeline_stats(
            pipeline,
            hist_bins=hist_bins,
            max_sources=max_sources,
            max_ticks_per_source=max_ticks_per_source,
            examples=examples,
        )
    if isinstance(subject, CurateResult):
        return _curate_result_stats(
            subject, hist_bins=hist_bins, examples=examples
        )
    if isinstance(subject, DatasetWriteResult):
        dataset = subject.get("dataset")
        if not dataset:
            raise ValueError(
                "DatasetWriteResult has no artifact path; pass the dataset directory "
                "to stats(...) explicitly"
            )
        return _artifact_stats(
            dataset,
            hist_bins=hist_bins,
            max_values=max_values,
            verify=verify,
            content=content,
            examples=examples,
            progress=progress,
            write_result=subject,
            registry=registry,
        )
    if isinstance(subject, DatasetPreview):
        return StatsReport(
            stage="uncurated",
            subject="preview",
            sections={"input": subject.to_dict()},
            warnings=list(subject.get("warnings") or ()),
        )
    if isinstance(subject, DatasetReport):
        return StatsReport(
            stage="uncurated",
            subject="candidate-report",
            sections={"candidates": subject.to_dict()},
            warnings=[],
        )
    if isinstance(subject, SelectionReview):
        return StatsReport(
            stage="curated",
            subject="selection-review",
            sections={"selection": subject.to_dict()},
            warnings=list(subject.get("warnings") or ()),
        )
    if isinstance(subject, HoursSelectionReview):
        return StatsReport(
            stage="output",
            subject="hours-selection-review",
            sections={"hours_selection": subject.to_dict()},
            warnings=list(subject.get("warnings") or ()),
        )
    if isinstance(subject, DatasetInspection):
        return StatsReport(
            stage="output",
            subject=str(subject.get("dataset")),
            sections={"artifact": subject.to_dict()},
            warnings=[],
        )
    if isinstance(subject, (str, Path)):
        return _artifact_stats(
            str(subject),
            hist_bins=hist_bins,
            max_values=max_values,
            verify=verify,
            content=content,
            examples=examples,
            progress=progress,
            registry=registry,
        )
    raise TypeError(
        "stats() accepts a Pipeline, CurateResult, DatasetWriteResult, typed report, "
        "or written dataset path"
    )


def stats(
    subject,
    *,
    hist_bins: int = 10,
    max_sources: int = 3,
    max_ticks_per_source: int = 10_000,
    examples: int = _DEFAULT_EXAMPLES,
    max_values: int = 100_000,
    content: bool = True,
    verify: bool = False,
    progress=None,
    registry=None,
) -> StatsReport:
    """Return extensible lifecycle statistics for a pipeline, selection, or artifact."""
    if progress is not None:
        from .progress import progress_callback

        progress = progress_callback(progress)
    report = _base_stats(
        subject,
        hist_bins=hist_bins,
        max_sources=max_sources,
        max_ticks_per_source=max_ticks_per_source,
        examples=examples,
        max_values=max_values,
        content=content,
        verify=verify,
        progress=progress,
        registry=registry,
    )
    if "pipeline" not in report.__dict__.get("_stats_context", {}) and hasattr(
        subject, "adapter"
    ) and hasattr(subject, "schema"):
        context = dict(report.__dict__.get("_stats_context") or {})
        context["pipeline"] = subject
        report._stats_context = context
    return _apply_enrichers(subject, report)


def stats_capabilities(report: StatsReport) -> dict:
    """Return discoverable actions supported by one concrete stats report."""
    sections = report.get("sections") or {}
    plots = []
    if _histogram_metric(report, None) is not None:
        plots.append("distribution")
    if (sections.get("input") or {}).get("fields"):
        plots.append("missingness")
    if (sections.get("artifact") or {}).get("splits"):
        plots.append("splits")
    for name in _STATS_PLOTTERS:
        section = sections.get(name) or {}
        if name in (section.get("plots") or ()):
            plots.insert(0, name)

    modes: list[str] = []
    selection = sections.get("selection") or {}
    if selection:
        evidence = selection.get("evidence") or {}
        modes.extend(
            mode
            for mode in ("selected", "rejected", "strongest", "weakest")
            if mode in evidence
        )
    hours = sections.get("hours_selection") or {}
    if hours.get("examples"):
        modes.extend(
            mode
            for mode in ("admitted", "rejected")
            if mode in hours["examples"]
        )
    if (sections.get("input") or {}).get("examples"):
        modes.append("input")
    modes = list(dict.fromkeys(modes))
    context = report.__dict__.get("_stats_context") or {}
    pipeline = context.get("pipeline")
    media = bool(
        _STATS_EXAMPLE_RENDERERS
        and pipeline is not None
        and getattr(pipeline.schema, "media", None)
    )
    return {
        "plots": plots,
        "default_plot": plots[0] if plots else None,
        "example_modes": modes,
        "comparisons": [
            comparison
            for comparison, required in (
                ("selected-vs-rejected", {"selected", "rejected"}),
                ("strongest-vs-weakest", {"strongest", "weakest"}),
            )
            if required <= set(modes)
        ],
        "media_examples": media,
        "html": True,
        "save": True,
    }


def _embedded_examples(
    report: StatsReport,
    mode: str,
    *,
    metric: str | None,
    n: int,
) -> tuple[list[dict], bool]:
    sections = report.get("sections") or {}
    selection = sections.get("selection") or {}
    evidence = selection.get("evidence") or {}
    if mode in evidence:
        items = [dict(item) for item in evidence[mode]]
        if metric is not None:
            # Check per item, like the live path: a metric on *some* items used to pass the
            # union check and then raise a bare KeyError inside the sort.
            available = sorted(
                {name for item in items for name in (item.get("features") or {})}
            )
            complete = bool(items) and all(
                metric in (item.get("features") or {}) for item in items
            )
            if not complete and not items:
                # No embedded item can vouch for the name; fall back to the metric namespace.
                available = sorted(selection.get("metrics") or {})
                complete = metric in available
            if not complete:
                raise ValueError(
                    f"unknown evidence metric {metric!r}; available: {available}"
                )
            items.sort(
                key=lambda item: float((item.get("features") or {})[metric]),
                reverse=mode != "weakest",
            )
            for item in items:
                item["rank_by"] = metric
                item["rank_value"] = float((item.get("features") or {})[metric])
        return items[:n], True
    hours = sections.get("hours_selection") or {}
    if mode in (hours.get("examples") or {}):
        if metric is not None:
            raise ValueError("hours evidence does not support metric ranking")
        return [dict(item) for item in hours["examples"][mode]][:n], True
    if mode == "input" and (sections.get("input") or {}).get("examples"):
        if metric is not None:
            raise ValueError("input examples do not support metric ranking")
        return [
            dict(item) for item in (sections["input"].get("examples") or ())
        ][:n], True
    raise ValueError(
        f"example mode {mode!r} is unavailable; "
        f"available: {stats_capabilities(report)['example_modes']}"
    )


def stats_examples(
    report: StatsReport,
    *,
    mode: str = "auto",
    metric: str | None = None,
    compare: str | None = None,
    n: int = 6,
    frames: int = 3,
) -> StatsExamples:
    """Select bounded human-review evidence from a lifecycle report."""
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 100:
        raise ValueError("examples n must be an integer in [1, 100]")
    if not isinstance(frames, int) or isinstance(frames, bool) or not 1 <= frames <= 12:
        raise ValueError("examples frames must be an integer in [1, 12]")
    capabilities = stats_capabilities(report)
    modes: tuple[str, ...]
    if compare is not None:
        comparison_modes = {
            "selected-vs-rejected": ("selected", "rejected"),
            "strongest-vs-weakest": ("strongest", "weakest"),
        }
        if compare not in comparison_modes:
            raise ValueError(
                "compare must be selected-vs-rejected or strongest-vs-weakest"
            )
        modes = comparison_modes[compare]
    else:
        if mode == "auto":
            modes = (
                (capabilities["example_modes"][0],)
                if capabilities["example_modes"]
                else ()
            )
        else:
            modes = (mode,)
    if not modes:
        raise ValueError("stats report has no examples")

    context = report.__dict__.get("_stats_context") or {}
    result = context.get("selection")
    groups = {}
    bounded = False
    for group in modes:
        if result is not None and group in {
            "selected", "rejected", "strongest", "weakest"
        }:
            from .review import selection_examples

            groups[group] = selection_examples(
                result,
                mode=group,
                metric=metric,
                n=n,
                seed=result.seed,
            )
        else:
            groups[group], embedded_bounded = _embedded_examples(
                report, group, metric=metric, n=n
            )
            bounded = bounded or embedded_bounded
    examples = StatsExamples(
        stage=report.get("stage"),
        metric=metric,
        compare=compare,
        groups=groups,
        bounded=bounded,
        frames_per_example=frames,
    )
    examples._stats_context = context
    return examples


def _histogram_metric(report: StatsReport, metric: str | None):
    selection = (report.get("sections") or {}).get("selection") or {}
    selection_metrics = selection.get("metrics") or {}
    if selection_metrics:
        name = metric or next(iter(selection_metrics))
        if name not in selection_metrics:
            raise ValueError(
                f"unknown metric {name!r}; available: {sorted(selection_metrics)}"
            )
        return name, selection_metrics[name], "selection"
    candidates = (report.get("sections") or {}).get("candidates") or {}
    candidate_metrics = candidates.get("metrics") or {}
    if candidate_metrics:
        name = metric or next(iter(candidate_metrics))
        if name not in candidate_metrics:
            raise ValueError(
                f"unknown metric {name!r}; available: {sorted(candidate_metrics)}"
            )
        return name, candidate_metrics[name], "candidates"
    contents = (report.get("sections") or {}).get("contents") or {}
    modalities = contents.get("modalities") or {}
    numeric = {
        name: value
        for name, value in modalities.items()
        if value.get("histogram")
    }
    if numeric:
        public_names = [name for name in numeric if not name.startswith("_")]
        name = metric or next(iter(public_names or numeric))
        if name not in numeric:
            raise ValueError(
                f"unknown modality {name!r}; available: {sorted(numeric)}"
            )
        return name, numeric[name], "contents"
    return None


def _stairs(ax, histogram: dict, *, label: str, alpha: float = 1.0) -> None:
    counts = histogram.get("counts") or ()
    edges = histogram.get("edges") or ()
    if not counts and not edges:
        return  # nothing was recorded (e.g. an empty selected pool) — draw nothing
    if len(edges) != len(counts) + 1:
        raise ValueError(
            f"malformed histogram for {label!r}: {len(counts)} counts with {len(edges)} "
            f"edges (expected {len(counts) + 1})"
        )
    ax.stairs(counts, edges, label=label, alpha=alpha, fill=alpha < 1.0)


def plot_stats(
    report: StatsReport,
    metric: str | None = None,
    *,
    kind: str | None = None,
    ax=None,
):
    """Plot one useful default view from a ``StatsReport``."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "stats plotting needs Matplotlib: pip install 'kuration-sdk[stats]'"
        ) from error
    if kind is None and metric is None:
        default = stats_capabilities(report).get("default_plot")
        if default in _STATS_PLOTTERS:
            return _STATS_PLOTTERS[default](report, metric=None, ax=ax)
    core_kinds = {None, "distribution", "missingness", "splits"}
    if kind not in core_kinds and kind not in _STATS_PLOTTERS:
        raise ValueError(
            "unknown stats plot kind; available: "
            f"{sorted({'distribution', 'missingness', 'splits'} | set(_STATS_PLOTTERS))}"
        )
    if kind in _STATS_PLOTTERS:
        return _STATS_PLOTTERS[kind](report, metric=metric, ax=ax)

    if ax is None:
        figure, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    else:
        figure = ax.figure

    sections = report.get("sections") or {}
    input_summary = sections.get("input") or {}
    artifact = sections.get("artifact") or {}
    has_metric_histograms = bool(
        (sections.get("selection") or {}).get("metrics")
        or (sections.get("candidates") or {}).get("metrics")
    )
    if kind == "missingness" or (
        kind is None
        and metric is None
        and input_summary.get("fields")
        and not has_metric_histograms
    ):
        fields = input_summary.get("fields") or {}
        names = list(fields)
        ax.bar(names, [fields[name].get("missing_fraction", 0.0) for name in names])
        ax.set_ylabel("Missing fraction")
        ax.set_ylim(0, 1)
        ax.set_title("Input field missingness")
        ax.tick_params(axis="x", rotation=30)
        return figure
    if kind == "splits":
        splits = artifact.get("splits") or {}
        if not splits:
            raise ValueError("stats report has no output split statistics")
        names = list(splits)
        ax.bar(names, [splits[name].get("steps", 0) for name in names])
        ax.set_ylabel("Emitted steps")
        ax.set_title("Output split sizes")
        return figure

    resolved = _histogram_metric(report, metric)
    if resolved is None:
        if artifact.get("splits"):
            splits = artifact["splits"]
            names = list(splits)
            ax.bar(names, [splits[name].get("steps", 0) for name in names])
            ax.set_ylabel("Emitted steps")
            ax.set_title("Output split sizes")
            return figure
        raise ValueError("stats report has no plottable numeric metrics or modalities")
    name, value, source = resolved
    if source == "selection":
        _stairs(ax, (value.get("candidate") or {}).get("histogram") or {},
                label="candidate", alpha=0.6)
        _stairs(ax, (value.get("selected") or {}).get("histogram") or {},
                label="selected")
        ax.legend()
    else:
        _stairs(ax, value.get("histogram") or {}, label=name)
    ax.set_xlabel(name)
    ax.set_ylabel("Count")
    ax.set_title(f"{name} distribution")
    return figure


def render_stats_examples_html(examples: StatsExamples) -> str:
    """Render bounded evidence with optional domain-owned media additions."""
    groups = examples.get("groups") or {}
    group_html = []
    for group, items in groups.items():
        cards = []
        for item in items:
            features = item.get("features") or {}
            metric = item.get("rank_by")
            rank_value = item.get("rank_value")
            feature_rows = "".join(
                f"<span><strong>{escape(str(name))}</strong> "
                f"{escape(f'{value:.4g}' if isinstance(value, (int, float)) else str(value))}"
                "</span>"
                for name, value in list(features.items())[:6]
            )
            location = escape(str(item.get("source", "input")))
            if item.get("start_idx") is not None:
                location += (
                    f" [{int(item['start_idx'])}, {int(item['end_idx'])})"
                )
            rank = (
                f"<span class=\"kur-evidence-rank\">{escape(str(metric))}: "
                f"{escape(f'{rank_value:.4g}' if isinstance(rank_value, (int, float)) else str(rank_value))}"
                "</span>"
                if metric and rank_value is not None
                else ""
            )
            cards.append(
                "<article class=\"kur-evidence-card\">"
                f"<header><strong>{location}</strong>{rank}</header>"
                f"<div class=\"kur-evidence-features\">{feature_rows}</div>"
                "</article>"
            )
        group_html.append(
            f"<section><h3>{escape(str(group).replace('_', ' ').title())}</h3>"
            f"<div class=\"kur-evidence-grid\">{''.join(cards)}</div></section>"
        )
    domain_html = []
    for renderer in _STATS_EXAMPLE_RENDERERS.values():
        rendered = renderer(examples)
        if rendered:
            domain_html.append(rendered)
    note = (
        "<p class=\"kur-evidence-note\">This detached report contains only the bounded "
        "examples captured when stats were generated.</p>"
        if examples.get("bounded")
        else ""
    )
    return (
        "<style>.kur-evidence-grid{display:grid;grid-template-columns:"
        "repeat(auto-fit,minmax(18rem,1fr));gap:.7rem}.kur-evidence-card{border:1px solid #8885;"
        "border-radius:.55rem;padding:.7rem}.kur-evidence-card header{display:flex;"
        "justify-content:space-between;gap:.7rem}.kur-evidence-rank{font-variant-numeric:"
        "tabular-nums}.kur-evidence-features{display:flex;flex-wrap:wrap;gap:.35rem .8rem;"
        "font-size:.88em;margin-top:.5rem}.kur-evidence-note{opacity:.75}</style>"
        "<section class=\"kur-evidence\"><h2>Stats evidence</h2>"
        f"{note}{''.join(group_html)}{''.join(domain_html)}</section>"
    )


def _number_cell(value) -> str:
    """Format one statistic for HTML, showing an em dash for a statistic that does not exist."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return escape("—" if value is None else str(value))
    return escape(f"{value:.5g}")


def _modality_notes(value: dict) -> str:
    """Describe how honest a modality's numbers are: masking, sampling, non-finite values."""
    notes = []
    if value.get("presence_masked"):
        notes.append(
            f"present {value.get('present_fraction', 1.0):.0%} "
            f"({value.get('absent_rows', 0)} absent rows excluded)"
        )
    if "histogram" in value:
        sampled = int((value.get("histogram") or {}).get("sampled_values", 0))
        notes.append(
            f"p50/p95 sampled ({sampled} values)"
            if value.get("percentiles_sampled")
            else "p50/p95 exact"
        )
    if value.get("non_finite"):
        notes.append(f"{value['non_finite']} non-finite")
    return " · ".join(notes)


def _summary_cards(report: StatsReport) -> str:
    cards = []
    for line in report.render().splitlines()[1:]:
        if line.startswith("warning "):
            continue
        label, _, value = line.partition(":")
        cards.append(
            f"<div class=\"kur-card\"><strong>{escape(label.title())}</strong>"
            f"<span>{escape(value.strip())}</span></div>"
        )
    return "".join(cards)


def render_stats_html(
    report: StatsReport,
    *,
    title: str = "kuration stats",
    full_document: bool = False,
) -> str:
    """Render a safe static HTML representation of a stats report."""
    sections = report.get("sections") or {}
    fields = (sections.get("input") or {}).get("fields") or {}
    field_rows = "".join(
        "<tr>"
        f"<td>{escape(name)}</td><td>{escape(str(value.get('kind', 'unknown')))}</td>"
        f"<td>{value.get('missing_fraction', 0.0):.1%}</td>"
        f"<td>{escape(str(value.get('min', '—')))}</td>"
        f"<td>{escape(str(value.get('max', '—')))}</td>"
        "</tr>"
        for name, value in fields.items()
    )
    metric_source = (sections.get("candidates") or {}).get("metrics") or {}
    selection_metrics = (sections.get("selection") or {}).get("metrics") or {}
    metric_rows = []
    for name, value in metric_source.items():
        metric_rows.append(
            f"<tr><td>{escape(name)}</td><td>{_number_cell(value.get('mean'))}</td>"
            f"<td>{_number_cell(value.get('std'))}</td>"
            f"<td>{_number_cell(value.get('p50'))}</td>"
            f"<td>{_number_cell(value.get('p95'))}</td></tr>"
        )
    for name, value in selection_metrics.items():
        candidate = value.get("candidate") or {}
        selected = value.get("selected") or {}
        cells = "".join(
            f"<td>{_number_cell(candidate.get(key))} → {_number_cell(selected.get(key))}</td>"
            for key in ("mean", "std", "p50", "p95")
        )
        metric_rows.append(f"<tr><td>{escape(name)}</td>{cells}</tr>")
    modalities = (sections.get("contents") or {}).get("modalities") or {}
    modality_rows = "".join(
        "<tr>"
        f"<td>{escape(name)}</td><td>{escape(value.get('dtype', 'unknown'))}</td>"
        f"<td>{escape(str(value.get('shape', [])))}</td>"
        f"<td>{value.get('rows', 0)}</td><td>{_number_cell(value.get('mean'))}</td>"
        f"<td>{escape(_modality_notes(value))}</td>"
        "</tr>"
        for name, value in modalities.items()
    )
    warnings = "".join(
        f"<li><strong>{escape(item.get('code', 'general'))}</strong>: "
        f"{escape(item['message'])}</li>"
        for item in report.get("warnings") or ()
    )
    standard_sections = {
        "input", "candidates", "selection", "budget_plan", "artifact", "contents",
        "hours_selection",
    }
    domain_rows = []
    for section_name, section in sections.items():
        if section_name in standard_sections or not isinstance(section, dict):
            continue
        for item in section.get("summary") or ():
            if not isinstance(item, dict):
                continue
            domain_rows.append(
                "<tr>"
                f"<td>{escape(section_name)}</td>"
                f"<td>{escape(str(item.get('label', 'summary')))}</td>"
                f"<td>{escape(str(item.get('value', '—')))}</td>"
                f"<td>{escape(str(item.get('detail', '')))}</td>"
                "</tr>"
            )
    raw = escape(json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=str))
    capabilities = stats_capabilities(report)
    capability_items = []
    if capabilities["plots"]:
        capability_items.append(
            "plots: " + ", ".join(capabilities["plots"])
        )
    if capabilities["example_modes"]:
        capability_items.append(
            "examples: " + ", ".join(capabilities["example_modes"])
        )
    if capabilities["media_examples"]:
        capability_items.append("media evidence")
    capability_html = (
        "<p class=\"kur-capabilities\"><strong>Available:</strong> "
        + escape(" · ".join(capability_items))
        + "</p>"
        if capability_items
        else ""
    )
    body = (
        f"<section class=\"kur-stats\"><h2>{escape(title)}</h2>"
        f"{capability_html}"
        f"<div class=\"kur-cards\">{_summary_cards(report)}</div>"
        + (
            "<h3>Input fields</h3><table><thead><tr><th>Field</th><th>Kind</th>"
            "<th>Missing</th><th>Min</th><th>Max</th></tr></thead>"
            f"<tbody>{field_rows}</tbody></table>"
            if field_rows
            else ""
        )
        + (
            "<h3>Metrics</h3><table><thead><tr><th>Metric</th><th>Mean</th>"
            "<th>Std</th><th>P50</th><th>P95</th></tr></thead>"
            f"<tbody>{''.join(metric_rows)}</tbody></table>"
            if metric_rows
            else ""
        )
        + (
            "<h3>Output modalities</h3><table><thead><tr><th>Modality</th>"
            "<th>Dtype</th><th>Shape</th><th>Rows</th><th>Mean</th><th>Notes</th></tr></thead>"
            f"<tbody>{modality_rows}</tbody></table>"
            if modality_rows
            else ""
        )
        + (
            "<h3>Domain insights</h3><table><thead><tr><th>Domain</th>"
            "<th>Signal</th><th>Value</th><th>Detail</th></tr></thead>"
            f"<tbody>{''.join(domain_rows)}</tbody></table>"
            if domain_rows
            else ""
        )
        + (f"<h3>Warnings</h3><ul>{warnings}</ul>" if warnings else "")
        + f"<details><summary>Structured data</summary><pre>{raw}</pre></details></section>"
    )
    if not full_document:
        return body
    style = """
    :root { color-scheme: light dark; font-family: system-ui, sans-serif; }
    body { max-width: 1100px; margin: 2rem auto; padding: 0 1rem; line-height: 1.45; }
    .kur-cards { display: grid; grid-template-columns: repeat(auto-fit,minmax(13rem,1fr));
                 gap: .75rem; margin: 1rem 0; }
    .kur-card { border: 1px solid #8886; border-radius: .6rem; padding: .8rem;
                display: flex; flex-direction: column; gap: .3rem; }
    table { width: 100%; border-collapse: collapse; margin: .7rem 0 1.3rem; }
    th, td { text-align: left; border-bottom: 1px solid #8885; padding: .45rem; }
    pre { overflow: auto; padding: 1rem; background: #8881; border-radius: .4rem; }
    summary { cursor: pointer; font-weight: 600; }
    """
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{escape(title)}</title><style>{style}</style></head><body>{body}</body></html>"
    )


def write_stats_html(path, report: StatsReport, *, title: str = "kuration stats") -> str:
    """Write a static stats document and return ``path`` as a string."""
    path = str(path)
    write_text(path, render_stats_html(report, title=title, full_document=True))
    return path
