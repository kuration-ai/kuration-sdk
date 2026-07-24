"""Unified, data-first statistics for pipelines, selections, and written artifacts."""
from __future__ import annotations

from collections import Counter
from html import escape
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .io import write_text
from .results import (
    DatasetInspection,
    DatasetPreview,
    DatasetReport,
    DatasetWriteResult,
    HoursSelectionReview,
    SelectionReview,
    StatsReport,
)


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
        review = pipeline.review(examples=examples, hist_bins=hist_bins)
        return StatsReport(
            stage="curated",
            subject="pipeline",
            sections={"input": preview.to_dict(), "selection": review.to_dict()},
            warnings=list(preview.get("warnings") or ())
            + list(review.get("warnings") or ()),
        )

    report = pipeline.report(hist_bins=hist_bins)
    warnings = list(preview.get("warnings") or ())
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
    return StatsReport(
        stage="curated",
        subject="selection",
        sections={"selection": review.to_dict()},
        warnings=list(review.get("warnings") or ()),
    )


def _sample_flat(array: np.ndarray, limit: int) -> np.ndarray:
    flat = np.asarray(array).reshape(-1)
    if len(flat) <= limit:
        return flat
    indexes = np.linspace(0, len(flat) - 1, num=limit, dtype=np.int64)
    return flat[indexes]


def _numeric_chunk(accumulator: dict, values: np.ndarray, sample_limit: int) -> None:
    numeric = np.asarray(values, dtype=np.float64).reshape(-1)
    accumulator["values"] += int(numeric.size)
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
    out = {
        "dtype": accumulator["dtype"],
        "shape": list(accumulator["shape"]),
        "rows": accumulator["rows"],
        "values": accumulator["values"],
    }
    if accumulator["numeric"]:
        finite = accumulator["finite"]
        out["finite"] = finite
        out["non_finite"] = accumulator["non_finite"]
        if finite:
            mean = accumulator["mean"]
            variance = max(0.0, accumulator["m2"] / finite)
            sample = np.asarray(accumulator["samples"][:max_values], dtype=np.float64)
            counts, edges = np.histogram(sample, bins=hist_bins)
            out.update(
                {
                    "min": accumulator["min"],
                    "max": accumulator["max"],
                    "mean": mean,
                    "std": math.sqrt(variance),
                    "p50": float(np.percentile(sample, 50)),
                    "p95": float(np.percentile(sample, 95)),
                    "histogram": {
                        "counts": counts.tolist(),
                        "edges": [float(edge) for edge in edges],
                        "sampled_values": int(sample.size),
                    },
                }
            )
    else:
        counts = Counter(accumulator["samples"][:max_values])
        out.update(
            {
                "unique_sampled": len(counts),
                "top_values": dict(counts.most_common(10)),
                "sampled_values": min(len(accumulator["samples"]), max_values),
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
) -> dict:
    from .loaders import read_shard_arrays, shard_entries
    from .progress import emit

    entries, container = shard_entries(dataset)
    per_shard_sample = max(1, math.ceil(max_values / max(1, len(entries))))
    accumulators: dict[str, dict] = {}
    emit(progress, "stats", "start", total=len(entries))
    for index, entry in enumerate(entries, 1):
        arrays = read_shard_arrays(
            entry, container, trim=True, verify=verify, registry=registry
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
                    "samples": [],
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
            accumulator["rows"] += int(array.shape[0])
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
    emit(
        progress,
        "stats",
        "complete",
        current=len(entries),
        total=len(entries),
        detail=f"{len(accumulators)} modalities",
    )
    return {
        "shards_scanned": len(entries),
        "verified": verify,
        "sampling": {
            "max_values_per_modality": max_values,
            "method": "even-within-shard",
        },
        "modalities": {
            name: _finalize_modality(
                accumulator, hist_bins=hist_bins, max_values=max_values
            )
            for name, accumulator in sorted(accumulators.items())
        },
    }


def _artifact_stats(
    dataset,
    *,
    hist_bins: int,
    max_values: int,
    verify: bool,
    content: bool,
    progress=None,
    write_result: DatasetWriteResult | None = None,
    registry=None,
) -> StatsReport:
    from .inspection import inspect_dataset

    inspection = inspect_dataset(dataset, verify=False, registry=registry)
    sections: dict[str, Any] = {"artifact": inspection.to_dict()}
    warnings: list[dict] = []
    if content:
        sections["contents"] = _artifact_contents(
            dataset,
            hist_bins=hist_bins,
            max_values=max_values,
            verify=verify,
            progress=progress,
            registry=registry,
        )
    if write_result is not None:
        results = write_result.get("results") or {}
        if results.get("budget"):
            from .review import hours_selection_review

            review = hours_selection_review(write_result)
            sections["hours_selection"] = review.to_dict()
            warnings.extend(review.get("warnings") or ())
    return StatsReport(
        stage="output",
        subject=str(dataset),
        sections=sections,
        warnings=warnings,
    )


def stats(
    subject,
    *,
    hist_bins: int = 10,
    max_sources: int = 3,
    max_ticks_per_source: int = 10_000,
    examples: int = 5,
    max_values: int = 100_000,
    content: bool = True,
    verify: bool = False,
    progress=None,
    registry=None,
) -> StatsReport:
    """Return unified statistics for a pipeline, selection result, or written artifact.

    Pipeline statistics infer their stage from the immutable fluent plan. Written artifacts
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
            progress=progress,
            registry=registry,
        )
    raise TypeError(
        "stats() accepts a Pipeline, CurateResult, DatasetWriteResult, typed report, "
        "or written dataset path"
    )


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
    if len(edges) != len(counts) + 1:
        return
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
    if kind not in {None, "distribution", "missingness", "splits"}:
        raise ValueError(
            "stats plot kind must be 'distribution', 'missingness', or 'splits'"
        )

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
            f"<tr><td>{escape(name)}</td><td>{value.get('mean', 0.0):.5g}</td>"
            f"<td>{value.get('std', 0.0):.5g}</td><td>{value.get('p50', 0.0):.5g}</td>"
            f"<td>{value.get('p95', 0.0):.5g}</td></tr>"
        )
    for name, value in selection_metrics.items():
        candidate = value.get("candidate") or {}
        selected = value.get("selected") or {}
        metric_rows.append(
            f"<tr><td>{escape(name)}</td><td>{candidate.get('mean', 0.0):.5g} → "
            f"{selected.get('mean', 0.0):.5g}</td>"
            f"<td>{candidate.get('std', 0.0):.5g} → "
            f"{selected.get('std', 0.0):.5g}</td>"
            f"<td>{candidate.get('p50', 0.0):.5g} → "
            f"{selected.get('p50', 0.0):.5g}</td>"
            f"<td>{candidate.get('p95', 0.0):.5g} → "
            f"{selected.get('p95', 0.0):.5g}</td></tr>"
        )
    modalities = (sections.get("contents") or {}).get("modalities") or {}
    modality_rows = "".join(
        "<tr>"
        f"<td>{escape(name)}</td><td>{escape(value.get('dtype', 'unknown'))}</td>"
        f"<td>{escape(str(value.get('shape', [])))}</td>"
        f"<td>{value.get('rows', 0)}</td><td>{escape(str(value.get('mean', '—')))}</td>"
        "</tr>"
        for name, value in modalities.items()
    )
    warnings = "".join(
        f"<li><strong>{escape(item.get('code', 'general'))}</strong>: "
        f"{escape(item['message'])}</li>"
        for item in report.get("warnings") or ()
    )
    raw = escape(json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=str))
    body = (
        f"<section class=\"kur-stats\"><h2>{escape(title)}</h2>"
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
            "<th>Dtype</th><th>Shape</th><th>Rows</th><th>Mean</th></tr></thead>"
            f"<tbody>{modality_rows}</tbody></table>"
            if modality_rows
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
