"""Single-dataset distribution / quality report.

The counterpart to the multi-arm `experiment` comparison: given one dataset's scored segments,
summarize what's actually there — per-metric distribution (mean/std/quantiles + a coarse
histogram), bin occupancy, and group/tag coverage. Use it to introspect what a strategy
selected, or to profile a raw corpus before curating.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import pickle
from typing import Optional, Sequence

import numpy as np

from .segment import Segment
from .results import DatasetReport
from .strategy import _feature_value


def _group_identity(value) -> str:
    """Return a deterministic token used only to disambiguate report labels."""
    kind = type(value)
    type_name = f"{kind.__module__}.{kind.__qualname__}"
    payload: object
    if value is None or isinstance(value, (bool, int, str)):
        payload = value
    elif isinstance(value, float):
        payload = value.hex()
    elif isinstance(value, bytes):
        payload = value.hex()
    elif isinstance(value, (list, tuple)):
        payload = [_group_identity(item) for item in value]
    elif isinstance(value, dict):
        payload = sorted((_group_identity(key), _group_identity(item))
                         for key, item in value.items())
    elif isinstance(value, (set, frozenset)):
        payload = sorted(_group_identity(item) for item in value)
    elif isinstance(value, np.generic):
        payload = _group_identity(value.item())
    else:
        try:
            payload = {"pickle_sha256": hashlib.sha256(
                pickle.dumps(value, protocol=5)).hexdigest()}
        except Exception as error:
            raise ValueError(
                f"cannot derive a stable report identity for group type {type_name}; "
                "convert group keys to stable strings") from error
    return json.dumps({"type": type_name, "value": payload}, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=True)


def _same_group(left, right) -> bool:
    if type(left) is not type(right):
        return False
    try:
        equal = left == right
    except Exception as error:
        raise ValueError("group keys must support deterministic equality") from error
    if isinstance(equal, (bool, np.bool_)):
        return bool(equal)
    raise ValueError("group key equality must produce one boolean value")


def _group_counts(segments: list[Segment]) -> dict[str, int]:
    """Return JSON-object counts without merging unequal values or Python numeric types."""
    buckets: list[dict] = []
    by_hashable_value: dict[tuple[type, object], dict] = {}
    unhashable_buckets: list[dict] = []
    by_identity: dict[str, dict] = {}

    def new_bucket(value) -> dict:
        identity = _group_identity(value)
        if identity in by_identity:
            raise ValueError(
                "unequal group keys have the same stable representation; convert group "
                "keys to distinct stable strings")
        kind = type(value)
        label = kind.__name__ if kind.__module__ == "builtins" \
            else f"{kind.__module__}.{kind.__qualname__}"
        bucket = {"value": value, "identity": identity, "label": label,
                  "text": str(value), "count": 0}
        buckets.append(bucket)
        by_identity[identity] = bucket
        return bucket

    for segment in segments:
        value = segment.group
        try:
            hash(value)
        except TypeError:
            bucket = next((candidate for candidate in unhashable_buckets
                           if _same_group(value, candidate["value"])), None)
            if bucket is None:
                bucket = new_bucket(value)
                unhashable_buckets.append(bucket)
        else:
            hash_key = (type(value), value)
            try:
                bucket = by_hashable_value.get(hash_key)
            except (TypeError, ValueError) as error:
                raise ValueError("group keys must support deterministic equality") from error
            if bucket is None:
                bucket = new_bucket(value)
                by_hashable_value[hash_key] = bucket
        bucket["count"] += 1

    by_text = Counter(bucket["text"] for bucket in buckets)
    by_label_text = Counter((bucket["label"], bucket["text"]) for bucket in buckets)
    out: dict[str, int] = {}
    for bucket in sorted(buckets, key=lambda item: (
            item["text"], item["label"], item["identity"])):
        label, text = bucket["label"], bucket["text"]
        if by_text[text] == 1:
            output_key = text
        elif by_label_text[(label, text)] == 1:
            output_key = f"{label}:{text}"
        else:
            digest = hashlib.sha256(bucket["identity"].encode()).hexdigest()[:12]
            output_key = f"{label}:{text}:{digest}"
        # A literal string group can itself look like a generated key. Keep the result lossless.
        while output_key in out:
            output_key = f"{label}:{output_key!r}"
        out[output_key] = bucket["count"]
    return out


def _histogram(values: np.ndarray, hist_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Histogram one metric, tolerating a degenerate spread.

    NumPy widens the range for exactly-constant data but raises "Too many bins for data
    range" when values differ only at float-noise scale — so a metric that barely varies
    (a near-flat field, a saturated score) would crash the whole report instead of showing
    up as the flat distribution it is. Widen such a range the same way NumPy already does
    for the constant case, and ignore non-finite values rather than propagating them.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(hist_bins, dtype=int), np.linspace(0.0, 1.0, hist_bins + 1)
    low, high = float(finite.min()), float(finite.max())
    # A bin narrower than the float spacing at this magnitude is what NumPy refuses, so the
    # widened span has to scale with the magnitude: `low - 0.5, high + 0.5` still crashed for
    # constant values around 2**53, where consecutive doubles are 2.0 apart.
    floor = max(0.5, hist_bins * float(np.spacing(max(abs(low), abs(high)))))
    if high - low <= floor:
        center = low + (high - low) / 2
        low, high = center - floor, center + floor
    return np.histogram(finite, bins=hist_bins, range=(low, high))


def _metric_summary(values: np.ndarray, hist_bins: int) -> dict:
    """Summarize one metric over its *finite* values, keeping the exclusion visible.

    The histogram has always ignored non-finite values; mean/std/min/max/p50/p95 used to keep
    them, so one NaN feature produced `mean=nan` beside a clean histogram — and `nan`/`inf`
    serialize as `NaN`/`Infinity`, which is not valid JSON (`kur-report --out`, `kur-stats
    --json`). Every statistic now describes the finite values only, `n_nonfinite` records how
    many were excluded, and a metric with no finite value at all reports `None` rather than a
    number nobody can trust.
    """
    finite = values[np.isfinite(values)]
    counts, edges = _histogram(values, hist_bins)
    summary: dict = {
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std()) if finite.size else None,
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "p50": float(np.percentile(finite, 50)) if finite.size else None,
        "p95": float(np.percentile(finite, 95)) if finite.size else None,
        "n_finite": int(finite.size),
        "n_nonfinite": int(values.size - finite.size),
        "histogram": {"counts": counts.tolist(), "edges": [float(e) for e in edges]},
    }
    return summary


def dataset_report(segments: list[Segment], *, metrics: Optional[Sequence[str]] = None,
                   hist_bins: int = 10) -> DatasetReport:
    """Distribution summary over `segments`. `metrics` defaults to every feature present."""
    n = len(segments)
    if n == 0:
        # keep the key set identical to the populated case so consumers (e.g. the report CLI)
        # don't KeyError on an empty pool.
        return DatasetReport(n_segments=0, n_sources=0, metrics={}, groups={}, tags={},
                             warnings=[])
    names = list(metrics) if metrics is not None else sorted({k for s in segments for k in s.features})

    metric_rep: dict = {}
    warnings: list[dict] = []
    for m in names:
        v = np.array([_feature_value(s, m) for s in segments], dtype=np.float64)
        metric_rep[m] = _metric_summary(v, hist_bins)
        if metric_rep[m]["n_nonfinite"]:
            warnings.append({
                "code": "non-finite-metric", "metric": m,
                "message": f"{m} has {metric_rep[m]['n_nonfinite']}/{n} non-finite value(s); "
                           "statistics describe the finite values only"})

    rep = {
        "n_segments": n,
        "n_sources": len({s.source_id for s in segments}),
        "metrics": metric_rep,
        "groups": _group_counts(segments),
        "tags": {str(k): v for k, v in Counter(t for s in segments for t in s.tags).items()},
        "warnings": warnings,
    }
    if any(s.bin for s in segments):
        bc = Counter(s.bin for s in segments)
        rep["bins"] = {"occupied": len(bc), "max_count": max(bc.values()), "min_count": min(bc.values())}
    return DatasetReport(rep)
