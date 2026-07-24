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


def dataset_report(segments: list[Segment], *, metrics: Optional[Sequence[str]] = None,
                   hist_bins: int = 10) -> dict:
    """Distribution summary over `segments`. `metrics` defaults to every feature present."""
    n = len(segments)
    if n == 0:
        # keep the key set identical to the populated case so consumers (e.g. the report CLI)
        # don't KeyError on an empty pool.
        return DatasetReport(n_segments=0, n_sources=0, metrics={}, groups={}, tags={})
    names = list(metrics) if metrics is not None else sorted({k for s in segments for k in s.features})

    metric_rep: dict = {}
    for m in names:
        v = np.array([_feature_value(s, m) for s in segments], dtype=np.float64)
        counts, edges = np.histogram(v, bins=hist_bins)
        metric_rep[m] = {
            "mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max()),
            "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
            "histogram": {"counts": counts.tolist(), "edges": [float(e) for e in edges]},
        }

    rep = {
        "n_segments": n,
        "n_sources": len({s.source_id for s in segments}),
        "metrics": metric_rep,
        "groups": _group_counts(segments),
        "tags": {str(k): v for k, v in Counter(t for s in segments for t in s.tags).items()},
    }
    if any(s.bin for s in segments):
        bc = Counter(s.bin for s in segments)
        rep["bins"] = {"occupied": len(bc), "max_count": max(bc.values()), "min_count": min(bc.values())}
    return DatasetReport(rep)
