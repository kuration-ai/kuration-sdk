"""Single-dataset distribution / quality report.

The counterpart to the multi-arm `experiment` comparison: given one dataset's scored segments,
summarize what's actually there — per-metric distribution (mean/std/quantiles + a coarse
histogram), bin occupancy, and group/tag coverage. Use it to introspect what a strategy
selected, or to profile a raw corpus before curating.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

import numpy as np

from .segment import Segment


def dataset_report(segments: list[Segment], *, metrics: Optional[Sequence[str]] = None,
                   hist_bins: int = 10) -> dict:
    """Distribution summary over `segments`. `metrics` defaults to every feature present."""
    n = len(segments)
    if n == 0:
        # keep the key set identical to the populated case so consumers (e.g. the report CLI)
        # don't KeyError on an empty pool.
        return {"n_segments": 0, "n_sources": 0, "metrics": {}, "groups": {}, "tags": {}}
    names = list(metrics) if metrics is not None else sorted({k for s in segments for k in s.features})

    metric_rep: dict = {}
    for m in names:
        v = np.array([s.features.get(m, 0.0) for s in segments], dtype=np.float64)
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
        "groups": dict(Counter(str(s.group) for s in segments)),
        "tags": {str(k): v for k, v in Counter(t for s in segments for t in s.tags).items()},
    }
    if any(s.bin for s in segments):
        bc = Counter(s.bin for s in segments)
        rep["bins"] = {"occupied": len(bc), "max_count": max(bc.values()), "min_count": min(bc.values())}
    return rep
