"""Equal-frequency (quantile) binning -> N-D bin coordinate over a strategy's axes.

Equal-width bins would pile most segments into one bucket (most windows are low-activity);
quantile bins give each occupied bin the same share of segments, so "slightly active" gets
the same selection budget as "very active". Vectorized: one `np.searchsorted` per axis over
the whole array, not a per-segment Python loop — which is why binning isn't parallelized
(vectorizing beats process-pool IPC here).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .segment import Segment


def quantile_bins(values: Sequence[float], n_bins: int) -> np.ndarray:
    """n_bins-1 edge values defining n_bins equal-frequency bins."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(max(n_bins - 1, 0), dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.quantile(arr, qs)


def bin_segments(segments: list[Segment], axes: Sequence[tuple[str, int]]) -> None:
    """Assign each segment an N-tuple `bin` over the axes (empty axes -> bin = ())."""
    n = len(segments)
    if n == 0 or not axes:
        for s in segments:
            s.bin = ()
        return
    cols: list[np.ndarray] = []
    for name, nbins in axes:
        vals = np.fromiter((s.features.get(name, 0.0) for s in segments),
                           dtype=np.float64, count=n)
        cols.append(np.searchsorted(quantile_bins(vals, nbins), vals, side="right"))
    for i, s in enumerate(segments):
        s.bin = tuple(int(c[i]) for c in cols)
