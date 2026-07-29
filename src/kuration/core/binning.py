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


def quantile_bins(values: Sequence[float] | np.ndarray, n_bins: int) -> np.ndarray:
    """n_bins-1 edge values defining n_bins equal-frequency bins."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(max(n_bins - 1, 0), dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.quantile(arr, qs)


def bin_codes(vals: np.ndarray, n_bins: int, *, axis_name: str = "?") -> np.ndarray:
    """Equal-frequency bin index per value, robust to ties. Skewed features produce duplicate
    quantile edges; deduping them and putting edge-equal values in the LOWER bin keeps a tied
    mass (e.g. 90% zeros) in its own bin instead of absorbing everything above it into one.
    NaN would silently poison the edges (all-NaN quantiles -> one bin), so it raises."""
    if np.isnan(vals).any():
        raise ValueError(
            f"binning axis {axis_name!r}: {int(np.isnan(vals).sum())} NaN feature value(s) — "
            f"a registered metric or scorer produced NaN; fix it or filter those segments")
    edges = np.unique(quantile_bins(vals, n_bins))
    if vals.size > 1 and np.unique(vals).size == 1:
        import warnings
        warnings.warn(f"binning axis {axis_name!r} is constant — every segment lands in one "
                      f"bin; the axis adds no selection signal", stacklevel=2)
    return np.searchsorted(edges, vals, side="left")


def bin_segments(segments: list[Segment], axes: Sequence[tuple[str, int]]) -> None:
    """Assign each segment an N-tuple `bin` over the axes (empty axes -> bin = ())."""
    n = len(segments)
    if n == 0 or not axes:
        for s in segments:
            s.bin = ()
        return
    cols: list[np.ndarray] = []
    for name, nbins in axes:
        missing = next((segment for segment in segments if name not in segment.features), None)
        if missing is not None:
            raise ValueError(
                f"binning axis {name!r}: segment {missing.source_id!r} "
                f"[{missing.start_idx}, {missing.end_idx}) has no such feature; "
                "ensure every metric/transform populates every candidate segment")
        vals = np.fromiter((s.features[name] for s in segments),
                           dtype=np.float64, count=n)
        cols.append(bin_codes(vals, nbins, axis_name=name))
    for i, s in enumerate(segments):
        s.bin = tuple(int(c[i]) for c in cols)
