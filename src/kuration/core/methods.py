"""Feature-space selection methods + pre-pass filters (optional, scikit-learn).

These extend the base methods (uniform-bins / weighted / top-k / target-distribution / filter)
with coverage- and representativeness-driven selection over the scalar metric vectors already
in `Segment.features` — the hand-crafted "telemetry embedding" the strategy bins/weights on.
No learned embeddings, no GPU.

This is the only module that imports scikit-learn (the optional `[methods]` extra), lazily and
with an install hint. Rationale (carried from the reference implementation):
  - **cluster-balanced** generalizes uniform-bins from per-axis quantile bins to joint k-means
    clusters (DINOv2 balanced subsampling): coverage over the *joint* space, not each axis.
  - **k-center** / **facility-location** order those clusters by a coverage criterion
    (farthest-point / submodular gain), then round-robin within — both run on the M centroids,
    not the n segments, bounding the O(n^2) blow-up.
  - **dsir** is multivariate importance resampling: a soft, joint target-distribution over a
    quantized feature space (flat, metric-shaped, or anchored to a high-quality subset).
  - **rare** / **typical** are kNN density (atypicality / prototypicality) — use as contrast
    arms; difficulty-only selection underperforms coverage at high pruning.

Determinism: callers pre-sort segments, so the feature matrix and every seeded step are
worker-count-independent. Scale caps are logged, never silent.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from .binning import bin_codes
from .segment import Segment
from .registry import default_registry as _default_registry
from .strategy import _feature_value

_DEFAULT_CLUSTERS = 2048
_DSIR_BINS = 8
logger = logging.getLogger(__name__)


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as e:
        raise ImportError("this method/filter needs scikit-learn (optional): "
                          "pip install 'kuration-sdk[methods]'") from e


def resolve_feature_space(strategy) -> list[str]:
    """Metric names spanning the vector space: explicit `feature_space`, else the binning
    axes. Deduped, order-stable. Raises if neither is set."""
    names = strategy.feature_space or [m for m, _ in strategy.axes]
    if not names:
        raise ValueError(f"method {strategy.method!r} needs 'feature_space' or binning.axes")
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def feature_matrix(segments: list[Segment], names: Sequence[str], *,
                   normalize: str = "zscore") -> np.ndarray:
    """Stack `names` features into a standardized (n, d) matrix. `zscore` (k-means/Euclidean),
    `quantile` (rank in [0,1], outlier-robust), or `none`."""
    n, d = len(segments), len(names)
    X = np.empty((n, d), dtype=np.float64)
    for j, name in enumerate(names):
        X[:, j] = np.fromiter((_feature_value(s, name) for s in segments),
                              dtype=np.float64, count=n)
    if not np.isfinite(X).all():
        raise ValueError("feature_space contains non-finite metric values; metrics/scorers must "
                         "return finite scalars")
    if normalize == "none" or n <= 1:
        return X
    if normalize == "quantile":
        from .strategy import _normalize
        for j in range(d):
            X[:, j] = _normalize(X[:, j], "quantile")
        return X
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def cluster_labels(X, n_clusters, *, seed=0, workers=1):
    """(labels, centroids). Exact KMeans for small n, MiniBatchKMeans past a threshold.
    n_clusters clamped to n and logged if reduced."""
    _require_sklearn()
    from sklearn.cluster import KMeans, MiniBatchKMeans
    n = X.shape[0]
    k = max(1, min(n_clusters, n))
    if k < n_clusters:
        logger.info("clusters capped %s -> %s (only %s segments)", n_clusters, k, n)
    # batch_size is fixed (not a function of `workers`): it changes the clustering, and the
    # selection must stay worker-count-independent (determinism invariant). `workers` only
    # tunes sklearn's n_jobs elsewhere, which doesn't affect results.
    km = (KMeans(n_clusters=k, random_state=seed, n_init=3) if n <= 20000
          else MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=1024))
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        labels = km.fit_predict(X)
    return labels.astype(np.int64), km.cluster_centers_


def _round_robin_priorities(labels, cluster_order, within_rank) -> np.ndarray:
    """Take rank-0 of each cluster (visited in cluster_order), then every rank-1, etc."""
    n_clusters = max(len(cluster_order), 1)
    co = np.array([cluster_order.get(int(l), 0) for l in labels], dtype=np.float64)
    return -(within_rank.astype(np.float64) * n_clusters + co)


def _within_cluster_rank(X, labels, centroids, *, seed, shuffle) -> np.ndarray:
    """Per-segment rank in its cluster: ascending distance to centroid (shuffle=False,
    representative-first) or seeded random (shuffle=True, balanced)."""
    n = X.shape[0]
    rank = np.empty(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    by_cluster: dict = {}
    for i, l in enumerate(labels):
        by_cluster.setdefault(int(l), []).append(i)
    for l, idxs in by_cluster.items():
        idxs_arr = np.array(idxs)
        if shuffle:
            order = idxs_arr[rng.permutation(len(idxs_arr))]
        else:
            d = np.linalg.norm(X[idxs_arr] - centroids[l], axis=1)
            order = idxs_arr[np.argsort(d, kind="stable")]
        rank[order] = np.arange(len(order))
    return rank


def cluster_balanced_priorities(segments, names, *, n_clusters=_DEFAULT_CLUSTERS,
                                normalize="zscore", seed=0, workers=1) -> np.ndarray:
    if not segments:
        return np.zeros(0)
    X = feature_matrix(segments, names, normalize=normalize)
    labels, centroids = cluster_labels(X, n_clusters, seed=seed, workers=workers)
    within = _within_cluster_rank(X, labels, centroids, seed=seed, shuffle=True)
    order = {c: i for i, c in enumerate(sorted(set(int(l) for l in labels)))}
    return _round_robin_priorities(labels, order, within)


def _farthest_point_order(C) -> list[int]:
    """Deterministic farthest-point traversal of the centroids (no RNG: the start is the
    centroid nearest the mean, then greedily the farthest remaining)."""
    k = C.shape[0]
    if k == 0:
        return []
    start = int(np.argmin(np.linalg.norm(C - C.mean(axis=0), axis=1)))
    chosen = [start]
    min_d = np.linalg.norm(C - C[start], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
        min_d = np.minimum(min_d, np.linalg.norm(C - C[nxt], axis=1))
    return chosen


def kcenter_priorities(segments, names, *, n_clusters=_DEFAULT_CLUSTERS,
                       normalize="zscore", seed=0, workers=1) -> np.ndarray:
    if not segments:
        return np.zeros(0)
    X = feature_matrix(segments, names, normalize=normalize)
    labels, centroids = cluster_labels(X, n_clusters, seed=seed, workers=workers)
    order = {int(c): i for i, c in enumerate(_farthest_point_order(centroids))}
    within = _within_cluster_rank(X, labels, centroids, seed=seed, shuffle=False)
    return _round_robin_priorities(labels, order, within)


def _facility_location_order(C, weights) -> list[int]:
    k = C.shape[0]
    if k == 0:
        return []
    norm = np.linalg.norm(C, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    S = (C / norm) @ (C / norm).T
    covered = np.zeros(k)
    chosen, remaining = [], set(range(k))
    while remaining:
        rem = np.fromiter(remaining, dtype=np.int64, count=len(remaining))
        gains = (weights[:, None] * np.maximum(S[:, rem] - covered[:, None], 0.0)).sum(axis=0)
        j = int(rem[int(np.argmax(gains))])
        chosen.append(j)
        covered = np.maximum(covered, S[:, j])
        remaining.discard(j)
    return chosen


def facility_location_priorities(segments, names, *, n_clusters=_DEFAULT_CLUSTERS,
                                 normalize="zscore", seed=0, workers=1) -> np.ndarray:
    if not segments:
        return np.zeros(0)
    X = feature_matrix(segments, names, normalize=normalize)
    labels, centroids = cluster_labels(X, n_clusters, seed=seed, workers=workers)
    counts = np.bincount(labels, minlength=centroids.shape[0]).astype(np.float64)
    order = {int(c): i for i, c in enumerate(_facility_location_order(centroids, counts))}
    within = _within_cluster_rank(X, labels, centroids, seed=seed, shuffle=False)
    return _round_robin_priorities(labels, order, within)


def dsir_priorities(segments, names, *, n_bins=_DSIR_BINS, target=None,
                    normalize="quantile", seed=0, workers=1) -> np.ndarray:
    """Importance resampling over a quantized joint feature space. Target p is uniform over
    occupied codes (flat), ∝ a metric per code (`{"metric": name}`), or the code distribution
    of a high-quality anchor subset (`{"anchor_metric": name, "anchor_quantile": q}`). Weight
    w = p/q; priority = log(w) + Gumbel (sampling without replacement ∝ w). numpy-only."""
    n = len(segments)
    if n == 0:
        return np.zeros(0)
    codes = np.zeros(n, dtype=np.int64)
    base = 1
    for name in names:
        vals = np.fromiter((_feature_value(s, name) for s in segments),
                           dtype=np.float64, count=n)
        codes = codes + base * bin_codes(vals, n_bins, axis_name=name)
        base *= (n_bins + 1)
    uniq, inv, counts = np.unique(codes, return_inverse=True, return_counts=True)
    q = counts[inv].astype(np.float64) / n
    if isinstance(target, dict) and "anchor_metric" in target:
        vals = np.fromiter((_feature_value(s, target["anchor_metric"]) for s in segments),
                           dtype=np.float64, count=n)
        anchor = vals >= np.quantile(vals, float(target.get("anchor_quantile", 0.75)))
        per_code = np.zeros(uniq.size)
        np.add.at(per_code, inv[anchor], 1.0)
        p = per_code[inv] / (per_code.sum() or 1.0)
    elif isinstance(target, dict) and "metric" in target:
        mvals = np.fromiter((max(0.0, _feature_value(s, target["metric"]))
                             for s in segments),
                            dtype=np.float64, count=n)
        per_code = np.zeros(uniq.size)
        np.add.at(per_code, inv, mvals)
        p = per_code[inv] / (per_code.sum() or 1.0)
    else:
        p = np.full(n, 1.0 / uniq.size)
    w = np.maximum(p / np.maximum(q, 1e-12), 1e-12)
    rng = np.random.default_rng(seed)
    return np.log(w) + (-np.log(-np.log(rng.random(n))))


def rare_priorities(segments, names, *, k=10, normalize="zscore", seed=0, workers=1,
                    sample=20000) -> np.ndarray:
    """Density-inverse rarity: priority = mean distance to k nearest neighbors (sparse =
    rare = admitted first). Reference set is a seeded sample at scale. Atypicality selection —
    use as a contrast arm, not alone."""
    if not segments:
        return np.zeros(0)
    _require_sklearn()
    from sklearn.neighbors import NearestNeighbors
    X = feature_matrix(segments, names, normalize=normalize)
    n = X.shape[0]
    if n > sample:
        rng = np.random.default_rng(seed)
        ref = X[np.sort(rng.choice(n, size=sample, replace=False))]
    else:
        ref = X
    if ref.shape[0] < 2:
        return np.zeros(n)
    # Query k+1 and drop each point's zero-distance self-hit (points inside the reference
    # sample would otherwise count themselves, deflating their rarity vs unsampled points).
    nn = NearestNeighbors(n_neighbors=min(k + 1, ref.shape[0]),
                          n_jobs=(workers or 1)).fit(ref)
    d, _ = nn.kneighbors(X)
    self_hit = d[:, 0] == 0.0
    out = np.empty(n)
    out[self_hit] = d[self_hit, 1:].mean(axis=1)
    out[~self_hit] = d[~self_hit, :-1].mean(axis=1)
    return out


def typical_priorities(segments, names, **kw) -> np.ndarray:
    """Prototypicality: the negative of `rare` (densest = most typical = admitted first)."""
    return -rare_priorities(segments, names, **kw)


PRIORITY_METHODS: dict[str, Any] = {
    "cluster-balanced": cluster_balanced_priorities,
    "k-center": kcenter_priorities,
    "facility-location": facility_location_priorities,
    "dsir": dsir_priorities,
    "rare": rare_priorities,
    "typical": typical_priorities,
}
METHOD_MODULE_METHODS = set(PRIORITY_METHODS)


def method_priorities(segments, strategy, *, seed=0, workers=1) -> np.ndarray:
    """Return one selection priority per segment for a feature-space strategy.

    The strategy chooses the method, feature space, normalization, and method parameters.
    These methods require the ``methods`` extra and raise ``ImportError`` when it is absent.
    """
    names = resolve_feature_space(strategy)
    fn = PRIORITY_METHODS[strategy.method]
    p = strategy.method_params or {}
    normalize = p.get("feature_normalize", "quantile" if strategy.method == "dsir" else "zscore")
    if strategy.method == "dsir":
        return fn(segments, names, n_bins=int(p.get("dsir_bins", _DSIR_BINS)),
                  target=strategy.target, normalize=normalize, seed=seed, workers=workers)
    if strategy.method in ("rare", "typical"):
        return fn(segments, names, k=int(p.get("knn_k", 10)), normalize=normalize,
                  seed=seed, workers=workers, sample=int(p.get("rare_sample", 20000)))
    return fn(segments, names, n_clusters=int(p.get("clusters", _DEFAULT_CLUSTERS)),
              normalize=normalize, seed=seed, workers=workers)


# ---- pre-pass filters -------------------------------------------------------
def anomaly_filter(segments, names, *, contamination=0.02, normalize="zscore",
                   seed=0, workers=1) -> list[Segment]:
    """Drop multivariate outliers (corrupted/degenerate windows) with an Isolation Forest."""
    if not segments:
        return segments
    _require_sklearn()
    from sklearn.ensemble import IsolationForest
    X = feature_matrix(segments, names, normalize=normalize)
    keep = IsolationForest(contamination=contamination, random_state=seed,
                           n_jobs=(workers or 1)).fit_predict(X) == 1
    kept = [s for s, k in zip(segments, keep) if k]
    logger.info("anomaly filter removed %s / %s segments (contamination=%s)",
                len(segments) - len(kept), len(segments), contamination)
    return kept


def dedup_filter(segments, names, *, threshold=0.98, n_clusters=_DEFAULT_CLUSTERS,
                 normalize="zscore", seed=0, workers=1) -> list[Segment]:
    """Feature-space SemDeDup: within each cluster keep one representative per near-duplicate
    group (cosine sim >= threshold), centroid-outward."""
    if len(segments) < 2:
        return segments
    X = feature_matrix(segments, names, normalize=normalize)
    labels, centroids = cluster_labels(X, n_clusters, seed=seed, workers=workers)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    Xn = X / norm
    keep_mask = np.ones(len(segments), dtype=bool)
    by_cluster: dict = {}
    for i, l in enumerate(labels):
        by_cluster.setdefault(int(l), []).append(i)
    for l, idxs in by_cluster.items():
        if len(idxs) < 2:
            continue
        idxs_arr = np.array(idxs)
        d = np.linalg.norm(X[idxs_arr] - centroids[l], axis=1)
        reps: list[int] = []
        for i in idxs_arr[np.argsort(d, kind="stable")]:
            if not reps or (Xn[i] @ Xn[reps].T).max() < threshold:
                reps.append(i)
            else:
                keep_mask[i] = False
    kept = [s for s, k in zip(segments, keep_mask) if k]
    logger.info("dedup filter removed %s / %s near-duplicates (cos>=%s)",
                len(segments) - len(kept), len(segments), threshold)
    return kept


_FILTERS: dict[str, Any] = {"anomaly": anomaly_filter, "dedup": dedup_filter}


def register_filter(name, fn, **metadata):
    """Register a pre-pass filter callable and its component metadata."""
    from .registry import register_filter as _register
    _register(name, fn, **metadata)


def apply_filters(segments: list[Segment], strategy, *, seed=0, workers=1) -> list[Segment]:
    """Chain `strategy.filters` ([{type: anomaly|dedup, ...knobs, feature_space?, per_group?}])
    over the pool before binning/selection. A filter may carry its own `feature_space` to gate
    on a different vector space than the selector. `per_group` scopes it within each
    `Segment.group` (avoids global dedup skewing the corpus), preserving order for determinism.
    """
    filters = getattr(strategy, "filters", None) or []
    if not filters:
        return segments
    default_names = None
    out = segments
    for spec in filters:
        ftype = spec.get("type")
        if ftype not in _FILTERS:
            from .registry import default_registry
            registry = getattr(strategy, "registry", None) or default_registry
            fn = registry.get("filter", ftype)
        else:
            fn = _FILTERS[ftype]
        names = spec.get("feature_space")
        if not names:
            if default_names is None:
                default_names = resolve_feature_space(strategy)
            names = default_names
        knobs = {k: v for k, v in spec.items() if k not in ("type", "feature_space", "per_group")}
        if spec.get("per_group"):
            from collections import defaultdict
            groups: dict = defaultdict(list)
            for s in out:
                groups[s.group].append(s)
            kept_ids = set()
            for g in sorted(groups, key=lambda x: (x is None, str(type(x)), str(x))):
                for s in fn(groups[g], names, seed=seed, workers=workers, **knobs):
                    kept_ids.add(id(s))
            out = [s for s in out if id(s) in kept_ids]
        else:
            out = fn(out, names, seed=seed, workers=workers, **knobs)
    return out


for _name, _fn in PRIORITY_METHODS.items():
    if _name not in _default_registry.names("method"):
        _default_registry.register("method", _name, _fn, capabilities={"whole_pool": True},
                                   extra="methods")
for _name, _fn in _FILTERS.items():
    if _name not in _default_registry.names("filter"):
        _default_registry.register("filter", _name, _fn, capabilities={"whole_pool": True},
                                   extra="methods")
