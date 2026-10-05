"""Operator wrappers around every built-in selection Method, pre-selection Filter,
and whole-pool Scorer.

Each function has the ordinary compositional-operator shape (``pipeline, **params ->
pipeline``): it validates nothing itself and adds no new execution machinery, it simply calls
the existing ``.curate()``/``.filter()``/``.score()`` methods that already implement the
underlying Method/Filter/Scorer contract. Registering them under the real ``"operator"`` kind
means every one of these becomes a named method (``pipeline.top_k(...)``,
``pipeline.dedup(...)``, ``pipeline.action_residual(...)``) exactly like
``kuration.selection``'s recipes or the gameplay domain's operators -- there is no longer a
separate "Method"/"Filter"/"Scorer" vocabulary a user needs to learn to find these by name.

The generic string-keyed forms (``.sample(method="top-k", ...)``, ``.filter("dedup", ...)``,
``.score("action_residual", ...)``) remain the escape hatch for a caller that has the method
name as data (a config file, a CLI flag) rather than as code -- the same relationship
``.transform()``/``.score()`` already have to their own named operator wrappers elsewhere in
the SDK. Both forms produce identical results by construction, since the named form is nothing
but the string-keyed call written out.

``uniform-bins``/``target-distribution`` need `.bin(...)` axes declared on the pipeline first;
that isn't duplicated here since `.curate()` already raises a clear, specific error
(``"method 'uniform-bins' requires binning.axes"``) when it's missing.

Filters (``anomaly``, ``dedup``) and the ``action_residual`` scorer keep their original
parameter names verbatim (``contamination``, ``n_clusters``, ``obs``/``action``/``k``) --
Method's surface is the one with a translated method_params vocabulary
(``clusters``/``feature_normalize``/``dsir_bins``/``knn_k``/``rare_sample`` instead of
``n_clusters``/``normalize``/``n_bins``/``k``/``sample``), confirmed empirically against
`strategy.py`'s `_resolve`/`Pipeline.sample`, and preserved here rather than "fixed" -- these
wrappers describe what `.sample()` actually accepts today, not a hypothetical cleaner surface.
"""
from __future__ import annotations

from typing import Any, Callable

_N_OR_HOURS_PROPERTIES = {
    "n": {"type": "integer", "minimum": 1},
    "hours": {"type": "number", "exclusiveMinimum": 0},
    "test_hours": {"type": "number", "minimum": 0},
    "group_cap": {
        "type": "object",
        "properties": {"max_group_overshoot": {"type": "number", "exclusiveMinimum": 0}},
        "additionalProperties": False,
    },
}

_WEIGHTS_BASE_PROPERTIES = {
    "weights": {"type": "object", "additionalProperties": {"type": "number"}},
    "normalize": {"type": "string", "enum": ["quantile", "minmax", "zscore", "none"]},
    **_N_OR_HOURS_PROPERTIES,
}

WEIGHTS_PARAMETERS = {
    "type": "object",
    "properties": {**_WEIGHTS_BASE_PROPERTIES},
    "required": ["weights"],
    "additionalProperties": False,
}

#: "weighted" only -- `floor` tunes the Gumbel-top-k sampler's non-negativity shift, which
#: `top-k` (a deterministic rank, no sampling) has no use for.
WEIGHTED_PARAMETERS = {
    "type": "object",
    "properties": {
        **_WEIGHTS_BASE_PROPERTIES,
        "floor": {"type": "number", "minimum": 0},
    },
    "required": ["weights"],
    "additionalProperties": False,
}

UNIFORM_BINS_PARAMETERS = {
    "type": "object",
    "properties": {**_N_OR_HOURS_PROPERTIES},
    "additionalProperties": False,
}

_TARGET_SCHEMA = {
    "anyOf": [
        {"type": "string", "const": "flat"},
        {
            "type": "object",
            "properties": {
                "metric": {"type": "string"},
                "anchor_metric": {"type": "string"},
                "anchor_quantile": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "additionalProperties": False,
        },
    ],
}

TARGET_DISTRIBUTION_PARAMETERS = {
    "type": "object",
    "properties": {
        "target": _TARGET_SCHEMA,
        **_N_OR_HOURS_PROPERTIES,
    },
    "additionalProperties": False,
}

_FEATURE_SPACE_BASE_PROPERTIES = {
    "feature_space": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    "clusters": {"type": "integer", "minimum": 1},
    "feature_normalize": {"type": "string", "enum": ["quantile", "minmax", "zscore", "none"]},
    **_N_OR_HOURS_PROPERTIES,
}

FEATURE_SPACE_CLUSTER_PARAMETERS = {
    "type": "object",
    "properties": {**_FEATURE_SPACE_BASE_PROPERTIES},
    "required": ["feature_space"],
    "additionalProperties": False,
}

DSIR_PARAMETERS = {
    "type": "object",
    "properties": {
        **_FEATURE_SPACE_BASE_PROPERTIES,
        "dsir_bins": {"type": "integer", "minimum": 2},
        "target": _TARGET_SCHEMA,
    },
    "required": ["feature_space"],
    "additionalProperties": False,
}

RARITY_PARAMETERS = {
    "type": "object",
    "properties": {
        **_FEATURE_SPACE_BASE_PROPERTIES,
        "knn_k": {"type": "integer", "minimum": 1},
        "rare_sample": {"type": "integer", "minimum": 1},
    },
    "required": ["feature_space"],
    "additionalProperties": False,
}

ANOMALY_FILTER_PARAMETERS = {
    "type": "object",
    "properties": {
        "contamination": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
        "normalize": {"type": "string", "enum": ["quantile", "minmax", "zscore", "none"]},
    },
    "additionalProperties": False,
}

DEDUP_FILTER_PARAMETERS = {
    "type": "object",
    "properties": {
        "feature_space": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "threshold": {"type": "number", "minimum": -1, "maximum": 1},
        "n_clusters": {"type": "integer", "minimum": 1},
        "normalize": {"type": "string", "enum": ["quantile", "minmax", "zscore", "none"]},
    },
    "required": ["feature_space"],
    "additionalProperties": False,
}

ACTION_RESIDUAL_PARAMETERS = {
    "type": "object",
    "properties": {
        "obs": {"type": "string"},
        "action": {"type": "string"},
        "k": {"type": "integer", "minimum": 1},
        "out": {"type": "string"},
    },
    "required": ["obs", "action"],
    "additionalProperties": False,
}


def top_k(pipeline, *, weights, normalize=None, n=None, hours=None, test_hours=None,
          group_cap=None):
    """Rank by a weighted linear combination of normalized features; take the highest
    priorities."""
    return pipeline.curate("top-k", weights=weights, normalize=normalize, n=n, hours=hours,
                           test_hours=test_hours, group_cap=group_cap)


def weighted(pipeline, *, weights, normalize=None, floor=None, n=None, hours=None,
            test_hours=None, group_cap=None):
    """Gumbel-top-k: sample without replacement, proportional to a weighted score.

    ``floor`` raises the non-negativity shift applied to the weighted score (default ``1e-9``)
    so the lowest-scoring segment in the pool keeps a real, non-negligible relative chance of
    selection instead of one near zero by construction.
    """
    extras = _method_params(floor=floor)
    return pipeline.curate("weighted", weights=weights, normalize=normalize, n=n, hours=hours,
                           test_hours=test_hours, group_cap=group_cap, **extras)


def uniform_bins(pipeline, *, n=None, hours=None, test_hours=None, group_cap=None):
    """Quantile-bin along declared `.bin(...)` axes and round-robin the bins for
    coverage-first selection. Requires at least one `.bin(...)` axis already declared."""
    return pipeline.curate("uniform-bins", n=n, hours=hours, test_hours=test_hours,
                           group_cap=group_cap)


def target_distribution(pipeline, *, target=None, n=None, hours=None, test_hours=None,
                        group_cap=None):
    """Match the pool to an explicit target distribution over a scalar feature. Requires at
    least one `.bin(...)` axis already declared."""
    return pipeline.curate("target-distribution", target=target, n=n, hours=hours,
                           test_hours=test_hours, group_cap=group_cap)


def _method_params(**kwargs) -> dict:
    """Drop unset (``None``) method_params extras before forwarding to `.curate()`.

    Unlike `.curate()`'s own named parameters (``n``/``hours``/``weights``/``target``/etc.,
    which default to and tolerate ``None``), extras routed through its ``**method_params``
    (``feature_normalize``/``clusters``/``dsir_bins``/``knn_k``/``rare_sample``) are validated
    strictly by `strategy._resolve` and reject an explicit ``None`` -- they must be entirely
    absent to fall back to the method's own default, confirmed empirically against
    `strategy.py`'s `_resolve`.
    """
    return {key: value for key, value in kwargs.items() if value is not None}


def cluster_balanced(pipeline, *, feature_space, clusters=None, feature_normalize=None,
                     n=None, hours=None, test_hours=None, group_cap=None):
    """Joint k-means-cluster round-robin coverage over the feature space (DINOv2-style
    balanced subsampling)."""
    extras = _method_params(clusters=clusters, feature_normalize=feature_normalize)
    return pipeline.curate("cluster-balanced", feature_space=feature_space, n=n, hours=hours,
                           test_hours=test_hours, group_cap=group_cap, **extras)


def k_center(pipeline, *, feature_space, clusters=None, feature_normalize=None, n=None,
            hours=None, test_hours=None, group_cap=None):
    """Farthest-point traversal of the cluster centroids ordered by coverage."""
    extras = _method_params(clusters=clusters, feature_normalize=feature_normalize)
    return pipeline.curate("k-center", feature_space=feature_space, n=n, hours=hours,
                           test_hours=test_hours, group_cap=group_cap, **extras)


def facility_location(pipeline, *, feature_space, clusters=None, feature_normalize=None,
                      n=None, hours=None, test_hours=None, group_cap=None):
    """Submodular facility-location cluster ordering (greedy coverage gain)."""
    extras = _method_params(clusters=clusters, feature_normalize=feature_normalize)
    return pipeline.curate("facility-location", feature_space=feature_space, n=n, hours=hours,
                           test_hours=test_hours, group_cap=group_cap, **extras)


def dsir(pipeline, *, feature_space, dsir_bins=None, target=None, feature_normalize=None,
        n=None, hours=None, test_hours=None, group_cap=None):
    """Multivariate importance resampling with Gumbel-top-k over a quantized joint feature
    space."""
    extras = _method_params(dsir_bins=dsir_bins, feature_normalize=feature_normalize)
    return pipeline.curate("dsir", feature_space=feature_space, target=target, n=n,
                           hours=hours, test_hours=test_hours, group_cap=group_cap, **extras)


def rare(pipeline, *, feature_space, knn_k=None, rare_sample=None, feature_normalize=None,
        n=None, hours=None, test_hours=None, group_cap=None):
    """Density-inverse kNN atypicality: sparse (rare) regions of the feature space are
    admitted first."""
    extras = _method_params(knn_k=knn_k, rare_sample=rare_sample,
                            feature_normalize=feature_normalize)
    return pipeline.curate("rare", feature_space=feature_space, n=n, hours=hours,
                           test_hours=test_hours, group_cap=group_cap, **extras)


def typical(pipeline, *, feature_space, knn_k=None, rare_sample=None, feature_normalize=None,
           n=None, hours=None, test_hours=None, group_cap=None):
    """Density kNN typicality: dense (prototype) regions of the feature space are admitted
    first."""
    extras = _method_params(knn_k=knn_k, rare_sample=rare_sample,
                            feature_normalize=feature_normalize)
    return pipeline.curate("typical", feature_space=feature_space, n=n, hours=hours,
                           test_hours=test_hours, group_cap=group_cap, **extras)


def anomaly(pipeline, *, contamination=None, normalize=None):
    """Drop multivariate outliers with an Isolation Forest, before selection runs."""
    kwargs = {}
    if contamination is not None:
        kwargs["contamination"] = contamination
    if normalize is not None:
        kwargs["normalize"] = normalize
    return pipeline.filter("anomaly", **kwargs)


def dedup(pipeline, *, feature_space, threshold=None, n_clusters=None, normalize=None):
    """Feature-space SemDeDup: within each cluster, keep one representative per
    near-duplicate group, before selection runs."""
    kwargs = {"feature_space": feature_space}
    if threshold is not None:
        kwargs["threshold"] = threshold
    if n_clusters is not None:
        kwargs["n_clusters"] = n_clusters
    if normalize is not None:
        kwargs["normalize"] = normalize
    return pipeline.filter("dedup", **kwargs)


def action_residual(pipeline, *, obs, action, k=None, out="action_residual"):
    """Action-conditioned k-nearest-neighbor predictive residual: how much better a
    window's own action predicts its own next observation than its k nearest neighbors'
    actions do (whole-pool, buffer-using)."""
    kwargs = {"obs": obs, "action": action}
    if k is not None:
        kwargs["k"] = k
    return pipeline.score("action_residual", out=out, **kwargs)


#: name -> (function, parameters schema, one-line description). Registered under the real
#: "operator" kind by :func:`register`, so every entry becomes a named Pipeline method.
_BUILTIN_OPERATORS: dict[str, tuple[Callable[..., Any], dict, str]] = {
    "top-k": (top_k, WEIGHTS_PARAMETERS,
             "Rank by a weighted score and take the highest priorities"),
    "weighted": (weighted, WEIGHTED_PARAMETERS,
                "Sample without replacement, proportional to a weighted score"),
    "uniform-bins": (uniform_bins, UNIFORM_BINS_PARAMETERS,
                    "Round-robin declared bins for coverage-first selection"),
    "target-distribution": (target_distribution, TARGET_DISTRIBUTION_PARAMETERS,
                            "Match the pool to an explicit target distribution"),
    "cluster-balanced": (cluster_balanced, FEATURE_SPACE_CLUSTER_PARAMETERS,
                         "Joint k-means-cluster round-robin coverage over the feature space"),
    "k-center": (k_center, FEATURE_SPACE_CLUSTER_PARAMETERS,
                "Farthest-point traversal of the cluster centroids"),
    "facility-location": (facility_location, FEATURE_SPACE_CLUSTER_PARAMETERS,
                          "Submodular facility-location cluster ordering"),
    "dsir": (dsir, DSIR_PARAMETERS,
            "Multivariate importance resampling over a quantized joint feature space"),
    "rare": (rare, RARITY_PARAMETERS, "Density-inverse kNN atypicality"),
    "typical": (typical, RARITY_PARAMETERS, "Density kNN typicality"),
    "anomaly": (anomaly, ANOMALY_FILTER_PARAMETERS,
               "Drop multivariate outliers with an Isolation Forest"),
    "dedup": (dedup, DEDUP_FILTER_PARAMETERS, "Feature-space SemDeDup near-duplicate removal"),
    "action-residual": (action_residual, ACTION_RESIDUAL_PARAMETERS,
                        "Action-conditioned k-nearest-neighbor predictive residual"),
}


def register() -> None:
    """Idempotently register every built-in Method/Filter/Scorer as a named operator."""
    from ._registration import needs_registration
    from .registry import default_registry, register_operator

    common = {"version": "1", "package": "kuration-sdk", "stability": "stable"}
    for name, (fn, parameters, description) in _BUILTIN_OPERATORS.items():
        if needs_registration(default_registry, "operator", name, fn):
            register_operator(name, fn, description=description, parameters=parameters,
                              **common)


# Registered eagerly on import, matching methods.py/scorers.py's own built-in registration --
# unlike the opt-in gameplay/selection domains, core's own built-ins are always available.
register()


__all__ = [
    "ACTION_RESIDUAL_PARAMETERS",
    "ANOMALY_FILTER_PARAMETERS",
    "DEDUP_FILTER_PARAMETERS",
    "DSIR_PARAMETERS",
    "FEATURE_SPACE_CLUSTER_PARAMETERS",
    "RARITY_PARAMETERS",
    "TARGET_DISTRIBUTION_PARAMETERS",
    "UNIFORM_BINS_PARAMETERS",
    "WEIGHTS_PARAMETERS",
    "WEIGHTED_PARAMETERS",
    "action_residual",
    "anomaly",
    "cluster_balanced",
    "dedup",
    "dsir",
    "facility_location",
    "k_center",
    "rare",
    "register",
    "target_distribution",
    "top_k",
    "typical",
    "uniform_bins",
    "weighted",
]
