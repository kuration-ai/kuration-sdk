"""Operator wrappers around the selection recipes.

Each function has the ordinary compositional-operator shape (``pipeline, **params ->
pipeline``) and simply builds the corresponding dataclass in
:mod:`kuration.selection.operations` and calls its ``.apply()``, so the method form
(``pipeline.balanced_selector(...)``), the direct dataclass form, and the config-dict
form (see :mod:`kuration.selection.plan`) always produce identical results.
"""
from __future__ import annotations

from .operations import BalancedSelector, DiverseSubset, RareEventCoverage

BALANCED_SELECTOR_PARAMETERS = {
    "type": "object",
    "properties": {
        "feature": {"type": "string"},
        "of": {"type": "string"},
        "kind": {"type": "string"},
        "bins": {"type": "integer"},
        "count": {"type": "integer"},
    },
    "required": ["feature", "of"],
    "additionalProperties": False,
}

RARE_EVENT_COVERAGE_PARAMETERS = {
    "type": "object",
    "properties": {
        "of": {"type": "string"},
        "feature": {"type": "string"},
        "count": {"type": "integer"},
        "weight": {"type": "number"},
    },
    "required": ["of"],
    "additionalProperties": False,
}

DIVERSE_SUBSET_PARAMETERS = {
    "type": "object",
    "properties": {
        "features": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "count": {"type": "integer"},
        "n_clusters": {"type": "integer"},
        "dedup_threshold": {"type": ["number", "null"]},
    },
    "required": ["features"],
    "additionalProperties": False,
}


def balanced_selector(pipeline, *, feature, of, kind="std", bins=8, count=100):
    """Compose the balanced-selector recipe into a pipeline."""
    return BalancedSelector(feature=feature, of=of, kind=kind, bins=bins, count=count).apply(
        pipeline
    )


def rare_event_coverage(pipeline, *, of, feature="event_rate", count=100, weight=1.0):
    """Compose the rare-event-coverage recipe into a pipeline."""
    return RareEventCoverage(of=of, feature=feature, count=count, weight=weight).apply(pipeline)


def diverse_subset(pipeline, *, features, count=100, n_clusters=32, dedup_threshold=None):
    """Compose the diverse-subset recipe into a pipeline."""
    return DiverseSubset(
        features=tuple(features),
        count=count,
        n_clusters=n_clusters,
        dedup_threshold=dedup_threshold,
    ).apply(pipeline)


__all__ = [
    "BALANCED_SELECTOR_PARAMETERS",
    "DIVERSE_SUBSET_PARAMETERS",
    "RARE_EVENT_COVERAGE_PARAMETERS",
    "balanced_selector",
    "diverse_subset",
    "rare_event_coverage",
]
