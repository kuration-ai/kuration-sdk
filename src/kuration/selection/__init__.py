"""Built-in dataset-selection operator domain.

Importing this module registers ``balanced_selector``, ``rare_event_coverage``, and
``diverse_subset`` as methods on the ordinary ``kuration.Pipeline``. Each is
also available as a standalone, configurable dataclass (``BalancedSelector``,
``RareEventCoverage``, ``DiverseSubset``) for direct use or config-driven pipelines
via :mod:`kuration.selection.plan`.

``DiverseSubset`` requires ``kuration-sdk[methods]`` (scikit-learn) at execution time.
"""
from __future__ import annotations

from . import operations
from .operations import BalancedSelector, DiverseSubset, RareEventCoverage
from .operators import (
    BALANCED_SELECTOR_PARAMETERS,
    DIVERSE_SUBSET_PARAMETERS,
    RARE_EVENT_COVERAGE_PARAMETERS,
    balanced_selector,
    diverse_subset,
    rare_event_coverage,
)
from .plan import Operation, describe, plan, register_operation_type, run

BALANCED_SELECTOR_NAME = "kuration/balanced-selector"
RARE_EVENT_COVERAGE_NAME = "kuration/rare-event-coverage"
DIVERSE_SUBSET_NAME = "kuration/diverse-subset"


def register() -> None:
    """Register every built-in selection operator, idempotently."""
    from kuration import default_registry, register_operator
    from ..core._registration import needs_registration

    common = {
        "version": "1",
        "package": "kuration-sdk",
        "stability": "experimental",
        "install_hint": "pip install kuration-sdk",
        "documentation": "https://aintropy-solutions.github.io/kuration-sdk/operators/",
    }
    if needs_registration(default_registry, "operator", BALANCED_SELECTOR_NAME,
                           balanced_selector):
        register_operator(
            BALANCED_SELECTOR_NAME,
            balanced_selector,
            description="Sample equally across quantile bins of a measured feature",
            parameters=BALANCED_SELECTOR_PARAMETERS,
            **common,
        )
    if needs_registration(default_registry, "operator", RARE_EVENT_COVERAGE_NAME,
                           rare_event_coverage):
        register_operator(
            RARE_EVENT_COVERAGE_NAME,
            rare_event_coverage,
            description="Weighted-sample toward windows containing a rare event",
            parameters=RARE_EVENT_COVERAGE_PARAMETERS,
            **common,
        )
    if needs_registration(default_registry, "operator", DIVERSE_SUBSET_NAME, diverse_subset):
        register_operator(
            DIVERSE_SUBSET_NAME,
            diverse_subset,
            description="Cluster-balanced selection over pre-measured features",
            parameters=DIVERSE_SUBSET_PARAMETERS,
            extra="methods",
            **{**common, "install_hint": 'pip install "kuration-sdk[methods]"'},
        )


# Direct selection imports are immediately usable without a separate discovery call.
register()

__all__ = [
    "BalancedSelector",
    "DiverseSubset",
    "Operation",
    "RareEventCoverage",
    "balanced_selector",
    "describe",
    "diverse_subset",
    "operations",
    "plan",
    "rare_event_coverage",
    "register",
    "register_operation_type",
    "run",
]
