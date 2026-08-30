"""Happy-path smoke tests for each gallery Operation.

One assertion per Operation: at ``seed=1`` on the shared fixture, the SDK
returns exactly the requested count of selected windows. If a refactor causes
any of these to drift, the responsible change should update the assertion
deliberately: silent drift is what this file exists to catch.
"""
from __future__ import annotations

from kuration_sdk_wrapper import BalancedSelector, DiverseSubset, RareEventCoverage


def test_balanced_selector_selects_expected_count(base_pipeline):
    op = BalancedSelector(feature="activity", of="signal", kind="std", bins=4, count=20)
    result = op.apply(base_pipeline).options(mode="strict", seed=1).collect()
    assert len(result.selected) == 20


def test_rare_event_coverage_selects_expected_count(base_pipeline):
    op = RareEventCoverage(of="fault", count=15)
    result = op.apply(base_pipeline).options(mode="strict", seed=1).collect()
    assert len(result.selected) == 15


def test_diverse_subset_selects_expected_count(measured_pipeline):
    op = DiverseSubset(features=("act", "temp_mean"), count=15, n_clusters=4)
    result = op.apply(measured_pipeline).options(mode="strict", seed=1).collect()
    assert len(result.selected) == 15


def test_diverse_subset_with_dedup_selects_expected_count(measured_pipeline):
    op = DiverseSubset(
        features=("act", "temp_mean"),
        count=15,
        n_clusters=4,
        dedup_threshold=0.98,
    )
    result = op.apply(measured_pipeline).options(mode="strict", seed=1).collect()
    assert len(result.selected) == 15


def test_describe_names_the_compiled_sdk_chain():
    """``.describe()`` should surface the SDK primitives the wrapper compiles to.

    This is the "no lock-in" guarantee from CLAUDE.md GTM tenet 4: a user who
    outgrows the kit reads ``.describe()`` and sees the raw SDK chain they
    could write by hand.
    """
    assert ".sample('equal'" in BalancedSelector(
        feature="activity", of="signal", bins=4, count=20
    ).describe()

    assert ".sample('weighted'" in RareEventCoverage(of="fault", count=10).describe()

    assert ".sample('cluster-balanced'" in DiverseSubset(
        features=("a", "b"), count=10, n_clusters=4
    ).describe()
