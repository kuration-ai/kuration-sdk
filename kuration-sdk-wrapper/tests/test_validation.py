"""Every ``ValueError`` branch in each Operation's ``__post_init__``.

Two properties are checked for each bad input: that a ``ValueError`` is raised
at construction time (fail-fast, before the pipeline is touched) and that its
message names the specific field that was invalid (so the error is actionable
for a beginner reading the traceback).
"""
from __future__ import annotations

import pytest

from kuration_sdk_wrapper import BalancedSelector, DiverseSubset, RareEventCoverage


# ---------- BalancedSelector ------------------------------------------------

@pytest.mark.parametrize(
    "kwargs, field",
    [
        (dict(feature="", of="signal", bins=4, count=20), "feature"),
        (dict(feature="activity", of="", bins=4, count=20), "of"),
        (dict(feature="activity", of="signal", bins=1, count=20), "bins"),
        (dict(feature="activity", of="signal", bins=4, count=0), "count"),
    ],
)
def test_balanced_selector_rejects_bad_input(kwargs, field):
    with pytest.raises(ValueError, match=field):
        BalancedSelector(**kwargs)


# ---------- RareEventCoverage -----------------------------------------------

@pytest.mark.parametrize(
    "kwargs, field",
    [
        (dict(of="", count=10), "of"),
        (dict(of="fault", feature="", count=10), "feature"),
        (dict(of="fault", count=0), "count"),
        (dict(of="fault", count=10, weight=-1), "weight"),
        (dict(of="fault", count=10, weight=0), "weight"),
    ],
)
def test_rare_event_coverage_rejects_bad_input(kwargs, field):
    with pytest.raises(ValueError, match=field):
        RareEventCoverage(**kwargs)


# ---------- DiverseSubset ---------------------------------------------------

@pytest.mark.parametrize(
    "kwargs, field",
    [
        (dict(features=(), count=10, n_clusters=4), "features"),
        (dict(features=("",), count=10, n_clusters=4), "features"),
        (dict(features=("a", "b"), count=0, n_clusters=4), "count"),
        (dict(features=("a", "b"), count=10, n_clusters=1), "n_clusters"),
        (dict(features=("a", "b"), count=10, n_clusters=4, dedup_threshold=1.5), "dedup_threshold"),
        (dict(features=("a", "b"), count=10, n_clusters=4, dedup_threshold=-1.5), "dedup_threshold"),
    ],
)
def test_diverse_subset_rejects_bad_input(kwargs, field):
    with pytest.raises(ValueError, match=field):
        DiverseSubset(**kwargs)
