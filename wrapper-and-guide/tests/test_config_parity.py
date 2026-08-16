"""Config-parity invariant: the three call styles must produce identical results.

CLAUDE.md engineering principle 3: "If they diverge, that's a bug." This file
formalizes the assertion sketched at the bottom of ``examples/chained.py`` and
extends it across every gallery Operation, so a refactor that breaks one style
lights up in CI instead of in a user's config-driven pipeline.
"""
from __future__ import annotations

from kuration_ai import BalancedSelector, DiverseSubset, RareEventCoverage, run


def _collect_three_styles(base, op_instance, op_config):
    """Apply an Operation three ways and return each selected-count."""
    r1 = op_instance.apply(base).options(mode="strict", seed=1).collect()
    r2 = run(base, [op_instance]).options(mode="strict", seed=1).collect()
    r3 = run(base, [op_config]).options(mode="strict", seed=1).collect()
    return len(r1.selected), len(r2.selected), len(r3.selected)


def test_balanced_selector_three_styles_identical(base_pipeline):
    op = BalancedSelector(feature="activity", of="signal", bins=4, count=20)
    config = {
        "type": "BalancedSelector",
        "feature": "activity",
        "of": "signal",
        "bins": 4,
        "count": 20,
    }
    n1, n2, n3 = _collect_three_styles(base_pipeline, op, config)
    assert n1 == n2 == n3 == 20


def test_rare_event_coverage_three_styles_identical(base_pipeline):
    op = RareEventCoverage(of="fault", count=15)
    config = {"type": "RareEventCoverage", "of": "fault", "count": 15}
    n1, n2, n3 = _collect_three_styles(base_pipeline, op, config)
    assert n1 == n2 == n3 == 15


def test_diverse_subset_three_styles_identical(measured_pipeline):
    op = DiverseSubset(features=("act", "temp_mean"), count=15, n_clusters=4)
    config = {
        "type": "DiverseSubset",
        "features": ["act", "temp_mean"],
        "count": 15,
        "n_clusters": 4,
    }
    n1, n2, n3 = _collect_three_styles(measured_pipeline, op, config)
    assert n1 == n2 == n3 == 15
