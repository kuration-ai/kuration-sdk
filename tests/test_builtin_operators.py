"""Every built-in Method/Filter/Scorer is also a named Pipeline operator, and the two call
styles -- `pipeline.<name>(...)` and the generic string-keyed form (`.sample(method=...)`,
`.filter(...)`, `.score(...)`) -- must select identical segments. No separate
"Method"/"Filter"/"Scorer" vocabulary should be needed to reach any of these by name.
"""
from __future__ import annotations

import pytest

from kuration import default_registry, input


def _rows(n=300):
    return [{"t": float(i), "x": float(i % 5), "y": float((i * 7) % 11),
             "fault": float(i % 13 == 0)} for i in range(n)]


@pytest.fixture
def pool():
    return (
        input({f"s{k}": _rows() for k in range(3)}, time="t")
        .window(8, stride=4)
        .measure("x_mean", "mean", of="x")
        .measure("y_mean", "mean", of="y")
    )


@pytest.fixture
def binned_pool(pool):
    return pool.bin("x_mean", bins=3)


def _ids(result):
    return [(s.source_id, s.start_idx) for s in result.selected]


def test_every_builtin_operator_is_registered():
    expected = {
        "top-k", "weighted", "uniform-bins", "target-distribution", "cluster-balanced",
        "k-center", "facility-location", "dsir", "rare", "typical", "anomaly", "dedup",
        "action-residual",
    }
    registered = set(default_registry.names("operator")) & expected
    assert registered == expected


@pytest.mark.parametrize("weights_method", ["top_k", "weighted"])
def test_weighted_methods_match_string_keyed_form(pool, weights_method):
    via_pipeline = getattr(pool, weights_method)(weights={"x_mean": 1.0}, n=8)
    string_keyed = pool.sample(weights_method.replace("_", "-"), weights={"x_mean": 1.0}, n=8)
    assert _ids(via_pipeline.options(seed=3).collect()) == _ids(string_keyed.options(seed=3).collect())


@pytest.mark.parametrize("weights_method", ["top_k", "weighted"])
def test_weighted_methods_accept_normalize_none(pool, weights_method):
    """normalize="none" (raw, unnormalized scale) was previously only accepted by filters
    (anomaly/dedup) and the classic curate()/write_dataset() functions -- not by .sample()/
    the fluent top_k/weighted operators -- even though the underlying _normalize() already
    implements it. Both call styles should accept it identically."""
    via_pipeline = getattr(pool, weights_method)(weights={"x_mean": 1.0}, normalize="none", n=8)
    string_keyed = pool.sample(weights_method.replace("_", "-"), weights={"x_mean": 1.0},
                               normalize="none", n=8)
    assert _ids(via_pipeline.options(seed=3).collect()) == _ids(string_keyed.options(seed=3).collect())


def test_weighted_floor_raises_the_lowest_scorers_selection_odds():
    """``floor`` shifts the Gumbel-top-k sampler's non-negativity offset (default ``1e-9``),
    which otherwise gives the single lowest-scoring segment in the pool a near-zero chance of
    selection by construction, regardless of how close its real score was to the rest."""
    # Two low-scoring segments (score 0) and eight high-scoring segments (score 10).
    sources = {name: [{"t": float(i), "x": 0.0} for i in range(16)] for name in ("low0", "low1")}
    sources.update({f"high{k}": [{"t": float(i), "x": 10.0} for i in range(16)]
                    for k in range(8)})
    pool = input(sources, time="t").window(16).measure("x_mean", "mean", of="x")

    def _low_selected(seeds, **kwargs):
        count = 0
        for seed in seeds:
            result = pool.weighted(weights={"x_mean": 1.0}, n=1, **kwargs).options(
                seed=seed).collect()
            if result.selected[0].source_id.startswith("low"):
                count += 1
        return count

    seeds = range(300)
    default_count = _low_selected(seeds)
    raised_count = _low_selected(seeds, floor=5.0)
    assert raised_count > default_count + 2


def test_uniform_bins_matches_string_keyed_form(binned_pool):
    via_pipeline = binned_pool.uniform_bins(n=8)
    string_keyed = binned_pool.sample("uniform-bins", n=8)
    assert _ids(via_pipeline.options(seed=3).collect()) == _ids(string_keyed.options(seed=3).collect())


def test_target_distribution_matches_string_keyed_form(binned_pool):
    via_pipeline = binned_pool.target_distribution(target={"metric": "x_mean"}, n=8)
    string_keyed = binned_pool.sample("target-distribution", target={"metric": "x_mean"}, n=8)
    assert _ids(via_pipeline.options(seed=3).collect()) == _ids(string_keyed.options(seed=3).collect())


def test_target_distribution_flat_default_needs_no_target(binned_pool):
    result = binned_pool.target_distribution(n=5).options(seed=1).collect()
    assert len(result.selected) == 5


@pytest.mark.parametrize("cluster_method", ["cluster_balanced", "k_center", "facility_location"])
def test_feature_space_cluster_methods_match_string_keyed_form(pool, cluster_method):
    via_pipeline = getattr(pool, cluster_method)(feature_space=["x_mean", "y_mean"], clusters=4, n=8)
    string_keyed = pool.sample(cluster_method.replace("_", "-"),
                               feature_space=["x_mean", "y_mean"], clusters=4, n=8)
    assert _ids(via_pipeline.options(seed=3).collect()) == _ids(string_keyed.options(seed=3).collect())


def test_dsir_matches_string_keyed_form(pool):
    via_pipeline = pool.dsir(feature_space=["x_mean"], dsir_bins=4, n=8)
    string_keyed = pool.sample("dsir", feature_space=["x_mean"], dsir_bins=4, n=8)
    assert _ids(via_pipeline.options(seed=3).collect()) == _ids(string_keyed.options(seed=3).collect())


@pytest.mark.parametrize("rarity_method", ["rare", "typical"])
def test_rarity_methods_match_string_keyed_form(pool, rarity_method):
    via_pipeline = getattr(pool, rarity_method)(feature_space=["x_mean"], knn_k=5, n=8)
    string_keyed = pool.sample(rarity_method, feature_space=["x_mean"], knn_k=5, n=8)
    assert _ids(via_pipeline.options(seed=3).collect()) == _ids(string_keyed.options(seed=3).collect())


def test_anomaly_operator_matches_string_keyed_filter(binned_pool):
    via_pipeline = binned_pool.anomaly(contamination=0.1).uniform_bins(n=5)
    string_keyed = binned_pool.filter("anomaly", contamination=0.1).sample("uniform-bins", n=5)
    assert _ids(via_pipeline.options(seed=1).collect()) == _ids(string_keyed.options(seed=1).collect())


def test_dedup_operator_matches_string_keyed_filter(binned_pool):
    via_pipeline = binned_pool.dedup(feature_space=["x_mean"], threshold=0.9).uniform_bins(n=5)
    string_keyed = (binned_pool.filter("dedup", feature_space=["x_mean"], threshold=0.9)
                    .sample("uniform-bins", n=5))
    assert _ids(via_pipeline.options(seed=1).collect()) == _ids(string_keyed.options(seed=1).collect())


def test_action_residual_operator_matches_string_keyed_scorer(pool):
    via_pipeline = pool.action_residual(obs="x", action="y", out="resid")
    string_keyed = pool.score("action_residual", out="resid", obs="x", action="y")
    via_pipeline_result = via_pipeline.bin("resid", bins=2).uniform_bins(n=5).options(seed=1).collect()
    string_result = string_keyed.bin("resid", bins=2).sample("uniform-bins", n=5).options(
        seed=1).collect()
    assert _ids(via_pipeline_result) == _ids(string_result)


def test_action_residual_default_output_name_is_the_operator_name(pool):
    result = pool.action_residual(obs="x", action="y").bin("action_residual", bins=2).uniform_bins(
        n=5).options(seed=1).collect()
    assert "action_residual" in result.selected[0].features
