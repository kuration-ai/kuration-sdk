"""Feature-space selection methods + pre-pass filters (scikit-learn extra)."""
from __future__ import annotations

import numpy as np
import pytest

from kuration import (
    Segment,
    SchemaSpec,
    Segmenter,
    Tick,
    apply_filters,
    curate,
    load_strategy,
    register_metric,
    segment_priorities,
)
from kuration.core.dsl.registry import METRICS

pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("sklearn") is None, reason="needs [methods] extra")

# Register no-op metrics so strategies can reference f1/f2 as a feature space; segment
# features are set by hand in these tests, so the bodies are never called.
for _nm in ("f1", "f2"):
    if _nm not in METRICS:
        register_metric(_nm)(lambda w: 0.0)

SCHEMA = SchemaSpec.from_dict({"name": "m", "entity_axis": "e", "sample_hz": 1.0,
                               "fields": {"act": {"kind": "scalar"}}})


def _blobs(seed=0):
    """30 segments in 3 well-separated blobs in (f1, f2)."""
    rng = np.random.default_rng(seed)
    centers = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
    segs = []
    for cx, cy in centers:
        for _ in range(10):
            segs.append(Segment("s", 0, 1, (0, 0),
                                features={"f1": cx + rng.normal() * 0.3,
                                          "f2": cy + rng.normal() * 0.3}))
    return segs


def _strat(method, **sel):
    return load_strategy({"name": method, "feature_space": ["f1", "f2"],
                          "selection": {"method": method, **sel}}, SCHEMA)


def _min_pairwise(points):
    d = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return d.min()


@pytest.mark.parametrize("method", ["cluster-balanced", "k-center", "facility-location"])
def test_coverage_methods_span_clusters(method):
    segs = _blobs()
    prio = segment_priorities(segs, _strat(method, clusters=3), seed=0)
    top3 = np.argsort(-prio, kind="stable")[:3]
    pts = np.array([[segs[i].features["f1"], segs[i].features["f2"]] for i in top3])
    assert _min_pairwise(pts) > 5     # one pick per blob (~10 apart), not 3 from one blob


def test_dsir_flat_runs():
    segs = _blobs()
    prio = segment_priorities(segs, _strat("dsir"), seed=0)
    assert prio.shape == (30,) and np.all(np.isfinite(prio))


def test_dsir_anchor_favors_anchor_region():
    segs = _blobs()
    strat = _strat("dsir", target={"anchor_metric": "f1", "anchor_quantile": 0.8})
    prio = segment_priorities(segs, strat, seed=0)
    top = np.argsort(-prio, kind="stable")[:10]
    mean_f1_top = np.mean([segs[i].features["f1"] for i in top])
    assert mean_f1_top > np.mean([s.features["f1"] for s in segs])   # high-f1 region favored


def test_rare_surfaces_outlier_typical_does_not():
    segs = _blobs() + [Segment("s", 0, 1, (0, 0), features={"f1": 100.0, "f2": 100.0})]
    rare = segment_priorities(segs, _strat("rare", knn_k=3), seed=0)
    typ = segment_priorities(segs, _strat("typical", knn_k=3), seed=0)
    assert int(np.argmax(rare)) == len(segs) - 1     # the outlier is rarest
    assert int(np.argmax(typ)) != len(segs) - 1      # ... and not the most typical


# ---- filters ----------------------------------------------------------------
def test_anomaly_filter_removes_outlier():
    segs = _blobs() + [Segment("s", 0, 1, (0, 0), features={"f1": 100.0, "f2": 100.0})]
    strat = load_strategy({"name": "a", "feature_space": ["f1", "f2"],
                           "filters": [{"type": "anomaly", "contamination": 0.1}],
                           "selection": {"method": "top-k", "weights": {"f1": 1.0}}}, SCHEMA)
    kept = apply_filters(segs, strat, seed=0)
    assert len(kept) < len(segs)
    assert not any(s.features["f1"] == 100.0 for s in kept)


def test_dedup_filter_collapses_duplicates():
    dup = [Segment("s", 0, 1, (0, 0), features={"f1": 1.0, "f2": 1.0}) for _ in range(8)]
    varied = _blobs()
    strat = load_strategy({"name": "d", "feature_space": ["f1", "f2"],
                           "filters": [{"type": "dedup", "threshold": 0.999, "n_clusters": 4}],
                           "selection": {"method": "top-k", "weights": {"f1": 1.0}}}, SCHEMA)
    kept = apply_filters(dup + varied, strat, seed=0)
    assert sum(1 for s in kept if s.features == {"f1": 1.0, "f2": 1.0}) <= 2   # near-dupes collapsed


# ---- end to end -------------------------------------------------------------
class MemAdapter:
    def __init__(self, schema, data):
        self.schema, self.data = schema, data
    def ticks(self, src):
        return self.data[src]


def test_curate_with_kcenter_method():
    data = {f"s{L}": [Tick(time=i, entities={0: {"act": float(L)}}) for i in range(8)]
            for L in range(1, 7)}
    strat = load_strategy({"name": "kc", "define": {"am": {"source": "e.act", "reduce": "mean"}},
                           "feature_space": ["am"],
                           "selection": {"method": "k-center", "clusters": 3}}, SCHEMA)
    res = curate(schema=SCHEMA, adapter=MemAdapter(SCHEMA, data), sources=list(data),
                 strategy=strat, segmenter=Segmenter(4, 4), target_size=3, seed=0)
    assert len(res.selected) == 3
    assert len({s.features["am"] for s in res.selected}) >= 2   # spreads across activity levels


def test_referenced_metrics_includes_feature_space():
    strat = _strat("k-center", clusters=3)
    assert {"f1", "f2"} <= strat.referenced_metrics()
