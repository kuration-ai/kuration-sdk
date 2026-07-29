"""SegmentScorer seam: a whole-pool/learned/action-aware score → feature → selection."""
from __future__ import annotations

import numpy as np
import pytest

from kuration import (
    SchemaSpec,
    Segmenter,
    Tick,
    apply_scorers,
    curate,
    load_strategy,
    register_scorer,
)

SCHEMA = SchemaSpec.from_dict({
    "name": "sc", "entity_axis": "e", "sample_hz": 1.0,
    "fields": {"obs": {"kind": "vector", "dims": ["a", "b"]},
               "action": {"kind": "scalar"}},
})


class MemAdapter:
    def __init__(self, schema, data):
        self.schema, self.data = schema, data
    def ticks(self, src):
        return self.data[src]


# A trivial whole-pool scorer: score = rank of the source's mean action (needs no buffers).
# Registered at import; guard against re-registration across test runs.
def _register_demo():
    if "rank_action" in __import__("kuration.core.scorers", fromlist=["_SCORERS"])._SCORERS:
        return

    class RankAction:
        needs_buffers = False
        def score(self, segments, ctx):
            v = np.array([s.features.get("am", 0.0) for s in segments])
            order = v.argsort().argsort()          # 0..n-1 rank
            return order.astype(float)
    register_scorer("rank_action", RankAction)


_register_demo()


def _data(levels, n=8):
    return {f"s{L}": [Tick(time=i, entities={0: {"obs": (float(i), float(L)),
                                                "action": float(L) + 0.1 * i}})
                      for i in range(n)] for L in levels}


# ---- the seam: scorer output becomes a selectable feature -------------------
def test_scorer_output_drives_selection():
    data = _data(range(1, 6))
    strat = load_strategy({
        "name": "byscore", "define": {"am": {"source": "e.action", "reduce": "mean"}},
        "scorers": [{"name": "rank_action", "as": "score"}],
        "selection": {"method": "top-k", "weights": {"score": 1.0}}}, SCHEMA)
    res = curate(schema=SCHEMA, adapter=MemAdapter(SCHEMA, data), sources=list(data),
                 strategy=strat, segmenter=Segmenter(4, 4), target_size=3, seed=0, on_error="raise")
    # non-empty pool (guards against vacuous pass if scoring silently skipped everything)
    assert res.all_segments and res.selected
    # every segment got the scorer feature, and selection ranked by it
    assert all("score" in s.features for s in res.all_segments)
    # the highest-scored segments are exactly the selected ones (scorer feature drove top-k)
    top = sorted(res.all_segments, key=lambda s: -s.features["score"])[:3]
    assert {id(s) for s in res.selected} == {id(s) for s in top}


def test_scorer_output_referenceable_at_load():
    # 'score' is produced by a scorer, so referencing it in weights must NOT be "undefined"
    load_strategy({"name": "ok", "scorers": [{"name": "rank_action", "as": "score"}],
                   "selection": {"method": "top-k", "weights": {"score": 1.0}}}, SCHEMA)


def test_unregistered_scorer_raises():
    with pytest.raises(ValueError):
        load_strategy({"name": "bad", "scorers": ["nope"],
                       "selection": {"method": "top-k", "weights": {"x": 1.0}},
                       "define": {"x": {"source": "e.action", "reduce": "mean"}}}, SCHEMA)


# ---- reference action-conditioned scorer (needs buffers + sklearn) ----------
def test_action_residual_scorer_runs():
    pytest.importorskip("sklearn")
    data = _data(range(1, 8))
    strat = load_strategy({
        "name": "ar", "define": {"am": {"source": "e.action", "reduce": "mean"}},
        "scorers": [{"name": "action_residual", "as": "resid", "obs": "obs", "action": "action", "k": 3}],
        "selection": {"method": "top-k", "weights": {"resid": 1.0}}}, SCHEMA)
    res = curate(schema=SCHEMA, adapter=MemAdapter(SCHEMA, data), sources=list(data),
                 strategy=strat, segmenter=Segmenter(4, 4), target_size=3, seed=0, on_error="raise")
    assert res.all_segments   # non-empty (the buffer-using scorer actually ran)
    assert all("resid" in s.features for s in res.all_segments)
    assert all(np.isfinite(s.features["resid"]) for s in res.all_segments)


def test_apply_scorers_validates_length():
    pytest.importorskip("sklearn")

    class Wrong:
        needs_buffers = False
        def score(self, segments, ctx):
            return np.zeros(len(segments) + 1)   # wrong length
    register_scorer("wrong_len", Wrong)
    data = _data([1, 2])
    # build a couple segments to score
    from kuration import Segment
    segs = [Segment("s", 0, 4, (0, 3)), Segment("s", 4, 8, (4, 7))]
    with pytest.raises(ValueError):
        apply_scorers([("wrong_len", "w", {})], segs, schema=SCHEMA, adapter=MemAdapter(SCHEMA, data))


@pytest.mark.parametrize("name,value", [
    ("wrong_scalar", np.asarray(1.0)),
    ("wrong_matrix", np.zeros((2, 1))),
    ("wrong_nan", np.asarray([0.0, np.nan])),
])
def test_apply_scorers_requires_finite_vector(name, value):
    class WrongShape:
        needs_buffers = False
        def score(self, segments, ctx):
            return value

    register_scorer(name, WrongShape)
    from kuration import Segment
    segs = [Segment("s", 0, 4, (0, 3)), Segment("s", 4, 8, (4, 7))]
    with pytest.raises(ValueError, match="finite 1-D vector"):
        apply_scorers([(name, "w", {})], segs, schema=SCHEMA,
                      adapter=MemAdapter(SCHEMA, _data([1, 2])))
