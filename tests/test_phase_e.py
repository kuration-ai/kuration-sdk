"""Phase E: the filter selection method + the multi-arm comparison harness."""
from __future__ import annotations

import pytest

from kuration import (
    Arm,
    SchemaSpec,
    Segmenter,
    Tick,
    curate,
    load_strategy,
    run_experiment,
)
from kuration.core.experiment import union_define

SCHEMA = SchemaSpec.from_dict({
    "name": "e", "entity_axis": "e", "sample_hz": 1.0, "fields": {"act": {"kind": "scalar"}},
})
DEFINE = {"am": {"source": "e.act", "reduce": "mean"}}


class MemAdapter:
    def __init__(self, schema, data):
        self.schema, self.data = schema, data
    def ticks(self, src):
        return self.data[src]


def _data():
    return {f"s{L}": [Tick(time=i, entities={0: {"act": float(L)}}) for i in range(12)]
            for L in range(1, 7)}


def _run(strategy, **kw):
    data = _data()
    return curate(schema=SCHEMA, adapter=MemAdapter(SCHEMA, data), sources=list(data),
                  strategy=strategy, segmenter=Segmenter(4, 4), target_size=100, **kw)


# ---- filter method ----------------------------------------------------------
def test_filter_keeps_only_matching_clips():
    strat = load_strategy({"name": "hi", "define": DEFINE,
                           "selection": {"method": "filter",
                                         "predicates": [{"metric": "am", "op": ">=", "value": 4}]}},
                          SCHEMA)
    res = _run(strat)
    assert res.selected and all(s.features["am"] >= 4 for s in res.selected)   # sources 4,5,6
    assert not any(s.features["am"] < 4 for s in res.selected)


def test_filter_compound_predicates():
    strat = load_strategy({"name": "band", "define": DEFINE,
                           "selection": {"method": "filter", "predicates": [
                               {"metric": "am", "op": ">=", "value": 3},
                               {"metric": "am", "op": "<=", "value": 5}]}}, SCHEMA)
    res = _run(strat)
    assert all(3 <= s.features["am"] <= 5 for s in res.selected)


@pytest.mark.parametrize("sel,msg", [
    ({"method": "filter"}, "requires selection.predicates"),
    ({"method": "filter", "predicates": [{"metric": "am", "op": "~", "value": 1}]}, "op must be"),
    ({"method": "filter", "predicates": [{"metric": "am", "value": 1}]}, "metric/op/value"),
])
def test_filter_validation(sel, msg):
    with pytest.raises(ValueError) as e:
        load_strategy({"name": "x", "define": DEFINE, "selection": sel}, SCHEMA)
    assert msg in str(e.value)


# ---- arm comparison ---------------------------------------------------------
def test_union_define_conflict_raises():
    a = load_strategy({"name": "a", "define": {"am": {"source": "e.act", "reduce": "mean"}},
                       "selection": {"method": "top-k", "weights": {"am": 1.0}}}, SCHEMA)
    b = load_strategy({"name": "b", "define": {"am": {"source": "e.act", "reduce": "max"}},
                       "selection": {"method": "top-k", "weights": {"am": 1.0}}}, SCHEMA)
    with pytest.raises(ValueError) as e:
        union_define([Arm("a", a), Arm("b", b)])
    assert "conflicting" in str(e.value)


def test_run_experiment_compares_arms():
    data = _data()
    base = load_strategy({"name": "base", "define": DEFINE,
                          "binning": {"axes": [{"metric": "am", "bins": 6}]},
                          "selection": {"method": "uniform-bins"}}, SCHEMA)
    hi = load_strategy({"name": "hi", "define": DEFINE,
                        "selection": {"method": "top-k", "weights": {"am": 1.0}}}, SCHEMA)
    lo = load_strategy({"name": "lo", "define": DEFINE,
                        "selection": {"method": "top-k", "weights": {"am": -1.0}}}, SCHEMA)
    out = run_experiment(schema=SCHEMA, adapter=MemAdapter(SCHEMA, data), sources=list(data),
                         arms=[Arm("base", base), Arm("hi", hi), Arm("lo", lo)],
                         segmenter=Segmenter(4, 4), target_size=6, baseline="base", seed=0)
    assert out["baseline"] == "base" and set(out["arms"]) == {"base", "hi", "lo"}
    # hi selects higher-activity windows than lo
    assert out["arms"]["hi"]["metrics"]["am"]["mean"] > out["arms"]["lo"]["metrics"]["am"]["mean"]
    # JS-vs-baseline reported for arms, not for the baseline itself
    assert "js_vs_baseline" in out["arms"]["hi"]["metrics"]["am"]
    assert "js_vs_baseline" not in out["arms"]["base"]["metrics"]["am"]
    assert out["arms"]["hi"]["metrics"]["am"]["js_vs_baseline"] >= 0.0
