"""Phase C: binning + declarative strategy + set-based selection + splits + config."""
from __future__ import annotations

import json

import pytest

from kuration import (
    Segment,
    Segmenter,
    SchemaSpec,
    Tick,
    bin_segments,
    build_config,
    curate,
    load_strategy,
    select,
)
from kuration.core.select import assign_splits, enforce_group_cap, enforce_tag_coverage

SCHEMA = SchemaSpec.from_dict({
    "name": "c", "entity_axis": "e", "sample_hz": 1.0,
    "fields": {"act": {"kind": "scalar"}},
})


class MemAdapter:
    """Tiny in-memory adapter: {source_id -> [Tick]}."""
    def __init__(self, schema, data):
        self.schema = schema
        self.data = data
    def ticks(self, src):
        return self.data[src]


def _sources(levels, n=12):
    # source sL has constant activity L -> every 4-tick window has act-mean = L
    return {f"s{L}": [Tick(time=i, entities={0: {"act": float(L)}}) for i in range(n)]
            for L in levels}


def _strategy(spec_overrides):
    base = {"name": "t", "define": {"am": {"source": "e.act", "reduce": "mean"}}}
    base.update(spec_overrides)
    return load_strategy(base, SCHEMA)


# ---- binning ----------------------------------------------------------------
def test_quantile_binning_balances_occupancy():
    segs = [Segment("s", 0, 1, (0, 0), features={"am": float(v)}) for v in range(100)]
    bin_segments(segs, [("am", 4)])
    from collections import Counter
    counts = Counter(s.bin for s in segs)
    assert len(counts) == 4 and max(counts.values()) - min(counts.values()) <= 1


# ---- selection methods ------------------------------------------------------
def _curate(strategy, target_size, **kw):
    data = _sources(range(1, 7))
    return curate(schema=SCHEMA, adapter=MemAdapter(SCHEMA, data), sources=list(data),
                  strategy=strategy, segmenter=Segmenter(4, 4), target_size=target_size, **kw)


def test_curate_rejects_empty_sources():
    strategy = _strategy(
        {"selection": {"method": "top-k", "weights": {"am": 1.0}}})

    with pytest.raises(ValueError, match="at least one source"):
        curate(
            schema=SCHEMA,
            adapter=MemAdapter(SCHEMA, {}),
            sources=[],
            strategy=strategy,
            segmenter=Segmenter(4, 4),
            target_size=1,
        )


def test_score_sources_rejects_an_empty_generator():
    from kuration.core.scoring import score_sources

    with pytest.raises(ValueError, match="at least one source"):
        score_sources(
            schema=SCHEMA,
            adapter=MemAdapter(SCHEMA, {}),
            sources=iter(()),
            define={},
            registered=[],
            segmenter=Segmenter(4, 4),
        )


def test_top_k_picks_highest():
    strat = _strategy({"selection": {"method": "top-k", "weights": {"am": 1.0}}})
    res = _curate(strat, target_size=6)
    assert len(res.selected) == 6
    assert all(s.features["am"] >= 5 for s in res.selected)   # sources 5 & 6, 3 windows each


def test_uniform_bins_covers_range():
    strat = _strategy({"binning": {"axes": [{"metric": "am", "bins": 6}]},
                       "selection": {"method": "uniform-bins"}})
    res = _curate(strat, target_size=6)
    assert len({s.features["am"] for s in res.selected}) >= 5   # spreads across levels


def test_weighted_and_target_distribution_run():
    w = _curate(_strategy({"selection": {"method": "weighted", "weights": {"am": 1.0}}}), 6)
    assert len(w.selected) == 6
    t = _curate(_strategy({"binning": {"axes": [{"metric": "am", "bins": 6}]},
                           "selection": {"method": "target-distribution"}}), 6)
    assert t.selected


def test_determinism_same_seed():
    strat = _strategy({"selection": {"method": "weighted", "weights": {"am": 1.0}}})
    a = _curate(strat, 6, seed=42)
    b = _curate(strat, 6, seed=42)
    key = lambda r: sorted((s.source_id, s.start_idx) for s in r.selected)
    assert key(a) == key(b)


# ---- post-passes ------------------------------------------------------------
def test_group_cap_drops_overshoot():
    segs = [Segment("s", i, i + 1, (0, 0), bin=(i % 3,), group=("A" if i < 8 else "B"))
            for i in range(10)]   # 8 in A, 2 in B
    capped = enforce_group_cap(segs, max_group_overshoot=1.0)  # fair share = 5/group
    from collections import Counter
    assert Counter(s.group for s in capped)["A"] <= 5


def test_tag_coverage_adds_rare_tag():
    selected = [Segment("s", i, i + 1, (0, 0), bin=(0,), tags=frozenset({"common"}))
                for i in range(10)]
    pool = selected + [Segment("s", 99, 100, (0, 0), bin=(1,), tags=frozenset({"rare"}))]
    out = enforce_tag_coverage(selected, pool, tag_vocab=["common", "rare"],
                               target_size=10, min_tag_frac=0.05)
    assert any("rare" in s.tags for s in out)   # the rare-tag segment got pulled in
    assert len(out) == 10                       # coverage swaps; it never expands the budget


# ---- splits -----------------------------------------------------------------
def test_assign_splits_leakage_free_and_seeded():
    groups = {f"s{i}": ("A" if i % 2 else "B") for i in range(10)}
    s1 = assign_splits(groups, test_frac=0.4, seed=0)
    s2 = assign_splits(groups, test_frac=0.4, seed=0)
    assert s1 == s2                                   # deterministic
    assert set(s1.values()) == {"train", "test"}
    assert sum(v == "test" for v in s1.values()) == 4  # 40% of 10


def test_curate_with_test_frac_assigns_splits():
    res = _curate(_strategy({"selection": {"method": "top-k", "weights": {"am": 1.0}}}),
                  6, test_frac=0.34)
    assert set(res.splits.values()) <= {"train", "test"} and len(res.splits) == 6


# ---- load-time validation ---------------------------------------------------
@pytest.mark.parametrize("spec,msg", [
    ({"selection": {"method": "nope"}}, "selection method"),
    ({"selection": {"method": "uniform-bins"}}, "binning.axes"),     # uniform-bins needs axes
    ({"selection": {"method": "top-k", "weights": {"ghost": 1.0}}}, "undefined metrics"),
    ({"define": {"bad": {"source": "e.nope", "reduce": "mean"}},
      "selection": {"method": "top-k", "weights": {"bad": 1.0}}}, "unknown field"),
])
def test_load_validation(spec, msg):
    base = {"name": "t", "define": {"am": {"source": "e.act", "reduce": "mean"}}}
    base.update(spec)
    with pytest.raises((ValueError, FileNotFoundError)) as e:
        load_strategy(base, SCHEMA)
    assert msg in str(e.value)


# ---- config reproducibility -------------------------------------------------
def test_config_embeds_schema_and_strategy():
    strat = _strategy({"selection": {"method": "top-k", "weights": {"am": 1.0}}})
    cfg = build_config(SCHEMA, strat, results={"selected": 6}, seed=0)
    # round-trips through JSON and the schema reloads to an equivalent spec
    again = json.loads(json.dumps(cfg, default=float))
    assert SchemaSpec.from_dict(again["schema"]).to_dict() == SCHEMA.to_dict()
    assert again["strategy"]["selection"]["method"] == "top-k"
