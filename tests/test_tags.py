"""Declarative tag derivation + tag-coverage working without a callback, both paths."""
from __future__ import annotations

import pytest

from kuration import (
    OutputSpec,
    SchemaSpec,
    Segmenter,
    Tick,
    build_window,
    curate,
    load_strategy,
    write_dataset,
)
from kuration.core.tags import compile_tagger
from kuration.core.select import enforce_tag_coverage

SCHEMA = SchemaSpec.from_dict({
    "name": "t", "entity_axis": "e", "sample_hz": 1.0,
    "fields": {"act": {"kind": "scalar"}, "kind": {"kind": "categorical"},
               "pos": {"kind": "vector", "dims": ["x", "y"]}},
})


class MemAdapter:
    def __init__(self, schema, data):
        self.schema, self.data = schema, data
    def ticks(self, src):
        return self.data[src]


def _data():
    # s1..s5 are high-activity "common"; s6 is low-activity "rare".
    d = {f"s{L}": [Tick(time=i, entities={0: {"act": float(L), "kind": "common",
                                              "pos": (float(i), float(L))}})
                   for i in range(8)] for L in range(1, 6)}
    d["s6"] = [Tick(time=i, entities={0: {"act": 0.5, "kind": "rare", "pos": (float(i), 0.0)}})
               for i in range(8)]
    return d


# ---- unit -------------------------------------------------------------------
def test_compile_tagger_distinct_values():
    ticks = [Tick(0, {0: {"kind": "a"}}), Tick(1, {0: {"kind": "b"}}), Tick(2, {0: {"kind": "a"}})]
    w = build_window(SCHEMA, ticks)
    assert compile_tagger({"from": ["kind"]}, SCHEMA)(w) == frozenset({"kind:a", "kind:b"})


def test_tags_validate_at_load():
    with pytest.raises(ValueError):   # 'act' is scalar, not categorical
        load_strategy({"name": "x", "tags": {"from": ["act"]},
                       "selection": {"method": "top-k", "weights": {"act": 1.0}}}, SCHEMA)


# ---- count path (curate.select) ---------------------------------------------
def test_tag_coverage_pulls_rare_tag_in_curate():
    data = _data()
    strat = load_strategy({
        "name": "hi", "define": {"am": {"source": "e.act", "reduce": "mean"}},
        "tags": {"from": ["kind"]},
        "selection": {"method": "top-k", "weights": {"am": 1.0},
                      "tag_coverage": {"min_tag_frac": 0.3}}}, SCHEMA)
    res = curate(schema=SCHEMA, adapter=MemAdapter(SCHEMA, data), sources=list(data),
                 strategy=strat, segmenter=Segmenter(4, 4), target_size=3, seed=0)
    # tags were derived declaratively (no tags_of callback)
    assert any(s.tags for s in res.all_segments)
    # top-k alone would take the 3 highest-act ("common") windows; coverage swaps in a "rare"
    assert any("kind:rare" in s.tags for s in res.selected)
    assert len(res.selected) == 3


def test_tag_coverage_accepts_a_single_pass_tag_vocabulary():
    from kuration import Segment

    common = Segment("common", 0, 1, (0, 0), bin=(0,), tags=frozenset({"common"}))
    rare = Segment("rare", 0, 1, (0, 0), bin=(1,), tags=frozenset({"rare"}))

    selected = enforce_tag_coverage(
        [common], [common, rare], tag_vocab=(tag for tag in ("common", "rare")),
        target_size=2, min_tag_frac=0.5)

    assert selected == [common, rare]


# ---- hours path (write_dataset / fill_to_hours reserve) ---------------------
def test_tag_coverage_reserve_in_write_dataset(tmp_path):
    data = _data()
    strat = load_strategy({
        "name": "hi", "define": {"am": {"source": "e.act", "reduce": "mean"}},
        "tags": {"from": ["kind"]},
        "selection": {"method": "top-k", "weights": {"am": 1.0},
                      "tag_coverage": {"min_tag_frac": 0.5}}}, SCHEMA)
    spec = OutputSpec.from_dict({"shard_length": 8, "container": "npz",
                                 "modalities": {"obs": {"dtype": "float32", "shape": [2],
                                                        "source": "field:pos"}}})
    # budget admits only ~2 tracks by priority (the high-act "common" ones); the reserve must
    # still pull in the lone "rare" source first.
    res = write_dataset(str(tmp_path / "out"), schema=SCHEMA, adapter=MemAdapter(SCHEMA, data),
                        sources=list(data), strategy=strat, segmenter=Segmenter(4, 4), spec=spec,
                        train_hours=24 / 3600, test_hours=0, seed=0)
    assert ("s6", 0, 0) in res["admitted"]   # the rare-tag source got reserved (src, entity, span)
