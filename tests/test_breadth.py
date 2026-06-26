"""Functional breadth: sub-track spans, NPZ/JSONL adapters, single-dataset report."""
from __future__ import annotations

import json

import numpy as np
import pytest

from kuration import (
    JsonlAdapter,
    NpzAdapter,
    OutputSpec,
    SchemaSpec,
    Segmenter,
    Tick,
    curate,
    dataset_report,
    load_strategy,
    write_dataset,
)

SCHEMA = SchemaSpec.from_dict({
    "name": "b", "entity_axis": "e", "sample_hz": 1.0,
    "fields": {"act": {"kind": "scalar"}, "pos": {"kind": "vector", "dims": ["x", "y"]},
               "kind": {"kind": "categorical"}},
})


class MemAdapter:
    def __init__(self, schema, data):
        self.schema, self.data = schema, data
    def ticks(self, src):
        return self.data[src]


# ---- sub-track spans --------------------------------------------------------
def test_spans_select_best_part_of_a_source(tmp_path):
    # one 16-step source: low activity first half, high activity second half
    ticks = [Tick(time=i, entities={0: {"act": (1.0 if i < 8 else 10.0),
                                        "pos": (float(i), 0.0), "kind": "k"}})
             for i in range(16)]
    strat = load_strategy({"name": "hi", "define": {"am": {"source": "e.act", "reduce": "mean"}},
                           "selection": {"method": "top-k", "weights": {"am": 1.0}}}, SCHEMA)
    spec = OutputSpec.from_dict({"shard_length": 4, "container": "npz",
                                 "modalities": {"obs": {"dtype": "float32", "shape": [2],
                                                        "source": "field:pos"}}})
    res = write_dataset(str(tmp_path / "out"), schema=SCHEMA, adapter=MemAdapter(SCHEMA, {"s": ticks}),
                        sources=["s"], strategy=strat, segmenter=Segmenter(4, 4), spec=spec,
                        span_steps=8, train_hours=8 / 3600, test_hours=0, seed=0)  # room for 1 span
    ids = res["admitted"]
    assert ("s", 0, 1) in ids and ("s", 0, 0) not in ids   # high-activity span kept, low dropped


def test_no_span_is_whole_track(tmp_path):
    ticks = [Tick(time=i, entities={0: {"act": 5.0, "pos": (float(i), 0.0), "kind": "k"}})
             for i in range(8)]
    strat = load_strategy({"name": "a", "define": {"am": {"source": "e.act", "reduce": "mean"}},
                           "selection": {"method": "top-k", "weights": {"am": 1.0}}}, SCHEMA)
    spec = OutputSpec.from_dict({"shard_length": 4, "container": "npz",
                                 "modalities": {"obs": {"dtype": "float32", "shape": [2],
                                                        "source": "field:pos"}}})
    res = write_dataset(str(tmp_path / "out"), schema=SCHEMA, adapter=MemAdapter(SCHEMA, {"s": ticks}),
                        sources=["s"], strategy=strat, segmenter=Segmenter(4, 4), spec=spec,
                        train_hours=100, test_hours=0, seed=0)
    assert res["admitted"] == {("s", 0, 0)}   # one whole-track span


# ---- NPZ / JSONL adapters ---------------------------------------------------
def test_npz_adapter(tmp_path):
    p = tmp_path / "src.npz"
    np.savez(p, xy=np.stack([np.arange(6), np.arange(6) * 2], axis=1).astype(float),
             a=np.arange(6, dtype=float))
    ticks = NpzAdapter(SCHEMA, fields={"pos": "xy", "act": "a"}).ticks(str(p))
    assert len(ticks) == 6 and ticks[3].entities[0]["pos"] == (3.0, 6.0)
    assert ticks[3].entities[0]["act"] == 3.0


def test_jsonl_adapter(tmp_path):
    p = tmp_path / "src.jsonl"
    p.write_text("\n".join(json.dumps({"t": i, "x": i, "y": 0, "act": i, "kind": "z"})
                           for i in range(5)))
    adapter = JsonlAdapter(SCHEMA, fields={"pos": ["x", "y"], "act": "act", "kind": "kind"},
                           time_column="t")
    ticks = adapter.ticks(str(p))
    assert len(ticks) == 5 and ticks[2].entities[0]["act"] == 2.0
    assert ticks[2].entities[0]["kind"] == "z"


# ---- report -----------------------------------------------------------------
def test_dataset_report_summarizes():
    data = {f"s{L}": [Tick(time=i, entities={0: {"act": float(L), "pos": (float(i), float(L)),
                                                 "kind": ("hot" if L > 3 else "cold")}})
                      for i in range(8)] for L in range(1, 6)}
    strat = load_strategy({"name": "a", "define": {"am": {"source": "e.act", "reduce": "mean"}},
                           "tags": {"from": ["kind"]},
                           "selection": {"method": "top-k", "weights": {"am": 1.0}}}, SCHEMA)
    res = curate(schema=SCHEMA, adapter=MemAdapter(SCHEMA, data), sources=list(data),
                 strategy=strat, segmenter=Segmenter(4, 4), target_size=100, seed=0)
    rep = dataset_report(res.all_segments)
    assert rep["n_segments"] > 0 and rep["n_sources"] == 5
    assert "am" in rep["metrics"] and rep["metrics"]["am"]["max"] == 5.0
    assert set(rep["tags"]) == {"kind:hot", "kind:cold"}
    assert len(rep["metrics"]["am"]["histogram"]["counts"]) == 10
    # empty pool must expose the SAME key set (no KeyError downstream, e.g. the report CLI)
    empty = dataset_report([])
    assert empty["n_segments"] == 0
    assert set(empty) >= {"n_segments", "n_sources", "metrics", "groups", "tags"}
