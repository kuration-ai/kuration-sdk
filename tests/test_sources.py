"""HDF5 input + curated-dataset-as-source: composition, re-curation, DSL filtering."""
from __future__ import annotations

import numpy as np

from kuration import (
    CuratedDatasetAdapter,
    FeatureEngine,
    HDF5Adapter,
    OutputSpec,
    SchemaSpec,
    Segmenter,
    Tick,
    assign_splits,
    curate,
    load_strategy,
    write_dataset,
)

RAW = SchemaSpec.from_dict({
    "name": "raw", "entity_axis": "e", "sample_hz": 1.0,
    "fields": {"pos": {"kind": "vector", "dims": ["x", "y"]}, "act": {"kind": "scalar"}},
})


class MemAdapter:
    def __init__(self, schema, data):
        self.schema, self.data = schema, data
    def ticks(self, src):
        return self.data[src]


# ---- HDF5 input -------------------------------------------------------------
def test_hdf5_adapter_reads_fields(tmp_path):
    import h5py
    p = tmp_path / "f.h5"
    with h5py.File(p, "w") as f:
        f["xy"] = np.stack([np.arange(10), np.arange(10) * 2], axis=1).astype(float)  # (10,2)
        f["a"] = np.arange(10, dtype=float)                                           # (10,)
    adapter = HDF5Adapter(RAW, fields={"pos": "xy", "act": "a"})
    ticks = adapter.ticks(str(p))
    assert len(ticks) == 10
    assert ticks[3].entities[0]["pos"] == (3.0, 6.0) and ticks[3].entities[0]["act"] == 3.0
    segs = Segmenter(4, 4).segments("f", ticks)
    FeatureEngine(RAW, define={"am": {"source": "e.act", "reduce": "mean"}}).compute(ticks, segs)
    assert segs[0].features["am"] == 1.5   # act 0,1,2,3


# ---- curated dataset as a source --------------------------------------------
def _make_dataset(tmp_path, levels, name, container="npz"):
    """Write a small curated dataset (obs=pos vector, act=scalar modalities)."""
    data = {f"{name}{L}": [Tick(time=i, entities={0: {"pos": (float(i), float(L)), "act": float(L)}})
                           for i in range(8)] for L in levels}
    strat = load_strategy({"name": "all", "define": {"am": {"source": "e.act", "reduce": "mean"}},
                           "selection": {"method": "top-k", "weights": {"am": 1.0}}}, RAW)
    spec = OutputSpec.from_dict({"shard_length": 8, "container": container, "modalities": {
        "obs": {"dtype": "float32", "shape": [2], "source": "field:pos"},
        "act": {"dtype": "float32", "shape": [], "source": "field:act"}}})
    out = tmp_path / name
    write_dataset(str(out), schema=RAW, adapter=MemAdapter(RAW, data), sources=list(data),
                  strategy=strat, segmenter=Segmenter(4, 4), spec=spec,
                  train_hours=100, test_hours=0, seed=0)   # big budget -> keep all
    return str(out)


import pytest


@pytest.mark.parametrize("container", ["npz", "json", "hdf5", "parquet"])
def test_curated_roundtrip_all_containers(tmp_path, container):
    if container == "parquet":
        pytest.importorskip("pyarrow")
    d = _make_dataset(tmp_path, [1, 2, 3], f"ds_{container}", container=container)
    adapter = CuratedDatasetAdapter(d)
    ticks = adapter.ticks(adapter.sources()[0])
    assert len(ticks) == 8
    assert len(ticks[0].entities[0]["obs"]) == 2 and ticks[0].entities[0]["act"] is not None


def test_curated_dataset_roundtrips_as_source(tmp_path):
    d = _make_dataset(tmp_path, [1, 2, 3], "ds")
    adapter = CuratedDatasetAdapter(d)
    # derived schema exposes the modalities as fields
    assert "act" in adapter.schema.fields and adapter.schema.fields["obs"].kind == "vector"
    srcs = adapter.sources()
    assert len(srcs) == 3                       # one shard per source track
    ticks = adapter.ticks(srcs[0])
    assert ticks[0].entities[0]["act"] is not None and len(ticks[0].entities[0]["obs"]) == 2


def test_filter_clips_within_a_curated_dataset(tmp_path):
    d = _make_dataset(tmp_path, [1, 2, 3, 4, 5], "ds")
    adapter = CuratedDatasetAdapter(d)
    strat = load_strategy({"name": "hot", "define": {"am": {"source": "e.act", "reduce": "mean"}},
                           "selection": {"method": "filter",
                                         "predicates": [{"metric": "am", "op": ">=", "value": 4}]}},
                          adapter.schema)
    res = curate(schema=adapter.schema, adapter=adapter, sources=adapter.sources(),
                 strategy=strat, segmenter=Segmenter(4, 4), target_size=100)
    assert res.selected and all(s.features["am"] >= 4 for s in res.selected)   # only act 4,5 clips


def test_random_split_across_two_datasets(tmp_path):
    d1 = _make_dataset(tmp_path, [1, 2, 3, 4], "a")
    d2 = _make_dataset(tmp_path, [5, 6, 7, 8], "b")
    adapter = CuratedDatasetAdapter([d1, d2])
    srcs = adapter.sources()
    assert len(srcs) == 8
    # 25/75 split, stratified by originating dataset, leakage-free + deterministic
    groups = {s: adapter.dataset_of[s] for s in srcs}
    split = assign_splits(groups, test_frac=0.25, seed=0)
    assert sum(v == "test" for v in split.values()) == 2          # 25% of 8
    assert split == assign_splits(groups, test_frac=0.25, seed=0)  # deterministic
