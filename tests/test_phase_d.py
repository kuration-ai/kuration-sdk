"""Phase D: output spec + shard writer + containers + hours fill + write_dataset."""
from __future__ import annotations

import json

import numpy as np
import pytest

from kuration import (
    OutputSpec,
    SchemaSpec,
    Segmenter,
    Tick,
    Track,
    fill_to_hours,
    load_strategy,
    write_dataset,
    write_track,
)
from kuration.core.output.shardwriter import _chunks

SCHEMA = SchemaSpec.from_dict({
    "name": "d", "entity_axis": "e", "sample_hz": 1.0,
    "fields": {"pos": {"kind": "vector", "dims": ["x", "y"]}, "act": {"kind": "scalar"}},
})


# ---- OutputSpec validation --------------------------------------------------
def test_outputspec_good_and_roundtrip():
    s = OutputSpec.from_dict({"shard_length": 4, "container": "npz",
                              "modalities": {"obs": {"dtype": "float32", "shape": [2],
                                                     "source": "field:pos"}}})
    assert s.shard_length == 4 and s.modalities[0].src_kind == "field"
    assert OutputSpec.from_dict(s.to_dict()).to_dict() == s.to_dict()


@pytest.mark.parametrize("spec,msg", [
    ({"shard_length": 4, "modalities": {}}, "at least one modality"),
    ({"shard_length": 4, "modalities": {"o": {"dtype": "float32", "source": "bogus"}}}, "kind"),
    ({"shard_length": 4, "modalities": {"o": {"dtype": "notatype", "source": "field:pos"}}}, "dtype"),
    ({"shard_length": 0, "modalities": {"o": {"dtype": "float32", "source": "field:pos"}}}, "shard_length"),
    ({"shard_length": 4, "short_policy": "x", "modalities": {"o": {"dtype": "f4", "source": "field:pos"}}}, "short_policy"),
])
def test_outputspec_bad(spec, msg):
    with pytest.raises(ValueError) as e:
        OutputSpec.from_dict(spec)
    assert msg in str(e.value)


# ---- chunking / short-policy ------------------------------------------------
def test_chunk_policies():
    recs = [{"i": i} for i in range(10)]
    keep = list(_chunks(recs, 4, "keep"))
    pad = list(_chunks(recs, 4, "pad"))
    drop = list(_chunks(recs, 4, "drop"))
    assert [r for _, r, _ in keep] == [4, 4, 2] and keep[-1][2] is False
    assert [r for _, r, _ in pad] == [4, 4, 2] and pad[-1][2] is True and len(pad[-1][0]) == 4
    assert [r for _, r, _ in drop] == [4, 4]     # short tail dropped


# ---- shard writing ----------------------------------------------------------
def _track(n=10):
    return Track("src", 0, [{"pos": (float(i), float(i * 2)), "act": float(i)} for i in range(n)])


def test_write_track_npz_field_modality(tmp_path):
    spec = OutputSpec.from_dict({"shard_length": 4, "container": "npz",
                                 "modalities": {"obs": {"dtype": "float32", "shape": [2],
                                                        "source": "field:pos"}}})
    entries = write_track(str(tmp_path), _track(10), spec, schema=SCHEMA, split="train")
    assert [e["n_steps"] for e in entries] == [4, 4, 2]
    z = np.load(entries[0]["path"])
    assert z["obs"].shape == (4, 2)
    assert list(z["obs"][1]) == [1.0, 2.0] and int(z["_n_steps"]) == 4


def test_write_track_pad_zeros_tail(tmp_path):
    spec = OutputSpec.from_dict({"shard_length": 4, "short_policy": "pad", "container": "npz",
                                 "modalities": {"obs": {"dtype": "float32", "shape": [2],
                                                        "source": "field:pos"}}})
    entries = write_track(str(tmp_path), _track(10), spec, schema=SCHEMA)
    last = np.load(entries[-1]["path"])
    assert entries[-1]["padded"] and entries[-1]["n_steps"] == 2
    assert last["obs"].shape == (4, 2) and not last["obs"][2:].any()   # pad rows zeroed


def test_write_track_encoder_modality(tmp_path):
    class Enc:
        def timestep(self, record, ctx):
            return {"speedish": np.asarray([record["act"] * 10.0], dtype=np.float32)}
    spec = OutputSpec.from_dict({"shard_length": 4, "container": "npz",
                                 "modalities": {"sp": {"dtype": "float32", "shape": [1],
                                                       "source": "encoder:speedish"}}})
    entries = write_track(str(tmp_path), _track(4), spec, encoder=Enc(), schema=SCHEMA)
    z = np.load(entries[0]["path"])
    assert list(z["sp"][3]) == [30.0]     # act=3 -> 30


def test_write_track_json_container(tmp_path):
    spec = OutputSpec.from_dict({"shard_length": 4, "container": "json",
                                 "modalities": {"obs": {"dtype": "float32", "shape": [2],
                                                        "source": "field:pos"}}})
    entries = write_track(str(tmp_path), _track(4), spec, schema=SCHEMA)
    data = json.loads(open(entries[0]["path"]).read())
    assert data["_n_steps"] == 4 and data["modalities"]["obs"][1] == [1.0, 2.0]


def test_write_track_hdf5_container(tmp_path):
    import h5py
    spec = OutputSpec.from_dict({"shard_length": 4, "container": "hdf5",
                                 "modalities": {"obs": {"dtype": "float32", "shape": [2],
                                                        "source": "field:pos"}}})
    entries = write_track(str(tmp_path), _track(10), spec, schema=SCHEMA)
    assert [e["n_steps"] for e in entries] == [4, 4, 2]
    with h5py.File(entries[0]["path"], "r") as f:
        assert f["obs"].shape == (4, 2) and list(f["obs"][1]) == [1.0, 2.0]
        assert f.attrs["n_steps"] == 4


def test_write_track_parquet_container(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    spec = OutputSpec.from_dict({"shard_length": 4, "container": "parquet",
                                 "modalities": {"obs": {"dtype": "float32", "shape": [2],
                                                        "source": "field:pos"},
                                                "act": {"dtype": "float32", "shape": [],
                                                        "source": "field:act"}}})
    entries = write_track(str(tmp_path), _track(4), spec, schema=SCHEMA)
    t = pq.read_table(entries[0]["path"])
    assert t.column("obs").to_pylist()[1] == [1.0, 2.0]      # tensor modality -> list column
    assert t.column("act").to_pylist()[3] == 3.0             # scalar modality -> scalar column
    assert int(t.schema.metadata[b"_n_steps"]) == 4


# ---- hours fill -------------------------------------------------------------
def _items(specs):
    return [{"id": i, "source_id": f"s{i}", "priority": p, "seconds": sec, "group": g}
            for i, (p, sec, g) in enumerate(specs)]


def test_fill_to_hours_takes_highest_priority():
    items = _items([(4, 3600, None), (3, 3600, None), (2, 3600, None), (1, 3600, None)])
    admitted, split = fill_to_hours(items, train_hours=2, test_hours=0, seed=0)
    assert admitted == {0, 1}                       # priorities 4 and 3 (2h of 1h items)
    assert all(v == "train" for v in split.values())


def test_fill_to_hours_group_cap():
    # 3 items group A, 1 group B, 1h each, 4h budget, overshoot 1.0 -> A capped at 2h
    items = _items([(9, 3600, "A"), (8, 3600, "A"), (7, 3600, "A"), (6, 3600, "B")])
    admitted, _ = fill_to_hours(items, train_hours=4, test_hours=0,
                                max_group_overshoot=1.0, seed=0)
    a_admitted = sum(1 for i in admitted if items[i]["group"] == "A")
    assert a_admitted <= 2


def test_fill_to_hours_deterministic():
    items = _items([(4, 3600, None), (3, 3600, None), (2, 3600, None)])
    assert fill_to_hours(items, train_hours=1, test_hours=1, seed=7)[0] == \
        fill_to_hours(items, train_hours=1, test_hours=1, seed=7)[0]


# ---- write_dataset end to end -----------------------------------------------
class MemAdapter:
    def __init__(self, schema, data):
        self.schema, self.data = schema, data
    def ticks(self, src):
        return self.data[src]


def test_write_dataset_end_to_end(tmp_path):
    data = {f"s{L}": [Tick(time=i, entities={0: {"pos": (float(i), float(L)), "act": float(L)}})
                      for i in range(12)] for L in range(1, 5)}
    adapter = MemAdapter(SCHEMA, data)
    strat = load_strategy({"name": "hi", "define": {"am": {"source": "e.act", "reduce": "mean"}},
                           "selection": {"method": "top-k", "weights": {"am": 1.0}}}, SCHEMA)
    spec = OutputSpec.from_dict({"shard_length": 4, "container": "npz",
                                 "modalities": {"obs": {"dtype": "float32", "shape": [2],
                                                        "source": "field:pos"}}})
    res = write_dataset(str(tmp_path), schema=SCHEMA, adapter=adapter, sources=list(data),
                        strategy=strat, segmenter=Segmenter(4, 4), spec=spec,
                        train_hours=12 / 3600 * 2, test_hours=0, seed=0)  # ~2 tracks of 12s
    # highest-activity source must be admitted (top-k by act); id = (source, entity, span)
    assert ("s4", 0, 0) in res["admitted"]
    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text())
    assert manifest["count"] == len(res["shards"]) > 0
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["schema"]["name"] == "d" and cfg["strategy"]["selection"]["method"] == "top-k"
    assert cfg["output"]["modalities"]["obs"]["source"] == "field:pos"
    # a shard actually exists on disk and reloads
    assert np.load(res["shards"][0]["path"])["obs"].shape[1] == 2
