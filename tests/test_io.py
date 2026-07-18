"""Path-agnostic I/O: local + any fsspec backend. Exercised over `memory://` (no creds)."""
from __future__ import annotations

import json

import pytest

import kuration.io as kio
from kuration import (
    CuratedDatasetAdapter,
    OutputSpec,
    SchemaSpec,
    Segmenter,
    TimeseriesAdapter,
    Tick,
    curate,
    load_strategy,
    write_dataset,
)

fsspec = pytest.importorskip("fsspec")

SCHEMA = SchemaSpec.from_dict({
    "name": "io", "entity_axis": "e", "sample_hz": 1.0,
    "fields": {"act": {"kind": "scalar"}, "pos": {"kind": "vector", "dims": ["x", "y"]}},
})


@pytest.fixture(autouse=True)
def _clean_memory_fs():
    fsspec.filesystem("memory").store.clear()   # isolate tests sharing the in-process memory fs
    yield


# ---- io primitives ----------------------------------------------------------
def test_text_bytes_exists_roundtrip():
    kio.write_text("memory://d/a.txt", "hi")
    assert kio.read_text("memory://d/a.txt") == "hi"
    assert kio.exists("memory://d/a.txt") and not kio.exists("memory://d/missing.txt")
    kio.write_bytes("memory://d/b.bin", b"\x00\x01")
    with kio.open_file("memory://d/b.bin", "rb") as f:
        assert f.read() == b"\x00\x01"


def test_local_copy_stages_remote():
    kio.write_text("memory://d/c.txt", "staged")
    with kio.local_copy("memory://d/c.txt") as p:
        assert "://" not in p and open(p).read() == "staged"   # real local path


def test_local_path_passthrough(tmp_path):
    p = tmp_path / "x.txt"
    kio.write_text(str(p), "local")
    assert kio.read_text(str(p)) == "local"
    with kio.local_copy(str(p)) as q:
        assert q == str(p)        # local: no copy


# ---- end to end: curate -> remote, then re-ingest from remote ---------------
def _data():
    return {f"s{L}": [Tick(time=i, entities={0: {"act": float(L), "pos": (float(i), float(L))}})
                      for i in range(8)] for L in range(1, 5)}


def test_write_dataset_to_memory_and_reingest():
    data = _data()

    class Mem:
        schema = SCHEMA
        def ticks(self, src):
            return data[src]

    strat = load_strategy({"name": "hi", "define": {"am": {"source": "e.act", "reduce": "mean"}},
                           "selection": {"method": "top-k", "weights": {"am": 1.0}}}, SCHEMA)
    spec = OutputSpec.from_dict({"shard_length": 8, "container": "npz",
                                 "modalities": {"obs": {"dtype": "float32", "shape": [2],
                                                        "source": "field:pos"}}})
    res = write_dataset("memory://ds", schema=SCHEMA, adapter=Mem(), sources=list(data),
                        strategy=strat, segmenter=Segmenter(4, 4), spec=spec,
                        train_hours=100, test_hours=0, seed=0)
    assert res["results"]["shards"] > 0
    # manifest + config + shards all landed on the remote fs
    man = json.loads(kio.read_text("memory://ds/dataset_manifest.json"))
    assert man["count"] == res["results"]["shards"]
    assert not man["shards"][0]["path"].startswith("memory://")
    assert kio.exists(f"memory://ds/{man['shards'][0]['path']}")

    # re-ingest the remote dataset as a source
    adapter = CuratedDatasetAdapter("memory://ds")
    ticks = adapter.ticks(adapter.sources()[0])
    assert len(ticks) == 8 and len(ticks[0].entities[0]["obs"]) == 2


def test_timeseries_adapter_reads_remote_csv():
    import csv, io as _io
    buf = _io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["t", "x", "y", "act"])
    w.writeheader()
    for i in range(6):
        w.writerow({"t": i, "x": i, "y": 0, "act": i})
    kio.write_text("memory://in/series.csv", buf.getvalue())
    adapter = TimeseriesAdapter(SCHEMA, fields={"pos": ["x", "y"], "act": "act"}, time_column="t")
    ticks = adapter.ticks("memory://in/series.csv")
    assert len(ticks) == 6 and ticks[3].entities[0]["act"] == 3.0
