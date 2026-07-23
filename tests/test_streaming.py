"""Out-of-core streaming: parity with the in-memory path + acceptance of lazy generators.

Bounded memory is a structural property (Segmenter.stream holds a deque(maxlen=window); the
streaming writer holds at most shard_length records per active (entity, span)) — these tests
prove *correctness equivalence* and that the streaming path accepts a source that is a pure
generator (never materialized / indexed).
"""
from __future__ import annotations

from collections import deque as stdlib_deque
import json

import numpy as np

from kuration import (
    OutputSpec,
    SchemaSpec,
    Segmenter,
    Tick,
    curate,
    load_strategy,
    write_dataset,
)

SCHEMA = SchemaSpec.from_dict({
    "name": "st", "entity_axis": "e", "sample_hz": 1.0,
    "fields": {"act": {"kind": "scalar"}, "pos": {"kind": "vector", "dims": ["x", "y"]}},
})
STRAT = {"name": "hi", "define": {"am": {"source": "e.act", "reduce": "mean"}},
         "selection": {"method": "top-k", "weights": {"am": 1.0}}}


def _data(levels, n=8):
    return {f"s{L}": [Tick(time=i, entities={0: {"act": float(L), "pos": (float(i), float(L))}})
                      for i in range(n)] for L in levels}


class ListAdapter:
    schema = SCHEMA
    def __init__(self, data): self.data = data
    def ticks(self, src): return self.data[src]            # a list


class GenAdapter:
    schema = SCHEMA
    def __init__(self, data): self.data = data
    def ticks(self, src): return (t for t in self.data[src])   # fresh generator each call


# ---- segmenter parity -------------------------------------------------------
def test_segmenter_stream_matches_segments():
    ticks = [Tick(time=i, entities={0: {"act": 1.0}}) for i in range(20)]
    sg = Segmenter(4, 2)
    a = sg.segments("s", ticks)
    streamed = list(sg.stream("s", iter(ticks)))
    b = [seg for seg, _ in streamed]
    assert [(s.start_idx, s.end_idx, s.time_range) for s in a] == \
           [(s.start_idx, s.end_idx, s.time_range) for s in b]
    assert all(len(w) == 4 for _, w in streamed)            # each window holds exactly W ticks


def test_segmenter_context_is_bounded_and_shortens_only_at_source_edges():
    ticks = [Tick(time=index, entities={0: {"act": 1.0}}) for index in range(8)]
    entries = list(Segmenter(4, 2).stream_with_context(
        "s", iter(ticks), lookbehind=1, lookahead=2))

    assert [(segment.start_idx, [tick.time for tick in before],
             [tick.time for tick in after])
            for segment, _window, before, after in entries] == [
        (0, [], [4, 5]),
        (2, [1], [6, 7]),
        (4, [3], []),
    ]


def test_segmenter_context_discards_sparse_stride_gaps(monkeypatch):
    import kuration.core.segment as segment_module

    class TrackingDeque(stdlib_deque):
        peak = 0

        def append(self, value):
            super().append(value)
            type(self).peak = max(type(self).peak, len(self))

    monkeypatch.setattr(segment_module, "deque", TrackingDeque)
    ticks = [Tick(time=index, entities={0: {"act": 1.0}}) for index in range(205)]

    entries = list(Segmenter(2, 100).stream_with_context(
        "s", iter(ticks), lookbehind=3, lookahead=4))

    assert [(segment.start_idx, [tick.time for tick in before],
             [tick.time for tick in after])
            for segment, _window, before, after in entries] == [
        (0, [], [2, 3, 4, 5]),
        (100, [97, 98, 99], [102, 103, 104, 105]),
        (200, [197, 198, 199], [202, 203, 204]),
    ]
    assert TrackingDeque.peak <= 2 + 3 + 4


# ---- score parity (+ generator source) --------------------------------------
def test_curate_streaming_matches_inmemory():
    data = _data(range(1, 6))
    strat = load_strategy(STRAT, SCHEMA)
    kw = dict(strategy=strat, segmenter=Segmenter(4, 4), target_size=8, seed=0)
    mem = curate(schema=SCHEMA, adapter=ListAdapter(data), sources=list(data), **kw)
    strm = curate(schema=SCHEMA, adapter=GenAdapter(data), sources=list(data),
                  streaming=True, **kw)
    key = lambda r: sorted((s.source_id, s.start_idx) for s in r.selected)
    feat = lambda r: sorted((s.source_id, s.start_idx, round(s.features["am"], 6))
                            for s in r.all_segments)
    assert key(mem) == key(strm)
    assert feat(mem) == feat(strm)


def test_streaming_accepts_pure_generator():
    data = _data(range(1, 4))
    res = curate(schema=SCHEMA, adapter=GenAdapter(data), sources=list(data),
                 strategy=load_strategy(STRAT, SCHEMA), segmenter=Segmenter(4, 4),
                 target_size=4, seed=0, streaming=True)
    assert res.selected   # ran end-to-end on generator sources


# ---- write parity -----------------------------------------------------------
def _spec():
    return OutputSpec.from_dict({"shard_length": 4, "container": "npz",
                                 "modalities": {"obs": {"dtype": "float32", "shape": [2],
                                                        "source": "field:pos"}}})


def _shards_by_name(out_dir):
    man = json.loads((out_dir / "dataset_manifest.json").read_text())
    out = {}
    for e in man["shards"]:
        arr = np.load(out_dir / e["path"])
        out[e["path"].split("/")[-1]] = (int(arr["_n_steps"]), arr["obs"])
    return out


def test_write_dataset_streaming_matches_inmemory(tmp_path):
    data = _data(range(1, 4))
    strat = load_strategy(STRAT, SCHEMA)
    base = dict(schema=SCHEMA, sources=list(data), strategy=strat, segmenter=Segmenter(4, 4),
                spec=_spec(), train_hours=100, test_hours=0, seed=0)
    write_dataset(str(tmp_path / "mem"), adapter=ListAdapter(data), **base)
    write_dataset(str(tmp_path / "str"), adapter=GenAdapter(data), streaming=True, **base)

    mem, strm = _shards_by_name(tmp_path / "mem"), _shards_by_name(tmp_path / "str")
    assert set(mem) == set(strm) and mem                      # same shard files
    for name in mem:
        assert mem[name][0] == strm[name][0]                  # same n_steps
        assert np.array_equal(mem[name][1], strm[name][1])    # identical arrays


def test_write_dataset_streaming_with_spans(tmp_path):
    # long single source, span_steps subdivides; streaming writer flushes per (entity, span)
    ticks = [Tick(time=i, entities={0: {"act": float(i), "pos": (float(i), 0.0)}}) for i in range(16)]
    strat = load_strategy(STRAT, SCHEMA)
    res = write_dataset(str(tmp_path / "out"), schema=SCHEMA, adapter=GenAdapter({"s": ticks}),
                        sources=["s"], strategy=strat, segmenter=Segmenter(4, 4), spec=_spec(),
                        span_steps=8, train_hours=100, test_hours=0, seed=0, streaming=True)
    assert res["results"]["shards"] > 0
    # spans appear in shard filenames
    man = json.loads((tmp_path / "out" / "dataset_manifest.json").read_text())
    assert any("_span" in e["path"] for e in man["shards"])
