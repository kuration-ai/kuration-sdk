"""Scale & robustness: parallel feature pass, per-source skip-and-log, resume."""
from __future__ import annotations

import json

import pytest

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
    "name": "s", "entity_axis": "e", "sample_hz": 1.0,
    "fields": {"pos": {"kind": "vector", "dims": ["x", "y"]}, "act": {"kind": "scalar"}},
})
STRAT = {"name": "hi", "define": {"am": {"source": "e.act", "reduce": "mean"}},
         "selection": {"method": "top-k", "weights": {"am": 1.0}}}


# Module-level so ProcessPoolExecutor can pickle instances to workers.
class MemAdapter:
    def __init__(self, schema, data):
        self.schema, self.data = schema, data
    def ticks(self, src):
        return self.data[src]

    def source_fingerprint(self, src):
        from kuration.core.provenance import _state_hash
        return _state_hash(self.data[src])


class FlakyAdapter(MemAdapter):
    def __init__(self, schema, data, bad):
        super().__init__(schema, data)
        self.bad = bad
    def ticks(self, src):
        if src == self.bad:
            raise RuntimeError("boom")
        return self.data[src]


def _data(levels, n=8):
    return {f"s{L}": [Tick(time=i, entities={0: {"pos": (float(i), float(L)), "act": float(L)}})
                      for i in range(n)] for L in levels}


# ---- parallel == serial -----------------------------------------------------
def test_parallel_feature_pass_matches_serial():
    data = _data(range(1, 13))
    strat = load_strategy(STRAT, SCHEMA)
    kw = dict(schema=SCHEMA, adapter=MemAdapter(SCHEMA, data), sources=list(data),
              strategy=strat, segmenter=Segmenter(4, 4), target_size=10, seed=0)
    serial = curate(workers=1, **kw)
    parallel = curate(workers=4, **kw)
    key = lambda r: sorted((s.source_id, s.start_idx) for s in r.selected)
    assert key(serial) == key(parallel)
    # features themselves identical regardless of worker count
    fs = lambda r: sorted((s.source_id, s.start_idx, round(s.features["am"], 6))
                          for s in r.all_segments)
    assert fs(serial) == fs(parallel)


# ---- skip-and-log -----------------------------------------------------------
def test_bad_source_skipped_and_logged(capsys):
    data = _data(range(1, 5))
    adapter = FlakyAdapter(SCHEMA, data, bad="s3")
    res = curate(schema=SCHEMA, adapter=adapter, sources=list(data),
                 strategy=load_strategy(STRAT, SCHEMA), segmenter=Segmenter(4, 4),
                 target_size=100, seed=0, on_error="skip")
    assert [e["source"] for e in res.errors] == ["s3"]
    assert res.errors[0]["error_type"] == "builtins.RuntimeError"
    assert res.errors[0]["message"] == "boom"
    assert "traceback" not in res.errors[0]  # portable artifacts omit private stack paths
    assert not any(s.source_id == "s3" for s in res.all_segments)   # the bad one is gone
    assert {s.source_id for s in res.all_segments} == {"s1", "s2", "s4"}
    assert "skip" in capsys.readouterr().err.lower()


def test_on_error_raise_propagates():
    data = _data(range(1, 5))
    adapter = FlakyAdapter(SCHEMA, data, bad="s3")
    with pytest.raises(RuntimeError):
        curate(schema=SCHEMA, adapter=adapter, sources=list(data),
               strategy=load_strategy(STRAT, SCHEMA), segmenter=Segmenter(4, 4),
               target_size=100, seed=0, on_error="raise")


@pytest.mark.parametrize("callback", ["group_of", "tags_of"])
def test_callback_failures_use_the_same_per_source_error_policy(callback):
    data = _data(range(1, 4))

    def group_of(source):
        if source == "s2":
            raise RuntimeError("bad grouping")
        return "ok"

    def tags_of(segment):
        if segment.source_id == "s2":
            raise RuntimeError("bad tagging")
        return ()

    kwargs = {callback: group_of if callback == "group_of" else tags_of}
    result = curate(
        schema=SCHEMA, adapter=MemAdapter(SCHEMA, data), sources=list(data),
        strategy=load_strategy(STRAT, SCHEMA), segmenter=Segmenter(4, 4),
        target_size=100, seed=0, on_error="skip", **kwargs)
    assert {segment.source_id for segment in result.all_segments} == {"s1", "s3"}
    assert [(error["source"], error["phase"]) for error in result.errors] == [
        ("s2", callback)]


# ---- resume -----------------------------------------------------------------
def _spec():
    return OutputSpec.from_dict({"shard_length": 4, "container": "npz",
                                 "modalities": {"obs": {"dtype": "float32", "shape": [2],
                                                        "source": "field:pos"}}})


def test_resume_repairs_corrupt_existing_shards(tmp_path):
    data = _data(range(1, 5))
    common = dict(schema=SCHEMA, adapter=MemAdapter(SCHEMA, data), sources=list(data),
                  strategy=load_strategy(STRAT, SCHEMA), segmenter=Segmenter(4, 4),
                  spec=_spec(), train_hours=100, test_hours=0, seed=0)
    first = write_dataset(str(tmp_path / "out"), **common)
    assert first["results"]["shards"] > 0

    assert all(entry.get("sha256") and entry.get("size_bytes", 0) > 0
               for entry in first["shards"])
    # A matching run may reuse valid shards, but a corrupt payload must be rewritten.
    victim = first["shards"][0]["path"]
    with open(victim, "wb") as f:
        f.write(b"CORRUPT")
    second = write_dataset(str(tmp_path / "out"), resume=True, **common)
    assert second["results"]["shards"] == first["results"]["shards"]   # same manifest
    assert open(victim, "rb").read() != b"CORRUPT"
    from kuration.core.io import file_integrity
    repaired = next(entry for entry in second["shards"] if entry["path"] == victim)
    assert file_integrity(victim) == {
        "sha256": repaired["sha256"], "size_bytes": repaired["size_bytes"]}


def test_force_overwrites_without_resuming(tmp_path):
    data = _data(range(1, 5))
    common = dict(schema=SCHEMA, adapter=MemAdapter(SCHEMA, data), sources=list(data),
                  strategy=load_strategy(STRAT, SCHEMA), segmenter=Segmenter(4, 4),
                  spec=_spec(), train_hours=100, test_hours=0, seed=0)
    first = write_dataset(str(tmp_path / "out"), **common)
    victim = first["shards"][0]["path"]
    with open(victim, "wb") as f:
        f.write(b"CORRUPT")
    write_dataset(str(tmp_path / "out"), resume=False, force=True, **common)
    assert open(victim, "rb").read() != b"CORRUPT"
