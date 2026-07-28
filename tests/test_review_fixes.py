"""Regression tests for the full-SDK review fixes (atomic writes, resume guards, shard-name
collisions, -inf fill admission, tie-robust binning, DSL spatial/jerk/vector-filter, manifest
determinism, parquet loudness, dtype restore, moved-dataset reload, tags normalization)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

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
    read_shard_arrays,
    shard_entries,
    write_dataset,
    write_track,
)
from kuration.core.adapters.curated import CuratedDatasetAdapter
from kuration.core.binning import bin_codes
from kuration.core.buffer import empty_like
from kuration.core.dsl import compile_metric
from kuration.core.io import staged_write

SCHEMA = SchemaSpec.from_dict({
    "name": "rf", "entity_axis": "e", "sample_hz": 1.0,
    "fields": {"pos": {"kind": "vector", "dims": ["x", "y"]}, "act": {"kind": "scalar"}},
})

GS = SchemaSpec.from_dict({
    "name": "g", "entity_axis": "agent", "sample_hz": 1.0,
    "fields": {
        "pos": {"kind": "vector", "dims": ["x", "y"], "semantics": "position"},
        "hp": {"kind": "scalar"},
        "weapon": {"kind": "categorical"},
    },
})

SPEC = OutputSpec.from_dict({"shard_length": 4, "container": "npz",
                             "modalities": {"obs": {"dtype": "float32", "shape": [2],
                                                    "source": "field:pos"}}})


class MemAdapter:
    def __init__(self, schema, data):
        self.schema, self.data = schema, data

    def ticks(self, src):
        return self.data[src]

    def source_fingerprint(self, src):
        from kuration.core.provenance import _state_hash
        return _state_hash(self.data[src])


def _data(n_sources=3, n_ticks=12, n_entities=2):
    return {f"s{L}": [Tick(time=i, entities={e: {"pos": (float(i), float(L + e)),
                                                 "act": float(L)}
                                             for e in range(n_entities)})
                      for i in range(n_ticks)] for L in range(1, n_sources + 1)}


def _strategy():
    return load_strategy({"name": "rf", "define": {"am": {"source": "e.act", "reduce": "mean"}},
                          "selection": {"method": "top-k", "weights": {"am": 1.0}}}, SCHEMA)


def _write(tmp_path, out="ds", spec=SPEC, resume=False, hours=None):
    data = _data()
    return write_dataset(str(tmp_path / out), schema=SCHEMA, adapter=MemAdapter(SCHEMA, data),
                         sources=list(data), strategy=_strategy(), segmenter=Segmenter(4, 4),
                         spec=spec, train_hours=hours if hours is not None else 1.0,
                         test_hours=0, seed=0, resume=resume)


# ---- C1: atomic local staged_write + resume guards ---------------------------
def test_staged_write_local_is_atomic(tmp_path):
    dest = tmp_path / "x.npz"
    with pytest.raises(RuntimeError):
        with staged_write(str(dest)) as p:
            Path(p).write_bytes(b"partial")
            raise RuntimeError("crash mid-write")
    assert not dest.exists()                      # no partial file at the final path
    assert list(tmp_path.iterdir()) == []         # temp cleaned up
    with staged_write(str(dest)) as p:
        Path(p).write_bytes(b"ok")
    assert dest.read_bytes() == b"ok"


def test_staged_write_keeps_npz_extension(tmp_path):
    # np.savez appends '.npz' when missing — the temp name must already carry it.
    dest = tmp_path / "y.npz"
    with staged_write(str(dest)) as p:
        np.savez(p, a=np.arange(3))
    assert np.load(dest)["a"].tolist() == [0, 1, 2]


def test_resume_rejects_changed_output_spec(tmp_path):
    _write(tmp_path)
    _write(tmp_path, resume=True)                 # same spec: fine
    other = OutputSpec.from_dict({"shard_length": 6, "container": "npz",
                                  "modalities": {"obs": {"dtype": "float32", "shape": [2],
                                                         "source": "field:pos"}}})
    with pytest.raises(ValueError, match="different output spec"):
        _write(tmp_path, spec=other, resume=True)


def test_resume_signature_includes_resolved_transforms(tmp_path):
    from kuration.core.output.dataset import _check_resume_run
    from kuration.core.provenance import build_run_signature

    data = _data()
    common = dict(schema=SCHEMA, strategy=_strategy(), spec=SPEC,
                  segmenter=Segmenter(4, 4), sources=list(data),
                  adapter=MemAdapter(SCHEMA, data), encoder=object(), seed=0,
                  sample_hz=1.0, train_hours=1.0, test_hours=0.0,
                  span_len=None, clip_aggregate="max")
    first = build_run_signature(**common, transforms=[{
        "name": "embed", "implementation": "vendor/embed", "outputs": ("z",),
        "inputs": ("field:act",), "batch_size": 8, "device": "cpu",
        "params": {"revision": "v1"}, "resources": {}}])
    second = build_run_signature(**common, transforms=[{
        "name": "embed", "implementation": "vendor/embed", "outputs": ("z",),
        "inputs": ("field:act",), "batch_size": 8, "device": "cpu",
        "params": {"revision": "v2"}, "resources": {}}])
    assert first["fingerprint"] != second["fingerprint"]
    _check_resume_run(str(tmp_path / "resume"), first, False)
    with pytest.raises(ValueError, match="different inputs or run configuration"):
        _check_resume_run(str(tmp_path / "resume"), second, True)


def test_directory_source_fingerprint_tracks_child_changes(tmp_path):
    from kuration.core.provenance import source_descriptor

    store = tmp_path / "store.zarr"
    store.mkdir()
    chunk = store / "0.0"
    chunk.write_bytes(b"old")
    before = source_descriptor(str(store), object())
    chunk.write_bytes(b"new payload")
    after = source_descriptor(str(store), object())
    assert before != after


# ---- C2: shard filenames cannot collide across (source, entity) --------------
def test_shard_names_do_not_collide(tmp_path):
    recs = [{"pos": (float(i), 0.0), "act": 0.0} for i in range(4)]
    e1 = write_track(str(tmp_path), Track("runs/a", "b", recs), SPEC, schema=SCHEMA)
    e2 = write_track(str(tmp_path), Track("runs", "a_b", recs), SPEC, schema=SCHEMA)
    assert {e["path"] for e in e1}.isdisjoint({e["path"] for e in e2})


# ---- C3: -inf priority is never admitted --------------------------------------
def test_fill_never_admits_neg_inf():
    items = [{"id": 0, "source_id": "a", "priority": 1.0, "seconds": 3600, "group": None},
             {"id": 1, "source_id": "b", "priority": float("-inf"), "seconds": 3600,
              "group": None, "tags": frozenset({"t"})}]
    admitted, _ = fill_to_hours(items, train_hours=10, test_hours=0, seed=0,
                                tag_coverage={"min_tag_frac": 0.5})
    assert admitted == {0}                        # slack budget + tag reserve still exclude it


# ---- M5: manifest entry order is hash-seed-independent ------------------------
def test_manifest_entries_sorted(tmp_path):
    res = _write(tmp_path)
    keys = [(e["source_id"], e["entity_id"], e["shard_index"]) for e in res["shards"]]
    assert len(keys) > 2 and keys == sorted(keys)


# ---- M6: parquet loudness + dtype restore -------------------------------------
def test_parquet_missing_n_steps_raises(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    from kuration.core.output.containers import read_shard, write_shard
    p = write_shard(str(tmp_path / "s"), "parquet", {"a": np.zeros(4, np.float32)}, 4)
    pq.write_table(pq.read_table(p).replace_schema_metadata(None), p)  # strip metadata
    with pytest.raises(ValueError, match="_n_steps"):
        read_shard(p, "parquet")


@pytest.mark.parametrize("container", ["json", "parquet"])
def test_dtype_restored_through_lossy_containers(tmp_path, container):
    if container == "parquet":
        pytest.importorskip("pyarrow")
    spec = OutputSpec.from_dict({"shard_length": 4, "container": container,
                                 "modalities": {"obs": {"dtype": "float32", "shape": [2],
                                                        "source": "field:pos"},
                                                "act": {"dtype": "uint8", "shape": [],
                                                        "source": "field:act"}}})
    recs = [{"pos": (float(i), 0.0), "act": float(i)} for i in range(4)]
    entries = write_track(str(tmp_path), Track("s", 0, recs), spec, schema=SCHEMA)
    out = read_shard_arrays(entries[0], container)
    assert out["obs"].dtype == np.float32 and out["act"].dtype == np.uint8
    assert out["act"].tolist() == [0, 1, 2, 3]


# ---- M1: binning is tie-robust and loud on NaN --------------------------------
def test_binning_separates_tied_mass():
    vals = np.array([0.0] * 90 + list(np.linspace(1, 10, 10)))
    codes = bin_codes(vals, 4)
    assert len(set(codes[:90])) == 1                       # tied mass in one bin
    assert set(codes[:90]).isdisjoint(set(codes[90:]))     # ...separate from the rest


def test_binning_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        bin_codes(np.array([1.0, np.nan]), 4, axis_name="m")


# ---- M2-M4: DSL spatial map/filter, jerk gaps, vector filters ------------------
def _gbuf():
    w = empty_like(GS, T=4, E=2)
    w.valid[:] = True
    for t in range(4):
        w.numeric["pos"][t, 0] = (0.0, 0.0)
        w.numeric["pos"][t, 1] = (10.0, 0.0)
        w.numeric["hp"][t, 0, 0] = 100.0
        w.numeric["hp"][t, 1, 0] = 50.0
    w.vocabs["weapon"] = ["A"]
    w.codes["weapon"][:] = 0
    return w


def test_spatial_reduce_rejects_map():
    with pytest.raises(ValueError, match="no map"):
        compile_metric({"source": "agent.pos", "reduce": "dispersion", "map": "abs"}, GS)


def test_spatial_filter_validated_and_applied():
    with pytest.raises(ValueError, match="unknown field"):
        compile_metric({"source": "agent.pos", "reduce": "dispersion", "filter": "nope>1"}, GS)
    unfiltered = compile_metric({"source": "agent.pos", "reduce": "dispersion"}, GS)(_gbuf())
    filtered = compile_metric({"source": "agent.pos", "reduce": "dispersion",
                               "filter": "hp>60"}, GS)(_gbuf())
    assert unfiltered == pytest.approx(5.0)
    assert filtered == pytest.approx(0.0)          # one entity left -> no pairwise spread


def test_jerk_does_not_bridge_gaps():
    w = _gbuf()
    w.numeric["hp"][:, 0, 0] = [100.0, 100.0, np.nan, 10.0]   # absent tick 2
    val = compile_metric({"source": "agent.hp", "reduce": "jerk"}, GS)(w)
    assert val == pytest.approx(0.0)               # the 100->10 jump spans a gap: not a step


def test_vector_comparison_filter_rejected():
    with pytest.raises(ValueError, match="vector"):
        compile_metric({"source": "agent.hp", "reduce": "mean", "filter": "pos>5"}, GS)


# ---- M7: a moved/copied dataset still loads ------------------------------------
def test_moved_dataset_reloads(tmp_path):
    res = _write(tmp_path, out="orig")
    assert res["shards"]
    shutil.move(str(tmp_path / "orig"), str(tmp_path / "moved"))
    entries, cont = shard_entries(str(tmp_path / "moved"), split="train")
    assert entries and read_shard_arrays(entries[0], cont)["obs"].shape[1] == 2
    ad = CuratedDatasetAdapter(str(tmp_path / "moved"))
    assert len(ad.ticks(ad.sources()[0])) > 0


def test_copied_dataset_prefers_its_own_shards_while_original_exists(tmp_path):
    _write(tmp_path, out="orig")
    manifest = json.loads((tmp_path / "orig" / "dataset_manifest.json").read_text())
    assert manifest["shards"][0]["path"].startswith("train/")
    assert not Path(manifest["shards"][0]["path"]).is_absolute()
    shutil.copytree(tmp_path / "orig", tmp_path / "copy")
    copied, cont = shard_entries(str(tmp_path / "copy"), split="train")
    original, _ = shard_entries(str(tmp_path / "orig"), split="train")
    assert copied and copied[0]["path"].startswith(str(tmp_path / "copy"))
    assert copied[0]["path"] != original[0]["path"]
    # Removing the original proves the already-resolved copied entries remain self-contained.
    shutil.rmtree(tmp_path / "orig")
    assert read_shard_arrays(copied[0], cont)["obs"].shape[1] == 2


def test_dataset_commit_sidecar_rejects_incomplete_or_changed_metadata(tmp_path):
    _write(tmp_path, out="complete")
    root = tmp_path / "complete"
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    commit = json.loads((root / "dataset.commit.json").read_text())
    assert manifest["commit_protocol"] == commit["protocol"] == "sidecar-v1"
    assert commit["state"] == "committed"
    assert set(commit["files"]) == {
        "dataset_manifest.json", "config.json", ".kuration_output_spec.json"}

    shutil.copytree(root, tmp_path / "writing")
    from kuration.core.manifest import begin_dataset
    from kuration.core.output.dataset import _resume_entries
    begin_dataset(str(tmp_path / "writing"))
    with pytest.raises(ValueError, match="commit state"):
        shard_entries(str(tmp_path / "writing"))
    assert _resume_entries(str(tmp_path / "writing"), True)

    shutil.copytree(root, tmp_path / "missing-commit")
    (tmp_path / "missing-commit" / "dataset.commit.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        shard_entries(str(tmp_path / "missing-commit"))

    shutil.copytree(root, tmp_path / "changed-after-commit")
    config = tmp_path / "changed-after-commit" / "config.json"
    config.write_text(config.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after the commit"):
        shard_entries(str(tmp_path / "changed-after-commit"))

    shutil.copytree(root, tmp_path / "changed-resume-marker")
    marker = tmp_path / "changed-resume-marker" / ".kuration_output_spec.json"
    marker.write_text(marker.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after the commit"):
        shard_entries(str(tmp_path / "changed-resume-marker"))


def test_shard_loader_rejects_missing_modality_and_count_mismatch(tmp_path):
    _write(tmp_path, out="ds")
    entries, cont = shard_entries(str(tmp_path / "ds"), split="train")
    with pytest.raises(KeyError, match="missing requested modalities"):
        read_shard_arrays(entries[0], cont, modalities=["typo"])
    bad = {**entries[0], "n_steps": entries[0]["n_steps"] + 1}
    with pytest.raises(ValueError, match="n_steps mismatch"):
        read_shard_arrays(bad, cont)


def test_curated_unknown_shard_raises(tmp_path):
    _write(tmp_path, out="d1")
    ad = CuratedDatasetAdapter(str(tmp_path / "d1"))
    with pytest.raises(KeyError, match="unknown shard"):
        ad.ticks("not/a/real/shard.npz")


# ---- M8: tags normalization ----------------------------------------------------
def test_tags_bare_list_normalized():
    strat = load_strategy({"name": "t", "define": {"am": {"source": "agent.hp",
                                                          "reduce": "mean"}},
                           "tags": ["weapon"],
                           "selection": {"method": "top-k", "weights": {"am": 1.0}}}, GS)
    assert strat.tags == {"from": ["weapon"]}


def test_tags_bad_shape_rejected():
    with pytest.raises(ValueError, match="tags"):
        load_strategy({"name": "t", "tags": {"frm": ["weapon"]},
                       "selection": {"method": "uniform-bins"},
                       "binning": {"axes": [{"metric": "am", "bins": 2}]},
                       "define": {"am": {"source": "agent.hp", "reduce": "mean"}}}, GS)


# ---- minors: loud-validation gaps ----------------------------------------------
def test_metric_spec_unknown_key_rejected():
    with pytest.raises(ValueError, match="fitler"):
        compile_metric({"source": "agent.hp", "reduce": "mean", "fitler": "hp>60"}, GS)


def test_scope_rejected_on_global_and_event():
    s = SchemaSpec.from_dict({"name": "ge", "entity_axis": "a",
                              "fields": {"hp": {"kind": "scalar"}},
                              "globals": {"alive": {"kind": "scalar"}},
                              "event_streams": {"ev": {"fields": {"code": {"kind": "scalar"}}}}})
    with pytest.raises(ValueError, match="scope"):
        compile_metric({"source": "alive", "reduce": "mean", "scope": "pooled"}, s)
    with pytest.raises(ValueError, match="scope"):
        compile_metric({"source": "ev.code", "reduce": "count", "scope": "pooled"}, s)


def test_event_numeric_filter_matches():
    s = SchemaSpec.from_dict({"name": "ev", "entity_axis": "a",
                              "fields": {"hp": {"kind": "scalar"}},
                              "event_streams": {"ev": {"fields": {"code": {"kind": "scalar"}}}}})
    w = empty_like(s, T=2, E=1)
    w.valid[:] = True
    w.events["ev"][0] = [{"code": 5}, {"code": 7}, {"other": 1}]
    fn = compile_metric({"source": "ev.code", "reduce": "count", "filter": "==5"}, s)
    assert fn(w) == pytest.approx(1.0)         # numeric 5 matches the '==5' literal
    fn_all = compile_metric({"source": "ev.code", "reduce": "count"}, s)
    assert fn_all(w) == pytest.approx(2.0)     # the code-less event is not a category value


def test_unknown_selection_key_rejected():
    with pytest.raises(ValueError, match="tag_covrage"):
        load_strategy({"name": "t", "define": {"am": {"source": "agent.hp", "reduce": "mean"}},
                       "binning": {"axes": [{"metric": "am", "bins": 2}]},
                       "selection": {"method": "uniform-bins",
                                     "tag_covrage": {"min_tag_frac": 0.01}}}, GS)


def test_target_dict_needs_metric_at_load():
    with pytest.raises(ValueError, match="metric"):
        load_strategy({"name": "t", "define": {"am": {"source": "agent.hp", "reduce": "mean"}},
                       "binning": {"axes": [{"metric": "am", "bins": 2}]},
                       "selection": {"method": "target-distribution",
                                     "target": {"anchor_metric": "am"}}}, GS)


def test_max_rate_validated_at_load():
    base = {"name": "s", "entity_axis": "a",
            "fields": {"pos": {"kind": "vector", "dims": ["x", "y"],
                               "semantics": "position", "max_rate": "fast"}}}
    with pytest.raises(ValueError, match="max_rate"):
        SchemaSpec.from_dict(base)


def test_scalar_into_vector_field_rejected():
    from kuration.core.buffer import build_window
    ticks = [Tick(time=0, entities={0: {"pos": 3.0}})]      # scalar for a 2-dim vector
    with pytest.raises(ValueError, match="vector"):
        build_window(GS, ticks)


def test_registered_metric_cannot_shadow_define():
    from kuration.core.dsl.registry import METRICS
    from kuration.core.features import FeatureEngine
    METRICS.setdefault("rf_shadow_test", lambda w: 0.0)
    try:
        with pytest.raises(ValueError, match="shadow"):
            FeatureEngine(GS, define={"rf_shadow_test": {"source": "agent.hp",
                                                         "reduce": "mean"}},
                          registered=["rf_shadow_test"])
    finally:
        METRICS.pop("rf_shadow_test", None)


# ---- minors: selection accounting ----------------------------------------------
def test_uniform_bins_respects_target_size():
    import random
    from kuration.core.segment import Segment
    from kuration.core.select import sample_uniform_bins
    segs = [Segment("s", i, i + 1, (i, i + 1), bin=(i,)) for i in range(500)]  # 500 bins
    out = sample_uniform_bins(segs, target_size=100, rng=random.Random(0))
    assert len(out) == 100                    # was: one PER BIN -> 500


def test_empty_pool_raises():
    from kuration.core.scoring import score_sources
    data = {"s": [Tick(time=i, entities={0: {"pos": (0.0, 0.0), "act": 0.0}})
                  for i in range(3)]}         # shorter than the window
    with pytest.raises(RuntimeError, match="EMPTY"):
        score_sources(schema=SCHEMA, adapter=MemAdapter(SCHEMA, data), sources=["s"],
                      define={"am": {"source": "e.act", "reduce": "mean"}}, registered=[],
                      segmenter=Segmenter(8, 8))


# ---- minors: adapters/containers -------------------------------------------------
def test_arrays_to_ticks_length_mismatch_raises():
    from kuration.core.adapters.base import arrays_to_ticks
    with pytest.raises(ValueError, match="length mismatch"):
        arrays_to_ticks(SCHEMA, {"act": np.zeros(3), "pos": np.zeros((5, 2))})


def test_arrays_to_ticks_accepts_shape_only_arrays():
    from kuration.core.adapters.base import arrays_to_ticks

    class ShapeOnlyArray:
        def __init__(self, values):
            self.values = np.asarray(values)
            self.shape = self.values.shape

        def __getitem__(self, index):
            return self.values[index]

    ticks = arrays_to_ticks(SCHEMA, {
        "act": ShapeOnlyArray([1.0, 2.0]),
        "pos": ShapeOnlyArray([[3.0, 4.0], [5.0, 6.0]]),
    })
    assert [tick.entities[0]["act"] for tick in ticks] == [1.0, 2.0]
    assert ticks[1].entities[0]["pos"] == (5.0, 6.0)


def test_timeseries_unordered_rows_raise():
    from kuration.core.adapters.timeseries import TimeseriesAdapter
    ad = TimeseriesAdapter(SCHEMA, fields={"act": "a"}, time_column="t", entity_column="e")
    rows = [{"t": 1, "e": "A", "a": 1}, {"t": 2, "e": "A", "a": 1},
            {"t": 1, "e": "B", "a": 2}, {"t": 2, "e": "B", "a": 2}]   # entity-major
    with pytest.raises(ValueError, match="not ordered"):
        ad.from_rows(rows)


def test_npz_underscore_modality_preserved(tmp_path):
    from kuration.core.output.containers import read_shard, write_shard
    p = write_shard(str(tmp_path / "s"), "npz", {"_mask": np.ones(4)}, 4)
    arrays, n = read_shard(p, "npz")
    assert "_mask" in arrays and n == 4


def test_span_drop_mismatch_rejected(tmp_path):
    data = _data()
    with pytest.raises(ValueError, match="zero shards"):
        write_dataset(str(tmp_path / "x"), schema=SCHEMA, adapter=MemAdapter(SCHEMA, data),
                      sources=list(data), strategy=_strategy(), segmenter=Segmenter(4, 4),
                      spec=OutputSpec.from_dict({"shard_length": 8, "short_policy": "drop",
                                                 "container": "npz",
                                                 "modalities": {"obs": {"dtype": "float32",
                                                                        "shape": [2],
                                                                        "source": "field:pos"}}}),
                      train_hours=1.0, span_steps=4)


# ---- 2026-07 follow-up review: contract validation and scale -----------------
def test_write_dataset_rejects_unknown_output_field_before_scoring(tmp_path):
    bad = OutputSpec.from_dict({"shard_length": 4, "modalities": {
        "obs": {"dtype": "float32", "shape": [], "source": "field:typo"}}})
    class NeverRead(MemAdapter):
        def ticks(self, src):
            raise AssertionError("validation must happen before data reads")
    with pytest.raises(ValueError, match="not declared"):
        write_dataset(str(tmp_path / "bad"), schema=SCHEMA,
                      adapter=NeverRead(SCHEMA, _data()), sources=["s1"],
                      strategy=_strategy(), segmenter=Segmenter(4, 4), spec=bad,
                      train_hours=1)


def test_output_field_shape_must_match_schema():
    bad = OutputSpec.from_dict({"shard_length": 4, "modalities": {
        "obs": {"dtype": "float32", "shape": [], "source": "field:pos"}}})
    with pytest.raises(ValueError, match="does not match"):
        bad.validate_schema(SCHEMA)


def test_hours_fill_has_no_implicit_group_cap():
    items = [{"id": i, "source_id": f"s{i}", "priority": 10 - i, "seconds": 10,
              "group": "A" if i < 9 else "B"} for i in range(10)]
    admitted, _ = fill_to_hours(items, train_hours=100 / 3600, test_hours=0)
    assert admitted == set(range(10))


def test_split_global_quota_survives_singleton_strata():
    from kuration.core.select import assign_splits
    split = assign_splits({f"s{i}": f"g{i}" for i in range(10)}, 0.2, 0)
    assert sum(v == "test" for v in split.values()) == 2


@pytest.mark.parametrize("mutator,match", [
    (lambda s: s["selection"].update(normalize="typo"), "normalize"),
    (lambda s: s.update(binning={"axes": [{"metric": "am", "bins": 0}]}), "bins"),
    (lambda s: s.update(filters=[{"type": "dedup", "thresholdd": 0.9}]), "unknown key"),
])
def test_strategy_rejects_silent_fallbacks(mutator, match):
    raw = {"define": {"am": {"source": "e.act", "reduce": "mean"}},
           "selection": {"method": "top-k", "weights": {"am": 1.0}}}
    mutator(raw)
    with pytest.raises(ValueError, match=match):
        load_strategy(raw, SCHEMA)


def test_strategy_rejects_scorer_output_collisions():
    raw = {"define": {"am": {"source": "e.act", "reduce": "mean"}},
           "scorers": [{"name": "action_residual", "as": "am", "obs": "pos",
                        "action": "act"}],
           "selection": {"method": "top-k", "weights": {"am": 1.0}}}
    with pytest.raises(ValueError, match="also appear"):
        load_strategy(raw, SCHEMA)


def test_array_adapters_reject_undeclared_globals():
    from kuration.core.adapters import HDF5Adapter, NpzAdapter, ZarrAdapter
    schema = SchemaSpec.from_dict({"name": "g", "fields": {"x": {"kind": "scalar"}},
                                   "globals": {"known": {"kind": "scalar"}}})
    for cls in (NpzAdapter, HDF5Adapter, ZarrAdapter):
        with pytest.raises(ValueError, match="undeclared schema global"):
            cls(schema, fields={"x": "x"}, globals={"typo": "g"})


def test_streaming_timeseries_uses_lazy_iterator(tmp_path):
    from kuration.core.adapters.timeseries import TimeseriesAdapter
    p = tmp_path / "data.csv"
    p.write_text("t,a,x,y\n" + "".join(f"{i},1,{i},0\n" for i in range(8)))
    class LazyOnly(TimeseriesAdapter):
        def ticks(self, source_uri):
            raise AssertionError("streaming path called materializing ticks()")
    ad = LazyOnly(SCHEMA, fields={"act": "a", "pos": ["x", "y"]}, time_column="t")
    from kuration.core.scoring import score_sources
    segs, _, _ = score_sources(schema=SCHEMA, adapter=ad, sources=[str(p)],
                               define=_strategy().define, registered=[],
                               segmenter=Segmenter(4, 4), streaming=True, on_error="raise")
    assert len(segs) == 2


def test_npz_streaming_rejected_loudly():
    from kuration.core.adapters import NpzAdapter
    from kuration.core.scoring import score_sources
    with pytest.raises(ValueError, match="cannot provide bounded-memory streaming"):
        score_sources(schema=SCHEMA, adapter=NpzAdapter(SCHEMA, fields={"act": "a"}),
                      sources=["unused.npz"], define={}, registered=[],
                      segmenter=Segmenter(4, 4), streaming=True)


def test_track_name_is_portable_and_bounded():
    from kuration.core.output.shardwriter import track_name
    name = track_name("C:\\bad?path/" + "x" * 400, "entity:*", "_span001")
    assert len(name) <= 109
    assert set(name) <= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def test_presence_mask_marks_entity_absence_in_memory_and_streaming(tmp_path):
    ticks = [Tick(time=i, entities=({0: {"pos": (float(i), 0.0), "act": 1.0}}
                                    if i not in (2, 3) else {})) for i in range(8)]
    data = {"s": ticks}
    common = dict(schema=SCHEMA, sources=["s"], strategy=_strategy(),
                  segmenter=Segmenter(4, 4), spec=SPEC, train_hours=1, test_hours=0)
    mem = write_dataset(str(tmp_path / "mem"), adapter=MemAdapter(SCHEMA, data), **common)
    stream = write_dataset(str(tmp_path / "stream"), adapter=MemAdapter(SCHEMA, data),
                           streaming=True, **common)
    for result in (mem, stream):
        first = np.load(result["shards"][0]["path"])
        assert first["_present"].tolist() == [True, True, False, False]


def test_presence_mask_can_be_disabled(tmp_path):
    spec = OutputSpec.from_dict({**SPEC.to_dict(), "presence_mask": False})
    entries = write_track(str(tmp_path), Track("s", 0, [{"pos": (0.0, 0.0)}]), spec,
                          schema=SCHEMA)
    assert "_present" not in np.load(entries[0]["path"]).files


def test_resume_rejects_changed_run_configuration(tmp_path):
    _write(tmp_path, out="resume", resume=False)
    changed = load_strategy({"name": "changed",
                             "define": {"am": {"source": "e.act", "reduce": "max"}},
                             "selection": {"method": "top-k", "weights": {"am": 1.0}}}, SCHEMA)
    data = _data()
    with pytest.raises(ValueError, match="different inputs or run configuration"):
        write_dataset(str(tmp_path / "resume"), schema=SCHEMA,
                      adapter=MemAdapter(SCHEMA, data), sources=list(data), strategy=changed,
                      segmenter=Segmenter(4, 4), spec=SPEC, train_hours=1, resume=True)


def test_budget_accounting_reports_emitted_underfill(tmp_path):
    data = _data(n_sources=1, n_ticks=6, n_entities=1)
    spec = OutputSpec.from_dict({"shard_length": 4, "short_policy": "drop",
                                 "modalities": {"obs": {"dtype": "float32", "shape": [2],
                                                          "source": "field:pos"}}})
    out = write_dataset(str(tmp_path / "budget"), schema=SCHEMA,
                        adapter=MemAdapter(SCHEMA, data), sources=list(data),
                        strategy=_strategy(), segmenter=Segmenter(4, 2), spec=spec,
                        train_hours=10 / 3600, test_hours=0)
    budget = out["results"]["budget"]
    assert budget["requested_seconds"]["train"] == 10
    assert budget["admitted_seconds"]["train"] == 6
    assert budget["emitted_seconds"]["train"] == 4
    assert budget["emission_loss_seconds"]["train"] == 2
    manifest = json.loads((tmp_path / "budget" / "dataset_manifest.json").read_text())
    assert manifest["budget"] == budget


# ---- 2026-07 full review: correctness, provenance, and loud output contracts ---------
@pytest.mark.parametrize("target", [0, -1, 1.5, True])
def test_select_rejects_invalid_target_size(target):
    from kuration import Segment, select
    segs = [Segment("s", i, i + 1, (i, i + 1), features={"am": float(i)})
            for i in range(3)]
    with pytest.raises(ValueError, match="target_size"):
        select(segs, _strategy(), target_size=target)


@pytest.mark.parametrize("train,test", [
    (float("nan"), 0), (float("inf"), 0), (-1, 0), (0, float("nan")),
])
def test_fill_rejects_invalid_budgets(train, test):
    items = [{"id": 0, "source_id": "s", "priority": 1.0,
              "seconds": 1.0, "group": None}]
    with pytest.raises(ValueError, match="hours"):
        fill_to_hours(items, train_hours=train, test_hours=test)


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_fill_rejects_invalid_item_seconds(seconds):
    items = [{"id": 0, "source_id": "s", "priority": 1.0,
              "seconds": seconds, "group": None}]
    with pytest.raises(ValueError, match="seconds"):
        fill_to_hours(items, train_hours=1, test_hours=0)


@pytest.mark.parametrize("kwargs,match", [
    ({"sample_hz": float("nan")}, "sample_hz"),
    ({"span_steps": -1}, "span_steps"),
    ({"span_seconds": float("inf")}, "span_seconds"),
    ({"span_steps": 4, "span_seconds": 4}, "only one"),
    ({"clip_aggregate": "median"}, "clip_aggregate"),
])
def test_write_dataset_rejects_invalid_numeric_options(tmp_path, kwargs, match):
    data = _data()
    with pytest.raises(ValueError, match=match):
        write_dataset(str(tmp_path / "bad"), schema=SCHEMA,
                      adapter=MemAdapter(SCHEMA, data), sources=list(data),
                      strategy=_strategy(), segmenter=Segmenter(4, 4), spec=SPEC,
                      train_hours=1, **kwargs)


def test_callable_provenance_includes_closure_state():
    from kuration.core.provenance import component_identity

    def make(value):
        return lambda source: (source, value)

    assert component_identity(make(1)) == component_identity(make(1))
    assert component_identity(make(1)) != component_identity(make(2))


def test_filter_uses_its_own_feature_space_without_strategy_default(monkeypatch):
    from types import SimpleNamespace
    import kuration.core.methods as methods
    from kuration import Segment

    seen = []
    monkeypatch.setitem(methods._FILTERS, "anomaly",
                        lambda segs, names, **kw: seen.append(names) or segs)
    strategy = SimpleNamespace(
        filters=[{"type": "anomaly", "feature_space": ["am"]}],
        feature_space=None, axes=[], method="top-k")
    segs = [Segment("s", 0, 1, (0, 1), features={"am": 1.0})]
    assert methods.apply_filters(segs, strategy) == segs
    assert seen == [["am"]]


def test_per_group_filter_accepts_heterogeneous_group_keys(monkeypatch):
    from types import SimpleNamespace
    import kuration.core.methods as methods
    from kuration import Segment

    monkeypatch.setitem(methods._FILTERS, "anomaly", lambda segs, names, **kw: segs)
    strategy = SimpleNamespace(
        filters=[{"type": "anomaly", "feature_space": ["am"], "per_group": True}],
        feature_space=None, axes=[], method="top-k")
    segs = [Segment("a", 0, 1, (0, 1), features={"am": 1.0}, group=1),
            Segment("b", 0, 1, (0, 1), features={"am": 2.0}, group="g")]
    assert methods.apply_filters(segs, strategy) == segs


def test_write_track_rejects_missing_present_field(tmp_path):
    entries_spec = OutputSpec.from_dict({"shard_length": 2, "modalities": {
        "act": {"dtype": "float32", "shape": [], "source": "field:act"}}})
    track = Track("s", 0, [{"pos": (0.0, 0.0)}])
    with pytest.raises(ValueError, match="missing value.*act"):
        write_track(str(tmp_path), track, entries_spec, schema=SCHEMA)


def test_write_track_rejects_missing_encoder_output(tmp_path):
    class MissingEncoder:
        def timestep(self, record, ctx):
            return {}

    spec = OutputSpec.from_dict({"shard_length": 2, "modalities": {
        "encoded": {"dtype": "float32", "shape": [], "source": "encoder:typo"}}})
    with pytest.raises(ValueError, match="missing value.*encoded"):
        write_track(str(tmp_path), Track("s", 0, [{"act": 1.0}]), spec,
                    encoder=MissingEncoder(), schema=SCHEMA)


def test_count_output_propagates_segment_features(tmp_path):
    from kuration import input
    rows = [{"t": i, "act": float(i), "pos": [float(i), 0.0]} for i in range(8)]
    result = (input(rows, time="t", hz=1)
              .window(4)
              .measure("am", "mean", of="act")
              .curate("top-k", n=1, weights={"am": 1.0})
              .output(tmp_path / "count", modalities={
                  "score": {"dtype": "float32", "shape": [], "source": "segment:am"}}))
    score = np.load(result["shards"][0]["path"])["score"]
    assert np.all(score == pytest.approx(5.5))


def test_hours_output_aggregates_segment_features_into_span_meta(tmp_path):
    data = _data(n_sources=1, n_ticks=8, n_entities=1)
    spec = OutputSpec.from_dict({"shard_length": 4, "modalities": {
        "score": {"dtype": "float32", "shape": [], "source": "segment:am"}}})
    out = write_dataset(str(tmp_path / "hours"), schema=SCHEMA,
                        adapter=MemAdapter(SCHEMA, data), sources=list(data),
                        strategy=_strategy(), segmenter=Segmenter(4, 4), spec=spec,
                        train_hours=1, span_steps=4, clip_aggregate="mean")
    assert out["shards"]
    for entry in out["shards"]:
        assert np.load(entry["path"])["score"].tolist() == [1.0] * entry["n_steps"]


def test_dataset_report_survives_a_near_constant_metric():
    """A metric varying only at float-noise scale must report as flat, not crash.

    NumPy widens the histogram range for exactly-constant data but raises for a spread too
    narrow to split, which took down `.review()` and `kur-report` for a near-flat field.
    """
    from kuration import Segment, dataset_report

    for values in ([6.0] * 20,                                    # exactly constant
                   [6.0, np.nextafter(6.0, 7.0)] * 10,            # adjacent floats
                   [6.0, 6.0 + 1e-15] * 10):                      # sub-epsilon spread
        segments = [Segment("s", i, i + 4, (i, i + 3), {"m": value})
                    for i, value in enumerate(values)]
        histogram = dataset_report(segments)["metrics"]["m"]["histogram"]
        assert sum(histogram["counts"]) == len(values)
        assert histogram["edges"][0] < histogram["edges"][-1]


def test_dataset_report_ignores_non_finite_metric_values():
    from kuration import Segment, dataset_report

    segments = [Segment("s", i, i + 4, (i, i + 3), {"m": float("nan")}) for i in range(8)]
    histogram = dataset_report(segments)["metrics"]["m"]["histogram"]
    assert sum(histogram["counts"]) == 0
    assert len(histogram["edges"]) == 11
