"""`check_source` — every built-in adapter's metadata-only preflight (no data rows read).

A mapping typo must fail at `kur-validate --check-sources` time with the offending source and
key named, instead of surfacing as every source skipped during a run.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from kuration import (
    CuratedDatasetAdapter,
    HDF5Adapter,
    HuggingFaceAdapter,
    JsonlAdapter,
    NpzAdapter,
    ParquetDatasetAdapter,
    SchemaSpec,
    TimeseriesAdapter,
)

SCHEMA = SchemaSpec.from_dict({
    "name": "d", "entity_axis": "e", "sample_hz": 1.0,
    "fields": {"pos": {"kind": "vector", "dims": ["x", "y"]}, "act": {"kind": "scalar"}},
})


# ---- csv (timeseries) --------------------------------------------------------
def test_csv_check_source(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("t,id,x,y,act\n0,e1,0.0,1.0,1\n")
    ok = TimeseriesAdapter(SCHEMA, fields={"pos": ["x", "y"], "act": "act"},
                           time_column="t", entity_column="id")
    ok.check_source(str(p))
    bad = TimeseriesAdapter(SCHEMA, fields={"pos": ["x", "y"], "act": "activity"},
                            time_column="t", entity_column="id")
    with pytest.raises(ValueError, match=r"missing mapped column\(s\).*activity"):
        bad.check_source(str(p))
    (tmp_path / "empty.csv").write_text("")
    with pytest.raises(ValueError, match="no header row"):
        ok.check_source(str(tmp_path / "empty.csv"))


# ---- jsonl -------------------------------------------------------------------
def test_jsonl_check_source(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"t": 0, "x": 0.0, "y": 1.0, "act": 1.0}) + "\n")
    ok = JsonlAdapter(SCHEMA, fields={"pos": ["x", "y"], "act": "act"}, time_column="t")
    ok.check_source(str(p))
    bad = JsonlAdapter(SCHEMA, fields={"pos": ["x", "z"], "act": "act"}, time_column="t")
    with pytest.raises(ValueError, match=r"missing mapped column\(s\).*z"):
        bad.check_source(str(p))
    (tmp_path / "empty.jsonl").write_text("\n\n")
    with pytest.raises(ValueError, match="no JSON lines"):
        ok.check_source(str(tmp_path / "empty.jsonl"))


# ---- npz ---------------------------------------------------------------------
def _npz(tmp_path, **arrays):
    p = tmp_path / "d.npz"
    np.savez(p, **arrays)
    return str(p)


def test_npz_check_source(tmp_path):
    p = _npz(tmp_path, xy=np.zeros((6, 2)), a=np.zeros(6))
    NpzAdapter(SCHEMA, fields={"pos": "xy", "act": "a"}).check_source(p)
    with pytest.raises(ValueError, match=r"missing mapped key\(s\).*axn"):
        NpzAdapter(SCHEMA, fields={"pos": "xy", "act": "axn"}).check_source(p)


def test_npz_check_source_catches_shape_problems(tmp_path):
    ragged = _npz(tmp_path, xy=np.zeros((6, 2)), a=np.zeros(9))       # step counts disagree
    with pytest.raises(ValueError, match="disagree on step count"):
        NpzAdapter(SCHEMA, fields={"pos": "xy", "act": "a"}).check_source(ragged)
    wide = _npz(tmp_path, xy=np.zeros((6, 3)), a=np.zeros(6))         # 3 cols into a 2-dim vec
    with pytest.raises(ValueError, match="per-step width 3"):
        NpzAdapter(SCHEMA, fields={"pos": "xy", "act": "a"}).check_source(wide)


# ---- hdf5 --------------------------------------------------------------------
def test_hdf5_check_source(tmp_path):
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "d.h5"
    with h5py.File(p, "w") as f:
        f["xy"] = np.zeros((6, 2))
        f["a"] = np.zeros(6)
    HDF5Adapter(SCHEMA, fields={"pos": "xy", "act": "a"}).check_source(str(p))
    with pytest.raises(ValueError, match=r"missing mapped key\(s\).*b"):
        HDF5Adapter(SCHEMA, fields={"pos": "xy", "act": "b"}).check_source(str(p))
    with pytest.raises(ValueError, match="per-step width 1"):
        HDF5Adapter(SCHEMA, fields={"pos": "a", "act": "a"}).check_source(str(p))


# ---- zarr --------------------------------------------------------------------
def test_zarr_check_source(tmp_path):
    zarr = pytest.importorskip("zarr")
    from kuration import ZarrAdapter
    store = str(tmp_path / "s.zarr")
    g = zarr.open(store, mode="w")
    g["px"] = np.zeros((6, 2))
    g["a"] = np.zeros(6)
    ZarrAdapter(SCHEMA, fields={"pos": "px", "act": "a"}).check_source(store)
    with pytest.raises(ValueError, match=r"missing mapped key\(s\).*nope"):
        ZarrAdapter(SCHEMA, fields={"pos": "px", "act": "nope"}).check_source(store)


# ---- parquet-dataset ---------------------------------------------------------
def _pq_part(path, cols=("t", "x", "y", "act")):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    pq.write_table(pa.table({c: [0.0, 1.0] for c in cols}), str(path))


def test_parquet_dataset_check_source(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    _pq_part(d / "part-0.parquet")
    _pq_part(d / "part-1.parquet", cols=("t", "x", "act"))            # one part lacks 'y'
    ad = ParquetDatasetAdapter(SCHEMA, path=str(d), source_per="dataset",
                               fields={"pos": ["x", "y"], "act": "act"}, time_column="t")
    assert ad.sources_are_paths is False        # the root/glob isn't a statable source path
    with pytest.raises(ValueError, match=r"part-1.parquet.*missing mapped column\(s\).*y"):
        ad.check_source(str(d))
    part = ParquetDatasetAdapter(SCHEMA, path=str(d),
                                 fields={"pos": ["x", "y"], "act": "act"}, time_column="t")
    assert part.sources_are_paths is True
    part.check_source(str(d / "part-0.parquet"))                      # per-part: inherited sniff


# ---- huggingface (cached-rows path — no network) ------------------------------
def test_hf_check_source_uses_cached_rows():
    ad = HuggingFaceAdapter(SCHEMA, dataset="org/name",
                            fields={"pos": ["x", "y"], "act": "act"},
                            time_column="t", source_column="ep")
    assert ad.sources_are_paths is False
    ad._rows = [{"t": 0, "x": 0.0, "y": 1.0, "act": 1.0, "ep": "ep1"}]
    ad.check_source("ep1")
    ad2 = HuggingFaceAdapter(SCHEMA, dataset="org/name",
                             fields={"pos": ["x", "y"], "act": "action"}, time_column="t")
    ad2._rows = list(ad._rows)
    with pytest.raises(ValueError, match=r"missing mapped column\(s\).*action"):
        ad2.check_source("org/name:train")


# ---- curated -----------------------------------------------------------------
def test_curated_check_source(tmp_path):
    from tests.test_sources import _make_dataset
    d = _make_dataset(tmp_path, [1, 2], "cur")
    ad = CuratedDatasetAdapter(d)
    shard = ad.sources()[0]
    ad.check_source(shard)
    with pytest.raises(KeyError, match="not in this adapter's manifests"):
        ad.check_source(str(tmp_path / "foreign_shard.npz"))
    import os
    os.remove(shard)                                                  # manifest now dangles
    with pytest.raises(ValueError, match="does not exist"):
        ad.check_source(shard)
