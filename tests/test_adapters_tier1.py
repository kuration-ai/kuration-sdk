"""Tier-1 input adapters: partitioned Parquet, HuggingFace, Zarr."""
from __future__ import annotations

import numpy as np
import pytest

from kuration import (
    HuggingFaceAdapter,
    ParquetDatasetAdapter,
    SchemaSpec,
    ZarrAdapter,
    get_adapter,
)

SCHEMA = SchemaSpec.from_dict({
    "name": "d", "entity_axis": "e", "sample_hz": 1.0,
    "fields": {"pos": {"kind": "vector", "dims": ["x", "y"]}, "act": {"kind": "scalar"}},
})


# ---- partitioned Parquet ----------------------------------------------------
def _write_part(path, *, traj, n=6):
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pa.table({"t": list(range(n)), "ep": [f"ep{traj}"] * n,
                  "x": [float(i) for i in range(n)],
                  "y": [float(traj)] * n, "act": [float(traj)] * n})
    pq.write_table(t, str(path))


def _pq_dataset(tmp_path, n_parts=3):
    d = tmp_path / "ds"
    (d / "sub").mkdir(parents=True)
    for k in range(n_parts):
        _write_part(d / "sub" / f"part-{k}.parquet", traj=k + 1)
    return d


def test_parquet_dataset_source_per_part(tmp_path):
    d = _pq_dataset(tmp_path, n_parts=3)
    ad = ParquetDatasetAdapter(SCHEMA, path=str(d),
                               fields={"pos": ["x", "y"], "act": "act"}, time_column="t")
    srcs = ad.sources()
    assert len(srcs) == 3 and all(s.endswith(".parquet") for s in srcs)
    ticks = ad.ticks(srcs[0])
    assert len(ticks) == 6
    assert ticks[0].entities[0]["pos"] == (0.0, 1.0)     # x=step, y=traj id (part-0 -> traj 1)


def test_parquet_dataset_source_per_dataset(tmp_path):
    d = _pq_dataset(tmp_path, n_parts=3)
    ad = ParquetDatasetAdapter(SCHEMA, path=str(d), source_per="dataset",
                               fields={"pos": ["x", "y"], "act": "act"}, time_column="t",
                               entity_column="ep")
    assert ad.sources() == [str(d)]
    ticks = ad.ticks(str(d))
    # 3 parts x 6 steps; rows sorted by time, equal-time rows from different parts share a tick
    assert len(ticks) == 6
    assert len(ticks[0].entities) == 3                   # one entity per part, sharing the tick


def test_parquet_dataset_colliding_rows_raise(tmp_path):
    # dataset mode without an entity_column: equal-time rows from different parts would
    # silently last-write-win — must raise instead.
    d = _pq_dataset(tmp_path, n_parts=2)
    ad = ParquetDatasetAdapter(SCHEMA, path=str(d), source_per="dataset",
                               fields={"pos": ["x", "y"], "act": "act"}, time_column="t")
    with pytest.raises(ValueError, match="duplicate row"):
        ad.ticks(str(d))


def test_parquet_dataset_via_get_adapter_and_empty(tmp_path):
    ad = get_adapter("parquet-dataset", schema=SCHEMA, path=str(tmp_path / "empty"),
                     fields={"act": "act"})
    with pytest.raises(ValueError, match="no .parquet parts"):
        ad.sources()


def test_parquet_dataset_bad_source_per(tmp_path):
    with pytest.raises(ValueError, match="source_per"):
        ParquetDatasetAdapter(SCHEMA, path=str(tmp_path), source_per="nope",
                              fields={"act": "act"})


# ---- HuggingFace (adapter logic, no `datasets` needed) ----------------------
def _hf_rows():
    return [{"t": i, "x": float(i), "y": float(traj), "act": float(traj), "ep": f"ep{traj}"}
            for traj in (1, 2) for i in range(4)]


def test_hf_source_column_splits_sources():
    ad = HuggingFaceAdapter(SCHEMA, dataset="dummy", fields={"pos": ["x", "y"], "act": "act"},
                            time_column="t", source_column="ep")
    ad._rows = _hf_rows()                                # inject; bypasses load_dataset
    assert ad.sources() == ["ep1", "ep2"]
    ticks = ad.ticks("ep2")
    assert len(ticks) == 4 and ticks[0].entities[0]["act"] == 2.0


def test_hf_single_source_default():
    ad = HuggingFaceAdapter(SCHEMA, dataset="org/name", split="train",
                            fields={"act": "act"}, time_column="t")
    ad._rows = [r for r in _hf_rows() if r["ep"] == "ep1"]   # one time-ordered trajectory
    assert ad.sources() == ["org/name:train"]
    assert len(ad.ticks("org/name:train")) == 4              # 4 distinct, ordered time values


def test_hf_real_dataset_object(monkeypatch):
    datasets = pytest.importorskip("datasets")
    dataset = datasets.Dataset.from_list(_hf_rows())
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: dataset)
    ad = HuggingFaceAdapter(SCHEMA, dataset="local-test",
                            fields={"pos": ["x", "y"], "act": "act"},
                            time_column="t", source_column="ep")
    assert ad.sources() == ["ep1", "ep2"]
    ticks = ad.ticks("ep2")
    assert len(ticks) == 4
    assert ticks[-1].entities[0]["pos"] == (3.0, 2.0)


def test_hf_missing_lib_is_loud(monkeypatch):
    # source_column set -> sources() must load (no cache), and absent `datasets` -> loud hint
    import sys
    ad = HuggingFaceAdapter(SCHEMA, dataset="x", fields={"act": "act"}, source_column="ep")
    monkeypatch.setitem(sys.modules, "datasets", None)   # force ImportError on `from datasets ...`
    with pytest.raises(ImportError, match=r"\[hf\]"):
        ad.sources()


# ---- Zarr (needs the lib) ---------------------------------------------------
def test_zarr_init_validates_fields():
    with pytest.raises(ValueError, match="undeclared schema field"):
        ZarrAdapter(SCHEMA, fields={"nope": "k"})


def test_zarr_roundtrip(tmp_path):
    zarr = pytest.importorskip("zarr")
    store = str(tmp_path / "s.zarr")
    g = zarr.open(store, mode="w")
    g["px"] = np.stack([np.arange(6.0), np.full(6, 3.0)], axis=1)   # (6, 2) -> pos
    g["a"] = np.arange(6.0)
    ad = ZarrAdapter(SCHEMA, fields={"pos": "px", "act": "a"})
    ticks = ad.ticks(store)
    assert len(ticks) == 6
    assert ticks[0].entities[0]["pos"] == (0.0, 3.0) and ticks[5].entities[0]["act"] == 5.0
