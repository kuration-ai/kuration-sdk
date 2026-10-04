"""Schema inference: sniff / load_rows / sample_source / infer_schema."""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from kuration.core.buffer import Tick
from kuration.core.infer import infer_schema, load_rows, sample_source, sniff


def _rows(n=6):
    rows = []
    for i in range(n):
        for ep in ("a", "b"):
            rows.append({"time": i * 0.5, "entity": ep, "x": float(i), "count": i,
                         "on": bool(i % 2), "mode": "run" if i % 2 else "walk",
                         "pos": [float(i), float(i) + 1.0, 0.0]})
    return rows


# ---- rows: classification + hints -------------------------------------------
def test_rows_classification_and_hints():
    schema, hints = infer_schema(_rows())
    assert schema.fields["x"].kind == "scalar"
    assert schema.fields["count"].kind == "scalar"
    assert schema.fields["on"].kind == "categorical"          # bool -> categorical
    assert schema.fields["mode"].kind == "categorical"
    assert schema.fields["pos"].kind == "vector"
    assert schema.fields["pos"].dims == ("d0", "d1", "d2")
    assert hints["fields"]["x"] == "x"
    assert hints["fields"]["pos"] == ["pos.0", "pos.1", "pos.2"]
    assert hints["list_columns"] == {"pos": 3}
    # time/entity detected by name and excluded from fields
    assert hints["time_column"] == "time" and hints["entity_column"] == "entity"
    assert "time" not in schema.fields and "entity" not in schema.fields
    assert schema.entity_axis == "entity"


def test_rows_explicit_time_and_entity():
    rows = [{"stamp": i, "who": "a", "x": float(i)} for i in range(4)]
    schema, hints = infer_schema(rows, time="stamp", entity="who")
    assert hints["time_column"] == "stamp" and hints["entity_column"] == "who"
    assert set(schema.fields) == {"x"}
    assert schema.entity_axis == "who"                        # named after the entity column


def test_rows_explicit_time_missing_raises():
    with pytest.raises(ValueError) as e:
        infer_schema([{"x": 1.0}], time="stamp")
    assert "stamp" in str(e.value)


def test_rows_entity_axis_default():
    schema, hints = infer_schema([{"t": i, "x": float(i)} for i in range(3)])
    assert schema.entity_axis == "entity" and hints["entity_column"] is None


# ---- rows: hz ----------------------------------------------------------------
def test_hz_from_repeated_multi_entity_times():
    schema, _ = infer_schema(_rows())                          # times repeat per entity
    assert schema.sample_hz == pytest.approx(2.0)              # dt=0.5 -> 2 Hz


def test_explicit_hz_wins():
    with pytest.warns(UserWarning, match="does not match the sample's own timestamps"):
        schema, _ = infer_schema(_rows(), hz=64.0)
    assert schema.sample_hz == 64.0


def test_explicit_hz_matching_timestamps_does_not_warn():
    # _rows() timestamps are 0.5s apart (2Hz) -- a declared hz of 2.0 should raise no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        schema, _ = infer_schema(_rows(), hz=2.0)
    assert schema.sample_hz == 2.0


def test_explicit_hz_mismatch_warning_names_the_observed_rate():
    """F18: a declared hz= that contradicts the actual timestamp deltas was accepted
    silently -- 25Hz data declared as 1Hz ran clean and threw off any hours budget by 25x.
    The warning now names both rates and the relative gap."""
    rows = [{"time": i * 0.04, "x": float(i)} for i in range(10)]  # 0.04s apart == 25Hz
    with pytest.warns(UserWarning, match=r"declared hz=1 does not match.*~25") as record:
        infer_schema(rows, hz=1.0)
    assert len(record) == 1


def test_hz_none_without_time_column():
    schema, hints = infer_schema([{"x": float(i)} for i in range(4)])
    assert schema.sample_hz is None and hints["time_column"] is None


def test_hz_none_nonnumeric_time():
    rows = [{"time": f"2026-07-{i:02d}", "x": float(i)} for i in range(1, 5)]
    schema, hints = infer_schema(rows)
    assert hints["time_column"] == "time" and schema.sample_hz is None


def test_hz_none_single_unique_time():
    rows = [{"t": 0.0, "entity": e, "x": 1.0} for e in "ab"]
    schema, _ = infer_schema(rows)
    assert schema.sample_hz is None


# ---- rows: kinds= override ----------------------------------------------------
def test_kinds_override_numeric_to_categorical():
    rows = [{"t": i, "lvl": i % 3, "x": float(i)} for i in range(6)]
    schema, hints = infer_schema(rows, kinds={"lvl": "categorical"})
    assert schema.fields["lvl"].kind == "categorical"
    assert hints["fields"]["lvl"] == "lvl"


def test_kinds_rescues_mixed_column():
    rows = [{"v": 1.0 if i % 2 else "off"} for i in range(4)]
    schema, _ = infer_schema(rows, kinds={"v": "categorical"})
    assert schema.fields["v"].kind == "categorical"


def test_kinds_vector_not_overridable():
    with pytest.raises(ValueError) as e:
        infer_schema([{"x": 1.0}], kinds={"x": "vector"})
    assert "vector" in str(e.value)


def test_kinds_unknown_column_raises():
    with pytest.raises(ValueError) as e:
        infer_schema([{"x": 1.0}], kinds={"nope": "scalar"})
    assert "nope" in str(e.value)


def test_kinds_cannot_override_list_column():
    with pytest.raises(ValueError) as e:
        infer_schema([{"p": [1.0, 2.0]}], kinds={"p": "scalar"})
    assert "p" in str(e.value) and "schema" in str(e.value)


# ---- rows: loud failures -------------------------------------------------------
def test_dotted_column_raises():
    with pytest.raises(ValueError) as e:
        infer_schema([{"a.b": 1.0}])
    assert "a.b" in str(e.value)


def test_mixed_type_column_raises():
    with pytest.raises(ValueError) as e:
        infer_schema([{"v": 1.0}, {"v": "x"}])
    assert "v" in str(e.value) and "kinds=" in str(e.value)


def test_ragged_vector_lengths_raise():
    with pytest.raises(ValueError) as e:
        infer_schema([{"p": [1.0, 2.0]}, {"p": [1.0]}])
    assert "p" in str(e.value)


def test_all_none_column_raises():
    with pytest.raises(ValueError) as e:
        infer_schema([{"t": 0, "v": None}, {"t": 1, "v": None}])
    assert "v" in str(e.value)


def test_unsupported_cell_type_raises():
    with pytest.raises(ValueError) as e:
        infer_schema([{"v": {"nested": 1}}])
    assert "v" in str(e.value)


def test_empty_sample_raises():
    with pytest.raises(ValueError):
        infer_schema([])
    with pytest.raises(ValueError):
        infer_schema({})


# ---- load_rows -----------------------------------------------------------------
def test_load_rows_array(tmp_path):
    p = tmp_path / "d.json"
    rows = [{"t": 0, "x": 1.0}, {"t": 1, "x": 2.0}]
    p.write_text(json.dumps(rows))
    assert load_rows(str(p)) == rows


def test_load_rows_columnar(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"t": [0, 1], "x": [1.0, 2.0]}))
    assert load_rows(str(p)) == [{"t": 0, "x": 1.0}, {"t": 1, "x": 2.0}]


def test_load_rows_scalar_doc_raises(tmp_path):
    p = tmp_path / "b.json"
    p.write_text("42")
    with pytest.raises(ValueError) as e:
        load_rows(str(p))
    assert "expected a JSON array of row objects or a columnar dict of equal-length lists" \
        in str(e.value)


def test_load_rows_ragged_columns_raise(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"t": [0, 1], "x": [1.0]}))
    with pytest.raises(ValueError) as e:
        load_rows(str(p))
    assert "ragged" in str(e.value)


def test_load_rows_non_dict_element_raises(tmp_path):
    p = tmp_path / "l.json"
    p.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError) as e:
        load_rows(str(p))
    assert "expected a JSON array of row objects" in str(e.value)


# ---- sample_source ---------------------------------------------------------------
def test_sample_source_json_array(tmp_path):
    p = tmp_path / "d.json"
    rows = [{"t": i, "x": float(i)} for i in range(5)]
    p.write_text(json.dumps(rows))
    kind, sample = sample_source(str(p), n=3)
    assert kind == "json-rows" and sample == rows[:3]


def test_sample_source_json_columnar(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"t": [0, 1], "x": [1.0, 2.0]}))
    kind, sample = sample_source(str(p))
    assert kind == "json-rows" and sample == [{"t": 0, "x": 1.0}, {"t": 1, "x": 2.0}]


def test_sample_source_csv_coerces_numerics(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("t,x,mode\n0,1.5,run\n1,2.5,walk\n")
    kind, sample = sample_source(str(p))
    assert kind == "timeseries"
    assert sample[0] == {"t": 0.0, "x": 1.5, "mode": "run"}    # numbers coerced, strings kept
    schema, hints = infer_schema(sample)
    assert schema.fields["x"].kind == "scalar"
    assert schema.fields["mode"].kind == "categorical"
    assert hints["time_column"] == "t"
    assert schema.sample_hz == pytest.approx(1.0)


def test_sample_source_csv_respects_n(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("t,x\n" + "".join(f"{i},{i}.0\n" for i in range(50)))
    _, sample = sample_source(str(p), n=7)
    assert len(sample) == 7


def test_sample_source_jsonl(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"t": 0, "x": 1.0}\n\n{"t": 1, "x": 2.0}\n{"t": 2, "x": 3.0}\n')
    kind, sample = sample_source(str(p), n=2)
    assert kind == "jsonl" and sample == [{"t": 0, "x": 1.0}, {"t": 1, "x": 2.0}]


def test_sample_source_npz(tmp_path):
    p = tmp_path / "d.npz"
    np.savez(p, x=np.arange(30, dtype=np.float64), pos=np.zeros((30, 2)))
    kind, sample = sample_source(str(p), n=10)
    assert kind == "npz"
    assert sample["x"].shape == (10,) and sample["pos"].shape == (10, 2)


def test_sample_source_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    p = tmp_path / "d.parquet"
    pq.write_table(pa.table({"t": [0, 1, 2], "x": [1.0, 2.0, 3.0]}), str(p))
    kind, sample = sample_source(str(p), n=2)
    assert kind == "timeseries" and sample == [{"t": 0, "x": 1.0}, {"t": 1, "x": 2.0}]


def test_sample_source_parquet_dataset(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    d = tmp_path / "ds"
    (d / "sub").mkdir(parents=True)
    for k in range(2):
        pq.write_table(pa.table({"t": [0, 1], "x": [float(k), float(k)]}),
                       str(d / "sub" / f"part-{k}.parquet"))
    kind, sample = sample_source(str(d), n=5)
    assert kind == "parquet-dataset"
    assert sample == [{"t": 0, "x": 0.0}, {"t": 1, "x": 0.0}]  # first part (sorted)


def test_sample_source_hdf5(tmp_path):
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "d.h5"
    with h5py.File(p, "w") as f:
        f["x"] = np.arange(20, dtype=np.float64)
        f["pos"] = np.ones((20, 3))
    kind, sample = sample_source(str(p), n=4)
    assert kind == "hdf5"
    assert sample["x"].shape == (4,) and sample["pos"].shape == (4, 3)


# ---- dict-of-arrays -----------------------------------------------------------
def test_arrays_scalar_vector_categorical():
    sample = {"x": np.arange(5, dtype=np.float64), "pos": np.zeros((5, 2)),
              "mode": np.array(["a", "b", "a", "b", "a"])}
    schema, hints = infer_schema(sample, hz=10.0)
    assert schema.fields["x"].kind == "scalar"
    assert schema.fields["pos"].kind == "vector" and schema.fields["pos"].dims == ("d0", "d1")
    assert schema.fields["mode"].kind == "categorical"
    assert schema.sample_hz == 10.0
    assert hints["fields"] == {"x": "x", "pos": "pos", "mode": "mode"}
    assert hints["time_column"] is None and hints["entity_column"] is None
    assert hints["list_columns"] == {}


def test_arrays_hz_none_without_hz_kwarg():
    schema, _ = infer_schema({"x": np.arange(5, dtype=np.float64)})
    assert schema.sample_hz is None


def test_arrays_ndim3_raises():
    with pytest.raises(ValueError) as e:
        infer_schema({"img": np.zeros((3, 4, 5))})
    assert "img" in str(e.value) and "schema" in str(e.value)


# ---- list[Tick] -----------------------------------------------------------------
def test_ticks_sample():
    ticks = [Tick(time=i * 0.25,
                  entities={0: {"x": float(i), "pos": (1.0, 2.0), "mode": "a"}})
             for i in range(8)]
    schema, hints = infer_schema(ticks)
    assert schema.fields["x"].kind == "scalar"
    assert schema.fields["pos"].kind == "vector" and schema.fields["pos"].dims == ("d0", "d1")
    assert schema.fields["mode"].kind == "categorical"
    assert schema.entity_axis == "entity"
    assert schema.sample_hz == pytest.approx(4.0)
    assert hints["fields"] == {"x": "x", "pos": "pos", "mode": "mode"}
    assert hints["time_column"] is None and hints["list_columns"] == {}


def test_ticks_explicit_hz_and_conflict():
    ticks = [Tick(time=i, entities={0: {"x": float(i)}}) for i in range(4)]
    with pytest.warns(UserWarning, match="does not match the sample's own timestamps"):
        schema, _ = infer_schema(ticks, hz=100.0)
    assert schema.sample_hz == 100.0
    bad = [Tick(time=0, entities={0: {"x": 1.0}}), Tick(time=1, entities={0: {"x": "s"}})]
    with pytest.raises(ValueError) as e:
        infer_schema(bad)
    assert "x" in str(e.value)


# ---- sniff -----------------------------------------------------------------------
@pytest.mark.parametrize("name,kind", [
    ("a.csv", "timeseries"), ("a.parquet", "timeseries"), ("a.jsonl", "jsonl"),
    ("a.npz", "npz"), ("a.h5", "hdf5"), ("a.hdf5", "hdf5"), ("a.json", "json-rows"),
    ("A.CSV", "timeseries"),
])
def test_sniff_extensions(name, kind):
    assert sniff(name) == kind


def test_sniff_parquet_dataset_dir(tmp_path):
    d = tmp_path / "ds"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "part-0.parquet").write_bytes(b"")
    assert sniff(str(d)) == "parquet-dataset"


def test_sniff_unknown_extension_raises():
    with pytest.raises(ValueError) as e:
        sniff("data.xyz")
    assert "adapter=" in str(e.value) and ".csv" in str(e.value)


def test_sniff_dir_without_parquet_raises(tmp_path):
    with pytest.raises(ValueError) as e:
        sniff(str(tmp_path))
    assert "adapter=" in str(e.value)
