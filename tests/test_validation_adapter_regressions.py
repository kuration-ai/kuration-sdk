"""Regression coverage for strict schema/DSL validation and adapter suffix handling."""
from __future__ import annotations

import pytest

from kuration.core.adapters.timeseries import TimeseriesAdapter
from kuration.core.dsl import compile_metric
from kuration.core.schema import FieldSpec, SchemaSpec


def test_speed_requires_sample_hz_at_compile_time():
    schema = SchemaSpec.from_dict({
        "name": "motion",
        "entity_axis": "agent",
        "fields": {
            "position": {
                "kind": "vector",
                "dims": ["x", "y"],
                "semantics": "position",
            },
        },
    })

    with pytest.raises(ValueError, match=r"speed.*sample_hz"):
        compile_metric(
            {"source": "agent.position", "map": "speed", "reduce": "mean"},
            schema,
        )


@pytest.mark.parametrize("literal", ["nan", "NaN", "inf", "+inf", "-inf", "Infinity"])
def test_numeric_filter_rejects_non_finite_literals_at_compile_time(literal):
    schema = SchemaSpec.from_dict({
        "name": "measurements",
        "entity_axis": "sensor",
        "fields": {"value": {"kind": "scalar"}},
    })

    with pytest.raises(ValueError, match=r"filter.*finite"):
        compile_metric(
            {"source": "sensor.value", "filter": f"value>{literal}", "reduce": "mean"},
            schema,
        )


@pytest.mark.parametrize("bad_value", [True, False, 0, -1, 1.0, "1"])
@pytest.mark.parametrize("from_dict", [False, True], ids=["direct", "from-dict"])
def test_bitmask_values_are_exact_positive_non_bool_ints(bad_value, from_dict):
    if from_dict:
        construct = lambda: SchemaSpec.from_dict({
            "name": "buttons",
            "fields": {
                "buttons": {
                    "kind": "scalar",
                    "semantics": "bitmask",
                    "bits": {"FIRE": bad_value},
                },
            },
        })
    else:
        construct = lambda: FieldSpec(
            "buttons",
            "scalar",
            semantics="bitmask",
            bits={"FIRE": bad_value},
        )

    with pytest.raises(ValueError, match="positive integer"):
        construct()


def test_positive_integer_bitmask_values_are_preserved():
    direct = FieldSpec(
        "buttons",
        "scalar",
        semantics="bitmask",
        bits={"FIRE": 3},
    )
    loaded = SchemaSpec.from_dict({
        "name": "buttons",
        "fields": {
            "buttons": {
                "kind": "scalar",
                "semantics": "bitmask",
                "bits": {"FIRE": 3},
            },
        },
    })

    assert direct.bits["FIRE"] == 3
    assert type(direct.bits["FIRE"]) is int
    assert loaded.fields["buttons"].bits["FIRE"] == 3
    assert type(loaded.fields["buttons"].bits["FIRE"]) is int


def test_mixed_case_parquet_suffix_is_consistent_across_adapter_paths(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "measurements.PaRqUeT"
    pq.write_table(pa.table({"time": [0, 1], "value": [1.5, 2.5]}), source)
    schema = SchemaSpec.from_dict({
        "name": "measurements",
        "fields": {"value": {"kind": "scalar"}},
    })
    adapter = TimeseriesAdapter(schema, fields={"value": "value"}, time_column="time")

    assert adapter.detect_source(str(source)) == 80
    adapter.check_source(str(source))
    ticks = adapter.ticks(str(source))

    assert [tick.time for tick in ticks] == [0, 1]
    assert [tick.entities[0]["value"] for tick in ticks] == [1.5, 2.5]


def test_nonnumeric_time_column_warns_once():
    """F22: a text time column (e.g. ISO timestamps) was silently accepted and used for plain
    string grouping/ordering -- hz-based or seconds-unit windowing then fails far from here, or
    silently misorders non-zero-padded strings. Warn once per ticks() call instead."""
    schema = SchemaSpec.from_dict({"name": "s", "fields": {"value": {"kind": "scalar"}}})
    adapter = TimeseriesAdapter(schema, fields={"value": "value"}, time_column="time")
    rows = [{"time": f"2026-07-0{i}T00:00:00", "value": float(i)} for i in range(1, 4)]

    with pytest.warns(UserWarning, match="time column 'time' is not numeric") as record:
        ticks = list(adapter.from_rows(rows))
    assert len(record) == 1  # one warning per ticks() call, not once per row
    assert [tick.time for tick in ticks] == [row["time"] for row in rows]
