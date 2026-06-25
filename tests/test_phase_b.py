"""Phase B: adapters + generic build_window + segmentation + feature engine, end to end."""
from __future__ import annotations

import numpy as np
import pytest

from kuration import (
    FeatureEngine,
    SchemaSpec,
    Segmenter,
    TimeseriesAdapter,
    Tick,
    build_window,
    get_adapter,
)

SCHEMA = SchemaSpec.from_dict({
    "name": "demo", "entity_axis": "agent", "sample_hz": 10.0,
    "fields": {
        "pos": {"kind": "vector", "dims": ["x", "y"], "semantics": "position", "max_rate": 50},
        "temp": {"kind": "scalar"},
        "label": {"kind": "categorical"},
    },
})

# Two agents over 10 timesteps: agent0 slides +1 in x/step, agent1 +1 in y/step.
ROWS = [
    {"t": t, "agent": a, "x": (t if a == "0" else 0), "y": (t if a == "1" else 0),
     "temp": 20, "label": ("A" if a == "0" else "B")}
    for t in range(10) for a in ("0", "1")
]
FIELDS = {"pos": ["x", "y"], "temp": "temp", "label": "label"}


def _adapter():
    return TimeseriesAdapter(SCHEMA, fields=FIELDS, time_column="t", entity_column="agent")


# ---- adapter ----------------------------------------------------------------
def test_from_rows_groups_into_ticks():
    ticks = _adapter().from_rows(ROWS)
    assert len(ticks) == 10
    assert set(ticks[0].entities) == {"0", "1"}
    assert ticks[3].entities["0"]["pos"] == (3.0, 0.0)   # vector assembled from x,y cols
    assert ticks[3].entities["1"]["pos"] == (0.0, 3.0)
    assert ticks[0].entities["0"]["label"] == "A"


def test_no_time_column_is_one_tick_per_row():
    a = TimeseriesAdapter(SCHEMA, fields=FIELDS, entity_column="agent")  # no time_column
    ticks = a.from_rows(ROWS)
    assert len(ticks) == len(ROWS)          # each row its own timestep
    assert all(len(t.entities) == 1 for t in ticks)


def test_csv_roundtrip(tmp_path):
    import csv
    p = tmp_path / "data.csv"
    with p.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["t", "agent", "x", "y", "temp", "label"])
        wr.writeheader()
        wr.writerows(ROWS)
    ticks = _adapter().ticks(str(p))
    assert len(ticks) == 10 and ticks[5].entities["0"]["pos"] == (5.0, 0.0)


def test_field_map_validates_against_schema():
    with pytest.raises(ValueError):
        TimeseriesAdapter(SCHEMA, fields={"nope": "x"})


def test_registry():
    a = get_adapter("timeseries", schema=SCHEMA, fields=FIELDS, time_column="t",
                    entity_column="agent")
    assert isinstance(a, TimeseriesAdapter)


# ---- generic build_window ---------------------------------------------------
def test_build_window_roster_and_validity():
    ticks = [
        Tick(time=0, entities={"a": {"temp": 1.0}}),
        Tick(time=1, entities={"a": {"temp": 2.0}, "b": {"temp": 9.0}}),  # b appears mid-window
    ]
    w = build_window(SCHEMA, ticks)
    assert w.T == 2 and w.E == 2
    assert w.valid[0, 0] and not w.valid[0, 1] and w.valid[1, 1]   # b absent at t0
    assert np.isnan(w.numeric["temp"][0, 1, 0])                    # absent cell is NaN
    assert w.numeric["temp"][1, 1, 0] == 9.0


def test_build_window_categorical_coding():
    ticks = _adapter().from_rows(ROWS)
    w = build_window(SCHEMA, ticks)
    codes, vocab = w.categorical_field("label")
    assert set(vocab) == {"A", "B"}


# ---- segmenter --------------------------------------------------------------
def test_segmenter_steps():
    ticks = _adapter().from_rows(ROWS)
    segs = Segmenter(window=4, stride=2).segments("src", ticks)
    assert [(s.start_idx, s.end_idx) for s in segs] == [(0, 4), (2, 6), (4, 8), (6, 10)]
    assert segs[0].time_range == (0.0, 3.0)


def test_segmenter_seconds_uses_sample_hz():
    ticks = _adapter().from_rows(ROWS)
    segs = Segmenter(window=0.4, stride=0.2, unit="seconds", sample_hz=SCHEMA.sample_hz) \
        .segments("src", ticks)
    assert segs[0].end_idx - segs[0].start_idx == 4   # 0.4s * 10hz
    assert len(segs) == 4


def test_segmenter_seconds_requires_hz():
    with pytest.raises(ValueError):
        Segmenter(window=0.4, stride=0.2, unit="seconds")


# ---- end to end -------------------------------------------------------------
def test_end_to_end_rows_to_features():
    ticks = _adapter().from_rows(ROWS)
    segs = Segmenter(window=4, stride=2).segments("src", ticks)
    eng = FeatureEngine(SCHEMA, define={
        "xmean": {"source": "agent.pos.x", "reduce": "mean"},
        "speed": {"source": "agent.pos", "map": "speed", "reduce": "mean"},
        "labels": {"source": "agent.label", "reduce": "nunique"},
    })
    eng.compute(ticks, segs)
    # speed: each agent moves 1 unit/step * 10hz = 10 (both axes), pooled mean = 10
    assert segs[0].features["speed"] == pytest.approx(10.0)
    # first window x: agent0 0,1,2,3 ; agent1 0,0,0,0 -> mean 6/8
    assert segs[0].features["xmean"] == pytest.approx(0.75)
    assert segs[0].features["labels"] == pytest.approx(2.0)
    assert all("speed" in s.features for s in segs)
