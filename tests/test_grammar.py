"""compile_metric over a synthetic WindowBuffer with hand-computed expected values.

The buffer (T=4, E=2 agents, sample_hz=2):
  agent0: pos.x = 0,1,2,3 (moving +1/tick), y=z=0; ang.yaw = 0,10,20,30; hp=100;
          btn FIRE(bit1) on ticks 0,1 else off; weapon "A"
  agent1: pos = (10,0,0) stationary; ang constant; hp=50; FIRE never; weapon "B"
  events: tick0 [type=x]; tick2 [type=y, type=x]
  global alive = [2,2,2,2]
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from kuration.core.buffer import empty_like
from kuration.core.dsl import compile_metric
from kuration.core.schema import SchemaSpec

SCHEMA = SchemaSpec.from_dict({
    "name": "demo", "entity_axis": "agent", "sample_hz": 2.0,
    "fields": {
        "pos": {"kind": "vector", "dims": ["x", "y", "z"], "semantics": "position", "max_rate": 100},
        "ang": {"kind": "vector", "dims": ["yaw", "pitch"], "semantics": "angular"},
        "hp": {"kind": "scalar"},
        "btn": {"kind": "scalar", "semantics": "bitmask", "bits": {"FIRE": 1, "JUMP": 2}},
        "weapon": {"kind": "categorical"},
    },
    "event_streams": {"event": {"fields": {"type": {"kind": "categorical"}}}},
    "globals": {"alive": {"kind": "scalar"}},
})


def _buffer():
    w = empty_like(SCHEMA, T=4, E=2)
    w.valid[:] = True
    for t in range(4):
        w.numeric["pos"][t, 0] = (t, 0, 0)
        w.numeric["pos"][t, 1] = (10, 0, 0)
        w.numeric["ang"][t, 0] = (10 * t, 0)
        w.numeric["ang"][t, 1] = (0, 0)
        w.numeric["hp"][t, 0, 0] = 100
        w.numeric["hp"][t, 1, 0] = 50
        w.numeric["btn"][t, 0, 0] = 1 if t < 2 else 0
        w.numeric["btn"][t, 1, 0] = 0
    w.codes["weapon"][:, 0] = 0
    w.codes["weapon"][:, 1] = 1
    w.vocabs["weapon"] = ["A", "B"]
    w.events["event"][0] = [{"type": "x"}]
    w.events["event"][2] = [{"type": "y"}, {"type": "x"}]
    w.globals["alive"][:] = 2
    return w


def m(spec):
    return compile_metric(spec, SCHEMA)(_buffer())


# ---- scalar / vector / maps -------------------------------------------------
def test_scalar_mean_pooled():
    # x values: agent0 0,1,2,3 ; agent1 10,10,10,10 -> mean = 46/8
    assert m({"source": "agent.pos.x", "reduce": "mean"}) == pytest.approx(46 / 8)


def test_speed_pooled_and_scaled():
    # agent0 moves 1/tick * sample_hz(2) = 2 ; agent1 0. 3 steps each -> pooled mean 1.0
    assert m({"source": "agent.pos", "map": "speed", "reduce": "mean"}) == pytest.approx(1.0)


def test_speed_per_entity_scope():
    # agent0 mean 2, agent1 mean 0 -> mean over entities 1.0
    assert m({"source": "agent.pos", "map": "speed", "reduce": "mean",
              "scope": "per_entity_mean"}) == pytest.approx(1.0)


def test_angle_delta_mean():
    # agent0 |dyaw|=10/step (3 steps); agent1 0 -> pooled mean over 6 values = 5
    assert m({"source": "agent.ang", "map": "angle_delta", "reduce": "mean"}) == pytest.approx(5.0)


def test_percentile_and_jerk():
    # x pooled sorted [0,1,2,3,10,10,10,10] -> median 6.5
    assert m({"source": "agent.pos.x", "reduce": "p50"}) == pytest.approx(6.5)
    # jerk: agent0 |dx|=1 mean 1 ; agent1 0 -> mean 0.5
    assert m({"source": "agent.pos.x", "reduce": "jerk"}) == pytest.approx(0.5)


def test_bitmask_rate():
    # FIRE on for agent0 ticks 0,1 -> pooled fraction nonzero = 2/8 = 0.25
    assert m({"source": "agent.btn", "map": "bit:FIRE", "reduce": "rate"}) == pytest.approx(0.25)


def test_filter_scalar_comparator():
    # mean hp over cells where hp>60 -> only agent0 (100). 4 cells -> 100
    assert m({"source": "agent.hp", "reduce": "mean", "filter": "hp>60"}) == pytest.approx(100.0)


# ---- categorical ------------------------------------------------------------
def test_categorical_nunique_and_entropy():
    assert m({"source": "agent.weapon", "reduce": "nunique"}) == pytest.approx(2.0)
    assert m({"source": "agent.weapon", "reduce": "entropy"}) == pytest.approx(1.0)  # 4 A / 4 B


def test_categorical_scope_is_applied_per_entity():
    w = _buffer()
    w.codes["weapon"][:, 0] = 0
    w.codes["weapon"][:, 1] = [1, 1, 2, 2]
    pooled = compile_metric(
        {"source": "agent.weapon", "reduce": "nunique", "scope": "pooled"}, SCHEMA)(w)
    per_entity = compile_metric(
        {"source": "agent.weapon", "reduce": "nunique",
         "scope": "per_entity_mean"}, SCHEMA)(w)
    assert pooled == pytest.approx(3.0)
    assert per_entity == pytest.approx(1.5)


# ---- spatial ----------------------------------------------------------------
def test_spatial_dispersion():
    # per tick dispersion = (10-t)/2 for t=0..3 -> mean of 5,4.5,4,3.5 = 4.25
    assert m({"source": "agent.pos", "reduce": "dispersion"}) == pytest.approx(4.25)


def test_spatial_distance_reducers_use_all_position_dimensions():
    w = _buffer()
    w.numeric["pos"][:] = 0
    w.numeric["pos"][:, 1, 2] = 2
    dispersion = compile_metric(
        {"source": "agent.pos", "reduce": "dispersion"}, SCHEMA)(w)
    min_distance = compile_metric(
        {"source": "agent.pos", "reduce": "min_pair_dist"}, SCHEMA)(w)
    assert dispersion == pytest.approx(1.0)
    assert min_distance == pytest.approx(2.0)


def test_spatial_hull_area_two_points_is_zero():
    assert m({"source": "agent.pos", "reduce": "hull_area"}) == pytest.approx(0.0)


# ---- globals + events -------------------------------------------------------
def test_global_mean():
    assert m({"source": "alive", "reduce": "mean"}) == pytest.approx(2.0)


def test_event_rate_count_nunique():
    assert m({"source": "event.type", "reduce": "rate"}) == pytest.approx(0.5)   # 2 of 4 ticks
    assert m({"source": "event.type", "reduce": "count"}) == pytest.approx(3.0)
    assert m({"source": "event.type", "reduce": "nunique"}) == pytest.approx(2.0)


def test_event_filter():
    assert m({"source": "event.type", "reduce": "count", "filter": "==x"}) == pytest.approx(2.0)


def test_max_rate_clamps_teleport():
    w = _buffer()
    w.numeric["pos"][2, 0] = (1000, 0, 0)  # teleport spike -> speed >> max_rate(100) -> dropped
    fn = compile_metric({"source": "agent.pos", "map": "speed", "reduce": "max"}, SCHEMA)
    assert fn(w) < 100  # the spike was clamped out, not the max


# ---- loud failures ----------------------------------------------------------
@pytest.mark.parametrize("spec,msg", [
    ({"source": "agent.pos", "reduce": "mean"}, "scalar"),            # vector needs a map/dim
    ({"source": "agent.nope", "reduce": "mean"}, "unknown field"),
    ({"source": "agent.pos.x", "reduce": "frobnicate"}, "unknown reduce"),
    ({"source": "agent.hp", "map": "bit:FIRE", "reduce": "rate"}, "bitmask"),  # hp not bitmask
    ({"source": "agent.ang", "reduce": "dispersion"}, "position"),   # spatial needs position
    ({"source": "agent.pos", "map": "speed", "reduce": "mean", "scope": "team"}, "scope"),
    ({"source": "ghost.x", "reduce": "mean"}, "namespace"),
    ({"source": "agent.hp", "reduce": "nunique"}, "numeric stream"),   # categorical-only reduce
    ({"source": "agent.hp", "reduce": "p150"}, "out of range"),        # percentile clamp
    ({"source": "alive", "reduce": "hull_area"}, "numeric stream"),    # spatial reduce on a global
    ({"source": "alive", "reduce": "nunique"}, "numeric stream"),
])
def test_loud_failures(spec, msg):
    with pytest.raises(ValueError) as e:
        compile_metric(spec, SCHEMA)
    assert msg in str(e.value)
