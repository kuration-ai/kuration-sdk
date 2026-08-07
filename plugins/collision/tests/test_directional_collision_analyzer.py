"""Tests for DirectionalCollisionAnalyzer -- the stateful, full-tick-sequence sibling of
CollisionMechanics. Covers the two mechanisms with no equivalent in the windowed transform
(the GRACE_TICKS combo-dropout tolerance and the fast-turn guard) plus the Analyzer-protocol
contract (`to_records()` row-count, `summary()` aggregation).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from kuration import default_registry, describe_component
from kuration_collision import register
from kuration_collision.directional_collision_analyzer import (
    COMPONENT_NAME,
    DirectionalCollisionAnalyzer,
)
from kuration_collision.mesh_cache import load_mesh_cached
from kuration_collision.physics import BIT_W, FL_FROZEN
from kuration_collision.ramp_fitting import save_ramp_models

register()

WEAPON = "weapon_ak47"


def _wall_glb(tmp_path):
    # A solid wall spanning x:[50,250], y:[-250,250], z:[-250,250] -- same fixture geometry
    # as test_collision_operator.py's wall, reused so a passing test here is comparable.
    box = trimesh.creation.box(extents=(200.0, 500.0, 500.0))
    box.apply_translation([150.0, 0.0, 0.0])
    path = tmp_path / "wall.glb"
    path.write_bytes(box.export(file_type="glb"))
    return str(path)


def _analyzer(tmp_path, models, **overrides):
    ramp_path = tmp_path / "ramps.json"
    save_ramp_models(models, str(ramp_path))
    kwargs = dict(
        map_for_source={"src1": "test_wall"},
        map_registry={"test_wall": [_wall_glb(tmp_path)]},
        ramp_models_path=str(ramp_path),
        disk_cache_dir=str(tmp_path / "cache"),
    )
    kwargs.update(overrides)
    return DirectionalCollisionAnalyzer(**kwargs)


# ---------------------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------------------


def test_registers_directional_collision_analyzer():
    assert COMPONENT_NAME in default_registry.names("analyzer")
    desc = describe_component("analyzer", COMPONENT_NAME, registry=default_registry.snapshot())
    assert desc["parameters"]["additionalProperties"] is False
    assert desc["package"] == "kuration-plugin-collision"
    assert desc["capabilities"] == {"stateful": True, "cross_segment": True}


# ---------------------------------------------------------------------------------------
# process_source -- residual-vs-wall collision detection
# ---------------------------------------------------------------------------------------


def test_flags_collision_when_player_stalls_against_a_wall(tmp_path):
    # Player holds W facing the wall (yaw=0 -> +x, wall at x=50) from x=40 -- close enough
    # to raycast-hit within the default 40 HU proximity -- but velocity never leaves 0, as
    # if genuinely blocked. The fitted model expects growing speed toward cap=200 HU/s.
    analyzer = _analyzer(tmp_path, {(BIT_W, WEAPON): (3.0, 200.0)})

    n = 4
    position = np.array([[40.0, 0.0, 0.0]] * n)
    yaw = np.zeros(n)
    btn_mask = np.full(n, BIT_W)
    velocity = np.zeros((n, 3))
    hp = np.full(n, 100.0)
    flags = np.zeros(n)
    weapon = np.array([WEAPON] * n, dtype=object)

    analyzer.process_source("src1", position=position, yaw=yaw, btn_mask=btn_mask,
                            velocity=velocity, hp=hp, flags=flags, weapon=weapon)
    records = analyzer.to_records()

    assert len(records) == n
    # Tick 0: v0 is captured FROM this (already-zero) actual velocity, so expected==actual.
    assert records[0]["collision"] is False
    assert records[0]["reason"] == "residual_too_small"
    # From tick 1 on the model expects growing speed the stalled player never gains.
    for r in records[1:]:
        assert r["collision"] is True
        assert r["reason"] == "flagged"
        assert r["residual_mag"] > 40.0


def test_no_collision_when_actual_velocity_matches_the_fitted_model(tmp_path):
    # Same setup, but this time actual velocity follows the exact ramp the model predicts
    # (a real, unobstructed acceleration) -- residual should stay ~0 the whole way.
    tau, cap = 3.0, 200.0
    analyzer = _analyzer(tmp_path, {(BIT_W, WEAPON): (tau, cap)})

    n = 5
    position = np.array([[1000.0, 1000.0, 0.0]] * n)  # nowhere near the wall -- irrelevant here
    yaw = np.zeros(n)
    btn_mask = np.full(n, BIT_W)
    velocity = np.zeros((n, 3))
    for t in range(n):
        velocity[t, 0] = cap * (1.0 - math.exp(-t / tau))
    hp = np.full(n, 100.0)
    flags = np.zeros(n)
    weapon = np.array([WEAPON] * n, dtype=object)

    analyzer.process_source("src1", position=position, yaw=yaw, btn_mask=btn_mask,
                            velocity=velocity, hp=hp, flags=flags, weapon=weapon)
    records = analyzer.to_records()

    assert len(records) == n
    for r in records:
        assert r["collision"] is False
        assert r["reason"] == "residual_too_small"
        assert r["residual_mag"] < 1.0


def test_dead_and_frozen_ticks_are_skipped_and_reset_tracking(tmp_path):
    analyzer = _analyzer(tmp_path, {(BIT_W, WEAPON): (3.0, 0.0)})  # cap=0 -- never flags

    n = 3
    position = np.array([[1000.0, 1000.0, 0.0]] * n)
    yaw = np.zeros(n)
    btn_mask = np.full(n, BIT_W)
    velocity = np.zeros((n, 3))
    hp = np.array([100.0, 0.0, 100.0])          # dead on tick 1
    flags = np.array([0.0, 0.0, float(FL_FROZEN)])  # frozen (round-freeze) on tick 2
    weapon = np.array([WEAPON] * n, dtype=object)

    analyzer.process_source("src1", position=position, yaw=yaw, btn_mask=btn_mask,
                            velocity=velocity, hp=hp, flags=flags, weapon=weapon)
    records = analyzer.to_records()

    assert records[1]["reason"] == "dead_or_skipped"
    assert records[1]["collision"] is False
    assert records[2]["reason"] == "dead_or_skipped"
    assert records[2]["collision"] is False


# ---------------------------------------------------------------------------------------
# GRACE_TICKS -- brief combo dropout tolerance (no equivalent in CollisionMechanics)
# ---------------------------------------------------------------------------------------


def test_grace_ticks_tolerate_a_brief_combo_dropout(tmp_path):
    # cap=0 keeps expected==v0==0 always (residual never crosses the flag threshold),
    # isolating tracking-state behavior from the collision-flagging logic entirely.
    analyzer = _analyzer(tmp_path, {(BIT_W, WEAPON): (3.0, 0.0)})

    n = 5
    position = np.array([[1000.0, 1000.0, 0.0]] * n)
    yaw = np.zeros(n)
    # W, W, <drop>, W, W -- a single-tick dropout, well within GRACE_TICKS=3.
    btn_mask = np.array([BIT_W, BIT_W, 0, BIT_W, BIT_W])
    velocity = np.zeros((n, 3))
    hp = np.full(n, 100.0)
    flags = np.zeros(n)
    weapon = np.array([WEAPON] * n, dtype=object)

    analyzer.process_source("src1", position=position, yaw=yaw, btn_mask=btn_mask,
                            velocity=velocity, hp=hp, flags=flags, weapon=weapon)
    records = analyzer.to_records()

    for r in records:
        assert r["tracking_combo"] == BIT_W
        assert r["reason"] != "not_modeled"


def test_combo_dropout_exceeding_grace_ticks_resets_tracking(tmp_path):
    analyzer = _analyzer(tmp_path, {(BIT_W, WEAPON): (3.0, 0.0)})

    n = 7
    position = np.array([[1000.0, 1000.0, 0.0]] * n)
    yaw = np.zeros(n)
    # W, W, <4-tick drop>, W -- exceeds GRACE_TICKS=3, so tracking must reset mid-drop.
    btn_mask = np.array([BIT_W, BIT_W, 0, 0, 0, 0, BIT_W])
    velocity = np.zeros((n, 3))
    hp = np.full(n, 100.0)
    flags = np.zeros(n)
    weapon = np.array([WEAPON] * n, dtype=object)

    analyzer.process_source("src1", position=position, yaw=yaw, btn_mask=btn_mask,
                            velocity=velocity, hp=hp, flags=flags, weapon=weapon)
    records = analyzer.to_records()

    # Ticks 2, 3, 4 are still within (or at) the GRACE_TICKS=3 tolerance window.
    for i in (2, 3, 4):
        assert records[i]["tracking_combo"] == BIT_W
    # Tick 5 is the 4th consecutive unmodeled tick -- grace exhausted, tracking drops.
    assert records[5]["tracking_combo"] is None
    assert records[5]["reason"] == "not_modeled"
    # Tick 6 re-acquires the combo from scratch.
    assert records[6]["tracking_combo"] == BIT_W


# ---------------------------------------------------------------------------------------
# fast-turn guard -- a spinning camera shouldn't be scored as a collision residual
# ---------------------------------------------------------------------------------------


def test_fast_turn_guard_suppresses_flagging_on_a_sharp_yaw_jump(tmp_path):
    analyzer = _analyzer(tmp_path, {(BIT_W, WEAPON): (3.0, 0.0)})  # cap=0, isolates the guard

    n = 5
    position = np.array([[1000.0, 1000.0, 0.0]] * n)
    yaw = np.array([0.0, 0.0, 30.0, 0.0, 0.0])  # a 30deg jump and back, > MAX_DYAW_LEARN=15
    btn_mask = np.full(n, BIT_W)
    velocity = np.zeros((n, 3))
    hp = np.full(n, 100.0)
    flags = np.zeros(n)
    weapon = np.array([WEAPON] * n, dtype=object)

    analyzer.process_source("src1", position=position, yaw=yaw, btn_mask=btn_mask,
                            velocity=velocity, hp=hp, flags=flags, weapon=weapon)
    records = analyzer.to_records()

    assert records[2]["reason"] == "fast_turn"
    assert records[3]["reason"] == "fast_turn"
    for i in (0, 1, 4):
        assert records[i]["reason"] != "fast_turn"


# ---------------------------------------------------------------------------------------
# to_records() / summary() -- Analyzer-protocol contract
# ---------------------------------------------------------------------------------------


def test_to_records_and_summary_shapes(tmp_path):
    analyzer = _analyzer(tmp_path, {(BIT_W, WEAPON): (3.0, 200.0)})

    n = 4
    position = np.array([[40.0, 0.0, 0.0]] * n)
    yaw = np.zeros(n)
    btn_mask = np.full(n, BIT_W)
    velocity = np.zeros((n, 3))
    hp = np.full(n, 100.0)
    flags = np.zeros(n)
    weapon = np.array([WEAPON] * n, dtype=object)

    analyzer.process_source("src1", position=position, yaw=yaw, btn_mask=btn_mask,
                            velocity=velocity, hp=hp, flags=flags, weapon=weapon)

    records = analyzer.to_records()
    assert len(records) == n  # one row per input tick, regardless of collision outcome

    summary = analyzer.summary()
    assert summary["sources_processed"] == 1
    assert summary["total_ticks"] == n
    assert summary["total_collisions"] == sum(r["collision"] for r in records)
    assert summary["per_source"]["src1"] == {
        "ticks": n, "collisions": summary["total_collisions"],
    }
