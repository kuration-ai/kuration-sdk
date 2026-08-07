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
from kuration.gameplay.collision import register
from kuration.gameplay.collision.directional_collision_analyzer import (
    COMPONENT_NAME,
    DirectionalCollisionAnalyzer,
)
from kuration.gameplay.collision.mesh_cache import load_mesh_cached
from kuration.gameplay.collision.physics import BIT_W, FL_FROZEN
from kuration.gameplay.collision.ramp_fitting import save_ramp_models

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
    assert desc["package"] == "kuration-sdk"
    assert desc["capabilities"] == {"stateful": True, "cross_segment": True,
                                    "tick_stream": True}


def test_registers_the_pipeline_preserving_companion_operator():
    assert COMPONENT_NAME in default_registry.names("operator")
    desc = describe_component("operator", COMPONENT_NAME, registry=default_registry.snapshot())
    assert desc["capabilities"] == {"stateful": True, "cross_segment": True,
                                    "pipeline_metadata": True}


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


# ---------------------------------------------------------------------------------------
# Construction contract -- explicit fallback ramp, explicit map configuration
# ---------------------------------------------------------------------------------------


def test_constructor_requires_some_map_configuration(tmp_path):
    from kuration.gameplay.collision.ramp_fitting import save_ramp_models as save

    ramp_path = tmp_path / "ramps.json"
    save({}, str(ramp_path))
    with pytest.raises(ValueError, match="map_for_source.*default_map|default_map"):
        DirectionalCollisionAnalyzer(
            map_registry={"m": ["x.glb"]}, ramp_models_path=str(ramp_path))


def test_fallback_ramp_is_an_explicit_documented_parameter(tmp_path):
    from kuration.gameplay.collision.directional_collision_analyzer import (
        DEFAULT_FALLBACK_RAMP,
    )

    # An empty fitted table used to silently borrow "whatever model happened to be first
    # in insertion order"; now every absent (combo, weapon) pair uses the documented
    # default, or an explicit override.
    analyzer = _analyzer(tmp_path, {})
    assert analyzer.fallback_ramp == DEFAULT_FALLBACK_RAMP

    override = _analyzer(tmp_path, {}, fallback_ramp=(2.0, 100.0))
    assert override.fallback_ramp == (2.0, 100.0)

    with pytest.raises(ValueError, match="fallback_ramp"):
        _analyzer(tmp_path, {}, fallback_ramp=(0.0, 100.0))


def test_unfitted_combo_weapon_pair_uses_the_fallback_ramp(tmp_path):
    # Fitted table knows only a rifle; the recorded weapon is a pistol. With a zero-cap
    # fallback the analyzer expects no motion (never flags); with a fast fallback it
    # expects cap-speed motion the stalled player never gains (flags) -- proving the
    # lookup actually used the fallback both times.
    n = 4
    arrays = dict(
        position=np.array([[40.0, 0.0, 0.0]] * n), yaw=np.zeros(n),
        btn_mask=np.full(n, BIT_W), velocity=np.zeros((n, 3)), hp=np.full(n, 100.0),
        flags=np.zeros(n), weapon=np.array(["weapon_deagle"] * n, dtype=object),
    )
    quiet = _analyzer(tmp_path, {(BIT_W, WEAPON): (3.0, 200.0)}, fallback_ramp=(3.0, 0.0))
    quiet.process_source("src1", **arrays)
    assert all(not r["collision"] for r in quiet.to_records())

    eager = _analyzer(tmp_path, {(BIT_W, WEAPON): (3.0, 0.0)}, fallback_ramp=(3.0, 200.0))
    eager.process_source("src1", **arrays)
    assert any(r["collision"] for r in eager.to_records())


# ---------------------------------------------------------------------------------------
# Tick-stream contract -- add_batch stitching, pipeline enrich(), direct-path parity
# ---------------------------------------------------------------------------------------


def _teapearce_npy(tmp_path, n=8):
    """One synthetic TeaPearce recording: player holds W facing the fixture wall
    (x=50) from x=40 with velocity pinned at zero -- genuinely blocked."""
    arr = np.empty((n, 6), dtype=object)
    for i in range(n):
        arr[i, 0] = None
        arr[i, 1] = {
            "localpos1": 40.0, "localpos2": 0.0, "localpos3": 0.0,
            "viewangle_xy": 0.0, "viewangle_vert": 0.0,
            "vel_1": 0.0, "vel_2": 0.0, "vel_3": 0.0,
            "obs_health": 100, "tp_wasd": ["w"], "obs_mode": 0,
            "gsi_weap_active": {"name": WEAPON},
        }
        for j in range(2, 6):
            arr[i, j] = None
    path = tmp_path / "recording.npy"
    np.save(path, arr, allow_pickle=True)
    return str(path)


def test_stream_path_matches_the_direct_process_source_path_exactly(tmp_path):
    # THE stitching guarantee: driving the analyzer through the real pipeline stream
    # driver (windowed, ownership-deduplicated, per-source reassembly) must produce
    # bit-identical records to handing process_source() the whole source directly.
    from kuration import input as kuration_input
    from kuration.gameplay.collision.teapearce_adapter import (
        TEAPEARCE_SCHEMA,
        TeaPearceDust2Adapter,
        load_directional_arrays,
    )

    source = _teapearce_npy(tmp_path)
    models = {(BIT_W, WEAPON): (3.0, 200.0)}
    ramp_path = tmp_path / "ramps.json"
    save_ramp_models(models, str(ramp_path))
    glb = _wall_glb(tmp_path)

    pipeline = (
        kuration_input(adapter=TeaPearceDust2Adapter(), sources=[source],
                       schema=TEAPEARCE_SCHEMA)
        .window(4, stride=4)
        .collision_directional(
            map_registry={"test_wall": [glb]},
            default_map="test_wall",
            ramp_models_path=str(ramp_path),
            disk_cache_dir=str(tmp_path / "cache_stream"),
        )
    )
    attachment = pipeline.metadata["gameplay"]["directional_collision"]
    summary = attachment["result"]["summary"]

    direct = _analyzer(tmp_path, models,
                       map_for_source={source: "test_wall"},
                       disk_cache_dir=str(tmp_path / "cache_direct"))
    direct.process_source(source, **load_directional_arrays(source))

    assert summary == direct.summary()
    assert summary["total_ticks"] == 8
    assert summary["total_collisions"] > 0
    assert summary["flagged_ticks"][source] == [
        r["tick"] for r in direct.to_records() if r["collision"]]
    assert attachment["mode"] == "stream"


def test_stream_enrich_summary_lists_flagged_ticks_in_source_coordinates(tmp_path):
    from kuration import input as kuration_input
    from kuration.gameplay.collision.teapearce_adapter import (
        TEAPEARCE_SCHEMA,
        TeaPearceDust2Adapter,
    )

    source = _teapearce_npy(tmp_path)
    ramp_path = tmp_path / "ramps.json"
    save_ramp_models({(BIT_W, WEAPON): (3.0, 200.0)}, str(ramp_path))

    result = (
        kuration_input(adapter=TeaPearceDust2Adapter(), sources=[source],
                       schema=TEAPEARCE_SCHEMA)
        .window(4, stride=4)
        .analyze("aintropy/collision-directional", {
            "map_registry": {"test_wall": [_wall_glb(tmp_path)]},
            "default_map": "test_wall",
            "ramp_models_path": str(ramp_path),
            "disk_cache_dir": str(tmp_path / "cache"),
        })
    )
    assert result["mode"] == "tick-stream"
    assert result["sources_analyzed"] == 1
    # Tick 0 captures v0 from the already-stalled velocity (expected==actual); from tick
    # 1 on the fitted ramp expects speed the blocked player never gains.
    assert result["summary"]["flagged_ticks"][source] == list(range(1, 8))


def test_add_batch_rejects_non_contiguous_windows(tmp_path):
    analyzer = _analyzer(tmp_path, {(BIT_W, WEAPON): (3.0, 0.0)})

    def batch_for(start, end):
        T = end - start
        return {
            "segment_ids": [("src1", start, end)],
            "field:position": np.zeros((1, T, 1, 3)),
            "field:angle": np.zeros((1, T, 1, 2)),
            "field:btn_mask": np.full((1, T, 1, 1), float(BIT_W)),
            "field:velocity": np.zeros((1, T, 1, 3)),
            "field:hp": np.full((1, T, 1, 1), 100.0),
            "field:flags": np.zeros((1, T, 1, 1)),
            "field:weapon": np.full((1, T, 1), WEAPON, dtype=object),
        }

    analyzer.add_batch(batch_for(0, 3), {})
    # The owned slice is walked immediately; raw window arrays are not retained until
    # summary()/to_records(), and only compact cross-tick state remains.
    assert len(analyzer._records) == 3
    assert set(analyzer._stream_states) == {"src1"}
    assert analyzer._stream_next_start == {"src1": 3}
    with pytest.raises(ValueError, match="contiguous"):
        analyzer.add_batch(batch_for(5, 8), {})  # a 2-tick gap the walk cannot survive
    analyzer.summary()
    assert analyzer._stream_states == {}
    assert analyzer._stream_next_start == {}
