"""Tests for CollisionMechanics: the learned-baseline anomaly scan (`_score_and_classify_
window`) and its registration/wiring as a `transform` + `operator` pair. Uses a synthetic
trimesh box wall and a minimal `action_state_dist.json` rather than real map/dataset files,
so every test here is deterministic and self-contained.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from kuration import default_registry, describe_component
from kuration.gameplay.collision import register
from kuration.gameplay.collision.asg_adapter import AsgCollisionAdapter
from kuration.gameplay.collision.collision_operator import (
    COMPONENT_NAME,
    OUTPUT_KEYS,
    CollisionMechanics,
    _score_and_classify_window,
)
from kuration.gameplay.collision.mesh_cache import load_mesh_cached

register()


def _wall_glb(tmp_path):
    # A solid wall spanning x:[50,250], y:[-250,250], z:[-250,250].
    box = trimesh.creation.box(extents=(200.0, 500.0, 500.0))
    box.apply_translation([150.0, 0.0, 0.0])
    path = tmp_path / "wall.glb"
    path.write_bytes(box.export(file_type="glb"))
    return str(path)


def _wall_mesh(tmp_path):
    # Load through the plugin's own production loader (its custom GLB accessor parser),
    # not trimesh.load() -- trimesh.load() returns a Scene wrapper for some GLB exports,
    # not a bare Trimesh, which is irrelevant to how mesh_cache.py actually reads GLBs.
    glb_path = _wall_glb(tmp_path)
    return load_mesh_cached("test_wall", [glb_path], disk_cache_dir=str(tmp_path / "wall_cache"))


def _dist_path(tmp_path):
    # Only dv_fwd is covered -- every other feature z-scores to 0.0 (z_score(_, None) == 0),
    # so this condition is the only thing that can push a tick's score above min_score.
    dist = {"W|ground": {"condition": "W|ground", "n": 100, "dv_fwd": {"mean": 0.0, "std": 0.01}}}
    path = tmp_path / "dist.json"
    path.write_text(json.dumps(dist))
    return str(path)


def _player_record(x, hp=100, btn=8, flags=1, yaw=0.0, pitch=0.0):
    return (np.array([x, 0.0, 0.0]), yaw, pitch, hp, btn, flags)


class _FakeCtx:
    def __init__(self, before):
        self._before = before

    def before(self, token):
        return self._before[token]


# ---------------------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------------------


def test_registers_transform_and_operator():
    assert COMPONENT_NAME in default_registry.names("transform")
    assert COMPONENT_NAME in default_registry.names("operator")
    transform_descriptor = describe_component("transform", COMPONENT_NAME,
                                              registry=default_registry.snapshot())
    assert transform_descriptor["parameters"]["additionalProperties"] is False
    assert transform_descriptor["package"] == "kuration-sdk"


# ---------------------------------------------------------------------------------------
# _score_and_classify_window -- direct unit test of the ported scan+classify math
# ---------------------------------------------------------------------------------------


def test_head_on_collision_is_detected_and_classified(tmp_path):
    mesh = _wall_mesh(tmp_path)
    with open(_dist_path(tmp_path)) as f:
        dist = json.load(f)

    # Player runs +x (yaw=0) from x=0 to x=2, then stops dead at x=2 (wall at x=50).
    # `vel` is the true HU/s velocity at each tick -- supplied directly (as an adapter
    # would), not derived from the position column here.
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    vel = np.array([[0.0, 0.0, 0.0], [128.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    yaw = np.zeros(3)
    pitch = np.zeros(3)
    hp = np.array([100.0, 100.0, 100.0])
    btn = np.array([8.0, 8.0, 8.0])       # IN_FORWARD held throughout
    flags = np.array([1.0, 1.0, 1.0])     # FL_ONGROUND

    out = _score_and_classify_window(
        pos, yaw, pitch, hp, btn, flags, vel,
        None, None, None, None, None, None, None,   # no lookbehind needed; window has 3 ticks
        mesh, dist,
        min_score=1.0, proximity=60.0, min_approach=100.0, max_residual=20.0,
        slide_angle=75.0, max_slide_speed=400.0, min_release_accel=100.0, max_release_vn=100.0,
    )

    assert out["collision_count"] == 1.0
    assert out["head_on_count"] == 1.0
    assert out["angled_count"] == 0.0
    assert out["w_slide_count"] == 0.0
    assert out["max_anomaly_score"] > 100.0     # dv_fwd z-score is enormous by construction
    assert 0.0 < out["min_dist_to_wall_hu"] < 60.0


def test_no_collision_when_player_never_approaches_wall(tmp_path):
    mesh = _wall_mesh(tmp_path)
    with open(_dist_path(tmp_path)) as f:
        dist = json.load(f)

    # Player stays still the whole time -- no dv_fwd anomaly, no collision.
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    vel = np.zeros((3, 3))
    yaw = np.zeros(3)
    pitch = np.zeros(3)
    hp = np.array([100.0, 100.0, 100.0])
    btn = np.array([8.0, 8.0, 8.0])
    flags = np.array([1.0, 1.0, 1.0])

    out = _score_and_classify_window(
        pos, yaw, pitch, hp, btn, flags, vel,
        None, None, None, None, None, None, None,
        mesh, dist,
        min_score=1.0, proximity=60.0, min_approach=100.0, max_residual=20.0,
        slide_angle=75.0, max_slide_speed=400.0, min_release_accel=100.0, max_release_vn=100.0,
    )
    assert out["collision_count"] == 0.0


def test_dead_player_ticks_are_skipped(tmp_path):
    mesh = _wall_mesh(tmp_path)
    with open(_dist_path(tmp_path)) as f:
        dist = json.load(f)

    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    vel = np.array([[0.0, 0.0, 0.0], [128.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    yaw = np.zeros(3)
    pitch = np.zeros(3)
    hp = np.array([100.0, 100.0, 0.0])   # dead on the final tick
    btn = np.array([8.0, 8.0, 8.0])
    flags = np.array([1.0, 1.0, 1.0])

    out = _score_and_classify_window(
        pos, yaw, pitch, hp, btn, flags, vel,
        None, None, None, None, None, None, None,
        mesh, dist,
        min_score=1.0, proximity=60.0, min_approach=100.0, max_residual=20.0,
        slide_angle=75.0, max_slide_speed=400.0, min_release_accel=100.0, max_release_vn=100.0,
    )
    assert out["collision_count"] == 0.0


# ---------------------------------------------------------------------------------------
# CollisionMechanics.transform_batch -- end-to-end wiring (mesh cache, batch/ctx contract)
# ---------------------------------------------------------------------------------------


def test_transform_batch_end_to_end_with_fake_ctx(tmp_path):
    glb_path = _wall_glb(tmp_path)
    dist_path = _dist_path(tmp_path)

    mech = CollisionMechanics(
        map_for_source={"match1::111": "de_dust2"},
        map_registry={"de_dust2": [glb_path]},
        dist_path=dist_path,
        disk_cache_dir=str(tmp_path / "cache"),
        min_score=1.0, proximity=60.0, min_approach=100.0, max_residual=20.0,
    )

    pos = np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])   # (1,3,3)
    vel = np.array([[[0.0, 0.0, 0.0], [128.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    angle = np.zeros((1, 3, 2))
    hp = np.array([[100.0, 100.0, 100.0]])
    btn = np.array([[8.0, 8.0, 8.0]])
    flags = np.array([[1.0, 1.0, 1.0]])

    batch = {
        "segment_ids": [("match1::111", 0, 3)],
        "field:position": pos,
        "field:angle": angle,
        "field:hp": hp,
        "field:btn_mask": btn,
        "field:flags": flags,
        "field:velocity": vel,
    }
    ctx = _FakeCtx({
        "field:position": [None], "field:angle": [None], "field:hp": [None],
        "field:btn_mask": [None], "field:flags": [None], "field:velocity": [None],
    })

    out = mech.transform_batch(batch, ctx)
    assert set(out) == set(OUTPUT_KEYS)
    assert out["collision_count"][0] == 1.0
    assert out["head_on_count"][0] == 1.0
    # Mesh cache populated exactly once per map, reused on a second call.
    assert "de_dust2" in mech._mesh_cache
    cached_mesh = mech._mesh_cache["de_dust2"]
    mech.transform_batch(batch, ctx)
    assert mech._mesh_cache["de_dust2"] is cached_mesh


def _write_asg_match(tmp_path):
    # Same head-on scenario as test_head_on_collision_is_detected_and_classified above
    # (run x=0 -> x=2, stop dead at x=2, wall at x=50) but expressed as real ASG data.json
    # ticks so AsgCollisionAdapter's own position-delta velocity synthesis (*64 HU/s) is
    # exercised for real, instead of a hand-built `vel` array: (2-0)*64=128, (2-2)*64=0 --
    # the exact vel this scenario's unit test supplies directly.
    data = {
        "header": {"map_name": "de_dust2"},
        "ticks": [
            {"tick": 100, "players": [{"id32": 111, "pos": {"x": 0.0, "y": 0.0, "z": 0.0},
                                       "ang": {"yaw": 0.0, "pitch": 0.0}, "hp": 100,
                                       "btn_mask": 8, "flags": 1, "weapon": "AK-47"}]},
            {"tick": 104, "players": [{"id32": 111, "pos": {"x": 2.0, "y": 0.0, "z": 0.0},
                                       "ang": {"yaw": 0.0, "pitch": 0.0}, "hp": 100,
                                       "btn_mask": 8, "flags": 1, "weapon": "AK-47"}]},
            {"tick": 108, "players": [{"id32": 111, "pos": {"x": 2.0, "y": 0.0, "z": 0.0},
                                       "ang": {"yaw": 0.0, "pitch": 0.0}, "hp": 100,
                                       "btn_mask": 8, "flags": 1, "weapon": "AK-47"}]},
        ],
    }
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data))
    return f"{path}::111"


def test_full_pipeline_input_window_apply_collect_end_to_end(tmp_path):
    # The other end-to-end test above drives CollisionMechanics.transform_batch() with a
    # hand-built batch dict + _FakeCtx, proving only the transform's own batch/ctx
    # contract. This drives the SAME scenario through the real, registered Pipeline --
    # input(adapter=...).window(...).apply(<registered operator name>...).collect() --
    # proving the operator actually works wired into the SDK a real caller would use, not
    # just its internal transform_batch() method in isolation.
    from kuration import input as kuration_input

    glb_path = _wall_glb(tmp_path)
    dist_path = _dist_path(tmp_path)
    source = _write_asg_match(tmp_path)

    # bins=1 is a deliberate no-op binning axis here (this test isn't exercising binning),
    # which F20 now warns on -- expected, not a bug.
    with pytest.warns(UserWarning, match="bins=1"):
        pipeline = (
            kuration_input(adapter=AsgCollisionAdapter(), sources=[source],
                           schema=AsgCollisionAdapter.schema)
            .window(3, stride=3)
            .apply(COMPONENT_NAME,
                   map_for_source={source: "de_dust2"},
                   map_registry={"de_dust2": [glb_path]},
                   dist_path=dist_path,
                   disk_cache_dir=str(tmp_path / "cache"),
                   min_score=1.0, proximity=60.0, min_approach=100.0, max_residual=20.0)
            .organize("collision_count", bins=1)
        )
    result = pipeline.collect(n=1)

    assert not result.errors
    assert len(result.all_segments) == 1
    features = result.all_segments[0].features
    assert features["collision_count"] == 1.0
    assert features["head_on_count"] == 1.0


@pytest.mark.filterwarnings("ignore:binning axis .* is constant")
def test_overlapping_windows_count_each_collision_tick_once(tmp_path):
    # The transform registers ownership="rightmost-start" (the same convention
    # pixel_shift uses for press instances): a window owns its steps up to where the
    # next window starts, so a collision tick inside a stride<window overlap is scored
    # by exactly one window -- the one whose owned prefix contains it, which sees it
    # with full lookbehind context. An earlier revision had no ownership convention and
    # double-counted the tick in both covering windows.
    from kuration import input as kuration_input

    glb_path = _wall_glb(tmp_path)
    dist_path = _dist_path(tmp_path)

    # pos = [0,2,2,2,2,2] -> vel = [0,128,0,0,0,0]: approach then stop-and-stay -- exactly
    # ONE physical deceleration/collision event, at tick index 2.
    xs = [0.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    data = {
        "header": {"map_name": "de_dust2"},
        "ticks": [
            {"tick": 100 + i * 4, "players": [{"id32": 111, "pos": {"x": x, "y": 0.0, "z": 0.0},
                                               "ang": {"yaw": 0.0, "pitch": 0.0}, "hp": 100,
                                               "btn_mask": 8, "flags": 1, "weapon": "AK-47"}]}
            for i, x in enumerate(xs)
        ],
    }
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data))
    source = f"{path}::111"

    # bins=1 is a deliberate no-op binning axis here (this test isn't exercising binning),
    # which F20 now warns on -- expected, not a bug.
    with pytest.warns(UserWarning, match="bins=1"):
        pipeline = (
            kuration_input(adapter=AsgCollisionAdapter(), sources=[source],
                           schema=AsgCollisionAdapter.schema)
            .window(4, stride=2)   # windows [0:4) and [2:6) both contain tick index 2
            .apply(COMPONENT_NAME,
                   map_for_source={source: "de_dust2"},
                   map_registry={"de_dust2": [glb_path]},
                   dist_path=dist_path,
                   disk_cache_dir=str(tmp_path / "cache"),
                   min_score=1.0, proximity=60.0, min_approach=100.0, max_residual=20.0)
            .organize("collision_count", bins=1)
        )
    result = pipeline.collect(n=10)

    assert not result.errors
    assert len(result.all_segments) == 2
    # One real collision event (tick index 2, inside both windows' extent): window
    # [0:4) owns ticks [0,2) only, window [2:6) owns [2,6) and scores the collision
    # with ticks 0-1 as lookbehind -- one event, counted once in the aggregate.
    by_start = sorted(result.all_segments, key=lambda seg: seg.start_idx)
    counts = [seg.features["collision_count"] for seg in by_start]
    assert counts == [0.0, 1.0]
    assert sum(counts) == 1.0
