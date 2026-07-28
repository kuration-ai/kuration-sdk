from __future__ import annotations

import json

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from kuration import default_registry, describe_component
from kuration_cs2_collision import register
from kuration_cs2_collision.collision_operator import (
    COMPONENT_NAME,
    OUTPUT_KEYS,
    CollisionMechanics,
    _score_and_classify_window,
)
from kuration_cs2_collision.mesh_cache import load_mesh_cached

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
    assert transform_descriptor["package"] == "kuration-plugin-cs2-collision"


# ---------------------------------------------------------------------------------------
# _score_and_classify_window -- direct unit test of the ported scan+classify math
# ---------------------------------------------------------------------------------------


def test_head_on_collision_is_detected_and_classified(tmp_path):
    mesh = _wall_mesh(tmp_path)
    with open(_dist_path(tmp_path)) as f:
        dist = json.load(f)

    # Player runs +x (yaw=0) from x=0 to x=2, then stops dead at x=2 (wall at x=50).
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    yaw = np.zeros(3)
    pitch = np.zeros(3)
    hp = np.array([100.0, 100.0, 100.0])
    btn = np.array([8.0, 8.0, 8.0])       # IN_FORWARD held throughout
    flags = np.array([1.0, 1.0, 1.0])     # FL_ONGROUND

    out = _score_and_classify_window(
        pos, yaw, pitch, hp, btn, flags,
        None, None, None, None, None, None,   # no lookbehind needed; window itself has 3 ticks
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
    yaw = np.zeros(3)
    pitch = np.zeros(3)
    hp = np.array([100.0, 100.0, 100.0])
    btn = np.array([8.0, 8.0, 8.0])
    flags = np.array([1.0, 1.0, 1.0])

    out = _score_and_classify_window(
        pos, yaw, pitch, hp, btn, flags,
        None, None, None, None, None, None,
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
    yaw = np.zeros(3)
    pitch = np.zeros(3)
    hp = np.array([100.0, 100.0, 0.0])   # dead on the final tick
    btn = np.array([8.0, 8.0, 8.0])
    flags = np.array([1.0, 1.0, 1.0])

    out = _score_and_classify_window(
        pos, yaw, pitch, hp, btn, flags,
        None, None, None, None, None, None,
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
    }
    ctx = _FakeCtx({
        "field:position": [None], "field:angle": [None], "field:hp": [None],
        "field:btn_mask": [None], "field:flags": [None],
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
