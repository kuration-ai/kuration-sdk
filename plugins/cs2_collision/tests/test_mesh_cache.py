from __future__ import annotations

import math

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from kuration_cs2_collision import mesh_cache as mc


def _box_glb(tmp_path, name="box"):
    box = trimesh.creation.box(extents=(200.0, 200.0, 200.0))
    box.metadata["name"] = name
    path = tmp_path / f"{name}.glb"
    path.write_bytes(box.export(file_type="glb"))
    return str(path)


def test_glb_round_trip_produces_nonempty_geometry(tmp_path):
    path = _box_glb(tmp_path)
    vertices, faces = mc._build_geometry_from_glb(path)
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3
    assert len(vertices) > 0
    assert len(faces) > 0


def test_build_map_geometry_writes_and_reuses_disk_cache(tmp_path):
    glb_path = _box_glb(tmp_path)
    cache_dir = tmp_path / "cache"

    v1, f1 = mc.build_map_geometry([glb_path], disk_cache_dir=str(cache_dir), map_name="test_map")
    cache_file = mc._disk_cache_path(str(cache_dir), "test_map")
    assert cache_file.exists()

    # Cache hit: parsing never runs again even if the raw file is now unreadable.
    v2, f2 = mc.build_map_geometry(["/nonexistent/path.glb"], disk_cache_dir=str(cache_dir),
                                   map_name="test_map")
    np.testing.assert_array_equal(v1, v2)
    np.testing.assert_array_equal(f1, f2)


def test_load_mesh_cached_returns_a_warmed_trimesh(tmp_path):
    glb_path = _box_glb(tmp_path)
    mesh = mc.load_mesh_cached("test_map2", [glb_path], disk_cache_dir=str(tmp_path / "cache2"))
    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.vertices) > 0


def test_cast_proximity_rays_hits_a_nearby_wall(tmp_path):
    glb_path = _box_glb(tmp_path)
    mesh = mc.load_mesh_cached("test_map3", [glb_path], disk_cache_dir=str(tmp_path / "cache3"))
    # Origin near the box's +x face (box spans -100..100 on each axis), facing +x (yaw=0
    # in this module's convention: dir = (cos(yaw), sin(yaw), 0)).
    origin = np.array([90.0, 0.0, 0.0])
    hit, dist, normal = mc.cast_proximity_rays(mesh, origin, yaw_deg=0.0, proximity=40.0)
    assert hit
    assert dist < 40.0
    assert normal is not None


def test_cast_proximity_rays_misses_when_far_from_geometry():
    # No mesh needed to prove the >proximity early-return path -- use a tiny mesh far away.
    mesh = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    mesh.apply_translation([10_000.0, 10_000.0, 0.0])
    origin = np.array([0.0, 0.0, 0.0])
    hit, dist, normal = mc.cast_proximity_rays(mesh, origin, yaw_deg=0.0, proximity=40.0)
    assert not hit
    assert normal is None


def test_wall_stopped_flags_approach_then_stop():
    vel_prev = np.array([5.0, 0.0, 0.0])   # approaching in +x at 5 units/sample
    vel_curr = np.array([0.0, 0.0, 0.0])   # stopped
    wall_normal = np.array([-1.0, 0.0, 0.0])  # wall faces back toward -x
    flagged, vn_prev, vn_curr = mc.wall_stopped(vel_prev, vel_curr, wall_normal,
                                                min_approach=250.0, max_residual=20.0)
    assert flagged
    assert vn_prev == pytest.approx(5.0 * 64)
    assert vn_curr == pytest.approx(0.0)


def test_wall_stopped_not_flagged_when_still_moving():
    vel_prev = np.array([5.0, 0.0, 0.0])
    vel_curr = np.array([5.0, 0.0, 0.0])
    flagged, vn_prev, vn_curr = mc.wall_stopped(vel_prev, vel_curr, None,
                                                min_approach=250.0, max_residual=20.0)
    assert not flagged
