"""Tests for mesh_cache.py: GLB/BSP geometry parsing (including displacement-surface and
static-prop reconstruction), the on-disk geometry cache, and the multi-height proximity
raycast (`cast_proximity_rays`) that backs both collision-detection algorithms.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from kuration.gameplay.collision import mesh_cache as mc


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


def test_build_geometry_from_glb_is_deterministic_across_repeated_parses(tmp_path):
    path = _box_glb(tmp_path)
    v1, f1 = mc._build_geometry_from_glb(path)
    v2, f2 = mc._build_geometry_from_glb(path)
    np.testing.assert_array_equal(v1, v2)
    np.testing.assert_array_equal(f1, f2)


def test_build_map_geometry_does_not_leak_state_across_different_maps(tmp_path):
    # Two distinct maps sharing one disk_cache_dir must not cross-contaminate: no shared
    # in-memory cache exists in mesh_cache.py (only the on-disk npz, keyed by map_name),
    # so this pins that the key -- not insertion order or a module-level slot -- is what
    # determines what a given map_name resolves to.
    small_glb = _box_glb(tmp_path, name="small")
    big_path = tmp_path / "big.glb"
    big_path.write_bytes(trimesh.creation.box(extents=(400.0, 400.0, 400.0)).export(file_type="glb"))
    cache_dir = tmp_path / "shared_cache"

    v_small, f_small = mc.build_map_geometry(
        [small_glb], disk_cache_dir=str(cache_dir), map_name="small_map")
    v_big, f_big = mc.build_map_geometry(
        [str(big_path)], disk_cache_dir=str(cache_dir), map_name="big_map")
    assert not np.array_equal(v_small, v_big)

    # Re-querying "small_map" with the same raw input after "big_map" was cached still
    # returns small_map's own geometry.
    v_small_again, f_small_again = mc.build_map_geometry(
        [small_glb], disk_cache_dir=str(cache_dir), map_name="small_map")
    np.testing.assert_array_equal(v_small, v_small_again)
    np.testing.assert_array_equal(f_small, f_small_again)


def test_build_map_geometry_writes_and_reuses_disk_cache(tmp_path):
    glb_path = _box_glb(tmp_path)
    cache_dir = tmp_path / "cache"

    v1, f1 = mc.build_map_geometry([glb_path], disk_cache_dir=str(cache_dir), map_name="test_map")
    cache_file = mc._disk_cache_path(str(cache_dir), "test_map")
    assert cache_file.exists()

    # Cache hit: the same unchanged raw input reuses the parsed geometry.
    v2, f2 = mc.build_map_geometry([glb_path], disk_cache_dir=str(cache_dir),
                                   map_name="test_map")
    np.testing.assert_array_equal(v1, v2)
    np.testing.assert_array_equal(f1, f2)


def test_build_map_geometry_rebuilds_when_same_named_source_changes(tmp_path):
    glb_path = tmp_path / "map.glb"
    cache_dir = tmp_path / "cache"

    glb_path.write_bytes(
        trimesh.creation.box(extents=(100.0, 100.0, 100.0)).export(file_type="glb"))
    v1, _ = mc.build_map_geometry(
        [str(glb_path)], disk_cache_dir=str(cache_dir), map_name="stable_name")

    glb_path.write_bytes(
        trimesh.creation.box(extents=(400.0, 400.0, 400.0)).export(file_type="glb"))
    v2, _ = mc.build_map_geometry(
        [str(glb_path)], disk_cache_dir=str(cache_dir), map_name="stable_name")

    np.testing.assert_allclose(np.ptp(v1, axis=0), [100.0, 100.0, 100.0])
    np.testing.assert_allclose(np.ptp(v2, axis=0), [400.0, 400.0, 400.0])


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
    # vel_prev/vel_curr are already true HU/s here -- wall_stopped no longer applies any
    # internal sample-rate scaling (see its docstring for why: a hardcoded conversion
    # baked in here was correct for one dataset's tick spacing and silently wrong for
    # another's; callers now supply already-correctly-scaled velocity themselves).
    vel_prev = np.array([320.0, 0.0, 0.0])   # approaching in +x at 320 HU/s
    vel_curr = np.array([0.0, 0.0, 0.0])   # stopped
    wall_normal = np.array([-1.0, 0.0, 0.0])  # wall faces back toward -x
    flagged, vn_prev, vn_curr = mc.wall_stopped(vel_prev, vel_curr, wall_normal,
                                                min_approach=250.0, max_residual=20.0)
    assert flagged
    assert vn_prev == pytest.approx(320.0)
    assert vn_curr == pytest.approx(0.0)


def test_wall_stopped_not_flagged_when_still_moving():
    vel_prev = np.array([320.0, 0.0, 0.0])
    vel_curr = np.array([320.0, 0.0, 0.0])
    flagged, vn_prev, vn_curr = mc.wall_stopped(vel_prev, vel_curr, None,
                                                min_approach=250.0, max_residual=20.0)
    assert not flagged


def test_cast_proximity_rays_hits_a_waist_high_barrier_missed_at_eye_height():
    # A barrier that only reaches 40 HU (below EYE_HEIGHT=64) was invisible before the
    # waist-level ray layer was added -- a horizontal fan cast only at eye height sails
    # clean over it.
    barrier = trimesh.creation.box(extents=(20.0, 200.0, 40.0))
    barrier.apply_translation([40.0, 0.0, 20.0])  # spans z in [0, 40], near face at x=30
    origin = np.array([0.0, 0.0, 0.0])

    # The eye-height-only layer alone still misses it entirely (this is the bug being fixed).
    eye_dist, _ = mc._cast_fan_at_height(barrier, origin, cy=1.0, sy=0.0, height=mc.EYE_HEIGHT)
    assert eye_dist == math.inf

    hit, dist, normal = mc.cast_proximity_rays(barrier, origin, yaw_deg=0.0, proximity=40.0)
    assert hit
    assert dist < 40.0


def test_prop_box_size_prefers_dimensions_embedded_in_filename():
    size = mc._prop_box_size(
        "models/props/de_dust/hr_dust/dust_crates/dust_crate_style_01_37x37x74.mdl")
    assert size == (37.0, 37.0, 74.0)


def test_prop_box_size_falls_back_to_keyword_default_without_dimensions():
    size = mc._prop_box_size("models/props/de_dust/hr_dust/dust_rusty_barrel_hr/dust_rusty_barrel_hr.mdl")
    assert size == mc._PROP_KEYWORDS["barrel"]


def test_prop_box_size_keyword_can_come_from_the_folder_not_the_filename():
    # "72x72b_tarp.mdl" itself has no "crate" substring -- only the parent folder does.
    size = mc._prop_box_size("models/props/de_dust/hr_dust/dust_crates/72x72b_tarp.mdl")
    assert size is not None


def test_prop_box_size_two_number_name_is_treated_as_a_wxh_panel():
    size = mc._prop_box_size("models/props/de_dust/hr_dust/dust_doors/dust_door_46x106_06.mdl")
    assert size == (46.0, mc._PROP_KEYWORDS["door"][1], 106.0)


def test_prop_box_size_returns_none_for_non_matching_decorative_prop():
    assert mc._prop_box_size("models/props/de_dust/hr_dust/dust_kasbah/window16x48_w1.mdl") is None


def test_box_geometry_has_correct_extents_and_base_anchored_origin():
    verts, faces = mc._box_geometry(np.array([0.0, 0.0, 0.0]), yaw_deg=0.0, size_xyz=(20.0, 10.0, 30.0))
    assert faces.shape == (12, 3)
    assert verts[:, 0].min() == pytest.approx(-10.0)
    assert verts[:, 0].max() == pytest.approx(10.0)
    assert verts[:, 1].min() == pytest.approx(-5.0)
    assert verts[:, 1].max() == pytest.approx(5.0)
    # Base-anchored: origin.z is the bottom of the box, not its center.
    assert verts[:, 2].min() == pytest.approx(0.0)
    assert verts[:, 2].max() == pytest.approx(30.0)


def test_box_geometry_yaw_rotates_the_footprint():
    verts, _ = mc._box_geometry(np.array([0.0, 0.0, 0.0]), yaw_deg=90.0, size_xyz=(20.0, 10.0, 30.0))
    # A 90-degree yaw swaps which horizontal axis the long/short sides line up with.
    assert verts[:, 0].max() == pytest.approx(5.0, abs=1e-9)
    assert verts[:, 1].max() == pytest.approx(10.0, abs=1e-9)


def test_displacement_triangles_reduce_to_a_flat_bilinear_grid_when_undisplaced():
    # Flat unit square base face, corners in CCW winding order starting near the origin.
    corners = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
    ])
    power = 1  # 3x3 grid
    n = (1 << power) + 1
    disp_verts = np.zeros((n * n, 4))  # zero displacement -> stays on the flat quad
    verts, tris = mc._displacement_triangles(corners, corners[0], 0, power, disp_verts)

    assert len(verts) == n * n
    assert tris.shape == (2 * (n - 1) * (n - 1), 3)
    # Grid corners should land exactly on the base face's 4 corners. Index i walks the
    # p0->p1/p3->p2 direction, j walks from the p0-p1 edge across to the p3-p2 edge -- so
    # verts[0]=p0, verts[n-1]=p3, verts[n*(n-1)]=p1, verts[-1]=p2 (matches
    # CCoreDispInfo::GenerateDispSurf's loop nesting exactly).
    np.testing.assert_allclose(verts[0], corners[0])
    np.testing.assert_allclose(verts[n - 1], corners[3])
    np.testing.assert_allclose(verts[n * (n - 1)], corners[1])
    np.testing.assert_allclose(verts[-1], corners[2])
    # Center of a 3x3 grid over a 10x10 square should be the square's center.
    center_idx = (n // 2) * n + (n // 2)
    np.testing.assert_allclose(verts[center_idx], [5.0, 5.0, 0.0])


def test_displacement_triangles_applies_vertex_field_displacement():
    corners = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
    ])
    power = 1
    n = 3
    disp_verts = np.zeros((n * n, 4))
    center_idx = 1 * n + 1
    disp_verts[center_idx] = [0.0, 0.0, 1.0, 25.0]  # push the center vertex straight up by 25
    verts, _ = mc._displacement_triangles(corners, corners[0], 0, power, disp_verts)
    np.testing.assert_allclose(verts[center_idx], [5.0, 5.0, 25.0])


def test_cache_entries_from_another_parser_version_are_rebuilt(tmp_path):
    # The parsers themselves changed once already (displacements + static props took a
    # real map from 8,494 to 68,302 faces) and an unversioned cache silently kept serving
    # the old geometry. Every entry now records GEOMETRY_CACHE_VERSION; an entry from a
    # different version -- or one predating the field -- must be re-parsed, not trusted.
    glb_path = _box_glb(tmp_path)
    cache_dir = tmp_path / "versioned_cache"

    v1, f1 = mc.build_map_geometry([glb_path], disk_cache_dir=str(cache_dir), map_name="m")
    cache_file = mc._disk_cache_path(str(cache_dir), "m")
    with np.load(cache_file) as data:
        assert int(data["cache_version"]) == mc.GEOMETRY_CACHE_VERSION
        assert "source_fingerprint" in data

    # Overwrite with a bogus unversioned (pre-SDK plugin layout) entry: a hit on it must
    # be treated as a miss and rebuilt from the raw file...
    np.savez(cache_file, vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]))
    v2, f2 = mc.build_map_geometry([glb_path], disk_cache_dir=str(cache_dir), map_name="m")
    np.testing.assert_array_equal(v1, v2)
    np.testing.assert_array_equal(f1, f2)

    # ...and the rebuild must repopulate the cache at the current version.
    with np.load(cache_file) as data:
        assert int(data["cache_version"]) == mc.GEOMETRY_CACHE_VERSION

    # A future-version entry is equally untrusted (downgrades happen too).
    np.savez(cache_file, vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]),
             cache_version=np.int64(mc.GEOMETRY_CACHE_VERSION + 1))
    v3, _ = mc.build_map_geometry([glb_path], disk_cache_dir=str(cache_dir), map_name="m")
    np.testing.assert_array_equal(v1, v3)
