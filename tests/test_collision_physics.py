"""Tests for physics.py: pure math/constant helpers (angle wrapping, local-frame velocity
conversion, z-scoring, condition-key/flag decoding) shared by both collision algorithms.
"""
from __future__ import annotations

from kuration.gameplay.collision import physics as ph


def test_wrap_angle_handles_wraparound():
    assert ph.wrap_angle(190.0) == -170.0
    assert ph.wrap_angle(-190.0) == 170.0
    assert ph.wrap_angle(10.0) == 10.0


def test_yaw_average_takes_circular_mean():
    assert ph.yaw_average(170.0, -170.0) == 180.0 or ph.yaw_average(170.0, -170.0) == -180.0


def test_to_local_forward_motion_is_pure_v_fwd():
    # Facing yaw=0 (local +x = crosshair direction per to_local's convention), moving
    # purely along +x should show up entirely as v_fwd with ~0 v_right.
    vf, vr, vu = ph.to_local(dx=1.0, dy=0.0, dz=0.0, yaw_deg=0.0)
    assert vf > 0
    assert abs(vr) < 1e-9
    assert vu == 0.0


def test_z_score_zero_when_std_negligible():
    assert ph.z_score(5.0, {"mean": 0.0, "std": 0.0}) == 0.0
    assert ph.z_score(5.0, None) == 0.0


def test_z_score_matches_manual_calc():
    assert ph.z_score(6.0, {"mean": 4.0, "std": 2.0}) == 1.0


def test_condition_key_excludes_weapon_and_encodes_ground_duck():
    assert ph.condition_key(ph.BIT_W, onground=True, ducking=False) == "W|ground"
    assert ph.condition_key(ph.BIT_W | ph.BIT_A, onground=False, ducking=True) == "W+A|air_duck"
    assert ph.condition_key(0, onground=True, ducking=False) == "idle|ground"


def test_is_skip_flags_true_for_frozen_or_noclip():
    assert ph.is_skip_flags(ph.FL_FROZEN)
    assert ph.is_skip_flags(ph.FL_NOCLIP)
    assert not ph.is_skip_flags(ph.FL_ONGROUND)
