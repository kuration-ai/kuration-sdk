"""Tests for kinematics.py: the reusable velocity/turn-rate feature extraction pulled out
of collision_operator.py's fused per-tick scan/classify loop.
"""
from __future__ import annotations

import numpy as np
import pytest

from kuration.gameplay.collision import physics as ph
from kuration.gameplay.collision.kinematics import KINEMATICS_KEYS, extract_kinematics


def _record(pos=(0.0, 0.0, 0.0), yaw=0.0, pitch=0.0, hp=100.0, btn=0, flags=0):
    return (np.array(pos, dtype=float), yaw, pitch, hp, btn, flags)


def test_extract_kinematics_returns_every_declared_key():
    result = extract_kinematics(_record(), _record(), _record())
    assert set(result) == set(KINEMATICS_KEYS)


def test_stationary_player_has_zero_kinematics():
    still = _record()
    result = extract_kinematics(still, still, still)
    for key in ("v_fwd", "v_right", "v_up", "speed_2d", "dv_fwd", "dv_right", "dv_up",
               "d_yaw", "d_pitch", "delta_hp", "pos_error", "dyaw10", "dyaw21"):
        assert result[key] == pytest.approx(0.0, abs=1e-9)


def test_forward_motion_at_zero_yaw_is_pure_v_fwd():
    # Facing yaw=0, moving purely along +x each tick: same convention as
    # physics.to_local's own test (facing yaw=0, +x is the forward axis).
    step = ph.DT  # one unit of dx per DT -> v_fwd == 1/DT... just needs a nonzero constant
    pm2 = _record(pos=(0.0, 0.0, 0.0))
    pm1 = _record(pos=(step, 0.0, 0.0))
    p0 = _record(pos=(2 * step, 0.0, 0.0))
    result = extract_kinematics(pm2, pm1, p0)
    assert result["v_fwd"] > 0
    assert result["v_right"] == pytest.approx(0.0, abs=1e-9)
    assert result["speed_2d"] == pytest.approx(result["v_fwd"], abs=1e-9)


def test_pos_error_is_zero_for_constant_velocity_motion():
    # Equal displacement each tick, no turning: the constant-velocity prediction is exact.
    pm2 = _record(pos=(0.0, 0.0, 0.0))
    pm1 = _record(pos=(5.0, 0.0, 0.0))
    p0 = _record(pos=(10.0, 0.0, 0.0))
    result = extract_kinematics(pm2, pm1, p0)
    assert result["pos_error"] == pytest.approx(0.0, abs=1e-6)


def test_pos_error_is_positive_when_motion_deviates_from_constant_velocity():
    pm2 = _record(pos=(0.0, 0.0, 0.0))
    pm1 = _record(pos=(5.0, 0.0, 0.0))
    p0 = _record(pos=(5.0, 5.0, 0.0))  # sudden lateral jump, not a continuation of +x motion
    result = extract_kinematics(pm2, pm1, p0)
    assert result["pos_error"] > 0


def test_delta_hp_is_health_change_since_the_previous_tick():
    pm2 = _record(hp=100.0)
    pm1 = _record(hp=80.0)
    p0 = _record(hp=65.0)
    result = extract_kinematics(pm2, pm1, p0)
    assert result["delta_hp"] == pytest.approx(-15.0)


def test_dyaw_tracks_absolute_turn_magnitude_since_each_predecessor():
    pm2 = _record(yaw=0.0)
    pm1 = _record(yaw=30.0)
    p0 = _record(yaw=50.0)
    result = extract_kinematics(pm2, pm1, p0)
    assert result["dyaw10"] == pytest.approx(20.0)
    assert result["dyaw21"] == pytest.approx(30.0)
    assert result["d_yaw"] == pytest.approx(20.0)


def test_dyaw_handles_wraparound_like_wrap_angle():
    pm2 = _record(yaw=0.0)
    pm1 = _record(yaw=170.0)
    p0 = _record(yaw=-170.0)   # 20 degrees further in the same turning direction, wrapped
    result = extract_kinematics(pm2, pm1, p0)
    assert result["dyaw10"] == pytest.approx(20.0)


def test_d_pitch_is_pitch_change_since_the_previous_tick():
    pm2 = _record(pitch=0.0)
    pm1 = _record(pitch=10.0)
    p0 = _record(pitch=25.0)
    result = extract_kinematics(pm2, pm1, p0)
    assert result["d_pitch"] == pytest.approx(15.0)
