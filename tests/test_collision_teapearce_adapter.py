"""Tests for TeaPearceDust2Adapter: reading TeaPearce dust2 `.npy` metadata files into a
canonical Tick stream (including the `weapon` field the directional analyzer needs), and
`load_directional_arrays`' Tick-stream-to-flat-arrays conversion for that same analyzer.
"""
from __future__ import annotations

import numpy as np
import pytest

from kuration.gameplay.collision.teapearce_adapter import (
    TEAPEARCE_SCHEMA,
    TeaPearceDust2Adapter,
    load_directional_arrays,
)


def _curr_vars(x, y, z, yaw, pitch, hp, wasd=None, obs_mode=0, weapon=None):
    cv = {
        "localpos1": x, "localpos2": y, "localpos3": z,
        "viewangle_xy": yaw, "viewangle_vert": pitch,
        "obs_health": hp, "tp_wasd": wasd or [], "obs_mode": obs_mode,
    }
    if weapon is not None:
        cv["gsi_weap_active"] = {"name": weapon}
    return cv


def _write_npy(tmp_path, rows):
    # Only column 1 (curr_vars) is read by the adapter -- other columns are cheap
    # placeholders matching the real file's (n_frames, 6) object-array shape.
    arr = np.empty((len(rows), 6), dtype=object)
    for i, cv in enumerate(rows):
        arr[i, 0] = None
        arr[i, 1] = cv
        arr[i, 2] = [[], 0.0, 0.0, 0, 0]
        arr[i, 3] = np.zeros(51)
        arr[i, 4] = np.zeros(54)
        arr[i, 5] = np.array([0.0, 0.0])
    path = tmp_path / "recording.npy"
    np.save(path, arr, allow_pickle=True)
    return str(path)


def test_ticks_yields_position_angle_hp_from_curr_vars(tmp_path):
    rows = [
        _curr_vars(100.0, 200.0, 30.0, 10.0, 5.0, 100, wasd=["w"]),
        _curr_vars(101.0, 200.0, 30.0, 10.0, 5.0, 100, wasd=["w", "a"]),
    ]
    path = _write_npy(tmp_path, rows)
    adapter = TeaPearceDust2Adapter()
    ticks = list(adapter.ticks(path))

    assert len(ticks) == 2
    assert ticks[0].time == pytest.approx(0 / TEAPEARCE_SCHEMA.sample_hz)
    assert ticks[1].time == pytest.approx(1 / TEAPEARCE_SCHEMA.sample_hz)
    np.testing.assert_allclose(ticks[0].entities["player"]["position"], [100.0, 200.0, 30.0])
    np.testing.assert_allclose(ticks[0].entities["player"]["angle"], [10.0, 5.0])
    assert ticks[0].entities["player"]["hp"] == 100.0


def test_btn_mask_encodes_held_movement_keys(tmp_path):
    rows = [
        _curr_vars(0, 0, 0, 0, 0, 100, wasd=["w"]),
        _curr_vars(0, 0, 0, 0, 0, 100, wasd=["w", "a"]),
        _curr_vars(0, 0, 0, 0, 0, 100, wasd=[]),
        _curr_vars(0, 0, 0, 0, 0, 100, wasd=["s", "d"]),
    ]
    path = _write_npy(tmp_path, rows)
    ticks = list(TeaPearceDust2Adapter().ticks(path))
    BIT_W, BIT_S, BIT_A, BIT_D = 8, 16, 512, 1024

    assert ticks[0].entities["player"]["btn_mask"] == BIT_W
    assert ticks[1].entities["player"]["btn_mask"] == BIT_W | BIT_A
    assert ticks[2].entities["player"]["btn_mask"] == 0
    assert ticks[3].entities["player"]["btn_mask"] == BIT_S | BIT_D


def test_observer_frames_are_marked_with_a_skip_flag(tmp_path):
    # obs_mode != 0 means the recorded position/velocity follow the OBSERVED player
    # while tp_wasd is still the recorder's keys -- exactly the telemetry/input mismatch
    # both collision algorithms would score as phantom collisions. The adapter marks
    # those frames FL_NOCLIP so is_skip_flags() excludes them; an earlier revision
    # reported flags=0.0 (merely "not onground"), which excluded nothing.
    from kuration.gameplay.collision.physics import FL_NOCLIP, FL_ONGROUND, is_skip_flags

    rows = [
        _curr_vars(0, 0, 0, 0, 0, 100, obs_mode=0),
        _curr_vars(0, 0, 0, 0, 0, 100, obs_mode=4),
    ]
    path = _write_npy(tmp_path, rows)
    ticks = list(TeaPearceDust2Adapter().ticks(path))

    assert ticks[0].entities["player"]["flags"] == FL_ONGROUND
    assert not is_skip_flags(int(ticks[0].entities["player"]["flags"]))
    assert ticks[1].entities["player"]["flags"] == float(FL_NOCLIP)
    assert is_skip_flags(int(ticks[1].entities["player"]["flags"]))


def test_check_source_passes_for_well_formed_file(tmp_path):
    rows = [_curr_vars(0, 0, 0, 0, 0, 100)]
    path = _write_npy(tmp_path, rows)
    TeaPearceDust2Adapter().check_source(path)  # no raise


def test_check_source_raises_for_missing_keys(tmp_path):
    arr = np.empty((1, 6), dtype=object)
    arr[0, 0] = None
    arr[0, 1] = {"obs_health": 100}  # missing position/angle keys
    for j in range(2, 6):
        arr[0, j] = None
    path = tmp_path / "bad.npy"
    np.save(path, arr, allow_pickle=True)

    with pytest.raises(ValueError, match="missing expected keys"):
        TeaPearceDust2Adapter().check_source(str(path))


def test_check_source_raises_for_wrong_shape(tmp_path):
    path = tmp_path / "wrong_shape.npy"
    np.save(path, np.zeros((10,)))
    with pytest.raises(ValueError, match="expected a"):
        TeaPearceDust2Adapter().check_source(str(path))


def test_weapon_field_is_categorical_and_reads_gsi_weap_active_name():
    assert TEAPEARCE_SCHEMA.fields["weapon"].kind == "categorical"


def test_ticks_yields_weapon_from_gsi_weap_active(tmp_path):
    rows = [
        _curr_vars(0, 0, 0, 0, 0, 100, weapon="weapon_ak47"),
        _curr_vars(0, 0, 0, 0, 0, 100),  # no active weapon this frame -- e.g. dead/observer
    ]
    path = _write_npy(tmp_path, rows)
    ticks = list(TeaPearceDust2Adapter().ticks(path))

    assert ticks[0].entities["player"]["weapon"] == "weapon_ak47"
    assert ticks[1].entities["player"]["weapon"] is None


def test_load_directional_arrays_flattens_the_tick_stream(tmp_path):
    rows = [
        _curr_vars(100.0, 200.0, 30.0, 10.0, 5.0, 100, wasd=["w"], weapon="weapon_ak47"),
        _curr_vars(101.0, 200.0, 30.0, 12.0, 5.0, 90, wasd=["w", "a"], weapon="weapon_ak47"),
    ]
    path = _write_npy(tmp_path, rows)
    arrs = load_directional_arrays(path)

    assert set(arrs) == {"position", "yaw", "btn_mask", "velocity", "hp", "flags", "weapon"}
    for key in arrs:
        assert len(arrs[key]) == 2
    np.testing.assert_allclose(arrs["position"][0], [100.0, 200.0, 30.0])
    np.testing.assert_allclose(arrs["yaw"], [10.0, 12.0])
    assert arrs["hp"].tolist() == [100.0, 90.0]
    assert list(arrs["weapon"]) == ["weapon_ak47", "weapon_ak47"]
    BIT_W, BIT_A = 8, 512
    assert arrs["btn_mask"].tolist() == [BIT_W, BIT_W | BIT_A]
