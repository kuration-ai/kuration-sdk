"""Tests for the acceleration-ramp fitting used by DirectionalCollisionAnalyzer.

`find_clean_press_ramps` accepts or rejects candidate presses on several independent
criteria; each rejection test below starts from one shared clean, accepted fixture
(`_clean_press_arrays`) and mutates exactly one thing so a failure points at the specific
gate that broke.
"""
from __future__ import annotations

import numpy as np
import pytest

from kuration.gameplay.collision.physics import BIT_A, BIT_W
from kuration.gameplay.collision.ramp_fitting import (
    find_clean_press_ramps,
    fit_exponential_ramp,
    fit_ramp_models,
    load_ramp_models,
    save_ramp_models,
)

RAMP_TICKS = 20
SETTLE_TICKS = 3
N = SETTLE_TICKS + 1 + 1 + RAMP_TICKS  # idle lead-in, transition tick, then the ramp window


def _clean_press_arrays(*, combo_bits=BIT_W, weapon="weapon_ak47"):
    """A minimal array set with exactly one clean press of `combo_bits` at tick
    SETTLE_TICKS+2 -> ... held through a full ramp window, everything else idle."""
    btn_mask = np.zeros(N, dtype=np.int64)
    btn_mask[SETTLE_TICKS + 2:] = combo_bits
    velocity = np.zeros((N, 2))
    hp = np.full(N, 100.0)
    weapons = np.array([weapon] * N, dtype=object)
    return btn_mask, velocity, hp, weapons


def test_find_clean_press_ramps_detects_a_clean_press():
    btn_mask, velocity, hp, weapon = _clean_press_arrays()
    by_weapon = find_clean_press_ramps(btn_mask, velocity, hp, weapon, combo_bits=BIT_W)
    assert list(by_weapon.keys()) == ["weapon_ak47"]
    assert len(by_weapon["weapon_ak47"]) == 1
    assert by_weapon["weapon_ak47"][0].shape == (RAMP_TICKS,)


def test_find_clean_press_ramps_rejects_pre_press_movement():
    btn_mask, velocity, hp, weapon = _clean_press_arrays()
    btn_mask[SETTLE_TICKS] = BIT_A  # a movement key held just before the "clean" press
    by_weapon = find_clean_press_ramps(btn_mask, velocity, hp, weapon, combo_bits=BIT_W)
    assert by_weapon == {}


def test_find_clean_press_ramps_rejects_high_pre_press_speed():
    btn_mask, velocity, hp, weapon = _clean_press_arrays()
    velocity[SETTLE_TICKS] = [50.0, 0.0]  # already moving fast despite no key held
    by_weapon = find_clean_press_ramps(btn_mask, velocity, hp, weapon, combo_bits=BIT_W)
    assert by_weapon == {}


def test_find_clean_press_ramps_rejects_weapon_change_mid_ramp():
    btn_mask, velocity, hp, weapon = _clean_press_arrays()
    weapon[SETTLE_TICKS + 10] = "weapon_m4a1"
    by_weapon = find_clean_press_ramps(btn_mask, velocity, hp, weapon, combo_bits=BIT_W)
    assert by_weapon == {}


def test_find_clean_press_ramps_rejects_death_mid_ramp():
    btn_mask, velocity, hp, weapon = _clean_press_arrays()
    hp[SETTLE_TICKS + 10] = 0.0
    by_weapon = find_clean_press_ramps(btn_mask, velocity, hp, weapon, combo_bits=BIT_W)
    assert by_weapon == {}


def test_find_clean_press_ramps_rejects_key_change_mid_ramp():
    btn_mask, velocity, hp, weapon = _clean_press_arrays()
    btn_mask[SETTLE_TICKS + 10] = BIT_W | BIT_A  # a second key gets added mid-ramp
    by_weapon = find_clean_press_ramps(btn_mask, velocity, hp, weapon, combo_bits=BIT_W)
    assert by_weapon == {}


def test_fit_exponential_ramp_recovers_a_known_analytic_curve():
    tau, cap = 5.0, 150.0
    t = np.arange(RAMP_TICKS, dtype=np.float64)
    curve = cap * (1.0 - np.exp(-t / tau))
    fit = fit_exponential_ramp([curve.copy() for _ in range(5)])
    assert fit["tau_ticks"] == pytest.approx(tau, abs=0.2)
    assert fit["cap_hu_s"] == pytest.approx(cap, abs=5.0)
    assert fit["n"] == 5
    assert len(fit["mean_curve"]) == RAMP_TICKS


def test_fit_ramp_models_pools_ramps_across_sources_and_respects_min_ramps():
    source_a = _clean_press_arrays()
    source_b = _clean_press_arrays()
    sources = [source_a, source_b]  # each contributes exactly 1 clean W-press -> 2 pooled

    models_loose = fit_ramp_models(sources, min_ramps=2)
    assert (BIT_W, "weapon_ak47") in models_loose

    models_strict = fit_ramp_models(sources, min_ramps=5)
    assert (BIT_W, "weapon_ak47") not in models_strict


def test_save_and_load_ramp_models_round_trips(tmp_path):
    models = {(BIT_W, "weapon_ak47"): (3.5, 210.0), (BIT_A, "weapon_m4a1"): (4.0, 180.0)}
    path = tmp_path / "ramp_models.json"
    save_ramp_models(models, str(path))
    loaded = load_ramp_models(str(path))
    assert loaded == models
