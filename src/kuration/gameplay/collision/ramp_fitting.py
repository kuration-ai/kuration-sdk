"""Acceleration-ramp model fitting for the directional-residual collision analyzer.

Ported (unmodified math) from the reference `acceleration_curve.py` script: for each
(movement-combo, weapon) pair, find "clean" presses -- held keys transitioning from
nothing held to exactly one combo, from a near-standstill, with the same weapon and no
other key change for the whole ramp window -- and fit an exponential
speed(t) = cap*(1 - exp(-t/tau)) curve to them by least squares over (tau, cap).

`directional_collision_analyzer.py`'s DirectionalCollisionAnalyzer needs this fitted
{(combo_bits, weapon): (tau_ticks, cap_hu_s)} table as a construction-time parameter, the
same role `action_state_dist.json` plays for the sibling `CollisionMechanics` transform:
it must be fit ONCE, pooled across many sources, and then reused across every analysis run
-- a single source rarely has enough clean presses of every combo+weapon pair to fit
reliably (confirmed in practice: pooling 190 whole TeaPearce matches still only produced
a reliable fit for 1 of 8 combos). Note this is a deliberate departure from
collision_operator.py's own precedent, which keeps baseline-fitting entirely out of the
SDK (its module docstring: "this plugin never fits a distribution itself, only consumes
one supplied via the dist_path parameter") -- that call was made because the reference
baseline-fitting pipeline is a large, multi-script external tool. This ramp fit is small,
pure-array, and self-contained enough that shipping it is worth the inconsistency.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .physics import BIT_A, BIT_D, BIT_S, BIT_W

# WASD strafe bits only -- distinct from physics.MOVE_BITS, which also folds in duck/walk
# for the sibling anomaly-baseline operator's condition_key(). The acceleration-ramp model
# here only ever models the 8 pure single/diagonal WASD combos below.
ALL_MOVE_BITS = BIT_W | BIT_S | BIT_A | BIT_D

COMBOS = {
    "W": BIT_W, "S": BIT_S, "A": BIT_A, "D": BIT_D,
    "W+A": BIT_W | BIT_A, "W+D": BIT_W | BIT_D,
    "S+A": BIT_S | BIT_A, "S+D": BIT_S | BIT_D,
}


def find_clean_press_ramps(
    btn_mask: np.ndarray, velocity: np.ndarray, hp: np.ndarray, weapon: np.ndarray,
    *, combo_bits: int, ramp_ticks: int = 20,
    settle_ticks: int = 3, max_pre_press_speed: float = 15.0,
) -> dict:
    """Find "clean" presses of one exact key combo (e.g. W+A): held keys transition from
    nothing held to EXACTLY `combo_bits` (no more, no less) at tick i->i+1, the player was
    slow/still and had no movement key held for `settle_ticks` beforehand (a real 0-start),
    the SAME exact combo (no key added/removed) and the SAME weapon are held for the whole
    ramp window, and the player stays alive throughout.

    Returns {weapon_name: [speed_array, ...]} -- ramps bucketed by the (constant, verified)
    weapon held during each one, since acceleration caps are weapon-dependent.
    """
    n = len(hp)
    speed = np.linalg.norm(velocity[:, :2], axis=1)  # horizontal speed, matches classify math
    held = btn_mask.astype(np.int64)
    is_combo = (held & ALL_MOVE_BITS) == combo_bits
    any_move = (held & ALL_MOVE_BITS) != 0

    by_weapon: dict = defaultdict(list)
    for i in range(settle_ticks, n - ramp_ticks):
        if is_combo[i] or not is_combo[i + 1]:
            continue  # need a clean transition into exactly this combo at i -> i+1
        pre = slice(i - settle_ticks, i + 1)
        if any_move[pre].any():
            continue
        if (speed[pre] > max_pre_press_speed).any():
            continue
        window = slice(i + 1, i + 1 + ramp_ticks)
        if hp[window].min() <= 0:
            continue
        if not np.all(is_combo[window]):
            continue  # keep the ramp clean of a key changing mid-press
        weapons_in_window = weapon[window]
        w0 = weapons_in_window[0]
        if w0 is None or not all(w == w0 for w in weapons_in_window):
            continue  # weapon must stay constant (and known) through the whole ramp
        by_weapon[w0].append(speed[window].copy())
    return dict(by_weapon)


def fit_exponential_ramp(ramps: list) -> dict:
    """Align ramps by ticks-since-press, average, and fit speed(t) = cap*(1 - exp(-t/tau))
    by least squares over (tau, cap). Returns {tau_ticks, cap_hu_s, mean_curve, n}."""
    arr = np.stack(ramps)  # (n_ramps, ramp_ticks)
    mean_curve = arr.mean(axis=0)
    t = np.arange(arr.shape[1], dtype=np.float64)

    best = None
    for tau in np.linspace(0.5, 30.0, 300):
        basis = 1.0 - np.exp(-t / tau)
        denom = float(np.dot(basis, basis))
        if denom < 1e-9:
            continue
        cap = float(np.dot(basis, mean_curve) / denom)
        pred = cap * basis
        sse = float(np.sum((mean_curve - pred) ** 2))
        if best is None or sse < best[0]:
            best = (sse, tau, cap)
    if best is None:  # defensive: the positive tau grid always produces a usable basis
        raise ValueError("could not fit an exponential ramp")
    _, tau, cap = best
    return {"tau_ticks": tau, "cap_hu_s": cap, "mean_curve": mean_curve.tolist(), "n": len(ramps)}


def fit_ramp_models(sources: list, *, min_ramps: int = 5) -> dict:
    """Fit a {(combo_bits, weapon_name): (tau_ticks, cap_hu_s)} table pooled across many
    sources at once. `sources` is a list of (btn_mask, velocity, hp, weapon) canonical
    per-source array tuples. Combos with fewer than `min_ramps` pooled clean presses are
    omitted from the result -- callers should supply a fallback ramp (e.g. the
    single best-populated model) for any (combo, weapon) pair absent from it."""
    all_ramps: dict = defaultdict(list)
    for btn_mask, velocity, hp, weapon in sources:
        for combo_bits in COMBOS.values():
            by_weapon = find_clean_press_ramps(btn_mask, velocity, hp, weapon, combo_bits=combo_bits)
            for weapon_name, ramps in by_weapon.items():
                all_ramps[(combo_bits, weapon_name)].extend(ramps)

    models = {}
    for (combo_bits, weapon_name), ramps in all_ramps.items():
        if len(ramps) < min_ramps:
            continue
        fit = fit_exponential_ramp(ramps)
        models[(combo_bits, weapon_name)] = (fit["tau_ticks"], fit["cap_hu_s"])
    return models


def save_ramp_models(models: dict, path: str) -> None:
    """Write a fitted {(combo_bits, weapon): (tau, cap)} table to disk as JSON. Tuple keys
    aren't JSON-safe, so this flattens to a list of records; `load_ramp_models` reverses it."""
    records = [
        {"combo_bits": combo_bits, "weapon": weapon, "tau_ticks": tau, "cap_hu_s": cap}
        for (combo_bits, weapon), (tau, cap) in models.items()
    ]
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def load_ramp_models(path: str) -> dict:
    """Load a ramp-models table written by `save_ramp_models` back into the
    {(combo_bits, weapon): (tau, cap)} shape DirectionalCollisionAnalyzer's runtime lookup
    needs."""
    with open(path) as f:
        records = json.load(f)
    return {(r["combo_bits"], r["weapon"]): (r["tau_ticks"], r["cap_hu_s"]) for r in records}
