"""Physics constants and math ported from the reference `physics_utils.py` /
`action_state_analyzer.py` / `collision_classifier.py` scripts (CS2 tick physics).

This is a read-only port: the reference scripts are untouched. Only the runtime pieces
needed to SCORE ticks against an already-learned `action_state_dist.json` and CLASSIFY
flagged ticks are ported here -- the learn-phase helpers (`fit`, `build_by_player`,
`build_tick_index`) are not, since this plugin never fits a distribution itself, only
consumes one supplied via the `dist_path` operator parameter.
"""
from __future__ import annotations

import math

TICK_RATE = 64.0
DT = 4 / TICK_RATE  # seconds per sample; see reference physics_utils.py for rationale

# Source Engine IN_* button bits present in the btn_mask field.
BIT_W = 8
BIT_S = 16
BIT_DUCK = 4
BIT_A = 512
BIT_D = 1024
BIT_WALK = 65536
MOVE_BITS = BIT_W | BIT_S | BIT_DUCK | BIT_A | BIT_D | BIT_WALK

# Source Engine FL_* entity flags present in the flags field.
FL_ONGROUND = 0x0001
FL_DUCKING = 0x0002
FL_FROZEN = 0x0020
FL_NOCLIP = 0x20000000

# Physics features scored per tick -- order matches the reference FEATURES tuple.
FEATURES = (
    "v_fwd", "v_right", "v_up",
    "speed_2d",
    "dv_fwd", "dv_right", "dv_up",
    "d_yaw", "d_pitch",
    "delta_hp",
    "pos_error",
)

MAX_LEARN_SPEED = 320.0   # HU/s; used for is_teleport_candidate parity with the reference
MAX_DYAW_LEARN = 15.0     # degrees; fast-turn ticks skip v_fwd/v_right/dv_fwd/dv_right z-scores


def wrap_angle(a: float) -> float:
    """Wrap an angle (degrees) to the range (-180, 180]."""
    return ((a + 180.0) % 360.0) - 180.0


def yaw_average(y1: float, y2: float) -> float:
    """Circular mean of two yaw angles (degrees), handling +/-180 deg wraparound."""
    diff = wrap_angle(y2 - y1)
    return wrap_angle(y1 + diff / 2.0)


def to_local(dx: float, dy: float, dz: float, yaw_deg: float) -> tuple:
    """Convert a world-frame displacement to local-frame (forward, right, up) velocity."""
    yaw = math.radians(yaw_deg)
    vx, vy = dx / DT, dy / DT
    vf = vx * math.cos(yaw) + vy * math.sin(yaw)
    vr = vx * math.sin(yaw) - vy * math.cos(yaw)
    vu = dz / DT
    return vf, vr, vu


def z_score(val: float, stats: dict | None) -> float:
    """Standardised score: (val - mean) / std. 0.0 if stats is missing or std ~ 0."""
    if stats is None or stats["std"] < 0.01:
        return 0.0
    return (val - stats["mean"]) / stats["std"]


def combo_label(move_bits: int) -> str:
    """Convert a MOVE_BITS-masked btn_mask to a human-readable string, e.g. 'W+A'."""
    parts = []
    if move_bits & BIT_W:
        parts.append("W")
    if move_bits & BIT_S:
        parts.append("S")
    if move_bits & BIT_A:
        parts.append("A")
    if move_bits & BIT_D:
        parts.append("D")
    if move_bits & BIT_WALK:
        parts.append("walk")
    if move_bits & BIT_DUCK:
        parts.append("duck")
    return "+".join(parts) if parts else "idle"


def condition_key(btn_mask: int, onground: bool, ducking: bool) -> str:
    """Canonical condition string for distribution lookup, e.g. 'W+A|ground'.

    Matches the reference `condition_key()` exactly: weapon is intentionally excluded
    from the key (pooling across weapon classes gives each condition more samples;
    speed-cap violations are a separate speedhack concern, not a collision concern).
    """
    combo = combo_label(int(btn_mask) & MOVE_BITS)
    gnd = "ground" if onground else "air"
    dk = "_duck" if ducking else ""
    return f"{combo}|{gnd}{dk}"


def is_skip_flags(flags: int) -> bool:
    """True if this tick should be excluded entirely (round-freeze or SourceTV observer)."""
    return bool(int(flags) & (FL_FROZEN | FL_NOCLIP))
