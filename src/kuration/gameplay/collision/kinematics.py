"""Player kinematics derived from three consecutive position/angle samples.

`extract_kinematics` is the one genuinely reusable seam inside `collision_operator.py`'s
fused per-tick scan/classify loop: local-frame velocity, turn rate, and pose-error features
assembled from `physics.py`'s primitives (`to_local`, `wrap_angle`, `yaw_average`). It has no
dependency on the anomaly-gating or raycast-classification stages that consume its output, so
it is usable on its own by anything that needs frame-relative velocity/turn features from raw
position+angle telemetry -- a movement-quality check or a speed-hack detector, for instance --
without pulling in a learned baseline or map geometry.

The anomaly-gate and raycast-classify stages that follow in `collision_operator.py` remain
fused: gating is a threshold immediately consumed by classification a few lines later on the
same tick, sharing local state (which features triggered, which button/flag bits are set) that
would only turn into an artificial `Segment`-shaped handoff if split into their own components.
"""
from __future__ import annotations

import math

from .physics import to_local, wrap_angle, yaw_average

#: Feature keys `extract_kinematics` always returns, in `physics.FEATURES` order plus the two
#: fast-turn detection keys (`dyaw10`, `dyaw21`) `collision_operator.py`'s gate reads directly.
KINEMATICS_KEYS = (
    "v_fwd", "v_right", "v_up",
    "speed_2d",
    "dv_fwd", "dv_right", "dv_up",
    "d_yaw", "d_pitch",
    "delta_hp",
    "pos_error",
    "dyaw10", "dyaw21",
)


def extract_kinematics(pm2: tuple, pm1: tuple, p0: tuple) -> dict:
    """Derive one tick's kinematics features from itself and its two predecessors.

    ``pm2``/``pm1``/``p0`` are ``(pos: array(3,), yaw, pitch, hp, btn_mask, flags)`` tuples
    for ``t-2``/``t-1``/``t`` respectively -- only ``pos``/``yaw``/``pitch``/``hp`` (indices
    0-3) are read; ``btn_mask``/``flags`` may be any value (collision_operator.py's callers
    pass the full per-tick record for convenience).

    Returns a dict with every key in :data:`KINEMATICS_KEYS`: local-frame forward/right/up
    velocity and speed at ``t`` (``v_fwd``/``v_right``/``v_up``/``speed_2d``), their change
    since ``t-1`` (``dv_fwd``/``dv_right``/``dv_up``), turn rate and pitch change
    (``d_yaw``/``d_pitch``), health delta (``delta_hp``), constant-velocity position-prediction
    error (``pos_error``), and the two raw (unwrapped-magnitude) yaw deltas used to detect a
    fast turn (``dyaw10`` = ``|yaw(t) - yaw(t-1)|``, ``dyaw21`` = ``|yaw(t-1) - yaw(t-2)|``).
    """
    pos0, pos1, pos2 = p0[0], pm1[0], pm2[0]
    yaw0, yaw1, yaw2 = p0[1], pm1[1], pm2[1]
    pitch0, pitch1 = p0[2], pm1[2]

    dx10, dy10, dz10 = pos0[0] - pos1[0], pos0[1] - pos1[1], pos0[2] - pos1[2]
    dx21, dy21, dz21 = pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2]

    dyaw10 = abs(wrap_angle(yaw0 - yaw1))
    dyaw21 = abs(wrap_angle(yaw1 - yaw2))

    yaw10 = yaw_average(yaw1, yaw0)
    yaw21 = yaw_average(yaw2, yaw1)

    vf0, vr0, vu0 = to_local(dx10, dy10, dz10, yaw10)
    vf1, vr1, vu1 = to_local(dx21, dy21, dz21, yaw21)

    spd0 = math.sqrt(vf0 ** 2 + vr0 ** 2)

    d_yaw = wrap_angle(yaw0 - yaw1)
    d_pitch = wrap_angle(pitch0 - pitch1)
    delta_hp = float(p0[3] - pm1[3])

    exp_x, exp_y, exp_z = pos1[0] + dx21, pos1[1] + dy21, pos1[2] + dz21
    pos_error = math.sqrt(
        (pos0[0] - exp_x) ** 2 + (pos0[1] - exp_y) ** 2 + (pos0[2] - exp_z) ** 2
    )

    return {
        "v_fwd": vf0, "v_right": vr0, "v_up": vu0,
        "speed_2d": spd0,
        "dv_fwd": vf0 - vf1, "dv_right": vr0 - vr1, "dv_up": vu0 - vu1,
        "d_yaw": d_yaw, "d_pitch": d_pitch,
        "delta_hp": delta_hp,
        "pos_error": pos_error,
        "dyaw10": dyaw10, "dyaw21": dyaw21,
    }
