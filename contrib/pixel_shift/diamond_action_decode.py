"""diamond_action_decode -- low-level decoding of the DIAMOND/TeaPearce 51-dim action
vector (11 keys + 2 clicks + 23-bin mouse_x one-hot + 15-bin mouse_y one-hot -- see
asg2diamond/action_map.py for the exact layout this mirrors) and isolated-press detection
+ pixel-displacement measurement against real video frames.

SHARED, DATA-FORMAT-SPECIFIC, NOT ASG-SPECIFIC: this is the reusable core both
pixel_shift_operator.py (which control, how much shift, per instance -- no notion of
"consistent") and any consistency/outlier operator built on top of it depend on. It knows
about DIAMOND's specific action encoding (so it can decode ANY dataset using that
convention, not just our own ASG captures) but knows nothing about reference values,
floors, or expected directions -- that's layered on top, not in here.

Mirrors batch_job_sensitivity_w.py's per-control logic exactly (same isolated-press
definition, same axis-appropriate flow projection), reimplemented with `control` as an
explicit argument instead of a module-global set from an env var, so one call can check
every control, not just whichever one a process was launched with.
"""
from __future__ import annotations

import cv2
import numpy as np

KEY_NAMES = ["W", "A", "S", "D", "SPACE", "CTRL", "SHIFT", "1", "2", "3", "R"]
N_KEYS = 11
N_MOUSE_X = 23
N_MOUSE_Y = 15
MOUSE_X_POSSIBLES = np.array(
    [-1000, -500, -300, -200, -100, -60, -30, -20, -10, -4, -2, 0, 2, 4, 10, 20, 30, 60, 100, 200, 300, 500, 1000],
    dtype=np.float64,
)
MOUSE_Y_POSSIBLES = np.array(
    [-200, -100, -50, -20, -10, -4, -2, 0, 2, 4, 10, 20, 50, 100, 200], dtype=np.float64
)
MX_OFF = N_KEYS + 2
MY_OFF = MX_OFF + N_MOUSE_X

CONTROL_AXES = {
    "W": {"opposing": "S", "axis": "longitudinal"},
    "S": {"opposing": "W", "axis": "longitudinal"},
    "A": {"opposing": "D", "axis": "lateral"},
    "D": {"opposing": "A", "axis": "lateral"},
    "MOUSE_X_POS": {"opposing": "MOUSE_X_NEG", "axis": "yaw"},
    "MOUSE_X_NEG": {"opposing": "MOUSE_X_POS", "axis": "yaw"},
    "MOUSE_Y_POS": {"opposing": "MOUSE_Y_NEG", "axis": "pitch"},
    "MOUSE_Y_NEG": {"opposing": "MOUSE_Y_POS", "axis": "pitch"},
}
MOUSE_CONTROLS = {"MOUSE_X_POS", "MOUSE_X_NEG", "MOUSE_Y_POS", "MOUSE_Y_NEG"}
MOUSE_SIGN = {"MOUSE_X_POS": 1.0, "MOUSE_X_NEG": -1.0, "MOUSE_Y_POS": 1.0, "MOUSE_Y_NEG": -1.0}
MOUSE_AXIS_OF = {"MOUSE_X_POS": "x", "MOUSE_X_NEG": "x", "MOUSE_Y_POS": "y", "MOUSE_Y_NEG": "y"}
BOTTOM_CROP_FRAC = 0.18
FARNEBACK_KWARGS = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

ALL_CONTROLS = list(CONTROL_AXES.keys())  # W, S, A, D, MOUSE_X_POS, MOUSE_X_NEG, MOUSE_Y_POS, MOUSE_Y_NEG


def mouse_delta_value(Y: np.ndarray, which: str) -> np.ndarray:
    if which == "x":
        idx = Y[:, MX_OFF:MX_OFF + N_MOUSE_X].argmax(axis=1)
        return MOUSE_X_POSSIBLES[idx]
    idx = Y[:, MY_OFF:MY_OFF + N_MOUSE_Y].argmax(axis=1)
    return MOUSE_Y_POSSIBLES[idx]


def mouse_delta_sign(Y: np.ndarray, which: str) -> np.ndarray:
    return np.sign(mouse_delta_value(Y, which))


def control_active_mask(Y: np.ndarray, control: str) -> np.ndarray:
    if control in MOUSE_CONTROLS:
        return mouse_delta_sign(Y, MOUSE_AXIS_OF[control]) == MOUSE_SIGN[control]
    return Y[:, KEY_NAMES.index(control)] > 0.5


def solo_labels(Y: np.ndarray) -> np.ndarray:
    """Per-frame label (0..14) of the sole active control, -1 if none or 2+ active."""
    keys_active = Y[:, :N_KEYS] > 0.5
    l_click = (Y[:, N_KEYS] > 0.5)[:, None]
    r_click = (Y[:, N_KEYS + 1] > 0.5)[:, None]
    mx_idx = Y[:, MX_OFF:MX_OFF + N_MOUSE_X].argmax(axis=1)
    my_idx = Y[:, MY_OFF:MY_OFF + N_MOUSE_Y].argmax(axis=1)
    mx_active = (MOUSE_X_POSSIBLES[mx_idx] != 0)[:, None]
    my_active = (MOUSE_Y_POSSIBLES[my_idx] != 0)[:, None]
    active = np.concatenate([keys_active, l_click, r_click, mx_active, my_active], axis=1)
    labels = np.full(Y.shape[0], -1, dtype=np.int8)
    indep_mask = active.sum(axis=1) == 1
    labels[indep_mask] = active[indep_mask].argmax(axis=1)
    return labels


def other_controls_active(Y: np.ndarray, control: str) -> np.ndarray:
    """True where any control is active that should end an isolated-press sequence for
    `control` -- mirrors batch_job_sensitivity_w.py's other_controls_active exactly (same
    per-axis aftermath rules: orthogonal keys stay free, mouse_x is additionally blocked
    for the lateral axis, etc.), reimplemented here to take `control` as an argument."""
    axis = CONTROL_AXES[control]["axis"]
    opposing = CONTROL_AXES[control]["opposing"]

    if control in MOUSE_CONTROLS:
        keys_active = (Y[:, :N_KEYS] > 0.5).any(axis=1)
        l_click = Y[:, N_KEYS] > 0.5
        r_click = Y[:, N_KEYS + 1] > 0.5
        opposing_active = mouse_delta_sign(Y, MOUSE_AXIS_OF[control]) == -MOUSE_SIGN[control]
        return keys_active | l_click | r_click | opposing_active

    opposing_idx = KEY_NAMES.index(opposing)
    non_orthogonal_keys = [opposing_idx, 4, 5, 6, 7, 8, 9, 10]
    keys_active = (Y[:, non_orthogonal_keys] > 0.5).any(axis=1)
    l_click = Y[:, N_KEYS] > 0.5
    r_click = Y[:, N_KEYS + 1] > 0.5
    blocked = keys_active | l_click | r_click

    if axis == "lateral":
        mx_idx = Y[:, MX_OFF:MX_OFF + N_MOUSE_X].argmax(axis=1)
        blocked = blocked | (MOUSE_X_POSSIBLES[mx_idx] != 0)
    return blocked


def find_isolated_press_sequences(Y: np.ndarray, control: str, press_len: int):
    """(start, end) inclusive frame ranges of a clean, isolated `press_len`-frame press of
    `control` followed by at least one real aftermath frame. Action-vector-only -- no pixel
    data touched here. Mirrors batch_job_sensitivity_w.py's find_press_sequences, with
    `control`/`press_len` as arguments instead of module globals."""
    if control in MOUSE_CONTROLS:
        solo_idx = MX_OFF if MOUSE_AXIS_OF[control] == "x" else MY_OFF
        solo = solo_labels(Y)
        is_solo = (solo == solo_idx) & (mouse_delta_sign(Y, MOUSE_AXIS_OF[control]) == MOUSE_SIGN[control])
    else:
        solo = solo_labels(Y)
        is_solo = solo == KEY_NAMES.index(control)

    n = len(is_solo)
    is_start = np.empty(n, dtype=bool)
    is_start[0] = True
    is_start[1:] = is_solo[1:] != is_solo[:-1]
    run_starts = np.where(is_start)[0]
    run_lengths = np.diff(np.append(run_starts, n))
    press_starts = run_starts[is_solo[run_starts] & (run_lengths == press_len)].tolist()

    blocked = other_controls_active(Y, control)
    active = control_active_mask(Y, control)

    sequences = []
    for t in press_starts:
        press_end = t + press_len - 1
        end = press_end
        for j in range(press_end + 1, n):
            if blocked[j] or active[j]:
                break
            end = j
        if end > press_end:
            sequences.append((t, end))
    return sequences


def _prep_gray(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    return gray[: int(h * (1 - BOTTOM_CROP_FRAC)), :]


_RADIAL_CACHE: dict = {}


def _radial_unit_vectors(h: int, w: int):
    key = (h, w)
    if key not in _RADIAL_CACHE:
        ys, xs = np.mgrid[0:h, 0:w]
        dx, dy = xs - w / 2.0, ys - h / 2.0
        r = np.sqrt(dx * dx + dy * dy)
        r_safe = np.where(r < 1e-6, 1e-6, r)
        _RADIAL_CACHE[key] = (dx / r_safe, dy / r_safe, r)
    return _RADIAL_CACHE[key]


def displacement_median(flow: np.ndarray, control: str) -> float:
    """Axis-appropriate median flow projection -- radial for W/S, horizontal for A/D and
    mouse_x, vertical for mouse_y. Mirrors displacement_median() in batch_job_sensitivity_w.py."""
    axis = CONTROL_AXES[control]["axis"]
    if axis == "longitudinal":
        h, w = flow.shape[:2]
        ux, uy, r = _radial_unit_vectors(h, w)
        radial = flow[..., 0] * ux + flow[..., 1] * uy
        return float(np.median(radial[r > 2.0]))
    if axis == "pitch":
        return float(np.median(flow[..., 1]))
    return float(np.median(flow[..., 0]))  # lateral or yaw


def release_transition_displacement(frames: np.ndarray, actions: np.ndarray, control: str, press_len: int) -> list:
    """List of (start_frame_idx, SIGNED px shift) -- one per isolated press instance found
    for `control` at `press_len` in this clip, using the release-transition frame pair
    (index `press_len - 1` within each sequence). Sign is kept (not abs'd) -- direction is
    meaningful data, not something this layer decides is good or bad.
    `start` is returned alongside the shift so callers can look up per-instance context
    (e.g. a mouse control's actual delta value at that frame) without re-deriving it."""
    sequences = find_isolated_press_sequences(actions, control, press_len)
    out = []
    for start, end in sequences:
        release_idx = start + press_len - 1
        if release_idx + 1 > end:
            continue
        gray0 = _prep_gray(frames[release_idx])
        gray1 = _prep_gray(frames[release_idx + 1])
        flow = cv2.calcOpticalFlowFarneback(gray0, gray1, None, **FARNEBACK_KWARGS)
        out.append((start, displacement_median(flow, control)))
    return out
