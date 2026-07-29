"""Low-level decoding of the DIAMOND/TeaPearce 51-dimensional action vector.

The layout is 11 keys + 2 clicks + 23-bin mouse_x one-hot + 15-bin mouse_y one-hot; see
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

from functools import lru_cache

import numpy as np


# cv2 is imported lazily, inside the two functions that actually touch a frame. The
# built-in operator is registered for every SDK installation, so an eager import would
# break core-only users that do not install the pixel-shift extra.
def _require_cv2():
    try:
        import cv2

        return cv2
    except ImportError as e:
        raise ImportError(
            "pixel_shift requires opencv (cv2) for optical-flow measurement. "
            "Install or repair it with `pip install \"kuration-sdk[pixel-shift]\"`."
        ) from e


KEY_NAMES = ["W", "A", "S", "D", "SPACE", "CTRL", "SHIFT", "1", "2", "3", "R"]
N_KEYS = 11
N_MOUSE_X = 23
N_MOUSE_Y = 15
MOUSE_X_POSSIBLES = np.array(
    [
        -1000,
        -500,
        -300,
        -200,
        -100,
        -60,
        -30,
        -20,
        -10,
        -4,
        -2,
        0,
        2,
        4,
        10,
        20,
        30,
        60,
        100,
        200,
        300,
        500,
        1000,
    ],
    dtype=np.float64,
)
MOUSE_Y_POSSIBLES = np.array(
    [-200, -100, -50, -20, -10, -4, -2, 0, 2, 4, 10, 20, 50, 100, 200], dtype=np.float64
)
MX_OFF = N_KEYS + 2
MY_OFF = MX_OFF + N_MOUSE_X

# solo_labels()'s `active` array is [keys(11), l_click, r_click, mx_active, my_active] --
# a 15-wide one-hot whose argmax is a label in [0, 14], NOT a byte-offset into the raw
# 51-dim action vector. MOUSE_X_LABEL/MOUSE_Y_LABEL are that label space; MX_OFF/MY_OFF
# (13 and 36) are a completely different numbering (offsets into Y's columns) that must
# never be compared against a solo_labels() output. MX_OFF happens to equal 13, the
# correct mouse_x label, purely by coincidence of N_KEYS+2 == N_KEYS+1+1 -- MY_OFF (36) has
# no such coincidence and comparing solo == MY_OFF is always False, since solo_labels()
# can only ever return -1..14. That silently disabled all isolated mouse_y-press detection
# (caught in review -- solo_idx must use MOUSE_Y_LABEL, not MY_OFF).
MOUSE_X_LABEL = N_KEYS + 2  # = 13: mx_active's column in solo_labels()'s `active`
MOUSE_Y_LABEL = N_KEYS + 3  # = 14: my_active's column in solo_labels()'s `active`

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
FARNEBACK_KWARGS = dict(
    pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

ALL_CONTROLS = list(
    CONTROL_AXES.keys()
)  # W, S, A, D, MOUSE_X_POS, MOUSE_X_NEG, MOUSE_Y_POS, MOUSE_Y_NEG
ACTION_WIDTH = N_KEYS + 2 + N_MOUSE_X + N_MOUSE_Y
BOUNDARY_POLICIES = ("exclude", "error", "assume-inactive")


def validate_actions(actions, *, name: str = "actions") -> np.ndarray:
    """Return a validated DIAMOND action matrix.

    The accepted representation is numeric ``(T, 51)`` data with finite values, binary
    key/click columns, and exactly one active bin in each mouse one-hot range per frame.
    """
    value = np.asarray(actions)
    if value.ndim != 2 or value.shape[1:] != (ACTION_WIDTH,):
        raise ValueError(f"{name} must have shape (T, {ACTION_WIDTH}), got {value.shape}")
    if value.dtype.kind not in "biuf":
        raise ValueError(f"{name} must have a numeric dtype, got {value.dtype}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    binary = value[:, :MX_OFF]
    if np.any((binary != 0) & (binary != 1)):
        raise ValueError(f"{name} key/click columns must contain only 0 or 1")
    for label, start, width in (("mouse_x", MX_OFF, N_MOUSE_X), ("mouse_y", MY_OFF, N_MOUSE_Y)):
        bins = value[:, start : start + width]
        if np.any((bins != 0) & (bins != 1)) or np.any(bins.sum(axis=1) != 1):
            raise ValueError(f"{name} {label} bins must be strict one-hot rows")
    return value


def validate_frames(frames, *, steps: int | None = None, name: str = "frames") -> np.ndarray:
    """Return validated RGB/BGR video frames suitable for Farneback optical flow."""
    value = np.asarray(frames)
    if value.ndim != 4 or value.shape[-1:] != (3,):
        raise ValueError(f"{name} must have shape (T, H, W, 3), got {value.shape}")
    if steps is not None and value.shape[0] != steps:
        raise ValueError(f"{name} has {value.shape[0]} steps but actions has {steps}")
    if value.shape[1] < 4 or value.shape[2] < 4:
        raise ValueError(f"{name} spatial dimensions must both be at least 4 pixels")
    if value.dtype not in (np.dtype(np.uint8), np.dtype(np.float32)):
        raise ValueError(f"{name} dtype must be uint8 or float32, got {value.dtype}")
    if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _validate_control(control: str) -> str:
    if not isinstance(control, str) or control not in CONTROL_AXES:
        raise ValueError(f"unknown control {control!r}; expected {ALL_CONTROLS}")
    return control


def _validate_axis(which: str) -> str:
    if not isinstance(which, str) or which not in {"x", "y"}:
        raise ValueError("mouse axis must be 'x' or 'y'")
    return which


def _validate_boundary(boundary: str) -> str:
    if not isinstance(boundary, str) or boundary not in BOUNDARY_POLICIES:
        raise ValueError(f"boundary must be one of {BOUNDARY_POLICIES}, got {boundary!r}")
    return boundary


def validate_boundary(boundary: str) -> str:
    """Return a supported segment-boundary policy or raise a clear configuration error."""
    return _validate_boundary(boundary)


def validate_color_space(color_space: str) -> str:
    """Normalize the two supported channel-order declarations."""
    if not isinstance(color_space, str):
        raise ValueError("color_space must be 'RGB' or 'BGR'")
    value = color_space.upper()
    if value not in {"RGB", "BGR"}:
        raise ValueError(f"unsupported color_space {color_space!r}; expected 'RGB' or 'BGR'")
    return value


def _neutral_actions(n: int = 1) -> np.ndarray:
    out = np.zeros((n, ACTION_WIDTH), dtype=np.float64)
    out[:, MX_OFF + int(np.where(MOUSE_X_POSSIBLES == 0)[0][0])] = 1
    out[:, MY_OFF + int(np.where(MOUSE_Y_POSSIBLES == 0)[0][0])] = 1
    return out


def _mouse_delta_value(Y: np.ndarray, which: str) -> np.ndarray:
    if which == "x":
        idx = Y[:, MX_OFF : MX_OFF + N_MOUSE_X].argmax(axis=1)
        return MOUSE_X_POSSIBLES[idx]
    idx = Y[:, MY_OFF : MY_OFF + N_MOUSE_Y].argmax(axis=1)
    return MOUSE_Y_POSSIBLES[idx]


def mouse_delta_value(Y: np.ndarray, which: str) -> np.ndarray:
    return _mouse_delta_value(validate_actions(Y), _validate_axis(which))


def mouse_delta_sign(Y: np.ndarray, which: str) -> np.ndarray:
    return np.sign(_mouse_delta_value(validate_actions(Y), _validate_axis(which)))


def _control_active_mask(Y: np.ndarray, control: str) -> np.ndarray:
    if control in MOUSE_CONTROLS:
        return np.sign(_mouse_delta_value(Y, MOUSE_AXIS_OF[control])) == MOUSE_SIGN[control]
    return Y[:, KEY_NAMES.index(control)] > 0.5


def control_active_mask(Y: np.ndarray, control: str) -> np.ndarray:
    return _control_active_mask(validate_actions(Y), _validate_control(control))


def _solo_labels(Y: np.ndarray) -> np.ndarray:
    keys_active = Y[:, :N_KEYS] > 0.5
    l_click = (Y[:, N_KEYS] > 0.5)[:, None]
    r_click = (Y[:, N_KEYS + 1] > 0.5)[:, None]
    mx_active = (_mouse_delta_value(Y, "x") != 0)[:, None]
    my_active = (_mouse_delta_value(Y, "y") != 0)[:, None]
    active = np.concatenate([keys_active, l_click, r_click, mx_active, my_active], axis=1)
    labels = np.full(Y.shape[0], -1, dtype=np.int8)
    indep_mask = active.sum(axis=1) == 1
    labels[indep_mask] = active[indep_mask].argmax(axis=1)
    return labels


def solo_labels(Y: np.ndarray) -> np.ndarray:
    """Per-frame label (0..14) of the sole active control, -1 if none or 2+ active."""
    return _solo_labels(validate_actions(Y))


def _other_controls_active(Y: np.ndarray, control: str) -> np.ndarray:
    axis = CONTROL_AXES[control]["axis"]
    opposing = CONTROL_AXES[control]["opposing"]

    if control in MOUSE_CONTROLS:
        keys_active = (Y[:, :N_KEYS] > 0.5).any(axis=1)
        l_click = Y[:, N_KEYS] > 0.5
        r_click = Y[:, N_KEYS + 1] > 0.5
        opposing_active = (
            np.sign(_mouse_delta_value(Y, MOUSE_AXIS_OF[control])) == -MOUSE_SIGN[control]
        )
        return keys_active | l_click | r_click | opposing_active

    opposing_idx = KEY_NAMES.index(opposing)
    non_orthogonal_keys = [opposing_idx, 4, 5, 6, 7, 8, 9, 10]
    keys_active = (Y[:, non_orthogonal_keys] > 0.5).any(axis=1)
    l_click = Y[:, N_KEYS] > 0.5
    r_click = Y[:, N_KEYS + 1] > 0.5
    blocked = keys_active | l_click | r_click
    if axis == "lateral":
        blocked = blocked | (_mouse_delta_value(Y, "x") != 0)
    return blocked


def other_controls_active(Y: np.ndarray, control: str) -> np.ndarray:
    """True where any control is active that should end an isolated-press sequence for
    `control` -- mirrors batch_job_sensitivity_w.py's other_controls_active exactly (same
    per-axis aftermath rules: orthogonal keys stay free, mouse_x is additionally blocked
    for the lateral axis, etc.), reimplemented here to take `control` as an argument."""
    return _other_controls_active(validate_actions(Y), _validate_control(control))


def _validate_press_len(press_len: int) -> int:
    if isinstance(press_len, bool) or not isinstance(press_len, (int, np.integer)):
        raise ValueError(f"press_len must be a positive integer, got {press_len!r}")
    value = int(press_len)
    if value < 1:
        raise ValueError(f"press_len must be a positive integer, got {press_len!r}")
    return value


def _find_isolated_press_sequences(
    Y: np.ndarray, control: str, press_len: int
) -> list[tuple[int, int]]:
    """Find sequences in already-validated actions with complete boundary context."""
    if len(Y) == 0:
        return []
    if control in MOUSE_CONTROLS:
        solo_label = MOUSE_X_LABEL if MOUSE_AXIS_OF[control] == "x" else MOUSE_Y_LABEL
        solo = _solo_labels(Y)
        is_solo = (solo == solo_label) & (
            np.sign(_mouse_delta_value(Y, MOUSE_AXIS_OF[control])) == MOUSE_SIGN[control]
        )
    else:
        solo = _solo_labels(Y)
        is_solo = solo == KEY_NAMES.index(control)

    n = len(is_solo)
    is_start = np.empty(n, dtype=bool)
    is_start[0] = True
    is_start[1:] = is_solo[1:] != is_solo[:-1]
    run_starts = np.where(is_start)[0]
    run_lengths = np.diff(np.append(run_starts, n))
    press_starts = run_starts[is_solo[run_starts] & (run_lengths == press_len)].tolist()

    blocked = _other_controls_active(Y, control)
    active = _control_active_mask(Y, control)

    sequences = []
    for t in press_starts:
        # A solo run can start mid-hold: e.g. none -> w+a -> w -> none reports a fresh
        # "solo W" run beginning at the w-only frame, even though W was already held
        # (just not alone) on the frame before -- that run is the tail of an
        # already-active press, not a genuine fresh press-len-N start. Require control
        # to have been inactive immediately before the run (or the run to start at t=0,
        # where boundary policy/lookbehind context governs instead).
        if t > 0 and active[t - 1]:
            continue
        press_end = t + press_len - 1
        end = press_end
        for j in range(press_end + 1, n):
            if blocked[j] or active[j]:
                break
            end = j
        if end > press_end:
            sequences.append((t, end))
    return sequences


def _actions_with_context(
    actions: np.ndarray,
    control: str,
    press_len: int,
    *,
    lookbehind_actions=None,
    lookahead_actions=None,
    boundary: str,
) -> tuple[np.ndarray, int, int]:
    """Build the smallest action window needed to disambiguate either clip edge."""
    before = lookbehind_actions
    if before is not None:
        if len(before):
            before = before[-1:]
        else:
            before = None

    after = lookahead_actions
    if after is not None:
        if not len(after):
            after = None

    active = _control_active_mask(actions, control)
    if boundary == "error" and len(actions):
        if before is None and active[0]:
            raise ValueError(
                f"{control} is active at the first frame; provide lookbehind_actions "
                "or choose another boundary policy"
            )
        if active[-1]:
            # Merely being non-None isn't "resolved" -- if control is still active in
            # every row of the provided lookahead too, we still don't know whether (or
            # where) the press ends; only a lookahead that shows an actual release
            # counts as sufficient context. Without this, any non-empty lookahead
            # silently suppressed the error regardless of whether it told us anything.
            resolved = after is not None and not _control_active_mask(after, control).all()
            if not resolved:
                raise ValueError(
                    f"{control} is active at the last frame and the provided "
                    "lookahead_actions never shows it becoming inactive; provide "
                    "lookahead that resolves the release or choose another boundary policy"
                )

    if after is not None:
        # Boundary validation above must inspect all supplied context: an exact one-frame
        # candidate may be disproved by the first lookahead row while a later row still
        # resolves the release for a two- or three-frame candidate. Only after validating
        # that context do we keep the bounded portion needed by this specific press_len.
        after = after[:press_len]

    if boundary == "assume-inactive":
        if before is None and active[0]:
            before = _neutral_actions()
        if after is None and active[-1]:
            after = _neutral_actions()

    parts = [part for part in (before, actions, after) if part is not None]
    window = np.concatenate(parts, axis=0) if parts else actions
    offset = 0 if before is None else len(before)
    return window, offset, len(actions)


def _owned_isolated_press_sequences(
    actions: np.ndarray,
    control: str,
    press_len: int,
    *,
    lookbehind_actions: np.ndarray | None,
    lookahead_actions: np.ndarray | None,
    boundary: str,
) -> list[tuple[int, int]]:
    if len(actions) == 0:
        return []
    window, offset, core_len = _actions_with_context(
        actions,
        control,
        press_len,
        lookbehind_actions=lookbehind_actions,
        lookahead_actions=lookahead_actions,
        boundary=boundary,
    )
    sequences = _find_isolated_press_sequences(window, control, press_len)
    owned = []
    for start, end in sequences:
        local_start = start - offset
        if not 0 <= local_start < core_len:
            continue
        if boundary == "exclude" and offset == 0 and local_start == 0:
            continue
        owned.append((local_start, end - offset))
    return owned


def find_isolated_press_sequences(
    Y: np.ndarray,
    control: str,
    press_len: int,
    *,
    lookbehind_actions=None,
    lookahead_actions=None,
    boundary: str = "exclude",
) -> list[tuple[int, int]]:
    """Return inclusive ranges for clean isolated presses owned by this action window.

    A sequence is owned by the window when its press starts inside ``Y``. Because a press
    already active at frame zero may have started in a previous segment, ``boundary`` is
    conservative by default: ``"exclude"`` ignores that ambiguous candidate. Supply a
    lookbehind/lookahead halo to classify edge presses exactly, choose ``"error"`` to fail
    on missing context, or explicitly choose ``"assume-inactive"`` for legacy clip-local
    behavior. Only one lookbehind row and at most ``press_len`` lookahead rows are used.

    Returned indexes are relative to ``Y``. An end index can extend into supplied
    lookahead context; the press start is always within ``Y``.
    """
    actions = validate_actions(Y)
    control = _validate_control(control)
    press_len = _validate_press_len(press_len)
    boundary = _validate_boundary(boundary)
    before = (
        None
        if lookbehind_actions is None
        else validate_actions(lookbehind_actions, name="lookbehind_actions")
    )
    after = (
        None
        if lookahead_actions is None
        else validate_actions(lookahead_actions, name="lookahead_actions")
    )
    return _owned_isolated_press_sequences(
        actions,
        control,
        press_len,
        lookbehind_actions=before,
        lookahead_actions=after,
        boundary=boundary,
    )


def _prep_gray(img: np.ndarray, color_space: str = "BGR") -> np.ndarray:
    """`color_space` picks the matching cv2 conversion flag -- it never reorders/copies
    `img` itself. Defaults to BGR, matching how ASG/DIAMOND HDF5 frames are stored, so
    every existing direct caller (raw HDF5 arrays, no SDK involved) is unaffected. Callers
    fed frames through a kuration-sdk media column (which may be RGB, e.g. `.join_video()`'s
    PyAV provider) should pass that column's real declared color_space instead."""
    image = validate_frames(np.asarray(img)[None, ...], name="img")[0]
    color_space = validate_color_space(color_space)
    cv2 = _require_cv2()
    if color_space == "BGR":
        flag = cv2.COLOR_BGR2GRAY
    elif color_space == "RGB":
        flag = cv2.COLOR_RGB2GRAY
    else:
        raise AssertionError("validated color space was not RGB/BGR")
    gray = cv2.cvtColor(image, flag)
    h = gray.shape[0]
    return gray[: int(h * (1 - BOTTOM_CROP_FRAC)), :]


# Each entry owns three float32 image-sized grids (about 12 * H * W bytes). Two entries cover
# the common full/cropped-resolution pair without an unbounded process-global cache; workloads
# cycling through more resolutions intentionally recompute rather than retain arbitrary RAM.
@lru_cache(maxsize=2)
def _radial_unit_vectors(h: int, w: int):
    ys, xs = np.mgrid[0:h, 0:w]
    dx = (xs - w / 2.0).astype(np.float32)
    dy = (ys - h / 2.0).astype(np.float32)
    r = np.sqrt(dx * dx + dy * dy, dtype=np.float32)
    r_safe = np.where(r < 1e-6, np.float32(1e-6), r)
    return dx / r_safe, dy / r_safe, r


def displacement_median(flow: np.ndarray, control: str) -> float:
    """Axis-appropriate median flow projection -- radial for W/S, horizontal for A/D and
    mouse_x, vertical for mouse_y. Mirrors displacement_median() in batch_job_sensitivity_w.py."""
    control = _validate_control(control)
    flow = np.asarray(flow)
    if flow.ndim != 3 or flow.shape[-1:] != (2,):
        raise ValueError(f"flow must have shape (H, W, 2), got {flow.shape}")
    if flow.shape[0] == 0 or flow.shape[1] == 0:
        # np.median on an empty spatial slice returns nan silently (with a runtime
        # warning) instead of erroring -- a degenerate flow must be rejected here, not
        # allowed to produce a NaN "measurement" downstream.
        raise ValueError(f"flow must have nonzero spatial dimensions, got {flow.shape}")
    if flow.dtype.kind not in "biuf" or not np.all(np.isfinite(flow)):
        raise ValueError("flow must contain finite numeric values")
    axis = CONTROL_AXES[control]["axis"]
    if axis == "longitudinal":
        h, w = flow.shape[:2]
        ux, uy, r = _radial_unit_vectors(h, w)
        radial = flow[..., 0] * ux + flow[..., 1] * uy
        return float(np.median(radial[r > 2.0]))
    if axis == "pitch":
        return float(np.median(flow[..., 1]))
    return float(np.median(flow[..., 0]))  # lateral or yaw


def _release_transition_displacement_validated(
    frames: np.ndarray,
    actions: np.ndarray,
    control: str,
    press_len: int,
    color_space: str,
    *,
    lookbehind_actions: np.ndarray | None,
    lookahead_actions: np.ndarray | None,
    lookahead_frames: np.ndarray | None,
    boundary: str,
) -> list[tuple[int, float]]:
    sequences = _owned_isolated_press_sequences(
        actions,
        control,
        press_len,
        lookbehind_actions=lookbehind_actions,
        lookahead_actions=lookahead_actions,
        boundary=boundary,
    )
    out: list[tuple[int, float]] = []
    if not sequences:
        return out
    available_frames = (
        frames
        if lookahead_frames is None
        else np.concatenate([frames, lookahead_frames[:press_len]], axis=0)
    )
    cv2 = _require_cv2()
    for start, end in sequences:
        release_idx = start + press_len - 1
        if release_idx + 1 > end:
            continue
        if release_idx + 1 >= len(available_frames):
            # ``assume-inactive`` can classify an action-only right-edge release, but it
            # cannot manufacture the corresponding image. Keep measurement honest: that
            # candidate remains unmeasured unless the caller supplies real lookahead.
            continue
        gray0 = _prep_gray(available_frames[release_idx], color_space)
        gray1 = _prep_gray(available_frames[release_idx + 1], color_space)
        flow = cv2.calcOpticalFlowFarneback(gray0, gray1, None, **FARNEBACK_KWARGS)
        out.append((start, displacement_median(flow, control)))
    return out


def release_transition_displacement(
    frames: np.ndarray,
    actions: np.ndarray,
    control: str,
    press_len: int,
    color_space: str = "BGR",
    *,
    lookbehind_actions=None,
    lookahead_actions=None,
    lookahead_frames=None,
    boundary: str = "exclude",
) -> list[tuple[int, float]]:
    """Measure the release-transition shift for each owned isolated press.

    Returns ``(start_frame_idx, signed_pixel_shift)`` pairs. Optional action/frame
    lookahead makes a release crossing the right segment boundary measurable; action-only
    lookbehind disambiguates a press at the left boundary. See
    :func:`find_isolated_press_sequences` for boundary-policy semantics.
    """
    actions = validate_actions(actions)
    frames = validate_frames(frames, steps=len(actions))
    control = _validate_control(control)
    press_len = _validate_press_len(press_len)
    boundary = _validate_boundary(boundary)
    color_space = validate_color_space(color_space)

    before = (
        None
        if lookbehind_actions is None
        else validate_actions(lookbehind_actions, name="lookbehind_actions")
    )
    if (lookahead_actions is None) != (lookahead_frames is None):
        raise ValueError("lookahead_actions and lookahead_frames must be provided together")
    future_actions = None
    future_frames = None
    if lookahead_actions is not None:
        future_actions = validate_actions(lookahead_actions, name="lookahead_actions")
        future_frames = validate_frames(
            lookahead_frames, steps=len(future_actions), name="lookahead_frames"
        )
        if future_frames.shape[1:] != frames.shape[1:]:
            raise ValueError("lookahead_frames must have the same spatial shape as frames")
        if future_frames.dtype != frames.dtype:
            raise ValueError("lookahead_frames must have the same dtype as frames")

    return _release_transition_displacement_validated(
        frames,
        actions,
        control,
        press_len,
        color_space,
        lookbehind_actions=before,
        lookahead_actions=future_actions,
        lookahead_frames=future_frames,
        boundary=boundary,
    )
