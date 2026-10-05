"""Tests for the built-in pixel_shift capability: action decoding, isolated-
press-detection logic, and pixel_shift_operator.py's aggregation, registration, and
pipeline integration. Mirrors this repo's own testing conventions (see
tests/test_media_transforms_execution.py's FakeVideoProvider pattern, and
tests/test_operator_registry.py's discovery-CLI test) rather than needing real video
files or real gameplay capture -- every test here is deterministic and self-contained.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from kuration import gameplay
from kuration.gameplay import pixel_shift
from kuration import (
    FrameRef,
    VideoMetadata,
    default_registry,
    describe_component,
    input,
    list_components,
)

from kuration.gameplay.pixel_shift import diamond_action_decode as dad
from kuration.gameplay.pixel_shift import pixel_shift_operator as pso
from kuration.gameplay.pixel_shift import register

register()

ACTION_DIM = dad.N_KEYS + 2 + dad.N_MOUSE_X + dad.N_MOUSE_Y  # 51, matches asg2diamond's layout


def test_radial_geometry_cache_is_bounded():
    dad._radial_unit_vectors.cache_clear()
    for size in ((8, 8), (9, 8), (10, 8)):
        vectors = dad._radial_unit_vectors(*size)
        assert all(array.dtype == np.float32 for array in vectors)
    assert dad._radial_unit_vectors.cache_info().currsize == 2


def _action(w=False, s=False, a=False, d=False, mouse_x=0.0, mouse_y=0.0):
    """One 51-dim DIAMOND action vector with the given keys/mouse bins active."""
    vec = np.zeros(ACTION_DIM, dtype=np.float64)
    for key, active in (("W", w), ("S", s), ("A", a), ("D", d)):
        if active:
            vec[dad.KEY_NAMES.index(key)] = 1.0
    mx_idx = int(np.argmin(np.abs(dad.MOUSE_X_POSSIBLES - mouse_x)))
    my_idx = int(np.argmin(np.abs(dad.MOUSE_Y_POSSIBLES - mouse_y)))
    vec[dad.MX_OFF + mx_idx] = 1.0
    vec[dad.MY_OFF + my_idx] = 1.0
    return vec


# ---------------------------------------------------------------------------------------
# diamond_action_decode.py -- pure action-vector logic, no image data needed
# ---------------------------------------------------------------------------------------


def test_mouse_delta_value_decodes_to_nearest_possible_bin():
    Y = np.stack([_action(mouse_x=19.0)])  # nearest of MOUSE_X_POSSIBLES to 19 is 20
    assert dad.mouse_delta_value(Y, "x")[0] == 20.0


def test_isolated_press_requires_the_exact_press_len():
    """W held for exactly 2 consecutive frames, one real aftermath frame -- an isolated
    press_len=2 instance, but NOT found when asking for press_len=1 or 3 (must match the
    real hold length exactly, per the README's isolated-press definition)."""
    Y = np.stack([_action(), _action(w=True), _action(w=True), _action()])
    assert dad.find_isolated_press_sequences(Y, "W", press_len=2) == [(1, 3)]
    assert dad.find_isolated_press_sequences(Y, "W", press_len=1) == []
    assert dad.find_isolated_press_sequences(Y, "W", press_len=3) == []


def test_isolated_press_needs_a_real_aftermath_frame():
    """A press with no frame after it at all can't produce a release-transition pair, so
    it isn't counted as an instance -- matches the README's "release-only" requirement:
    there must be something after the press to measure the transition against."""
    Y = np.stack([_action(w=True)])
    assert dad.find_isolated_press_sequences(Y, "W", press_len=1) == []


def test_aftermath_breaks_on_the_opposing_control():
    """W tap immediately followed by S (the opposing control on the same axis) blocks the
    aftermath on the very first potential frame, leaving no valid pair."""
    Y = np.stack([_action(w=True), _action(s=True)])
    assert dad.find_isolated_press_sequences(Y, "W", press_len=1) == []


def test_aftermath_allows_orthogonal_keys_to_vary_freely():
    """W tap, then A (orthogonal axis) becomes active in the aftermath -- still a valid W
    instance, per the aftermath rule documented in diamond_action_decode.py."""
    Y = np.stack([_action(), _action(w=True), _action(a=True), _action(a=True)])
    assert dad.find_isolated_press_sequences(Y, "W", press_len=1) == [(1, 3)]


def test_held_control_briefly_sharing_a_frame_is_not_a_false_new_press():
    """Regression test for a review-caught bug: none -> w+a -> w -> none reported a fresh
    press_len=1 "W" press starting at the w-only frame, even though W was already held
    (continuously, just not alone) on the frame before. A solo run that begins mid-hold is
    the tail of an already-active press, not a genuine new press-len-N start, and must not
    be reported as one."""
    Y = np.stack([_action(), _action(w=True, a=True), _action(w=True), _action()])
    assert dad.find_isolated_press_sequences(Y, "W", press_len=1) == []

    # Sanity: a genuinely fresh, isolated W press (not preceded by an already-held W) is
    # still detected correctly.
    fresh = np.stack([_action(), _action(w=True), _action()])
    assert dad.find_isolated_press_sequences(fresh, "W", press_len=1) == [(1, 2)]


def test_isolated_mouse_y_press_is_actually_detected():
    """Regression test for a review-caught bug: find_isolated_press_sequences compared
    solo_labels() output (a label in [0, 14]) against MY_OFF (=36, a completely different
    numbering -- an offset into the raw action vector's columns), which can never match,
    silently disabling ALL isolated mouse_y-press detection. MOUSE_X_POS/NEG worked only by
    coincidence (MX_OFF happens to equal the correct label, 13); MOUSE_Y_POS/NEG did not
    (MY_OFF=36 vs the correct label, 14). Fixed by introducing MOUSE_X_LABEL/MOUSE_Y_LABEL
    as their own, correct numbering, separate from MX_OFF/MY_OFF."""
    Y = np.stack([_action(), _action(mouse_y=20.0), _action()])
    assert dad.find_isolated_press_sequences(Y, "MOUSE_Y_POS", press_len=1) == [(1, 2)]

    Y_neg = np.stack([_action(), _action(mouse_y=-20.0), _action()])
    assert dad.find_isolated_press_sequences(Y_neg, "MOUSE_Y_NEG", press_len=1) == [(1, 2)]


def test_empty_input_does_not_raise():
    """Regression test for a review-caught bug: `is_start[0] = True` on a 0-length array
    raised IndexError for an empty window instead of returning no instances found."""
    empty = np.zeros((0, ACTION_DIM), dtype=np.float64)
    assert dad.find_isolated_press_sequences(empty, "W", press_len=1) == []
    assert dad.find_isolated_press_sequences(empty, "MOUSE_Y_POS", press_len=1) == []


def test_boundary_policy_is_conservative_but_explicitly_flexible():
    core = np.stack([_action(w=True), _action()])
    assert dad.find_isolated_press_sequences(core, "W", press_len=1) == []
    assert dad.find_isolated_press_sequences(
        core, "W", press_len=1, boundary="assume-inactive"
    ) == [(0, 1)]
    assert dad.find_isolated_press_sequences(
        core, "W", press_len=1, lookbehind_actions=np.stack([_action()])
    ) == [(0, 1)]

    # A real active lookbehind proves this press began outside the owned window.
    assert (
        dad.find_isolated_press_sequences(
            core, "W", press_len=1, lookbehind_actions=np.stack([_action(w=True)])
        )
        == []
    )


def test_lookahead_recovers_a_press_clipped_by_the_right_boundary():
    core = np.stack([_action(), _action(w=True), _action(w=True)])
    after = np.stack([_action()])
    assert dad.find_isolated_press_sequences(core, "W", press_len=2) == []
    assert dad.find_isolated_press_sequences(core, "W", press_len=2, lookahead_actions=after) == [
        (1, 3)
    ]


def test_error_boundary_requires_context_for_targeted_active_edges():
    left = np.stack([_action(w=True), _action()])
    with pytest.raises(ValueError, match="first frame"):
        dad.find_isolated_press_sequences(left, "W", 1, boundary="error")

    right = np.stack([_action(), _action(w=True)])
    with pytest.raises(ValueError, match="last frame"):
        dad.find_isolated_press_sequences(right, "W", 1, boundary="error")


def test_error_boundary_is_not_suppressed_by_lookahead_that_does_not_resolve():
    """Regression test for a review-caught bug: any non-empty lookahead_actions suppressed
    the right-edge "error" check, regardless of whether it actually showed the press
    ending. Lookahead where W is STILL active throughout tells us nothing new -- the
    ambiguity remains, so "error" must still raise. Lookahead that genuinely shows W
    becoming inactive, by contrast, resolves it and must not raise."""
    right = np.stack([_action(), _action(w=True)])
    unresolving_lookahead = np.stack([_action(w=True), _action(w=True)])
    with pytest.raises(ValueError, match="last frame"):
        dad.find_isolated_press_sequences(
            right, "W", 1, lookahead_actions=unresolving_lookahead, boundary="error"
        )

    resolving_lookahead = np.stack([_action()])
    dad.find_isolated_press_sequences(
        right, "W", 1, lookahead_actions=resolving_lookahead, boundary="error"
    )  # must not raise

    # The release may be later than the bounded context needed to disprove this specific
    # press length. Boundary validation must inspect the complete supplied lookahead before
    # sequence detection truncates it to press_len rows.
    later_release = np.stack([_action(w=True), _action()])
    assert (
        dad.find_isolated_press_sequences(
            right, "W", 1, lookahead_actions=later_release, boundary="error"
        )
        == []
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value[:, :-1], "shape"),
        (lambda value: value.astype(object), "numeric dtype"),
        (lambda value: np.where(np.eye(1, value.shape[1], 0), np.nan, value), "finite"),
    ],
)
def test_action_validation_rejects_malformed_arrays(mutate, message):
    actions = np.stack([_action()])
    with pytest.raises(ValueError, match=message):
        dad.validate_actions(mutate(actions))


def test_action_validation_rejects_non_binary_and_non_one_hot_fields():
    non_binary = np.stack([_action()])
    non_binary[0, 0] = 0.25
    with pytest.raises(ValueError, match="key/click"):
        dad.validate_actions(non_binary)

    non_one_hot = np.stack([_action()])
    non_one_hot[0, dad.MX_OFF : dad.MX_OFF + dad.N_MOUSE_X] = 0
    with pytest.raises(ValueError, match="mouse_x"):
        dad.validate_actions(non_one_hot)


def test_decode_api_rejects_unknown_semantics_and_invalid_lengths():
    actions = np.stack([_action()])
    with pytest.raises(ValueError, match="unknown control"):
        dad.find_isolated_press_sequences(actions, "JUMP", 1)
    with pytest.raises(ValueError, match="positive integer"):
        dad.find_isolated_press_sequences(actions, "W", 0)
    with pytest.raises(ValueError, match="mouse axis"):
        dad.mouse_delta_value(actions, "z")
    with pytest.raises(ValueError, match="boundary"):
        dad.find_isolated_press_sequences(actions, "W", 1, boundary="guess")


def test_displacement_median_rejects_degenerate_empty_flow():
    """Regression test for a review-caught bug: an empty spatial dimension passed the
    shape check (ndim==3, last axis==2) and reached np.median, which silently returns nan
    for an empty array instead of erroring -- a degenerate flow must be rejected explicitly,
    not allowed to produce a NaN "measurement" downstream."""
    empty_h = np.zeros((0, 10, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="nonzero spatial dimensions"):
        dad.displacement_median(empty_h, "W")

    empty_w = np.zeros((10, 0, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="nonzero spatial dimensions"):
        dad.displacement_median(empty_w, "W")


class TestDisplacementSignConvention:
    """Validates README §Method point 4 (coordinate/sign convention) directly against
    hand-built flow fields -- no real optical flow needed to check the projection math."""

    def test_radial_projection_positive_for_outward_flow(self):
        h, w = 20, 30
        ux, uy, _ = dad._radial_unit_vectors(h, w)
        outward_flow = np.stack([ux, uy], axis=-1).astype(np.float32)
        assert dad.displacement_median(outward_flow, "W") > 0
        assert dad.displacement_median(-outward_flow, "W") < 0  # inward = S's direction

    def test_horizontal_projection_for_lateral_and_yaw_controls(self):
        h, w = 20, 30
        rightward_flow = np.zeros((h, w, 2), dtype=np.float32)
        rightward_flow[..., 0] = 1.0
        # A and D share the same raw horizontal projection (this function doesn't decide
        # which sign is "correct" per control -- that's the reference/expected_sign layer,
        # not this one); both read the same measurement off the same flow field.
        assert dad.displacement_median(rightward_flow, "A") == dad.displacement_median(
            rightward_flow, "D"
        )
        assert dad.displacement_median(rightward_flow, "MOUSE_X_POS") == dad.displacement_median(
            rightward_flow, "A"
        )

    def test_vertical_projection_for_pitch_controls(self):
        h, w = 20, 30
        downward_flow = np.zeros((h, w, 2), dtype=np.float32)
        downward_flow[..., 1] = 1.0
        assert dad.displacement_median(downward_flow, "MOUSE_Y_POS") > 0
        assert dad.displacement_median(-downward_flow, "MOUSE_Y_POS") < 0


def test_crop_removes_exactly_the_configured_bottom_fraction():
    """Validates README §Method point 1: BOTTOM_CROP_FRAC=0.18 drops the bottom 18% of the
    frame before optical flow ever sees it."""
    h, w = 100, 50
    img = np.zeros((h, w, 3), dtype=np.uint8)
    gray = dad._prep_gray(img)
    expected_h = int(h * (1 - dad.BOTTOM_CROP_FRAC))
    assert gray.shape == (expected_h, w)
    assert expected_h == 82


def test_release_transition_measures_exactly_one_frame_pair_per_instance():
    """Validates README §Method point 5: for a press_len=N instance, only the transition
    at index N-1 -> N is measured, not every frame in the press+aftermath sequence, even
    when the aftermath is much longer than one frame."""
    rng = np.random.default_rng(0)
    h, w = 40, 60
    # W held for press_len=2 (frames 0,1), then 5 aftermath frames (2..6) -- 6 possible
    # frame-pairs exist in this sequence, but exactly one should be measured.
    actions = np.stack([_action()] + [_action(w=True)] * 2 + [_action()] * 5)
    frames = rng.integers(0, 255, size=(actions.shape[0], h, w, 3), dtype=np.uint8)

    results = dad.release_transition_displacement(frames, actions, "W", press_len=2)
    assert len(results) == 1
    start, _shift = results[0]
    assert start == 1  # sequence start; release transition is start + press_len - 1 = 2 -> 3


# ---------------------------------------------------------------------------------------
# pixel_shift_operator.py -- per-segment aggregation (min/max/mean/n_instances)
# ---------------------------------------------------------------------------------------


def test_segment_scalars_reports_true_min_max_mean_not_an_average_that_hides_outliers():
    rng = np.random.default_rng(1)
    h, w = 40, 60
    # Two isolated D taps (press_len=1) in one segment: one near-zero, one large --
    # min/max must capture both, not just their average.
    actions = np.stack(
        [
            _action(),
            _action(d=True),
            _action(),  # tap 1: frames 0,1,2
            _action(),
            _action(d=True),
            _action(),  # tap 2: frames 3,4,5
        ]
    )
    frames = rng.integers(0, 255, size=(actions.shape[0], h, w, 3), dtype=np.uint8)

    scalars = pso.segment_scalars(frames, actions, controls=["D"])
    assert scalars["d_n_instances"] == 2.0
    assert scalars["d_shift_min"] <= scalars["d_shift_mean"] <= scalars["d_shift_max"]


def test_segment_scalars_sentinel_when_no_instances_found():
    h, w = 40, 60
    actions = np.stack([_action(), _action(), _action()])
    frames = np.zeros((actions.shape[0], h, w, 3), dtype=np.uint8)

    scalars = pso.segment_scalars(frames, actions, controls=["W"])
    assert scalars["w_n_instances"] == 0.0
    assert scalars["w_shift_min"] == 0.0
    assert scalars["w_shift_max"] == 0.0
    assert scalars["w_shift_mean"] == 0.0
    assert scalars["w_requested"] == 1.0


def test_segment_scalars_flags_whether_a_shift_was_actually_measured():
    """Regression test for a review-caught finding: with n_instances == 0 the shift
    statistics were emitted as 0.0, indistinguishable from a genuine zero-motion
    measurement (the core transform contract admits only finite scalars, so NaN is not
    available). `{control}_shift_measured` is the explicit 0/1 companion a `.filter()`
    can select on."""
    frames = np.stack(_shifting_noise_frames(3))
    idle = np.stack([_action(), _action(), _action()])
    pressed = np.stack([_action(), _action(w=True), _action()])

    untested = pso.segment_scalars(frames, idle, controls=["W"])
    assert untested["w_shift_measured"] == 0.0
    assert untested["w_n_instances"] == 0.0
    assert untested["w_shift_mean"] == 0.0  # placeholder, only trustworthy via the flag

    measured = pso.segment_scalars(frames, pressed, controls=["W"])
    assert measured["w_shift_measured"] == 1.0
    assert measured["w_n_instances"] == 1.0
    # A control that was never requested is not "measured" either.
    assert measured["s_shift_measured"] == 0.0
    assert all(f"{control.lower()}_shift_measured" in pso.OUTPUT_KEYS
               for control in dad.ALL_CONTROLS)


def test_iter_instances_validates_its_own_arrays_for_direct_callers():
    """Regression test for a review-caught finding: `iter_instances` relied on a docstring
    instruction to call `validate_segment_inputs` first, so a direct caller feeding
    unvalidated arrays got silent garbage instead of an error."""
    frames = np.stack(_shifting_noise_frames(3))
    actions = np.stack([_action(), _action(w=True), _action()])

    with pytest.raises(ValueError, match="shape"):
        list(pixel_shift.iter_instances(frames, actions[:, :-1], "W"))
    with pytest.raises(ValueError, match="steps"):
        list(pixel_shift.iter_instances(frames[:2], actions, "W"))
    with pytest.raises(ValueError, match="validated"):
        list(pixel_shift.iter_instances(frames, actions, "W", validated="yes"))

    # Pre-validated arrays may skip the repeated per-control revalidation explicitly.
    validated = pixel_shift.validate_segment_inputs(frames, actions)
    assert len(list(pixel_shift.iter_instances(
        validated[0], validated[1], "W", validated=True))) == 1


def test_segment_scalars_distinguishes_unrequested_from_requested_without_events():
    actions = np.stack([_action(), _action(), _action()])
    frames = np.zeros((len(actions), 40, 60, 3), dtype=np.uint8)

    scalars = pso.segment_scalars(frames, actions, controls=("W",))
    assert scalars["w_requested"] == 1.0
    assert scalars["w_n_instances"] == 0.0
    assert scalars["s_requested"] == 0.0
    assert scalars["s_n_instances"] == 0.0


def test_segment_scalars_validates_controls_for_direct_callers():
    actions = np.stack([_action()])
    frames = np.zeros((1, 40, 60, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="non-empty"):
        pso.segment_scalars(frames, actions, controls=[])
    with pytest.raises(ValueError, match="unknown pixel-shift control"):
        pso.segment_scalars(frames, actions, controls=["JUMP"])


def test_public_instance_measurement_contract_is_exported_and_validated():
    frames = np.stack(_shifting_noise_frames(3))
    actions = np.stack([_action(), _action(w=True), _action()])
    validated = pixel_shift.validate_segment_inputs(frames, actions)

    rows = list(
        pixel_shift.iter_instances(
            validated[0],
            validated[1],
            "W",
            lookbehind_actions=validated[2],
            lookahead_actions=validated[3],
            lookahead_frames=validated[4],
        )
    )

    assert len(rows) == 1
    assert rows[0][0:2] == ("1", 1)
    assert np.isfinite(rows[0][2])
    with pytest.raises(ValueError, match="unknown pixel-shift control"):
        list(pixel_shift.iter_instances(validated[0], validated[1], "JUMP"))


def test_segment_scalars_validates_frame_contract_before_processing():
    actions = np.stack([_action(), _action()])
    with pytest.raises(ValueError, match="steps"):
        pso.segment_scalars(np.zeros((1, 40, 60, 3), dtype=np.uint8), actions, controls=["W"])
    with pytest.raises(ValueError, match="dtype"):
        pso.segment_scalars(np.zeros((2, 40, 60, 3), dtype=np.float64), actions, controls=["W"])
    with pytest.raises(ValueError, match="shape"):
        pso.segment_scalars(np.zeros((2, 40, 60), dtype=np.uint8), actions, controls=["W"])


def test_release_lookahead_requires_matching_actions_and_frames():
    actions = np.stack([_action(), _action(w=True)])
    frames = np.zeros((2, 40, 60, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="provided together"):
        dad.release_transition_displacement(
            frames,
            actions,
            "W",
            1,
            lookahead_actions=np.stack([_action()]),
        )


def test_release_lookahead_measures_a_right_boundary_release():
    rng = np.random.default_rng(12)
    actions = np.stack([_action(), _action(w=True), _action(w=True)])
    frames = rng.integers(0, 255, size=(3, 40, 60, 3), dtype=np.uint8)
    after_actions = np.stack([_action()])
    after_frames = rng.integers(0, 255, size=(1, 40, 60, 3), dtype=np.uint8)

    result = dad.release_transition_displacement(
        frames,
        actions,
        "W",
        2,
        lookahead_actions=after_actions,
        lookahead_frames=after_frames,
    )
    assert len(result) == 1
    assert result[0][0] == 1


def test_segment_scalars_error_boundary_accepts_resolved_multiframe_release():
    """The aggregator checks key press lengths in ascending order. A valid two-frame
    boundary press must not fail during the preceding one-frame check merely because that
    check only needs one lookahead row for sequence classification."""
    rng = np.random.default_rng(13)
    actions = np.stack([_action(), _action(w=True)])
    frames = rng.integers(0, 255, size=(2, 40, 60, 3), dtype=np.uint8)
    after_actions = np.stack([_action(w=True), _action()])
    after_frames = rng.integers(0, 255, size=(2, 40, 60, 3), dtype=np.uint8)

    scalars = pso.segment_scalars(
        frames,
        actions,
        controls=["W"],
        lookahead_actions=after_actions,
        lookahead_frames=after_frames,
        boundary="error",
    )
    assert scalars["w_n_instances"] == 1.0


def test_assumed_right_boundary_never_invents_a_missing_frame():
    actions = np.stack([_action(), _action(w=True)])
    frames = np.zeros((2, 40, 60, 3), dtype=np.uint8)
    assert (
        dad.release_transition_displacement(frames, actions, "W", 1, boundary="assume-inactive")
        == []
    )


def test_transform_rejects_unknown_or_duplicate_controls():
    with pytest.raises(ValueError, match="unknown pixel-shift control"):
        pso.PixelShift(["JUMP"])
    with pytest.raises(ValueError, match="duplicates"):
        pso.PixelShift(["W", "W"])
    with pytest.raises(ValueError, match="boundary"):
        pso.PixelShift(["W"], boundary="guess")


def test_segment_scalars_has_no_side_effects():
    """segment_scalars must be a pure function -- no shared/global state to leak between
    calls or across worker processes. Calling it twice with identical inputs must give
    identical outputs, and must not depend on (or mutate) anything besides its arguments."""
    rng = np.random.default_rng(1)
    h, w = 40, 60
    actions = np.stack([_action(), _action(d=True), _action()])
    frames = rng.integers(0, 255, size=(actions.shape[0], h, w, 3), dtype=np.uint8)

    first = pso.segment_scalars(frames, actions, controls=["D"])
    second = pso.segment_scalars(frames, actions, controls=["D"])
    assert first == second


def test_pixel_shift_report_records_clip_provenance():
    rng = np.random.default_rng(2)
    h, w = 40, 60
    actions = np.stack([_action(), _action(a=True), _action()])
    frames = rng.integers(0, 255, size=(actions.shape[0], h, w, 3), dtype=np.uint8)

    report = pso.PixelShiftReport()
    report.add_segment(frames, actions, ("demo", 100, 103), controls=["A"])
    assert "A" in report._ledger
    assert "1" in report._ledger["A"]  # press_len=1
    record = report._ledger["A"]["1"][0]
    assert record["source_id"] == "demo"
    assert record["frame"] == 101  # start_idx (100) + local instance start (1, where A is held)


def test_add_segment_rejects_malformed_source_id_and_negative_indexes():
    """Regression test for a review-caught bug: source_id was never validated (None, "",
    or any object would satisfy every existing check), and start_idx/end_idx were only
    checked to be integers, not non-negative -- a malformed segment_id could silently
    poison print_report()'s clip/frame provenance with a plausible-looking but meaningless
    location instead of erroring."""
    rng = np.random.default_rng(2)
    h, w = 40, 60
    actions = np.stack([_action(), _action(a=True), _action()])
    frames = rng.integers(0, 255, size=(actions.shape[0], h, w, 3), dtype=np.uint8)
    report = pso.PixelShiftReport()

    with pytest.raises(ValueError, match="source_id"):
        report.add_segment(frames, actions, (None, 100, 103), controls=["A"])
    with pytest.raises(ValueError, match="source_id"):
        report.add_segment(frames, actions, ("", 100, 103), controls=["A"])
    with pytest.raises(ValueError, match="non-negative"):
        report.add_segment(frames, actions, ("demo", -5, -2), controls=["A"])


def test_pixel_shift_report_preserves_distinct_instances_within_one_segment():
    """Regression test for a review-caught bug: every instance found in one add_segment()
    call used to share the SAME segment-level clip label, discarding each instance's own
    local start frame. print_report() then deduped labels through a set, silently
    collapsing multiple real, distinct instances from the same segment into a single
    printed entry. Two isolated A-taps in one segment must now show up as two distinct
    (source_id, frame) locations, not one."""
    rng = np.random.default_rng(4)
    h, w = 40, 60
    # Two isolated A taps (press_len=1) in one segment, at different local frames.
    actions = np.stack(
        [
            _action(),
            _action(a=True),
            _action(),  # tap 1: local frames 0,1,2
            _action(),
            _action(a=True),
            _action(),  # tap 2: local frames 3,4,5
        ]
    )
    frames = rng.integers(0, 255, size=(actions.shape[0], h, w, 3), dtype=np.uint8)

    report = pso.PixelShiftReport()
    report.add_segment(frames, actions, ("demo", 100, 106), controls=["A"])
    entries = report._ledger["A"]["1"]
    assert len(entries) == 2
    frames_seen = sorted(e["frame"] for e in entries)
    assert frames_seen == [101, 104]  # start_idx(100) + local starts (1, 4) -- distinct


def test_pixel_shift_report_instances_are_independent_per_object():
    """Two separate PixelShiftReport instances must never share state -- this is the
    entire point of making it an explicit object instead of a module-level singleton."""
    rng = np.random.default_rng(2)
    h, w = 40, 60
    actions = np.stack([_action(), _action(a=True), _action()])
    frames = rng.integers(0, 255, size=(actions.shape[0], h, w, 3), dtype=np.uint8)

    report_1 = pso.PixelShiftReport()
    report_1.add_segment(frames, actions, ("one", 0, 3), controls=["A"])
    report_2 = pso.PixelShiftReport()
    assert report_2._ledger == {}
    assert "A" in report_1._ledger


def test_report_record_exports_have_one_concat_safe_schema():
    """PixelShiftReport and AnomalyChecker exports use the same keys and source-level
    instance-count semantics, so combining their returned lists needs no reshaping."""
    report = pso.PixelShiftReport()
    report._ledger = {
        "W": {
            "1": [
                {"source_id": "clip2", "frame": 9, "shift": 1.5},
                {"source_id": "clip1", "frame": 4, "shift": 2.5},
            ],
        },
        "A": {
            "2": [{"source_id": "clip1", "frame": 2, "shift": -3.5}],
        },
    }
    pixel_records = report.to_records()

    from kuration.gameplay.action_state_consistency import AnomalyChecker, validate_config

    checker = AnomalyChecker(
        validate_config(
            {
                "anomaly_type": "synthetic_demo",
                "baselines": {
                    "X": {"1": {"expected_sign": 1, "min": 0.0, "max": 10.0}},
                },
            }
        )
    )
    checker.resolve_segment(("clip3", 0, 5), expected_length=5)
    checker.record_instance("clip3", "X", "1", 1, 3.0)
    anomaly_records = checker.to_records()

    expected_fields = {
        "source_id",
        "control",
        "cell_key",
        "frame",
        "shift",
        "consistent",
        "verdict",
        "n_instances",
        "frame_space",
        "frame_space_note",
    }
    combined = pixel_records + anomaly_records
    assert all(set(record) == expected_fields for record in combined)
    assert [(record["source_id"], record["frame"]) for record in pixel_records] == [
        ("clip1", 2),
        ("clip1", 4),
        ("clip2", 9),
    ]
    assert [record["n_instances"] for record in pixel_records] == [2, 2, 1]
    assert all(record["frame_space"] == "source" for record in pixel_records)
    assert all(record["consistent"] is None and record["verdict"] is None
               for record in pixel_records)


def test_pixel_shift_report_deduplicates_the_same_press_across_overlapping_windows():
    """Regression test for a review-caught bug: a press's global start can fall within
    more than one overlapping window's local range, so each window's add_segment() call
    independently "owns" and records it -- overlapping windows do not, by themselves,
    prevent duplicate counting (see the README's "Input and segment-boundary contract").
    Two overlapping windows here both genuinely see the SAME global W press (global frame
    5); PixelShiftReport must record it once, not twice, because both calls resolve to the
    same (source_id, control, cell_key, frame) instance key."""
    rng = np.random.default_rng(9)
    h, w = 40, 60
    # Global timeline, frames 0..9: an isolated W press at global frame 5.
    global_actions = np.stack([_action()] * 5 + [_action(w=True)] + [_action()] * 4)
    global_frames = rng.integers(0, 255, size=(10, h, w, 3), dtype=np.uint8)

    # Window A: global [3:8) -- local start of the press = 5 - 3 = 2.
    # Window B: global [4:9) -- local start of the press = 5 - 4 = 1.
    # Both windows own the press (boundary="exclude" only rejects local_start == 0).
    window_a = (global_actions[3:8], global_frames[3:8], ("clip", 3, 8))
    window_b = (global_actions[4:9], global_frames[4:9], ("clip", 4, 9))

    report = pso.PixelShiftReport()
    for actions, frames, segment_id in (window_a, window_b):
        report.add_segment(frames, actions, segment_id, controls=["W"])

    entries = report._ledger["W"]["1"]
    assert len(entries) == 1  # not 2 -- deduplicated across the two overlapping windows
    assert entries[0]["frame"] == 5  # the true global frame, matching both windows' math


# ---------------------------------------------------------------------------------------
# Color-space handling -- _prep_gray assumes BGR by default (matching ASG/DIAMOND HDF5
# storage), but kuration-sdk's MediaSpec defaults color_space to RGB and its own
# first-party PyAV video provider actually decodes rgb24. `color_space` picks the matching
# cv2 conversion flag (COLOR_BGR2GRAY / COLOR_RGB2GRAY) -- it must never reorder or copy
# the frame array itself; PixelShift.transform_batch reads ctx["schema"] to get the real
# declared color_space and passes it straight through (see its comment for the mechanism).
# ---------------------------------------------------------------------------------------


def _asymmetric_channel_image(h=8, w=6):
    """A synthetic frame where each channel has a distinct, non-symmetric value pattern --
    reordering channels (or picking the wrong cv2 conversion flag) changes the grayscale
    result, unlike e.g. a grayscale-already image where channel order wouldn't matter."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def test_prep_gray_defaults_to_bgr():
    img = _asymmetric_channel_image()
    cv2 = dad._require_cv2()
    expected = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    expected = expected[: int(expected.shape[0] * (1 - dad.BOTTOM_CROP_FRAC)), :]
    assert np.array_equal(dad._prep_gray(img), expected)


def test_prep_gray_rgb_flag_equals_bgr_flag_on_reversed_channels():
    """Picking COLOR_RGB2GRAY directly must give the identical result to reversing the
    channel axis first and picking COLOR_BGR2GRAY -- confirms the flag-selection approach
    (which never touches/copies the input array) is mathematically equivalent to the
    reorder-then-convert approach, without paying for the reorder."""
    img = _asymmetric_channel_image()
    rgb_result = dad._prep_gray(img, color_space="RGB")
    bgr_of_reversed = dad._prep_gray(img[..., ::-1], color_space="BGR")
    assert np.array_equal(rgb_result, bgr_of_reversed)
    # sanity: channel order actually matters for this image (not a no-op comparison)
    assert not np.array_equal(rgb_result, dad._prep_gray(img, color_space="BGR"))


def test_prep_gray_rejects_unknown_color_space():
    img = _asymmetric_channel_image()
    with pytest.raises(ValueError, match="unsupported color_space"):
        dad._prep_gray(img, color_space="YUV")


# ---------------------------------------------------------------------------------------
# Full pipeline integration -- FakeVideoProvider, same pattern as
# tests/test_media_transforms_execution.py, so no real video file is needed
# ---------------------------------------------------------------------------------------


class FakeVideoProvider:
    capabilities = {"streaming": True, "variable_frame_rate": True}

    def __init__(self, frames):
        self.frames = frames  # list of (H, W, 3) uint8 arrays

    def probe(self, uri):
        return VideoMetadata(
            uri,
            self.frames[0].shape[1],
            self.frames[0].shape[0],
            len(self.frames),
            float(len(self.frames) - 1),
            1.0,
        )

    def iter_refs(self, uri, *, source_id=None):
        for i in range(len(self.frames)):
            yield FrameRef(uri, source_id or uri, i, float(i))

    def decode_batch(self, refs):
        return np.stack([self.frames[r.frame_index] for r in refs])


def _shifting_noise_frames(n, h=40, w=60, seed=3):
    """Textured (not solid-color) synthetic frames with a real, recoverable shift between
    consecutive frames -- Farneback needs actual texture to find flow, unlike a solid
    color image, which has no gradient to track at all."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h, w + n, 3), dtype=np.uint8)
    return [base[:, i : i + w] for i in range(n)]  # each frame is the base shifted by 1px


def test_full_pipeline_with_fake_video_provider():
    """Real cv2 Farneback + real crop + real release-only-pair logic, run through the
    actual registered aintropy/pixel-shift transform via a real (fake-video-backed)
    kuration Pipeline -- not a unit test of an isolated function."""
    n_frames = 6
    frames = _shifting_noise_frames(n_frames)
    provider = FakeVideoProvider(frames)

    action_cols = [f"a{i}" for i in range(ACTION_DIM)]
    actions = [_action(), _action(w=True), _action(), _action(), _action(), _action()]
    rows = [
        {"t": i, **{col: float(v) for col, v in zip(action_cols, actions[i], strict=True)}}
        for i in range(n_frames)
    ]

    from kuration import SchemaSpec

    schema = SchemaSpec.from_dict(
        {
            "name": "pixel_shift_fake_video_test",
            "fields": {"action": {"kind": "vector", "dims": action_cols}},
        }
    )
    # bins=1 is a deliberate no-op binning axis here (this test isn't exercising binning),
    # which F20 now warns on -- expected, not a bug.
    with pytest.warns(UserWarning, match="bins=1"):
        pipeline = (
            input(rows, schema=schema, fields={"action": action_cols}, time="t")
            .join_video({"mem": "fake.mp4"}, provider=provider, on="frame_index")
            .window(n_frames, stride=n_frames)
            .pixel_shift(controls=["W"])
            .organize("w_n_instances", bins=1)
        )
    result = pipeline.collect(n=10)
    assert not result.errors
    assert len(result.all_segments) == 1
    features = result.all_segments[0].features
    assert features["w_n_instances"] == 1.0
    assert features["w_requested"] == 1.0
    assert features["s_requested"] == 0.0
    assert features["w_shift_min"] == features["w_shift_max"] == features["w_shift_mean"]


def _pixel_shift_pipeline(rows, schema, provider, *, media_column, action_field, inputs=None,
                          window=6):
    from kuration import input as kur_input

    pipeline = (
        kur_input(rows, schema=schema, fields={action_field: [
            column for column in schema.fields[action_field].dims]}, time="t")
        .join_video({"mem": "fake.mp4"}, provider=provider, column=media_column,
                    on="frame_index")
        .window(window, stride=window)
    )
    if inputs is None:
        return pipeline.pixel_shift(controls=["W"])
    return pipeline.transform("aintropy/pixel-shift", inputs=inputs, controls=["W"])


def test_pixel_shift_transform_runs_on_remapped_input_tokens():
    """Regression test for a review-caught finding: `transform_batch` hardcoded the
    "media:frame"/"field:action" batch tokens and `ctx["schema"].media["frame"]`, while the
    core validates a per-use `inputs=` remapping successfully -- so a remapped use passed
    pipeline construction and then died mid-run with a bare KeyError. The transform now
    resolves every input through the roles the context publishes."""
    from kuration import SchemaSpec

    n_frames = 6
    provider = FakeVideoProvider(_shifting_noise_frames(n_frames))
    action_cols = [f"a{i}" for i in range(ACTION_DIM)]
    actions = [_action(), _action(w=True)] + [_action() for _ in range(4)]
    rows = [
        {"t": index, **{column: float(value)
                        for column, value in zip(action_cols, actions[index], strict=True)}}
        for index in range(n_frames)
    ]
    schema = SchemaSpec.from_dict({
        "name": "pixel_shift_remapped_inputs",
        "fields": {"controls": {"kind": "vector", "dims": action_cols}},
    })

    # bins=1 is a deliberate no-op binning axis here (this test isn't exercising binning),
    # which F20 now warns on -- expected, not a bug.
    with pytest.warns(UserWarning, match="bins=1"):
        result = (_pixel_shift_pipeline(
            rows, schema, provider, media_column="frames2", action_field="controls",
            inputs=("media:frames2", "field:controls"))
            .organize("w_n_instances", bins=1)
            .collect(n=10))

    assert not result.errors
    features = result.all_segments[0].features
    assert features["w_n_instances"] == 1.0
    assert features["w_shift_measured"] == 1.0


def test_pixel_shift_does_not_decode_an_unused_lookbehind_frame():
    """Regression test for a review-caught finding: `lookbehind=1` applied to every input, so
    the runner decoded one lookbehind video frame per segment that `transform_batch` never
    reads (it only consumes the lookbehind of the action field)."""
    from kuration import SchemaSpec

    n_frames = 8
    frames = _shifting_noise_frames(n_frames)

    class CountingProvider(FakeVideoProvider):
        def __init__(self, frames):
            super().__init__(frames)
            self.decode_calls = []

        def decode_batch(self, refs):
            refs = list(refs)
            self.decode_calls.append([ref.frame_index for ref in refs])
            return super().decode_batch(refs)

    provider = CountingProvider(frames)
    action_cols = [f"a{i}" for i in range(ACTION_DIM)]
    actions = [_action() for _ in range(n_frames)]
    actions[5] = _action(w=True)
    rows = [
        {"t": index, **{column: float(value)
                        for column, value in zip(action_cols, actions[index], strict=True)}}
        for index in range(n_frames)
    ]
    schema = SchemaSpec.from_dict({
        "name": "pixel_shift_lookbehind_decode",
        "fields": {"action": {"kind": "vector", "dims": action_cols}},
    })

    # bins=1 is a deliberate no-op binning axis here (this test isn't exercising binning),
    # which F20 now warns on -- expected, not a bug.
    with pytest.warns(UserWarning, match="bins=1"):
        result = (_pixel_shift_pipeline(rows, schema, provider, media_column="frame",
                                       action_field="action", window=4)
                  .organize("w_n_instances", bins=1)
                  .collect(n=10))

    assert not result.errors
    # Owned windows plus the first window's three lookahead frames. The second window would
    # additionally decode [3] as its (never-read) lookbehind frame under a shared count.
    assert provider.decode_calls == [[0, 1, 2, 3], [4, 5, 6], [4, 5, 6, 7]]
    assert [3] not in provider.decode_calls


@pytest.mark.parametrize(
    "window,stride,press_start,expected_by_start",
    [
        (4, 4, 3, {0: 1.0, 4: 0.0}),
        (4, 2, 2, {0: 0.0, 2: 1.0, 4: 0.0}),
    ],
)
def test_pipeline_halo_counts_boundary_press_once(
        window, stride, press_start, expected_by_start):
    n_frames = 8
    frames = _shifting_noise_frames(n_frames)
    provider = FakeVideoProvider(frames)
    actions = [_action() for _ in range(n_frames)]
    actions[press_start] = _action(w=True)
    action_cols = [f"a{i}" for i in range(ACTION_DIM)]
    rows = [
        {"t": index, **{column: float(value) for column, value in zip(
            action_cols, actions[index], strict=True)}}
        for index in range(n_frames)
    ]
    from kuration import SchemaSpec

    schema = SchemaSpec.from_dict({
        "name": "pixel_shift_boundary_test",
        "fields": {"action": {"kind": "vector", "dims": action_cols}},
    })
    # bins=1 is a deliberate no-op binning axis here (this test isn't exercising binning),
    # which F20 now warns on -- expected, not a bug.
    with pytest.warns(UserWarning, match="bins=1"):
        result = (input(rows, schema=schema, fields={"action": action_cols}, time="t")
                  .join_video({"mem": "fake.mp4"}, provider=provider, on="frame_index")
                  .window(window, stride=stride)
                  .pixel_shift(controls=["W"])
                  .organize("w_n_instances", bins=1)
                  .collect(n=10))

    assert {segment.start_idx: segment.features["w_n_instances"]
            for segment in result.all_segments} == expected_by_start
    assert sum(segment.features["w_n_instances"] for segment in result.all_segments) == 1.0


# ---------------------------------------------------------------------------------------
# Registration + discovery -- both built-in components resolve through the ordinary
# Python API and the kur-list/kur-describe CLI without plugin entry-point scanning.
# ---------------------------------------------------------------------------------------


def test_pixel_shift_registered_as_both_operator_and_transform():
    names_by_kind = {
        kind: {row["name"] for row in list_components(kind)} for kind in ("operator", "transform")
    }
    assert "aintropy/pixel-shift" in names_by_kind["operator"]
    assert "aintropy/pixel-shift" in names_by_kind["transform"]

    op = describe_component("operator", "aintropy/pixel-shift")
    assert op["version"] == "1"
    assert op["package"] == "kuration-sdk"
    assert op["extra"] == "pixel-shift"
    assert "controls" in op["parameters"]["properties"]
    assert op["parameters"]["properties"]["boundary"]["default"] == "exclude"

    transform = describe_component("transform", "aintropy/pixel-shift")
    assert transform["version"] == "1"
    assert transform["description"]  # non-empty; was missing before this PR's fix


def test_builtin_and_isolated_snapshot_contract():
    isolated = default_registry.snapshot()
    assert isolated.get("operator", "aintropy/pixel-shift") is pso.pixel_shift
    assert isolated.get("transform", "aintropy/pixel-shift") is pso.PixelShift


def test_supported_direct_api_is_exported_from_the_package_root():
    assert pixel_shift.ALL_CONTROLS == dad.ALL_CONTROLS
    assert pixel_shift.find_isolated_press_sequences is dad.find_isolated_press_sequences
    assert pixel_shift.release_transition_displacement is dad.release_transition_displacement
    assert pixel_shift.validate_actions is dad.validate_actions
    assert pixel_shift.validate_frames is dad.validate_frames


def test_pixel_shift_discoverable_through_the_kur_list_cli(monkeypatch, capsys):
    from kuration.core.cli.discovery import list_main

    monkeypatch.setattr(sys, "argv", ["kur-list", "operators", "--json"])
    list_main()
    rows = json.loads(capsys.readouterr().out)
    assert "aintropy/pixel-shift" in {row["name"] for row in rows}
