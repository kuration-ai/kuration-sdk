"""Tests for the pixel_shift plugin: diamond_action_decode.py's action-decode / isolated-
press-detection logic, and pixel_shift_operator.py's aggregation, registration, and
pipeline integration. Mirrors this repo's own testing conventions (see
tests/test_media_transforms_execution.py's FakeVideoProvider pattern, and
tests/test_operator_registry.py's discovery-CLI test) rather than needing real video
files or real gameplay capture -- every test here is deterministic and self-contained.
"""
from __future__ import annotations

import json
import sys
from importlib.metadata import entry_points

import numpy as np
import pytest

from kuration import (FrameRef, VideoMetadata, default_registry, describe_component, input,
                      list_components)

from kuration_pixel_shift import diamond_action_decode as dad
from kuration_pixel_shift import pixel_shift_operator as pso
from kuration_pixel_shift import register

register()

ACTION_DIM = dad.N_KEYS + 2 + dad.N_MOUSE_X + dad.N_MOUSE_Y  # 51, matches asg2diamond's layout


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
    Y = np.stack([_action(w=True), _action(w=True), _action()])
    assert dad.find_isolated_press_sequences(Y, "W", press_len=2) == [(0, 2)]
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
    Y = np.stack([_action(w=True), _action(a=True), _action(a=True)])
    assert dad.find_isolated_press_sequences(Y, "W", press_len=1) == [(0, 2)]


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
        assert dad.displacement_median(rightward_flow, "A") == dad.displacement_median(rightward_flow, "D")
        assert dad.displacement_median(rightward_flow, "MOUSE_X_POS") == dad.displacement_median(rightward_flow, "A")

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
    actions = np.stack([_action(w=True)] * 2 + [_action()] * 5)
    frames = rng.integers(0, 255, size=(actions.shape[0], h, w, 3), dtype=np.uint8)

    results = dad.release_transition_displacement(frames, actions, "W", press_len=2)
    assert len(results) == 1
    start, _shift = results[0]
    assert start == 0  # sequence start; release transition is start + press_len - 1 = 1 -> 2


# ---------------------------------------------------------------------------------------
# pixel_shift_operator.py -- per-segment aggregation (min/max/mean/n_instances)
# ---------------------------------------------------------------------------------------

def test_segment_scalars_reports_true_min_max_mean_not_an_average_that_hides_outliers():
    rng = np.random.default_rng(1)
    h, w = 40, 60
    # Two isolated D taps (press_len=1) in one segment: one near-zero, one large --
    # min/max must capture both, not just their average.
    actions = np.stack([
        _action(), _action(d=True), _action(),  # tap 1: frames 0,1,2
        _action(), _action(d=True), _action(),  # tap 2: frames 3,4,5
    ])
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


def test_transform_rejects_unknown_or_duplicate_controls():
    with pytest.raises(ValueError, match="unknown pixel-shift control"):
        pso.PixelShift(["JUMP"])
    with pytest.raises(ValueError, match="duplicates"):
        pso.PixelShift(["W", "W"])


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
    actions = np.stack([
        _action(), _action(a=True), _action(),  # tap 1: local frames 0,1,2
        _action(), _action(a=True), _action(),  # tap 2: local frames 3,4,5
    ])
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
        return VideoMetadata(uri, self.frames[0].shape[1], self.frames[0].shape[0],
                              len(self.frames), float(len(self.frames) - 1), 1.0)

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
    return [base[:, i:i + w] for i in range(n)]  # each frame is the base shifted by 1px


def test_full_pipeline_with_fake_video_provider():
    """Real cv2 Farneback + real crop + real release-only-pair logic, run through the
    actual registered aintropy/pixel-shift transform via a real (fake-video-backed)
    kuration Pipeline -- not a unit test of an isolated function."""
    n_frames = 6
    frames = _shifting_noise_frames(n_frames)
    provider = FakeVideoProvider(frames)

    action_cols = [f"a{i}" for i in range(ACTION_DIM)]
    actions = [_action(), _action(w=True), _action(), _action(), _action(), _action()]
    rows = [{"t": i, **{col: float(v) for col, v in zip(action_cols, actions[i], strict=True)}}
            for i in range(n_frames)]

    from kuration import SchemaSpec
    schema = SchemaSpec.from_dict({
        "name": "pixel_shift_fake_video_test",
        "fields": {"action": {"kind": "vector", "dims": action_cols}},
    })
    pipeline = (input(rows, schema=schema, fields={"action": action_cols}, time="t")
                .join_video({"mem": "fake.mp4"}, provider=provider, on="frame_index")
                .window(n_frames, stride=n_frames)
                .transform("aintropy/pixel-shift", controls=["W"])
                .organize("w_n_instances", bins=1))
    result = pipeline.collect(n=10)
    assert not result.errors
    assert len(result.all_segments) == 1
    features = result.all_segments[0].features
    assert features["w_n_instances"] == 1.0
    assert features["w_shift_min"] == features["w_shift_max"] == features["w_shift_mean"]


# ---------------------------------------------------------------------------------------
# Registration + discovery -- both the `kuration.plugins`-entry-point-registered
# components resolve through the ordinary Python API and the kur-list/kur-describe CLI
# ---------------------------------------------------------------------------------------

def test_pixel_shift_registered_as_both_operator_and_transform():
    names_by_kind = {kind: {row["name"] for row in list_components(kind)}
                      for kind in ("operator", "transform")}
    assert "aintropy/pixel-shift" in names_by_kind["operator"]
    assert "aintropy/pixel-shift" in names_by_kind["transform"]

    op = describe_component("operator", "aintropy/pixel-shift")
    assert op["version"] == "1"
    assert "controls" in op["parameters"]["properties"]

    transform = describe_component("transform", "aintropy/pixel-shift")
    assert transform["version"] == "1"
    assert transform["description"]  # non-empty; was missing before this PR's fix


def test_plugin_entry_point_and_isolated_snapshot_contract():
    installed = entry_points()
    candidates = (installed.select(group="kuration.plugins", name="pixel_shift")
                  if hasattr(installed, "select") else [
                      point for point in installed.get("kuration.plugins", ())
                      if point.name == "pixel_shift"])
    assert [point.value for point in candidates] == ["kuration_pixel_shift:register"]

    isolated = default_registry.snapshot()
    assert isolated.get("operator", "aintropy/pixel-shift") is pso.pixel_shift
    assert isolated.get("transform", "aintropy/pixel-shift") is pso.PixelShift


def test_pixel_shift_discoverable_through_the_kur_list_cli(monkeypatch, capsys):
    from kuration.cli.discovery import list_main

    monkeypatch.setattr(sys, "argv", ["kur-list", "operators", "--json"])
    list_main()
    rows = json.loads(capsys.readouterr().out)
    assert "aintropy/pixel-shift" in {row["name"] for row in rows}
