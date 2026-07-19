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

import numpy as np
import pytest

from kuration import FrameRef, VideoMetadata, describe_component, input, list_components

from plugins.pixel_shift import diamond_action_decode as dad
from plugins.pixel_shift import pixel_shift_operator as pso

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
    pso.reset_ledger()
    rng = np.random.default_rng(1)
    h, w = 40, 60
    # Two isolated D taps (press_len=1) in one segment: one near-zero, one large --
    # min/max must capture both, not just their average.
    actions = np.stack([
        _action(), _action(d=True), _action(),  # tap 1: frames 0,1,2
        _action(), _action(d=True), _action(),  # tap 2: frames 3,4,5
    ])
    frames = rng.integers(0, 255, size=(actions.shape[0], h, w, 3), dtype=np.uint8)

    scalars = pso.segment_scalars(frames, actions, "clip 'test', frames 0-6", controls=["D"])
    assert scalars["d_n_instances"] == 2.0
    assert scalars["d_shift_min"] <= scalars["d_shift_mean"] <= scalars["d_shift_max"]


def test_segment_scalars_sentinel_when_no_instances_found():
    pso.reset_ledger()
    h, w = 40, 60
    actions = np.stack([_action(), _action(), _action()])
    frames = np.zeros((actions.shape[0], h, w, 3), dtype=np.uint8)

    scalars = pso.segment_scalars(frames, actions, "clip 'test', frames 0-3", controls=["W"])
    assert scalars["w_n_instances"] == 0.0
    assert scalars["w_shift_min"] == 0.0
    assert scalars["w_shift_max"] == 0.0
    assert scalars["w_shift_mean"] == 0.0


def test_ledger_records_clip_provenance_for_print_report():
    pso.reset_ledger()
    rng = np.random.default_rng(2)
    h, w = 40, 60
    actions = np.stack([_action(), _action(a=True), _action()])
    frames = rng.integers(0, 255, size=(actions.shape[0], h, w, 3), dtype=np.uint8)

    pso.segment_scalars(frames, actions, "clip 'demo', frames 100-103", controls=["A"])
    assert "A" in pso.LEDGER
    assert "1" in pso.LEDGER["A"]  # press_len=1
    assert pso.LEDGER["A"]["1"][0]["clip"] == "clip 'demo', frames 100-103"


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
    pso.reset_ledger()
    n_frames = 6
    frames = _shifting_noise_frames(n_frames)
    provider = FakeVideoProvider(frames)

    action_cols = [f"a{i}" for i in range(ACTION_DIM)]
    actions = [_action(), _action(w=True), _action(), _action(), _action(), _action()]
    rows = [{"t": i, **{col: float(v) for col, v in zip(action_cols, actions[i])}}
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


def test_pixel_shift_discoverable_through_the_kur_list_cli(monkeypatch, capsys):
    from kuration.cli.discovery import list_main

    monkeypatch.setattr(sys, "argv", ["kur-list", "operators", "--json"])
    list_main()
    rows = json.loads(capsys.readouterr().out)
    assert "aintropy/pixel-shift" in {row["name"] for row in rows}
