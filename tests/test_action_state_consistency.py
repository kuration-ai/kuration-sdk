"""Tests for the action_state adapter: baseline validation against pixel_shift's control
vocabulary, per-instance consistency checking via real optical flow, per-clip verdict
aggregation. OpenCV remains an optional SDK extra for optical-flow execution.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from kuration import Segment, get_analyzer, import_dataset, run_dataset_analyzer
from kuration.gameplay import action_state_consistency as asc
from kuration.gameplay.action_state_consistency import (
    ActionStateAnomalyScorer,
    ActionStateConsistencyChecker,
    load_baselines_from_reference,
    validate_action_state_baselines,
    validate_action_state_config,
)
from kuration.gameplay.pixel_shift import diamond_action_decode as dad

ACTION_DIM = dad.ACTION_WIDTH


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


def _w_tap_frames_and_actions(h=40, w=60, seed=3):
    """One isolated W tap (frames 0,1,2: neutral, W, neutral) with real, textured frames
    that produce a genuine, recoverable optical-flow shift -- same pattern as pixel_shift's
    own _shifting_noise_frames."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h, w + 3, 3), dtype=np.uint8)
    frames = np.stack([base[:, i : i + w] for i in range(3)])
    actions = np.stack([_action(), _action(w=True), _action()])
    return frames, actions


def test_action_state_baselines_rejects_unknown_control():
    with pytest.raises(ValueError, match="unknown control"):
        validate_action_state_baselines({"JUMP": {"1": {"expected_sign": 1, "min": 0.0, "max": 1.0}}})


def test_action_state_config_accepts_well_formed_config():
    config = validate_action_state_config(
        {"anomaly_type": "action_state", "baselines": {"W": {"1": {"expected_sign": 1, "min": 0.0, "max": 1.0}}}}
    )
    assert config["anomaly_type"] == "action_state"


def test_action_state_config_rejects_unknown_anomaly_type():
    # A structurally valid baselines dict, so this isolates the anomaly_type check itself
    # -- generic validation (baseline shape) runs before the action_state-specific
    # anomaly_type restriction, so an empty/malformed baselines dict would otherwise be
    # reported first.
    with pytest.raises(ValueError, match="anomaly_type"):
        validate_action_state_config(
            {"anomaly_type": "frame_drop", "baselines": {"W": {"1": {"expected_sign": 1, "min": 0.0, "max": 1.0}}}}
        )


def test_action_state_config_rejects_missing_baselines():
    with pytest.raises(ValueError, match="baselines"):
        validate_action_state_config({"anomaly_type": "action_state"})


def test_sdk_registry_constructs_the_real_action_state_checker():
    baselines = {
        "W": {"1": {"expected_sign": 1, "min": 0.0, "max": 1.0}},
    }
    asc.register()

    checker = get_analyzer(
        asc.ANALYZER_NAME,
        anomaly_type="action_state",
        baselines=baselines,
    )

    assert isinstance(checker, ActionStateConsistencyChecker)
    assert checker.baselines == baselines


def test_top_level_action_state_checker_needs_only_the_sdk_import():
    baselines = {
        "W": {"1": {"expected_sign": 1, "min": 0.0, "max": 1.0}},
    }

    checker = asc.action_state_consistency(
        anomaly_type="action_state",
        baselines=baselines,
    )

    assert isinstance(checker, ActionStateConsistencyChecker)
    assert checker.baselines == baselines


def _fake_reference():
    return {
        "cells": {
            "w": {
                "1": {"reliable": True, "floor": 0.1, "ceiling": 2.0, "expected_sign": 1},
                "2": {"reliable": False, "floor": None, "ceiling": None, "expected_sign": None,
                      "reason": "n=3 < MIN_N=50"},
            },
            "s": {
                "1": {"reliable": True, "floor": 0.05, "ceiling": 1.5, "expected_sign": -1},
            },
        }
    }


def test_load_baselines_from_reference_skips_unreliable_cells():
    baselines = load_baselines_from_reference(_fake_reference())
    assert baselines["W"] == {"1": {"expected_sign": 1, "min": 0.1, "max": 2.0}}
    assert "2" not in baselines["W"]  # unreliable -- not silently trusted
    assert baselines["S"] == {"1": {"expected_sign": -1, "min": 0.05, "max": 1.5}}


def test_load_baselines_from_reference_respects_controls_filter():
    baselines = load_baselines_from_reference(_fake_reference(), controls=["W"])
    assert set(baselines) == {"W"}


def test_load_baselines_from_reference_result_is_directly_valid():
    baselines = load_baselines_from_reference(_fake_reference())
    validate_action_state_baselines(baselines)  # must not raise


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ({}, "cells"),
        ({"cells": {"w": {"1": {}}}}, "reliable"),
        (
            {"cells": {"w": {"1": {
                "reliable": True,
                "floor": 0.0,
                "ceiling": 1.0,
            }}}},
            "missing",
        ),
        (
            {"cells": {"jump": {"1": {
                "reliable": True,
                "floor": 0.0,
                "ceiling": 1.0,
                "expected_sign": 1,
            }}}},
            "unknown control",
        ),
    ],
)
def test_load_baselines_from_reference_rejects_malformed_reports(reference, message):
    with pytest.raises(ValueError, match=message):
        load_baselines_from_reference(reference)


def test_load_baselines_from_reference_rejects_invalid_control_filters():
    reference = _fake_reference()
    with pytest.raises(ValueError, match="non-empty sequence"):
        load_baselines_from_reference(reference, controls="W")
    with pytest.raises(ValueError, match="unknown"):
        load_baselines_from_reference(reference, controls=["JUMP"])
    with pytest.raises(ValueError, match="duplicates"):
        load_baselines_from_reference(reference, controls=["W", "W"])
    with pytest.raises(ValueError, match="no reliable"):
        load_baselines_from_reference(reference, controls=["A"])


# ---------------------------------------------------------------------------------------
# ActionStateConsistencyChecker -- per-clip aggregation over real pixel_shift measurements
# ---------------------------------------------------------------------------------------


def _checker(min_=-1e9, max_=1e9, expected_sign=1):
    """A checker with a deliberately WIDE-open (or, via min_/max_/expected_sign, a
    deliberately narrow) W baseline covering all three press lengths, so tests can
    control whether a real measured shift is trivially consistent or not without needing
    to predict optical flow's exact numeric output."""
    baselines = {
        "W": {
            "1": {"expected_sign": expected_sign, "min": max(0.0, min_), "max": max_},
            "2": {"expected_sign": expected_sign, "min": max(0.0, min_), "max": max_},
            "3": {"expected_sign": expected_sign, "min": max(0.0, min_), "max": max_},
        }
    }
    return ActionStateConsistencyChecker({"anomaly_type": "action_state", "baselines": baselines})


def test_add_segment_rejects_control_without_a_configured_baseline():
    checker = _checker()
    frames, actions = _w_tap_frames_and_actions()
    with pytest.raises(ValueError, match="no baseline configured"):
        checker.add_segment(frames, actions, ("clip1", 0, 3), controls=["S"])


def test_clip_with_only_consistent_instances_is_consistent():
    checker = _checker()  # wide-open range, real W's positive sign
    frames, actions = _w_tap_frames_and_actions()
    checker.add_segment(frames, actions, ("clip1", 0, 3), controls=["W"])
    assert checker.clip_verdict("clip1") == "consistent"
    assert checker.summary() == {"consistent": ["clip1"], "inconsistent": [], "inconclusive": []}


def test_clip_with_a_wrong_sign_instance_is_inconsistent():
    checker = _checker(expected_sign=-1)  # W's real measured shift is positive -> violates a -1 baseline
    frames, actions = _w_tap_frames_and_actions()
    checker.add_segment(frames, actions, ("clip1", 0, 3), controls=["W"])
    assert checker.clip_verdict("clip1") == "inconsistent"


def test_clip_with_no_qualifying_instances_is_inconclusive():
    checker = _checker()
    actions = np.stack([_action(), _action(), _action()])  # W never pressed
    frames = np.zeros((3, 40, 60, 3), dtype=np.uint8)
    checker.add_segment(frames, actions, ("clip1", 0, 3), controls=["W"])
    assert checker.clip_verdict("clip1") == "inconclusive"


def test_one_bad_instance_makes_the_whole_clip_inconsistent_even_with_others_good():
    """Matches the established rule: within one clip, if multiple instances are found,
    ALL must be consistent for the clip to be consistent -- one violation poisons the
    whole clip, not just that one instance. Uses a press_len=1 baseline that's trivially
    satisfied and a press_len=2 baseline that's impossible to satisfy (min far beyond any
    real optical-flow magnitude), so the outcome doesn't depend on predicting real noise."""
    baselines = {
        "W": {
            "1": {"expected_sign": 1, "min": 0.0, "max": 1e9},  # trivially satisfied
            "2": {"expected_sign": 1, "min": 1e9, "max": 1e9 + 1},  # impossible to satisfy
        }
    }
    checker = ActionStateConsistencyChecker(
        {"anomaly_type": "action_state", "baselines": baselines}
    )
    rng = np.random.default_rng(1)
    h, w = 40, 60
    # One press_len=1 tap (frames 0,1,2) and one press_len=2 tap (frames 3,4,5,6), both
    # attributed to the same clip.
    actions = np.stack(
        [_action(), _action(w=True), _action()]
        + [_action(), _action(w=True), _action(w=True), _action()]
    )
    frames = rng.integers(0, 255, size=(len(actions), h, w, 3), dtype=np.uint8)
    checker.add_segment(frames, actions, ("clip1", 0, len(actions)), controls=["W"])
    assert checker.clip_verdict("clip1") == "inconsistent"


def test_multiple_segments_of_the_same_clip_aggregate_into_one_verdict():
    checker = _checker()
    frames1, actions1 = _w_tap_frames_and_actions(seed=1)
    frames2, actions2 = _w_tap_frames_and_actions(seed=2)
    checker.add_segment(frames1, actions1, ("clip1", 0, 3), controls=["W"])
    checker.add_segment(frames2, actions2, ("clip1", 3, 6), controls=["W"])
    assert checker.clip_verdict("clip1") == "consistent"
    records = checker._clip_records["clip1"]
    assert len(records) == 2  # both segments' instances attributed to the same clip


def test_control_never_observed_in_a_segment_is_silently_not_checked():
    """A baselined control that's simply never active in a given segment (here,
    MOUSE_X_POS while only W is pressed) must not crash -- zero instances are found for
    it, nothing is recorded, and the clip falls back to inconclusive if that's all that
    was checked."""
    baselines = {"MOUSE_X_POS": {"2": {"expected_sign": 1, "min": 0.0, "max": 1e9}}}
    checker = ActionStateConsistencyChecker({"anomaly_type": "action_state", "baselines": baselines})
    frames, actions = _w_tap_frames_and_actions()  # a W tap; mouse never moves
    checker.add_segment(frames, actions, ("clip1", 0, 3), controls=["MOUSE_X_POS"])
    assert checker.clip_verdict("clip1") == "inconclusive"


def test_instance_at_an_unconfigured_cell_key_is_silently_not_checked():
    """A control CAN have a baseline dict without covering every cell_key it might
    produce -- e.g. only press_len=1 configured, not 2 or 3. An instance at an
    unconfigured cell_key (press_len=2 here) must not crash and must not count as a
    violation; it's simply invisible to this checker, while a DIFFERENT, configured
    cell_key (press_len=1) in the very same segment is still checked normally."""
    baselines = {"W": {"1": {"expected_sign": 1, "min": 0.0, "max": 1e9}}}  # no "2" entry
    checker = ActionStateConsistencyChecker({"anomaly_type": "action_state", "baselines": baselines})
    rng = np.random.default_rng(5)
    h, w = 40, 60
    # One press_len=1 tap (frames 0,1,2; configured, checked) and one press_len=2 tap
    # (frames 3,4,5,6; unconfigured cell_key, must be skipped, not a violation).
    actions = np.stack(
        [_action(), _action(w=True), _action()]
        + [_action(), _action(w=True), _action(w=True), _action()]
    )
    frames = rng.integers(0, 255, size=(len(actions), h, w, 3), dtype=np.uint8)
    checker.add_segment(frames, actions, ("clip1", 0, len(actions)), controls=["W"])
    assert checker.clip_verdict("clip1") == "consistent"
    assert {r["cell_key"] for r in checker._clip_records["clip1"]} == {"1"}


def test_clip_verdict_raises_for_an_unknown_clip():
    checker = _checker()
    with pytest.raises(KeyError):
        checker.clip_verdict("never-seen")


def test_deduplicates_the_same_press_across_overlapping_windows():
    """Same overlapping-window scenario as pixel_shift's own regression test:
    a press's global start can fall within more than one overlapping window's local
    range, so each window's add_segment() call would otherwise independently record it.
    Global timeline, frames 0..9: an isolated W press at global frame 5. Window A: global
    [3:8) -- local start of the press = 5-3 = 2. Window B: global [4:9) -- local start =
    5-4 = 1. Both windows own the press (boundary="exclude" only rejects local_start==0)."""
    checker = _checker()
    rng = np.random.default_rng(9)
    h, w = 40, 60
    global_actions = np.stack([_action()] * 5 + [_action(w=True)] + [_action()] * 4)
    global_frames = rng.integers(0, 255, size=(10, h, w, 3), dtype=np.uint8)

    checker.add_segment(global_frames[3:8], global_actions[3:8], ("clip", 3, 8), controls=["W"])
    checker.add_segment(global_frames[4:9], global_actions[4:9], ("clip", 4, 9), controls=["W"])

    records = checker._clip_records["clip"]
    assert len(records) == 1  # not 2 -- deduplicated across the two overlapping windows
    assert records[0]["frame"] == 5  # the true global frame, matching both windows' math


def test_print_report_includes_counts_and_clip_ids(capsys):
    checker = _checker()
    frames, actions = _w_tap_frames_and_actions()
    checker.add_segment(frames, actions, ("clip1", 0, 3), controls=["W"])
    checker.print_report()
    out = capsys.readouterr().out
    assert "CONSISTENT: 1 clip(s)" in out
    assert "clip1" in out


def test_print_report_includes_violation_detail_for_inconsistent_clips(capsys):
    """The inconsistent-clip branch (control/cell/frame/shift detail) is a distinct code
    path from the consistent/inconclusive branches -- exercise it directly rather than
    trusting it by inspection."""
    checker = _checker(expected_sign=-1)  # real W shift is positive -> violates a -1 baseline
    frames, actions = _w_tap_frames_and_actions()
    checker.add_segment(frames, actions, ("bad_clip", 0, 3), controls=["W"])
    assert checker.clip_verdict("bad_clip") == "inconsistent"

    checker.print_report()
    out = capsys.readouterr().out
    assert "INCONSISTENT: 1 clip(s)" in out
    assert "bad_clip" in out
    assert "1 violation(s)" in out
    assert "W/1@frame1" in out  # control/cell_key@frame from the one recorded instance
    assert "shift=" in out


def test_action_state_analyzer_runs_on_curated_dataset_modalities(tmp_path):
    source = tmp_path / "clip.npz"
    dataset = tmp_path / "dataset"
    frames, actions = _w_tap_frames_and_actions()
    np.savez(source, video=frames, input=actions)
    import_dataset(
        source,
        dataset,
        source_id="clip1",
    )

    result = run_dataset_analyzer(
        asc.ANALYZER_NAME,
        dataset,
        parameters={
            "anomaly_type": "action_state",
            "baselines": {
                "W": {"1": {"expected_sign": 1, "min": 0.0, "max": 1e9}},
            },
        },
        options={"frames": "video", "actions": "input", "controls": ["W"]},
    )

    assert result["passed"] is True
    assert result["shards_checked"] == 1
    assert result["summary"]["consistent"] == ["clip1"]


def test_check_dataset_reports_shard_relative_positions_and_no_cross_shard_dedup(
        tmp_path, capsys):
    """Regression test for a review-caught finding: `check_dataset` assigns synthetic
    monotonic per-source offsets, which are not source frames and cannot support the
    documented (source_id, control, cell_key, frame) cross-window de-duplication. A shard
    manifest entry records path/source_id/entity_id/split/shard_index/n_steps only -- no
    source start offset -- so real offsets are unrecoverable; the synthetic space is now
    named, mapped per shard, and its lost guarantee is stated instead of silently inverted."""
    source = tmp_path / "clip.npz"
    dataset = tmp_path / "dataset"
    rng = np.random.default_rng(11)
    base = rng.integers(0, 255, size=(40, 66, 3), dtype=np.uint8)
    frames = np.stack([base[:, index : index + 60] for index in range(6)])
    actions = np.stack([_action(), _action(w=True), _action(),
                        _action(), _action(w=True), _action()])
    np.savez(source, video=frames, input=actions)
    # Two shards for one clip: exactly the case where a synthetic offset stands in for a
    # source position, one W tap in each shard.
    import_dataset(source, dataset, source_id="clip1", shard_length=3)

    checker = asc.action_state_consistency(
        anomaly_type="action_state",
        baselines={"W": {"1": {"expected_sign": 1, "min": 0.0, "max": 1e9}}},
    )
    result = checker.check_dataset(dataset, frames="video", actions="input", controls=["W"])

    assert result["shards_checked"] == 2
    assert result["frame_space"] == "shard-concatenation"
    assert result["cross_shard_dedup"] is False
    assert "not source frames" in result["frame_space_note"]
    # Every shard's occupied range is reported, so a caller can map a printed position back.
    assert [(entry["start"], entry["end"]) for entry in result["shard_ranges"]] == [(0, 3), (3, 6)]
    assert all(entry["path"] and entry["source_id"] == "clip1"
               for entry in result["shard_ranges"])
    assert [record["frame"] for record in checker._clip_records["clip1"]] == [1, 4]

    checker.print_report()
    out = capsys.readouterr().out
    assert "frame space: shard-concatenation" in out
    assert "no cross-shard de-duplication in dataset mode" in out


# ---------------------------------------------------------------------------------------
# ActionStateAnomalyScorer: the SegmentScorer rehost, reading Segment.records instead of
# re-deriving instances from raw frames.
# ---------------------------------------------------------------------------------------

def _record(control="W", cell_key="1", local_start=0, shift=0.5):
    return {"control": control, "cell_key": cell_key, "local_start": local_start,
            "shift": shift}


def _segment(source_id, start_idx, end_idx, records=()):
    seg = Segment(source_id, start_idx, end_idx, (float(start_idx), float(end_idx)))
    if records:
        seg.records[asc.PIXEL_SHIFT_RECORDS_KEY] = list(records)
    return seg


CONSISTENT_BASELINE = {"W": {"1": {"expected_sign": 1, "min": 0.0, "max": 1e9}}}
FLIPPED_BASELINE = {"W": {"1": {"expected_sign": -1, "min": 0.0, "max": 1e9}}}


def test_action_state_anomaly_scorer_counts_inconsistent_records_per_segment():
    scorer = ActionStateAnomalyScorer(baselines=CONSISTENT_BASELINE, controls=["W"])
    segments = [
        _segment("clip1", 0, 4, [_record(shift=0.5)]),          # one consistent record
        _segment("clip1", 4, 8),                                # no records at all
        _segment("clip1", 8, 12, [_record(shift=0.5), _record(local_start=1, shift=0.6)]),
    ]
    scores = scorer.score(segments, {})
    assert list(scores) == [0.0, 0.0, 0.0]

    flipped_scorer = ActionStateAnomalyScorer(baselines=FLIPPED_BASELINE, controls=["W"])
    scores = flipped_scorer.score(segments, {})
    assert list(scores) == [1.0, 0.0, 2.0]


def test_action_state_anomaly_scorer_ignores_records_for_unrequested_controls():
    scorer = ActionStateAnomalyScorer(baselines=FLIPPED_BASELINE, controls=["W"])
    segments = [_segment("clip1", 0, 4, [_record(control="S", shift=0.5)])]
    assert list(scorer.score(segments, {})) == [0.0]


def test_action_state_anomaly_scorer_needs_controls_when_self_calibrated():
    with pytest.raises(ValueError, match="needs controls="):
        ActionStateAnomalyScorer()


def test_action_state_anomaly_scorer_rejects_control_without_a_configured_baseline():
    with pytest.raises(ValueError, match="no baseline configured for control"):
        ActionStateAnomalyScorer(baselines=CONSISTENT_BASELINE, controls=["S"])


def test_action_state_anomaly_scorer_self_calibrates_across_the_whole_pool():
    # 4 normal clips with a +shift press, one outlier clip with a -shift press: enough
    # independent evidence (min_samples=3) for the outlier's leave-one-out consensus to flag it.
    calibration = {"min_samples": 3, "sign_consensus": 0.7}
    scorer = ActionStateAnomalyScorer(controls=["W"], calibration=calibration)
    segments = [_segment(f"clip{i}", 0, 4, [_record(shift=0.5)]) for i in range(1, 5)]
    segments.append(_segment("clip5", 0, 4, [_record(shift=-0.5)]))
    scores = scorer.score(segments, {})
    assert list(scores[:4]) == [0.0, 0.0, 0.0, 0.0]
    assert scores[4] == 1.0


def test_resolved_baselines_for_clip_returns_supplied_baselines_verbatim():
    config = validate_action_state_config(
        {"anomaly_type": "action_state", "baselines": CONSISTENT_BASELINE})
    checker = ActionStateConsistencyChecker(config)
    checker.resolve_segment(("clip1", 0, 4), expected_length=4)
    assert checker.resolved_baselines_for_clip("clip1") == CONSISTENT_BASELINE


def test_resolved_baselines_for_clip_derives_leave_one_out_consensus():
    config = validate_action_state_config(
        {"anomaly_type": "action_state", "baselines": None,
         "calibration": {"min_samples": 3, "sign_consensus": 0.7}})
    checker = ActionStateConsistencyChecker(config)
    for i in range(1, 6):
        source_id = f"clip{i}"
        checker.resolve_segment((source_id, 0, 4), expected_length=4)
        checker.record_instance(source_id, "W", "1", 0, 0.5 if source_id != "clip5" else -0.5)
    # clip5 is excluded from its own consensus: the other 4 clips unanimously agree on +1.
    assert checker.resolved_baselines_for_clip("clip5")["W"]["1"]["expected_sign"] == 1
    # A normal clip's consensus pool includes the outlier, diluting but not flipping consensus.
    assert checker.resolved_baselines_for_clip("clip1")["W"]["1"]["expected_sign"] == 1


def test_is_instance_consistent_matches_record_instance_judgment():
    config = validate_action_state_config(
        {"anomaly_type": "action_state", "baselines": CONSISTENT_BASELINE})
    checker = ActionStateConsistencyChecker(config)
    checker.resolve_segment(("clip1", 0, 4), expected_length=4)
    assert checker.is_instance_consistent("clip1", "W", "1", 0.5) is True
    assert checker.is_instance_consistent("clip1", "W", "1", -0.5) is False
    # No baseline configured for this cell_key -> not checked, matching record_instance.
    assert checker.is_instance_consistent("clip1", "W", "999", 0.5) is None
