"""Tests for the generic anomaly_checker core -- baseline validation, the sign+magnitude
consistency check, and per-clip verdict aggregation. Deliberately uses only synthetic
(source_id, control, cell_key, frame, shift) tuples and remains independent of OpenCV.
"""
from __future__ import annotations

import numpy as np
import pytest

from kuration.gameplay.action_state_consistency import (
    AnomalyChecker,
    validate_baselines,
    validate_config,
)
from kuration.gameplay.action_state_consistency.anomaly_checker import _is_consistent


def test_generic_validate_baselines_accepts_any_string_control():
    """The generic core has no opinion on what a "control" is -- an arbitrary label is
    fine, unlike validate_action_state_baselines which checks against pixel_shift's
    vocabulary specifically."""
    baselines = validate_baselines({"anything": {"1": {"expected_sign": 1, "min": 0.1, "max": 2.0}}})
    assert baselines == {"anything": {"1": {"expected_sign": 1, "min": 0.1, "max": 2.0}}}


def test_generic_validate_baselines_rejects_empty_dict():
    with pytest.raises(ValueError, match="non-empty"):
        validate_baselines({})


def test_generic_validate_baselines_rejects_missing_fields():
    with pytest.raises(ValueError, match="missing"):
        validate_baselines({"X": {"1": {"expected_sign": 1, "min": 0.0}}})


@pytest.mark.parametrize("bad_sign", [0, 2, -2, 1.5, True, "positive"])
def test_generic_validate_baselines_rejects_invalid_expected_sign(bad_sign):
    with pytest.raises(ValueError, match="expected_sign"):
        validate_baselines({"X": {"1": {"expected_sign": bad_sign, "min": 0.0, "max": 1.0}}})


def test_generic_validate_baselines_rejects_negative_min_or_max():
    with pytest.raises(ValueError, match="non-negative"):
        validate_baselines({"X": {"1": {"expected_sign": 1, "min": -0.1, "max": 1.0}}})
    with pytest.raises(ValueError, match="non-negative"):
        validate_baselines({"X": {"1": {"expected_sign": 1, "min": 0.0, "max": -1.0}}})


def test_generic_validate_baselines_rejects_min_greater_than_max():
    with pytest.raises(ValueError, match="min must not exceed max"):
        validate_baselines({"X": {"1": {"expected_sign": 1, "min": 2.0, "max": 1.0}}})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("bound", ["min", "max"])
def test_generic_validate_baselines_rejects_nonfinite_bounds(bound, bad):
    cell = {"expected_sign": 1, "min": 0.0, "max": 1.0}
    cell[bound] = bad
    with pytest.raises(ValueError, match="finite"):
        validate_baselines({"X": {"1": cell}})


def test_generic_validate_baselines_accepts_numpy_real_scalars():
    validated = validate_baselines({
        "X": {"1": {"expected_sign": 1, "min": np.float32(0.25), "max": np.int64(2)}}
    })
    assert validated["X"]["1"] == {"expected_sign": 1, "min": 0.25, "max": 2.0}


def test_generic_validate_baselines_rejects_non_string_cell_key():
    with pytest.raises(ValueError, match="cell keys must be non-empty strings"):
        validate_baselines({"X": {1: {"expected_sign": 1, "min": 0.0, "max": 1.0}}})


def test_generic_validate_baselines_rejects_empty_cell_key():
    with pytest.raises(ValueError, match="non-empty strings"):
        validate_baselines({"X": {"": {"expected_sign": 1, "min": 0.0, "max": 1.0}}})


def test_generic_validate_config_accepts_any_non_empty_anomaly_type():
    config = validate_config(
        {"anomaly_type": "literally_anything", "baselines": {"X": {"1": {"expected_sign": 1, "min": 0.0, "max": 1.0}}}}
    )
    assert config["anomaly_type"] == "literally_anything"


def test_generic_validate_config_rejects_missing_baselines():
    with pytest.raises(ValueError, match="baselines"):
        validate_config({"anomaly_type": "x"})


def test_generic_is_consistent_accepts_matching_sign_and_in_range_magnitude():
    baseline = {"expected_sign": 1, "min": 0.5, "max": 2.0}
    assert _is_consistent(1.0, baseline) is True


def test_generic_is_consistent_rejects_wrong_sign():
    baseline = {"expected_sign": 1, "min": 0.5, "max": 2.0}
    assert _is_consistent(-1.0, baseline) is False


def test_generic_is_consistent_rejects_too_small_magnitude():
    baseline = {"expected_sign": 1, "min": 0.5, "max": 2.0}
    assert _is_consistent(0.1, baseline) is False


def test_generic_is_consistent_rejects_too_large_magnitude():
    baseline = {"expected_sign": 1, "min": 0.5, "max": 2.0}
    assert _is_consistent(5.0, baseline) is False


def test_generic_is_consistent_exact_zero_shift_has_no_sign_to_violate():
    """Regression-shaped edge case: a naive (shift >= 0) == (expected_sign == 1) sign
    check would wrongly flag an exact-zero shift as violating a negative-sign baseline,
    even though zero has no direction. Only the magnitude bound applies for a zero shift."""
    zero_inclusive = {"expected_sign": -1, "min": 0.0, "max": 1.0}
    assert _is_consistent(0.0, zero_inclusive) is True

    positive_floor = {"expected_sign": -1, "min": 0.5, "max": 1.0}
    assert _is_consistent(0.0, positive_floor) is False  # fails on magnitude, not sign


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_generic_is_consistent_rejects_nonfinite_measurements(bad):
    with pytest.raises(ValueError, match="shift must be finite"):
        _is_consistent(bad, {"expected_sign": 1, "min": 0.0, "max": 1.0})


def test_generic_anomaly_checker_end_to_end_with_synthetic_instances():
    """The generic AnomalyChecker never touches frames/actions/pixel_shift at all --
    exercised here with hand-fed (source_id, control, cell_key, frame, shift) tuples,
    proving the whole clip-verdict mechanism (resolve_segment, record_instance,
    clip_verdict, summary, print_report) works independently of any measurement source."""
    config = validate_config(
        {"anomaly_type": "synthetic_demo", "baselines": {"X": {"1": {"expected_sign": 1, "min": 0.0, "max": 10.0}}}}
    )
    checker = AnomalyChecker(config)
    source_id, start_idx, _end_idx = checker.resolve_segment(("clip1", 0, 5), expected_length=5)
    checker.record_instance(source_id, "X", "1", start_idx + 2, 3.0)
    assert checker.clip_verdict("clip1") == "consistent"
    assert checker.summary() == {"consistent": ["clip1"], "inconsistent": [], "inconclusive": []}


def test_resolve_segment_accepts_numpy_integer_indexes_and_normalizes_them():
    """Regression test for a review-caught finding: `_validate_segment_id` required plain
    `int` indexes and rejected `np.integer`, so an SDK-provided `segment_ids` entry (which
    carries NumPy scalars) failed here even though the sibling
    `PixelShiftReport.add_segment` accepts them. Indexes are now normalized to `int` so
    recorded frame keys have one canonical type."""
    config = validate_config(
        {"anomaly_type": "synthetic_demo",
         "baselines": {"X": {"1": {"expected_sign": 1, "min": 0.0, "max": 10.0}}}}
    )
    checker = AnomalyChecker(config)

    source_id, start_idx, end_idx = checker.resolve_segment(
        ("clip1", np.int64(4), np.int64(9)), expected_length=5)

    assert (start_idx, end_idx) == (4, 9)
    assert type(start_idx) is int and type(end_idx) is int
    checker.record_instance(source_id, "X", "1", start_idx + 1, 3.0)
    # The same instance seen again through plain ints must still de-duplicate.
    checker.record_instance(source_id, "X", "1", 5, 3.0)
    assert len(checker._clip_records["clip1"]) == 1
    with pytest.raises(ValueError, match="non-negative integers"):
        checker.resolve_segment(("clip1", 0.5, 5.5), expected_length=5)


def test_declared_frame_space_is_printed_and_cannot_change_mid_run(capsys):
    """A caller measuring in a coordinate space other than real source frames must declare
    it, so the printed report never presents synthetic positions as source frames."""
    config = validate_config(
        {"anomaly_type": "synthetic_demo",
         "baselines": {"X": {"1": {"expected_sign": 1, "min": 0.0, "max": 1.0}}}}
    )
    checker = AnomalyChecker(config)
    assert checker.frame_space == "source"
    with pytest.raises(ValueError, match="caveat"):
        checker.declare_frame_space("shard-concatenation", "")
    checker.declare_frame_space("shard-concatenation", "offsets are shard-relative")

    checker.resolve_segment(("clip1", 0, 5), expected_length=5)
    checker.record_instance("clip1", "X", "1", 2, 5.0)  # out of range -> violation
    checker.print_report()
    out = capsys.readouterr().out
    assert "frame space: shard-concatenation -- offsets are shard-relative" in out
    assert "@shard-concatenation-offset2" in out
    with pytest.raises(ValueError, match="cannot redeclare"):
        checker.declare_frame_space("source", "")


def test_to_records_exports_flat_per_instance_rows_without_printing(capsys):
    """to_records() is the unprinted, structured sibling of print_report() -- same
    source/control/cell/frame schema as PixelShiftReport.to_records(), plus the
    action-state judgment and source-level aggregate repeated on every row."""
    config = validate_config(
        {"anomaly_type": "synthetic_demo",
         "baselines": {"X": {"1": {"expected_sign": 1, "min": 0.0, "max": 10.0}}}}
    )
    checker = AnomalyChecker(config)
    checker.declare_frame_space("shard-concatenation", "offsets are shard-relative")

    # clip1: consistent, two instances.
    checker.resolve_segment(("clip1", 0, 5), expected_length=5)
    checker.record_instance("clip1", "X", "1", 4, 3.0)
    checker.record_instance("clip1", "X", "1", 1, 5.0)
    # clip2: inconsistent, one violation.
    checker.resolve_segment(("clip2", 0, 5), expected_length=5)
    checker.record_instance("clip2", "X", "1", 2, 20.0)
    # clip3: inconclusive, nothing ever observed.
    checker.resolve_segment(("clip3", 0, 5), expected_length=5)

    records = checker.to_records()
    assert capsys.readouterr().out == ""  # nothing printed

    assert records == [
        {
            "source_id": "clip1", "control": "X", "cell_key": "1", "frame": 1,
            "shift": 5.0, "consistent": True, "verdict": "consistent", "n_instances": 2,
            "frame_space": "shard-concatenation",
            "frame_space_note": "offsets are shard-relative",
        },
        {
            "source_id": "clip1", "control": "X", "cell_key": "1", "frame": 4,
            "shift": 3.0, "consistent": True, "verdict": "consistent", "n_instances": 2,
            "frame_space": "shard-concatenation",
            "frame_space_note": "offsets are shard-relative",
        },
        {
            "source_id": "clip2", "control": "X", "cell_key": "1", "frame": 2,
            "shift": 20.0, "consistent": False, "verdict": "inconsistent",
            "n_instances": 1, "frame_space": "shard-concatenation",
            "frame_space_note": "offsets are shard-relative",
        },
        {
            "source_id": "clip3", "control": None, "cell_key": None, "frame": None,
            "shift": None, "consistent": None, "verdict": "inconclusive",
            "n_instances": 0, "frame_space": "shard-concatenation",
            "frame_space_note": "offsets are shard-relative",
        },
    ]


def test_generic_anomaly_checker_record_instance_rejects_unconfigured_control():
    config = validate_config({"anomaly_type": "synthetic_demo", "baselines": {"X": {"1": {"expected_sign": 1, "min": 0.0, "max": 10.0}}}})
    checker = AnomalyChecker(config)
    checker.resolve_segment(("clip1", 0, 5), expected_length=5)
    with pytest.raises(KeyError):
        checker.record_instance("clip1", "Y", "1", 2, 3.0)  # "Y" was never configured


def test_generic_anomaly_checker_record_instance_requires_resolved_valid_coordinates():
    config = validate_config({
        "anomaly_type": "synthetic_demo",
        "baselines": {"X": {"1": {"expected_sign": 1, "min": 0.0, "max": 10.0}}},
    })
    checker = AnomalyChecker(config)
    with pytest.raises(ValueError, match=r"resolve_segment\(\)"):
        checker.record_instance("clip1", "X", "1", 2, 3.0)

    checker.resolve_segment(("clip1", 0, 5), expected_length=5)
    with pytest.raises(ValueError, match="source_id"):
        checker.record_instance("", "X", "1", 2, 3.0)
    with pytest.raises(ValueError, match="cell_key"):
        checker.record_instance("clip1", "X", "", 2, 3.0)
    with pytest.raises(ValueError, match="non-negative integer"):
        checker.record_instance("clip1", "X", "1", -1, 3.0)
    checker.record_instance("clip1", "X", "1", np.int64(2), 3.0)
    assert checker._clip_records["clip1"][0]["frame"] == 2
    assert type(checker._clip_records["clip1"][0]["frame"]) is int


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_generic_anomaly_checker_record_instance_rejects_nonfinite_shift(bad):
    config = validate_config(
        {
            "anomaly_type": "synthetic_demo",
            "baselines": {"X": {"1": {"expected_sign": 1, "min": 0.0, "max": 10.0}}},
        }
    )
    checker = AnomalyChecker(config)
    checker.resolve_segment(("clip1", 0, 5), expected_length=5)
    with pytest.raises(ValueError, match="shift must be finite"):
        checker.record_instance("clip1", "X", "1", 2, bad)
    checker.record_instance("clip1", "X", "1", 2, 3.0)
    assert checker.clip_verdict("clip1") == "consistent"


def test_register_adds_the_real_builtin_analyzer():
    """The built-in analyzer is present without plugin discovery."""
    from kuration import default_registry
    from kuration.gameplay import action_state_consistency as asc

    asc.register()  # idempotent -- must not raise even if already registered
    assert asc.ANALYZER_NAME in default_registry.names("analyzer")
    descriptor = default_registry.descriptor("analyzer", asc.ANALYZER_NAME)
    # Dataset-mode only: the tick-stream path (redundantly re-deriving pixel-shift's own
    # measurements from raw frames) is retired in favor of the scorer below.
    assert descriptor.capabilities == {"stateful": True, "cross_segment": True,
                                       "dataset": True}
    assert descriptor.package == "kuration-sdk"
    assert descriptor.extra == "action-state"
    assert asc.SCORER_NAME in default_registry.names("scorer")
    scorer = default_registry.descriptor("scorer", asc.SCORER_NAME)
    assert scorer.package == "kuration-sdk"
    assert set(scorer.parameters["properties"]) == {"baselines", "controls", "calibration"}
    assert asc.OPERATOR_NAME in default_registry.names("operator")
    operator = default_registry.descriptor("operator", asc.OPERATOR_NAME)
    assert operator.capabilities["pipeline_metadata"] is True
    assert operator.capabilities["dataset"] is True
    # baselines is no longer schema-required: omitting it (with controls=) selects
    # self-calibrated mode. The handler itself enforces "baselines or controls".
    assert "required" not in operator.parameters
    assert "calibration" in operator.parameters["properties"]
    assert "aintropy/consistency-check" not in default_registry.names("transform")
    assert "aintropy/consistency-check" not in default_registry.names("operator")


def test_generic_anomaly_checker_is_usable_without_opencv(monkeypatch):
    """Generic tuple aggregation does not need the pixel measurement runtime."""
    monkeypatch.setitem(__import__("sys").modules, "cv2", None)
    from kuration.gameplay.action_state_consistency import anomaly_checker

    config = anomaly_checker.validate_config(
        {"anomaly_type": "x", "baselines": {"X": {"1": {"expected_sign": 1, "min": 0.0, "max": 10.0}}}}
    )
    checker = anomaly_checker.AnomalyChecker(config)
    source_id, start_idx, _ = checker.resolve_segment(("clip1", 0, 5), expected_length=5)
    checker.record_instance(source_id, "X", "1", start_idx + 1, 5.0)
    assert checker.clip_verdict("clip1") == "consistent"


def test_builtin_action_state_constructs_without_opencv(monkeypatch):
    """Discovery and analyzer construction stay safe in a minimal SDK install."""
    monkeypatch.setitem(__import__("sys").modules, "cv2", None)
    from kuration import get_analyzer
    from kuration.gameplay import action_state_consistency as asc

    checker = get_analyzer(
        asc.ANALYZER_NAME,
        anomaly_type="action_state",
        baselines={"W": {"1": {"expected_sign": 1, "min": 0.0, "max": 1.0}}},
    )
    assert isinstance(checker, asc.ActionStateConsistencyChecker)


def test_pixel_shift_execution_has_sdk_extra_hint_without_opencv(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "cv2", None)
    from kuration.gameplay.pixel_shift import diamond_action_decode

    with pytest.raises(ImportError, match=r"kuration-sdk\[pixel-shift\]"):
        diamond_action_decode._require_cv2()


# ---------------------------------------------------------------------------------------
# Self-calibrated mode: leave-one-out consensus outlier detection, no external baselines
# ---------------------------------------------------------------------------------------
from kuration.gameplay.action_state_consistency import (  # noqa: E402
    DEFAULT_CALIBRATION,
    derive_baselines,
    validate_calibration,
)


def _self_calibrated(calibration=None):
    return AnomalyChecker(validate_config(
        {"anomaly_type": "synthetic", "baselines": None, "calibration": calibration}))


def _record(checker, clip, shifts, *, control="X", cell="1"):
    checker.resolve_segment((clip, 0, 10), expected_length=10)
    for frame, shift in enumerate(shifts):
        checker.record_instance(clip, control, cell, frame, shift)


def test_validate_calibration_fills_defaults_and_rejects_bad_knobs():
    assert validate_calibration(None) == DEFAULT_CALIBRATION
    assert validate_calibration({"min_clips": 4})["min_clips"] == 4
    assert validate_calibration({"min_clips": 4})["min_samples"] == 5
    with pytest.raises(ValueError, match="unknown calibration knob"):
        validate_calibration({"quantile": 0.1})
    with pytest.raises(ValueError, match="min_clips"):
        validate_calibration({"min_clips": 1})
    with pytest.raises(ValueError, match="sign_consensus"):
        validate_calibration({"sign_consensus": 0.5})
    with pytest.raises(ValueError, match="tolerance"):
        validate_calibration({"tolerance": 0.9})


def test_validate_config_separates_the_two_calibration_modes():
    self_calibrated = validate_config({"anomaly_type": "t", "baselines": None})
    assert self_calibrated["baselines"] is None
    assert self_calibrated["calibration"] == DEFAULT_CALIBRATION
    with pytest.raises(ValueError, match="self-calibrated"):
        validate_config({
            "anomaly_type": "t",
            "baselines": {"X": {"1": {"expected_sign": 1, "min": 0.0, "max": 1.0}}},
            "calibration": {"min_clips": 4},
        })


def test_derive_baselines_skips_underpowered_and_contested_cells():
    derived = derive_baselines({
        ("X", "1"): [1.0, 1.1, 0.9, 1.0, 1.05],       # clean positive consensus
        ("X", "2"): [1.0, 1.0],                        # too few samples
        ("Y", "1"): [1.0, -1.0, 1.0, -1.0, 1.0],       # contested direction (0.6 < 0.8)
        ("Z", "1"): [0.0, 0.0, 0.0, 0.0, 0.0],         # no nonzero evidence
    })
    assert set(derived) == {"X"}
    baseline = derived["X"]["1"]
    assert baseline["expected_sign"] == 1
    # 5th/95th percentile of |shift| widened by the 1.5x tolerance.
    assert baseline["min"] == pytest.approx(np.quantile([1.0, 1.1, 0.9, 1.0, 1.05], 0.05) / 1.5)
    assert baseline["max"] == pytest.approx(np.quantile([1.0, 1.1, 0.9, 1.0, 1.05], 0.95) * 1.5)


def test_self_calibrated_checker_flags_the_relative_outlier_leave_one_out():
    checker = _self_calibrated({"min_samples": 5, "sign_consensus": 0.7})
    for clip in ("a", "b", "c"):
        _record(checker, clip, [1.0, 1.1, 0.9])
    _record(checker, "d", [-1.0, -1.05])          # moves the wrong way
    _record(checker, "e", [1.0], control="Y")     # a cell nobody else measured

    summary = checker.summary()
    assert summary["inconsistent"] == ["d"]
    assert summary["consistent"] == ["a", "b", "c"]
    # e's only instances live in a cell with no independent evidence: unjudged instances
    # carry no verdict weight, so the clip is inconclusive rather than vacuously blessed.
    assert summary["inconclusive"] == ["e"]

    rows = {(row["source_id"], row["frame"]): row for row in checker.to_records()}
    assert rows[("d", 0)]["consistent"] is False
    assert rows[("e", 0)]["consistent"] is None
    assert rows[("e", 0)]["verdict"] == "inconclusive"

    report = checker.calibration_report()
    assert report["mode"] == "self-calibrated"
    assert report["sign_consensus"] == 0.7
    # The pooled consensus is the reportable record of inferred normality.
    assert report["derived_baselines"]["X"]["1"]["expected_sign"] == 1


def test_self_calibrated_checker_refuses_pools_below_min_clips():
    checker = _self_calibrated()
    _record(checker, "a", [1.0])
    _record(checker, "b", [1.0])
    with pytest.raises(ValueError, match="at least 3 clips"):
        checker.summary()
    with pytest.raises(ValueError, match="at least 3 clips"):
        checker.calibration_report()


def test_supplied_baselines_have_nothing_derived():
    checker = AnomalyChecker(validate_config({
        "anomaly_type": "t",
        "baselines": {"X": {"1": {"expected_sign": 1, "min": 0.0, "max": 2.0}}},
    }))
    assert checker.self_calibrated is False
    assert checker.calibration_report() is None
    with pytest.raises(ValueError, match="nothing was derived"):
        checker.derived_baselines()
