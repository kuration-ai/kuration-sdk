"""Tests for the generic anomaly_checker core -- baseline validation, the sign+magnitude
consistency check, and per-clip verdict aggregation. Deliberately uses only synthetic
(source_id, control, cell_key, frame, shift) tuples, never kuration_pixel_shift -- this
file must be able to run and pass with kuration-plugin-pixel-shift NOT installed at all,
proving the generic core is genuinely measurement-source-agnostic. See
test_action_state.py for the pixel_shift-specific adapter's own tests.
"""
from __future__ import annotations

import sys

import pytest

from kuration_action_state_consistency import (
    AnomalyChecker,
    validate_baselines,
    validate_config,
)
from kuration_action_state_consistency.anomaly_checker import _is_consistent


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


def test_generic_validate_baselines_rejects_non_string_cell_key():
    with pytest.raises(ValueError, match="cell keys must be strings"):
        validate_baselines({"X": {1: {"expected_sign": 1, "min": 0.0, "max": 1.0}}})


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


def test_generic_anomaly_checker_record_instance_rejects_unconfigured_control():
    config = validate_config({"anomaly_type": "synthetic_demo", "baselines": {"X": {"1": {"expected_sign": 1, "min": 0.0, "max": 10.0}}}})
    checker = AnomalyChecker(config)
    checker.resolve_segment(("clip1", 0, 5), expected_length=5)
    with pytest.raises(KeyError):
        checker.record_instance("clip1", "Y", "1", 2, 3.0)  # "Y" was never configured


def test_consistency_check_transform_counts_consistent_and_inconsistent_values():
    """The one component this package registers with kuration-sdk -- built entirely on
    the generic core, no pixel_shift involved. A plain, direct call (not through a real
    Pipeline) is enough to exercise its actual counting logic."""
    from kuration_action_state_consistency import ConsistencyCheck

    transform = ConsistencyCheck(expected_sign=1, min_magnitude=0.5, max_magnitude=2.0)
    batch = {"field:value": [[1.0, -1.0, 0.1, 5.0]]}  # one segment, 4 ticks
    result = transform.transform_batch(batch, ctx={})
    assert result["n_consistent"].tolist() == [1.0]  # only 1.0 satisfies sign+range
    assert result["n_inconsistent"].tolist() == [3.0]


def test_register_adds_the_transform_and_operator_without_pixel_shift():
    """register() must succeed and register something real using ONLY this package's
    hard dependencies (kuration-sdk, numpy) -- kuration-sdk's plugin-discovery contract
    calls register() unconditionally, regardless of which optional extras are installed,
    so it must never depend on kuration_pixel_shift."""
    from kuration import default_registry
    import kuration_action_state_consistency as asc

    asc.register()  # idempotent -- must not raise even if already registered
    assert "aintropy/consistency-check" in default_registry.names("transform")
    assert "aintropy/consistency-check" in default_registry.names("operator")


def test_generic_anomaly_checker_is_importable_and_usable_without_pixel_shift_installed(monkeypatch):
    """The core bifurcation guarantee: block kuration_pixel_shift from being importable at
    all, then confirm the generic anomaly_checker module (and AnomalyChecker itself) still
    imports and works fully -- it never references kuration_pixel_shift anywhere."""
    monkeypatch.setitem(sys.modules, "kuration_pixel_shift", None)
    for name in list(sys.modules):
        if name.startswith("kuration_action_state_consistency"):
            monkeypatch.delitem(sys.modules, name)

    import kuration_action_state_consistency.anomaly_checker as fresh_anomaly_checker

    config = fresh_anomaly_checker.validate_config(
        {"anomaly_type": "x", "baselines": {"X": {"1": {"expected_sign": 1, "min": 0.0, "max": 10.0}}}}
    )
    checker = fresh_anomaly_checker.AnomalyChecker(config)
    source_id, start_idx, _ = checker.resolve_segment(("clip1", 0, 5), expected_length=5)
    checker.record_instance(source_id, "X", "1", start_idx + 1, 5.0)
    assert checker.clip_verdict("clip1") == "consistent"


def test_register_succeeds_without_pixel_shift_installed(monkeypatch):
    """The exact scenario kuration-sdk's plugin discovery hits in a core-only install:
    kuration_pixel_shift unavailable, register() still called unconditionally. Must
    succeed and register the generic component -- this is what keeps `kur-list`/
    `kur-describe`/plugin validation from crashing for anyone who installed this package
    without the [action_state] extra."""
    monkeypatch.setitem(sys.modules, "kuration_pixel_shift", None)
    for name in list(sys.modules):
        if name.startswith("kuration_action_state_consistency"):
            monkeypatch.delitem(sys.modules, name)

    import kuration_action_state_consistency as fresh_asc
    from kuration import default_registry

    fresh_asc.register()
    assert "aintropy/consistency-check" in default_registry.names("transform")


def test_action_state_module_functions_raise_a_clear_error_without_pixel_shift_installed(monkeypatch):
    """The flip side: block kuration_pixel_shift, and confirm the action_state-specific
    functions (which DO need it) fail with a clear, actionable ImportError -- not an
    opaque ModuleNotFoundError, and not a silent wrong answer."""
    monkeypatch.setitem(sys.modules, "kuration_pixel_shift", None)
    for name in list(sys.modules):
        if name.startswith("kuration_action_state_consistency"):
            monkeypatch.delitem(sys.modules, name)

    import kuration_action_state_consistency.action_state as fresh_action_state

    with pytest.raises(ImportError, match="kuration-plugin-pixel-shift"):
        fresh_action_state.validate_action_state_baselines(
            {"W": {"1": {"expected_sign": 1, "min": 0.0, "max": 1.0}}}
        )
    with pytest.raises(ImportError, match="kuration-plugin-pixel-shift"):
        fresh_action_state.load_baselines_from_reference({"cells": {}})
