"""Coverage for the kuration.selection domain: recipes, Pipeline operators, and the
config-driven plan/run/describe runner.

Ported from the standalone kuration-ai package (preserved untouched on the
feature/kuration-ai-legacy branch) when its three curation recipes and config runner
were promoted into the SDK as first-class components.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from kuration import default_registry, input as kuration_input
from kuration.selection import BalancedSelector, DiverseSubset, RareEventCoverage
from kuration.selection.plan import _USER_REGISTRY, _available_operations
from kuration.selection.plan import describe, plan, register_operation_type, run


def _sessions() -> dict[str, list[dict]]:
    """Three synthetic sessions with a signal, a temp, and a rare fault flag.

    Session i's fault fires on ticks where ``step % 30 == i``, so there are ~5
    faults per session: enough for RareEventCoverage to have something to
    upweight without dominating the window pool.
    """
    return {
        f"session-{i}": [
            {
                "t": float(step),
                "signal": float(i * step % 7),
                "temp": float(20 + step % 10),
                "fault": 1 if step % 30 == i else 0,
            }
            for step in range(150)
        ]
        for i in range(3)
    }


@pytest.fixture
def base_pipeline():
    """Windowed SDK pipeline with no measurements applied yet.

    Yields 36 windows per session x 3 sessions = 108 windows total, well above
    any count= value used below.
    """
    return kuration_input(_sessions(), time="t", hz=1.0).window(8, stride=4)


@pytest.fixture
def measured_pipeline(base_pipeline):
    """Windowed pipeline with two features already measured.

    DiverseSubset expects features to be measured on the pipeline before it
    runs, so this fixture stages ``act`` and ``temp_mean`` for cluster-balanced
    sampling.
    """
    return (
        base_pipeline
        .measure("act", "std", of="signal")
        .measure("temp_mean", "mean", of="temp")
    )


# ---------- Registration -----------------------------------------------------

def test_recipes_are_registered_as_operators():
    names = default_registry.names("operator")
    assert "kuration/balanced-selector" in names
    assert "kuration/rare-event-coverage" in names
    assert "kuration/diverse-subset" in names


# ---------- Happy-path smoke tests -------------------------------------------
#
# One assertion per recipe. At seed=1 on the shared fixture, the SDK returns
# exactly the requested count of selected windows. If a refactor causes any of
# these to drift, the responsible change should update the assertion
# deliberately: silent drift is what this file exists to catch.

def test_balanced_selector_selects_expected_count(base_pipeline):
    op = BalancedSelector(feature="activity", of="signal", kind="std", bins=4, count=20)
    result = op.apply(base_pipeline).options(mode="strict", seed=1).collect()
    assert len(result.selected) == 20


def test_balanced_selector_operator_matches_dataclass(base_pipeline):
    op = BalancedSelector(feature="activity", of="signal", kind="std", bins=4, count=20)
    direct = op.apply(base_pipeline).options(mode="strict", seed=1).collect()
    via_pipeline = base_pipeline.balanced_selector(
        feature="activity", of="signal", kind="std", bins=4, count=20
    ).options(mode="strict", seed=1).collect()
    assert len(direct.selected) == len(via_pipeline.selected) == 20


def test_rare_event_coverage_selects_expected_count(base_pipeline):
    op = RareEventCoverage(of="fault", count=15)
    result = op.apply(base_pipeline).options(mode="strict", seed=1).collect()
    assert len(result.selected) == 15


def test_rare_event_coverage_operator_matches_dataclass(base_pipeline):
    op = RareEventCoverage(of="fault", count=15)
    direct = op.apply(base_pipeline).options(mode="strict", seed=1).collect()
    via_pipeline = base_pipeline.rare_event_coverage(of="fault", count=15).options(
        mode="strict", seed=1
    ).collect()
    assert len(direct.selected) == len(via_pipeline.selected) == 15


def test_diverse_subset_selects_expected_count(measured_pipeline):
    op = DiverseSubset(features=("act", "temp_mean"), count=15, n_clusters=4)
    result = op.apply(measured_pipeline).options(mode="strict", seed=1).collect()
    assert len(result.selected) == 15


def test_diverse_subset_with_dedup_selects_expected_count(measured_pipeline):
    op = DiverseSubset(
        features=("act", "temp_mean"),
        count=15,
        n_clusters=4,
        dedup_threshold=0.98,
    )
    result = op.apply(measured_pipeline).options(mode="strict", seed=1).collect()
    assert len(result.selected) == 15


def test_diverse_subset_operator_matches_dataclass(measured_pipeline):
    op = DiverseSubset(features=("act", "temp_mean"), count=15, n_clusters=4)
    direct = op.apply(measured_pipeline).options(mode="strict", seed=1).collect()
    via_pipeline = measured_pipeline.diverse_subset(
        features=("act", "temp_mean"), count=15, n_clusters=4
    ).options(mode="strict", seed=1).collect()
    assert len(direct.selected) == len(via_pipeline.selected) == 15


def test_describe_names_the_compiled_sdk_chain():
    """.describe() should surface the SDK primitives the recipe compiles to, so a user
    who outgrows the recipe can read it and see the raw SDK chain they could write by
    hand."""
    assert ".sample('equal'" in BalancedSelector(
        feature="activity", of="signal", bins=4, count=20
    ).describe()

    assert ".sample('weighted'" in RareEventCoverage(of="fault", count=10).describe()

    assert ".sample('cluster-balanced'" in DiverseSubset(
        features=("a", "b"), count=10, n_clusters=4
    ).describe()


# ---------- Validation --------------------------------------------------------
#
# Every ValueError branch in each recipe's __post_init__: raised at construction time
# (fail-fast, before the pipeline is touched), and naming the specific field that was
# invalid.

@pytest.mark.parametrize(
    "kwargs, field",
    [
        (dict(feature="", of="signal", bins=4, count=20), "feature"),
        (dict(feature="activity", of="", bins=4, count=20), "of"),
        (dict(feature="activity", of="signal", bins=1, count=20), "bins"),
        (dict(feature="activity", of="signal", bins=4, count=0), "count"),
    ],
)
def test_balanced_selector_rejects_bad_input(kwargs, field):
    with pytest.raises(ValueError, match=field):
        BalancedSelector(**kwargs)


@pytest.mark.parametrize(
    "kwargs, field",
    [
        (dict(of="", count=10), "of"),
        (dict(of="fault", feature="", count=10), "feature"),
        (dict(of="fault", count=0), "count"),
        (dict(of="fault", count=10, weight=-1), "weight"),
        (dict(of="fault", count=10, weight=0), "weight"),
    ],
)
def test_rare_event_coverage_rejects_bad_input(kwargs, field):
    with pytest.raises(ValueError, match=field):
        RareEventCoverage(**kwargs)


@pytest.mark.parametrize(
    "kwargs, field",
    [
        (dict(features=(), count=10, n_clusters=4), "features"),
        (dict(features=("",), count=10, n_clusters=4), "features"),
        (dict(features=("a", "b"), count=0, n_clusters=4), "count"),
        (dict(features=("a", "b"), count=10, n_clusters=1), "n_clusters"),
        (dict(features=("a", "b"), count=10, n_clusters=4, dedup_threshold=1.5),
         "dedup_threshold"),
        (dict(features=("a", "b"), count=10, n_clusters=4, dedup_threshold=-1.5),
         "dedup_threshold"),
    ],
)
def test_diverse_subset_rejects_bad_input(kwargs, field):
    with pytest.raises(ValueError, match=field):
        DiverseSubset(**kwargs)


# ---------- Config-parity invariant -------------------------------------------
#
# The three call styles (direct .apply, run() with an instance, run() with a config
# dict) must produce identical results. If they diverge, that's a bug.

def _collect_three_styles(base, op_instance, op_config):
    r1 = op_instance.apply(base).options(mode="strict", seed=1).collect()
    r2 = run(base, [op_instance]).options(mode="strict", seed=1).collect()
    r3 = run(base, [op_config]).options(mode="strict", seed=1).collect()
    return len(r1.selected), len(r2.selected), len(r3.selected)


def test_balanced_selector_three_styles_identical(base_pipeline):
    op = BalancedSelector(feature="activity", of="signal", bins=4, count=20)
    config = {
        "type": "BalancedSelector",
        "feature": "activity",
        "of": "signal",
        "bins": 4,
        "count": 20,
    }
    n1, n2, n3 = _collect_three_styles(base_pipeline, op, config)
    assert n1 == n2 == n3 == 20


def test_rare_event_coverage_three_styles_identical(base_pipeline):
    op = RareEventCoverage(of="fault", count=15)
    config = {"type": "RareEventCoverage", "of": "fault", "count": 15}
    n1, n2, n3 = _collect_three_styles(base_pipeline, op, config)
    assert n1 == n2 == n3 == 15


def test_diverse_subset_three_styles_identical(measured_pipeline):
    op = DiverseSubset(features=("act", "temp_mean"), count=15, n_clusters=4)
    config = {
        "type": "DiverseSubset",
        "features": ["act", "temp_mean"],
        "count": 15,
        "n_clusters": 4,
    }
    n1, n2, n3 = _collect_three_styles(measured_pipeline, op, config)
    assert n1 == n2 == n3 == 15


# ---------- Config-resolution error messages ----------------------------------
#
# Every config-resolution failure names the config-item index, the specific problem,
# and (when the input is close to something valid) a "did you mean X?" hint. These
# tests pin the message shape so future changes to the resolver don't silently
# regress the DX.

def test_missing_type_key():
    with pytest.raises(ValueError, match=r"config item 0: missing 'type' key"):
        plan([{"feature": "activity"}])


def test_type_must_be_string():
    with pytest.raises(ValueError, match=r"'type' must be a string"):
        plan([{"type": 123}])


def test_unknown_type_gets_fuzzy_hint():
    with pytest.raises(ValueError, match=r"Did you mean 'BalancedSelector'"):
        plan([{"type": "balancedselector"}])


def test_unknown_type_lists_available_when_no_close_match():
    with pytest.raises(ValueError) as exc:
        plan([{"type": "ZzzzCompletelyUnrelated"}])
    assert "Did you mean" not in str(exc.value)
    assert "Available:" in str(exc.value)
    assert "BalancedSelector" in str(exc.value)


def test_wrong_kwarg_gets_fuzzy_hint():
    with pytest.raises(ValueError, match=r"Did you mean 'bins'"):
        plan([{"type": "BalancedSelector", "feature": "x", "of": "y", "bin": 4}])


def test_item_index_tracked_across_multi_item_config():
    with pytest.raises(ValueError, match=r"config item 1: unknown 'type'"):
        plan([
            {"type": "BalancedSelector", "feature": "x", "of": "y"},
            {"type": "TypoInSecondItem"},
        ])


def test_plan_resolves_dicts_without_applying():
    config = [{"type": "BalancedSelector", "feature": "activity", "of": "signal"}]
    resolved = plan(config)
    assert len(resolved) == 1
    assert isinstance(resolved[0], BalancedSelector)
    assert resolved[0].feature == "activity"
    assert resolved[0].of == "signal"


def test_plan_preserves_instances_as_is():
    op = BalancedSelector(feature="a", of="b")
    resolved = plan([op])
    assert resolved[0] is op


def test_plan_supports_mixed_instances_and_dicts():
    config = [
        BalancedSelector(feature="a", of="b"),
        {"type": "BalancedSelector", "feature": "c", "of": "d"},
    ]
    resolved = plan(config)
    assert len(resolved) == 2
    assert resolved[0].feature == "a"
    assert resolved[1].feature == "c"


def test_describe_names_the_sdk_primitives():
    config = [{"type": "BalancedSelector", "feature": "activity", "of": "signal"}]
    output = describe(config)
    assert "selection plan:" in output
    assert ".measure(" in output
    assert ".bin(" in output
    assert ".sample(" in output


def test_describe_indexes_operations():
    config = [
        {"type": "BalancedSelector", "feature": "a", "of": "b"},
        {"type": "BalancedSelector", "feature": "c", "of": "d"},
    ]
    output = describe(config)
    assert "  1. " in output
    assert "  2. " in output


def test_describe_empty_plan():
    assert describe([]) == "selection plan: (empty)"


# ---------- Extension seam: register_operation_type() -------------------------

@dataclass(frozen=True)
class _DummyOp:
    """A dummy Operation for tests. Does nothing to the pipeline."""

    threshold: float = 0.5

    EXTRAS: ClassVar[tuple[str, ...]] = ()

    def apply(self, pipeline):
        return pipeline


@dataclass(frozen=True)
class _OtherDummyOp:
    """A second dummy Operation, used to test collision detection."""

    def apply(self, pipeline):
        return pipeline


class _NoApplyClass:
    """Deliberately missing .apply: should be rejected by register_operation_type()."""


class TestRegisterOperationType:
    def setup_method(self):
        _USER_REGISTRY.clear()

    def test_register_makes_class_usable_via_config_dict(self):
        register_operation_type("DummyOp", _DummyOp)
        resolved = plan([{"type": "DummyOp", "threshold": 0.7}])
        assert len(resolved) == 1
        assert isinstance(resolved[0], _DummyOp)
        assert resolved[0].threshold == 0.7

    def test_register_is_idempotent_for_same_class(self):
        register_operation_type("DummyOp", _DummyOp)
        register_operation_type("DummyOp", _DummyOp)
        assert _USER_REGISTRY["DummyOp"] is _DummyOp

    def test_register_raises_on_name_collision_with_different_class(self):
        register_operation_type("DummyOp", _DummyOp)
        with pytest.raises(ValueError, match=r"already registered with a different class"):
            register_operation_type("DummyOp", _OtherDummyOp)

    def test_register_rejects_classes_without_apply(self):
        with pytest.raises(ValueError, match=r"no 'apply' method"):
            register_operation_type("NoApply", _NoApplyClass)

    def test_register_rejects_empty_name(self):
        with pytest.raises(ValueError, match=r"non-empty string"):
            register_operation_type("", _DummyOp)

    def test_register_rejects_non_string_name(self):
        with pytest.raises(ValueError, match=r"non-empty string"):
            register_operation_type(123, _DummyOp)  # type: ignore[arg-type]

    def test_user_op_composes_with_builtin_in_same_plan(self):
        register_operation_type("DummyOp", _DummyOp)
        resolved = plan([
            BalancedSelector(feature="a", of="b"),
            {"type": "DummyOp"},
        ])
        assert len(resolved) == 2
        assert isinstance(resolved[0], BalancedSelector)
        assert isinstance(resolved[1], _DummyOp)

    def test_user_registered_class_appears_in_available_operations(self):
        register_operation_type("DummyOp", _DummyOp)
        assert "DummyOp" in _available_operations()
        assert "BalancedSelector" in _available_operations()
