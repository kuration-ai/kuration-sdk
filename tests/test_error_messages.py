"""Actionable error messages per design/03 spec.

Every config-resolution failure names the config-item index, the specific
problem, and (when the input is close to something valid) a `did you mean X?`
hint. These tests pin the message shape so future changes to the resolver
don't silently regress the DX.
"""
import pytest

from kuration_ai import BalancedSelector, describe, plan


class TestConfigResolutionErrors:
    def test_missing_type_key(self):
        with pytest.raises(ValueError, match=r"config item 0: missing 'type' key"):
            plan([{"feature": "activity"}])

    def test_type_must_be_string(self):
        with pytest.raises(ValueError, match=r"'type' must be a string"):
            plan([{"type": 123}])

    def test_unknown_type_gets_fuzzy_hint(self):
        with pytest.raises(ValueError, match=r"Did you mean 'BalancedSelector'"):
            plan([{"type": "balancedselector"}])

    def test_unknown_type_lists_available_when_no_close_match(self):
        with pytest.raises(ValueError) as exc:
            plan([{"type": "ZzzzCompletelyUnrelated"}])
        assert "Did you mean" not in str(exc.value)
        assert "Available:" in str(exc.value)
        assert "BalancedSelector" in str(exc.value)

    def test_wrong_kwarg_gets_fuzzy_hint(self):
        with pytest.raises(ValueError, match=r"Did you mean 'bins'"):
            plan([{"type": "BalancedSelector", "feature": "x", "of": "y", "bin": 4}])

    def test_item_index_tracked_across_multi_item_config(self):
        with pytest.raises(ValueError, match=r"config item 1: unknown 'type'"):
            plan([
                {"type": "BalancedSelector", "feature": "x", "of": "y"},
                {"type": "TypoInSecondItem"},
            ])


class TestPlanAndDescribe:
    def test_plan_resolves_dicts_without_applying(self):
        config = [{"type": "BalancedSelector", "feature": "activity", "of": "signal"}]
        resolved = plan(config)
        assert len(resolved) == 1
        assert isinstance(resolved[0], BalancedSelector)
        assert resolved[0].feature == "activity"
        assert resolved[0].of == "signal"

    def test_plan_preserves_instances_as_is(self):
        op = BalancedSelector(feature="a", of="b")
        resolved = plan([op])
        assert resolved[0] is op

    def test_plan_supports_mixed_instances_and_dicts(self):
        config = [
            BalancedSelector(feature="a", of="b"),
            {"type": "BalancedSelector", "feature": "c", "of": "d"},
        ]
        resolved = plan(config)
        assert len(resolved) == 2
        assert resolved[0].feature == "a"
        assert resolved[1].feature == "c"

    def test_describe_names_the_sdk_primitives(self):
        config = [{"type": "BalancedSelector", "feature": "activity", "of": "signal"}]
        output = describe(config)
        assert "Kuration AI plan:" in output
        assert ".measure(" in output
        assert ".bin(" in output
        assert ".sample(" in output

    def test_describe_indexes_operations(self):
        config = [
            {"type": "BalancedSelector", "feature": "a", "of": "b"},
            {"type": "BalancedSelector", "feature": "c", "of": "d"},
        ]
        output = describe(config)
        assert "  1. " in output
        assert "  2. " in output

    def test_describe_empty_plan(self):
        assert describe([]) == "Kuration AI plan: (empty)"
