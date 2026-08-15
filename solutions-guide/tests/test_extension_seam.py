"""Extension seam tests: register() contract + user Operation usage."""
from dataclasses import dataclass
from typing import ClassVar

import pytest

from kuration_ai import BalancedSelector, plan, register
from kuration_ai.operation import _USER_REGISTRY


@dataclass(frozen=True)
class DummyOp:
    """A dummy Operation for tests. Does nothing to the pipeline."""

    threshold: float = 0.5

    EXTRAS: ClassVar[tuple[str, ...]] = ()

    def apply(self, pipeline):
        return pipeline


@dataclass(frozen=True)
class OtherDummyOp:
    """A second dummy Operation, used to test collision detection."""

    def apply(self, pipeline):
        return pipeline


class NoApplyClass:
    """Deliberately missing .apply — should be rejected by register()."""


class TestRegister:
    def setup_method(self):
        _USER_REGISTRY.clear()

    def test_register_makes_class_usable_via_config_dict(self):
        register("DummyOp", DummyOp)
        resolved = plan([{"type": "DummyOp", "threshold": 0.7}])
        assert len(resolved) == 1
        assert isinstance(resolved[0], DummyOp)
        assert resolved[0].threshold == 0.7

    def test_register_is_idempotent_for_same_class(self):
        register("DummyOp", DummyOp)
        register("DummyOp", DummyOp)
        assert _USER_REGISTRY["DummyOp"] is DummyOp

    def test_register_raises_on_name_collision_with_different_class(self):
        register("DummyOp", DummyOp)
        with pytest.raises(ValueError, match=r"already registered with a different class"):
            register("DummyOp", OtherDummyOp)

    def test_register_rejects_classes_without_apply(self):
        with pytest.raises(ValueError, match=r"no 'apply' method"):
            register("NoApply", NoApplyClass)

    def test_register_rejects_empty_name(self):
        with pytest.raises(ValueError, match=r"non-empty string"):
            register("", DummyOp)

    def test_register_rejects_non_string_name(self):
        with pytest.raises(ValueError, match=r"non-empty string"):
            register(123, DummyOp)  # type: ignore[arg-type]

    def test_user_op_composes_with_builtin_in_same_plan(self):
        register("DummyOp", DummyOp)
        resolved = plan([
            BalancedSelector(feature="a", of="b"),
            {"type": "DummyOp"},
        ])
        assert len(resolved) == 2
        assert isinstance(resolved[0], BalancedSelector)
        assert isinstance(resolved[1], DummyOp)

    def test_user_registered_class_appears_in_available_operations(self):
        from kuration_ai.operation import _available_operations
        register("DummyOp", DummyOp)
        assert "DummyOp" in _available_operations()
        assert "BalancedSelector" in _available_operations()
