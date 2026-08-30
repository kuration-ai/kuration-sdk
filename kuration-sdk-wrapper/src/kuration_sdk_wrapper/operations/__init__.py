"""Pre-built operations.

Each operation is a small, self-contained composition of Kuration SDK primitives
(`.measure`, `.filter`, `.bin`, `.sample`, ...) exposed as a single configurable unit.

Add your own by writing a class with `.apply(pipeline) -> pipeline` and importing it
here.
"""
from kuration_sdk_wrapper.operations.balance import BalancedSelector
from kuration_sdk_wrapper.operations.diverse_subset import DiverseSubset
from kuration_sdk_wrapper.operations.rare_events import RareEventCoverage

__all__ = ["BalancedSelector", "DiverseSubset", "RareEventCoverage"]
