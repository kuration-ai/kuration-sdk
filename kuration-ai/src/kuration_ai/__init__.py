"""kuration-ai: curated training-dataset selection for Python.

Three built-in curation operations plus an extension pattern for writing your
own. Each operation is a standalone, configurable unit: use one alone, chain
several, or drive them from a config dict.
"""
# Re-exported so users import the pipeline builder from a single place.
from kuration import input

from kuration_ai.operation import (
    Operation,
    describe,
    plan,
    register,
    run,
)
from kuration_ai.operations import BalancedSelector, DiverseSubset, RareEventCoverage

__all__ = [
    "Operation",
    "input",
    "run",
    "plan",
    "describe",
    "register",
    "BalancedSelector",
    "DiverseSubset",
    "RareEventCoverage",
]
__version__ = "0.0.1"
