"""Discovery and construction for stateful, direct-API analyzers.

Unlike a batch transform, an analyzer may accumulate evidence across segments before it
can return a result. The SDK therefore discovers and validates analyzer factories but does
not schedule them as implicit pipeline stages: callers own the analyzer instance and drive
its capability-specific update methods explicitly.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class Analyzer(Protocol):
    """Minimum result contract for a user-owned stateful analyzer."""

    def summary(self) -> Any: ...


def register_analyzer(name: str, factory: Callable | None = None, **metadata):
    """Register a discoverable analyzer factory, directly or as a decorator."""

    def decorate(component: Callable):
        from .registry import default_registry

        default_registry.register_analyzer(name, component, **metadata)
        return component

    if factory is None:
        return decorate
    return decorate(factory)


def get_analyzer(name: str, *, registry=None, **params) -> Analyzer:
    """Construct a named analyzer after validating its descriptor parameters.

    Plugin entry points are loaded on demand for the default registry. The returned object
    is intentionally user-owned; callers drive its documented update methods and call
    ``summary()`` when all relevant segments have been supplied.
    """
    from .plugins import load_plugins
    from .registry import default_registry, validate_parameters

    registry = registry or default_registry
    if name not in registry.names("analyzer") and registry is default_registry:
        load_plugins()
    if name not in registry.names("analyzer"):
        raise ValueError(f"unknown analyzer {name!r}; registered: {registry.names('analyzer')}")
    descriptor = registry.descriptor("analyzer", name)
    validate_parameters(descriptor, params)
    analyzer = registry.get("analyzer", name)(**params)
    if not isinstance(analyzer, Analyzer):
        raise ValueError(
            f"analyzer {name!r} factory must return an object with a callable summary()"
        )
    return analyzer


SPLIT_LEAKAGE_PARAMETERS = {
    "type": "object",
    "properties": {
        "check_groups": {
            "type": "boolean",
            "default": True,
            "description": "Also flag groups whose sources cross split boundaries.",
        },
    },
    "additionalProperties": False,
}


class SplitLeakageAnalyzer:
    """Audit source and optional group assignments across dataset splits.

    Repeated observations of the same assignment are allowed because a source may produce
    many segments or shards. A source observed in more than one split is leakage. When
    ``check_groups`` is enabled, a group spanning splits or one source claiming multiple
    groups is also a failure.
    """

    def __init__(self, *, check_groups: bool = True):
        if not isinstance(check_groups, bool):
            raise ValueError("check_groups must be true or false")
        self.check_groups = check_groups
        self._observations = 0
        self._assignments: set[tuple[str, str, str | None]] = set()
        self._source_splits: dict[str, set[str]] = defaultdict(set)
        self._source_groups: dict[str, set[str]] = defaultdict(set)
        self._group_splits: dict[str, set[str]] = defaultdict(set)
        self._group_sources: dict[str, set[str]] = defaultdict(set)

    def add_assignment(self, source_id: str, split: str, *, group: str | None = None) -> None:
        """Record one source/split observation and its optional leakage group."""
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string")
        if not isinstance(split, str) or not split:
            raise ValueError("split must be a non-empty string")
        if group is not None and (not isinstance(group, str) or not group):
            raise ValueError("group must be a non-empty string or None")

        self._observations += 1
        self._assignments.add((source_id, split, group))
        self._source_splits[source_id].add(split)
        if group is not None:
            self._source_groups[source_id].add(group)
            self._group_splits[group].add(split)
            self._group_sources[group].add(source_id)

    def add_assignments(
        self,
        source_splits: Mapping[str, str],
        *,
        source_groups: Mapping[str, str] | None = None,
    ) -> None:
        """Record an SDK-style ``{source_id: split}`` mapping in one call."""
        if not isinstance(source_splits, Mapping):
            raise ValueError("source_splits must be a mapping")
        if source_groups is not None and not isinstance(source_groups, Mapping):
            raise ValueError("source_groups must be a mapping or None")
        if source_groups is not None:
            unknown = set(source_groups) - set(source_splits)
            if unknown:
                raise ValueError(
                    f"source_groups contains source(s) absent from source_splits: {sorted(unknown)!r}"
                )
        for source_id, split in source_splits.items():
            group = None if source_groups is None else source_groups.get(source_id)
            self.add_assignment(source_id, split, group=group)

    def summary(self) -> dict:
        """Return a deterministic, JSON-safe leakage report."""
        source_leaks = [
            {"source_id": source_id, "splits": sorted(splits)}
            for source_id, splits in sorted(self._source_splits.items())
            if len(splits) > 1
        ]
        group_leaks = []
        source_group_conflicts = []
        if self.check_groups:
            group_leaks = [
                {
                    "group": group,
                    "splits": sorted(splits),
                    "sources": sorted(self._group_sources[group]),
                }
                for group, splits in sorted(self._group_splits.items())
                if len(splits) > 1
            ]
            source_group_conflicts = [
                {"source_id": source_id, "groups": sorted(groups)}
                for source_id, groups in sorted(self._source_groups.items())
                if len(groups) > 1
            ]
        splits = sorted({split for values in self._source_splits.values() for split in values})
        failures = source_leaks or group_leaks or source_group_conflicts
        return {
            "passed": not failures,
            "check_groups": self.check_groups,
            "observations": self._observations,
            "unique_assignments": len(self._assignments),
            "sources": len(self._source_splits),
            "groups": len(self._group_splits),
            "splits": splits,
            "source_leaks": source_leaks,
            "group_leaks": group_leaks,
            "source_group_conflicts": source_group_conflicts,
        }


@register_analyzer(
    "kuration/split-leakage",
    description="Audit source and group assignments for train/test split leakage",
    parameters=SPLIT_LEAKAGE_PARAMETERS,
    capabilities={"stateful": True, "cross_segment": True},
    documentation="https://aintropy-ai.github.io/kuration-sdk/customization/#stateful-analyzers",
    version="1",
)
def _split_leakage_analyzer(*, check_groups: bool = True) -> SplitLeakageAnalyzer:
    return SplitLeakageAnalyzer(check_groups=check_groups)
