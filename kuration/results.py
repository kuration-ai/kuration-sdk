"""Typed public results with temporary mapping compatibility."""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from typing import Any


class TypedResult(dict):
    """Dict-compatible typed result used during the compatibility transition."""
    def to_dict(self) -> dict:
        return dict(self)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


class DatasetWriteResult(TypedResult):
    pass


class ExperimentResult(TypedResult):
    pass


class DatasetReport(TypedResult):
    pass


class ValidationResult(TypedResult):
    pass


@dataclass(frozen=True)
class ExecutionPlan(Mapping[str, Any]):
    source: dict
    schema: dict
    window: dict
    strategy: dict
    output: Any
    options: dict
    stages: tuple[dict, ...] = ()
    warnings: tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def render(self, *, detailed: bool = False) -> str:
        lines = [f"source: {self.source['kind']} ({self.source['count']} source(s))",
                 f"schema: {self.schema.get('name', 'unnamed')}",
                 f"window: {self.window['window']} {self.window['unit']} "
                 f"(stride {self.window['stride']})",
                 f"method: {self.strategy.get('selection', {}).get('method', 'unresolved')}"]
        if self.output:
            lines.append(f"output: {self.output.get('container', 'unresolved')}")
        lines.extend(f"warning: {w['message']}" for w in self.warnings)
        if detailed:
            lines.extend(f"stage: {s['name']} (streaming={s['streaming']})" for s in self.stages)
        return "\n".join(lines)
