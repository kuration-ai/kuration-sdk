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


class DatasetInspection(TypedResult):
    """Metadata and optional integrity results for one written dataset artifact."""


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
    physical_plan: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def render(self, *, detailed: bool = False) -> str:
        selection = self.strategy.get("selection", {})
        request = self.options.get("selection_budget") or {}
        budget = (f"count={request['value']}" if request.get("kind") == "count"
                  else f"hours={request['value']}" if request.get("kind") == "hours"
                  else "budget unresolved")
        physical = self.physical_plan or {}
        stages = physical.get("stages", self.stages)
        materializes = [stage["name"] for stage in stages
                        if stage.get("materializes_total_pool")]
        extras = self.options.get("optional_extras") or []
        estimate = self.options.get("source_estimate")
        source_kind = {"_RowsAdapter": "in-memory rows", "_TickAdapter": "in-memory ticks",
                       "_JsonRowsAdapter": "JSON rows"}.get(
                           self.source["kind"], self.source["kind"])
        lines = [f"source: {source_kind} ({self.source['count']} source(s))",
                 f"schema: {self.schema.get('name', 'unnamed')}",
                 f"window: {self.window['window']} {self.window['unit']} "
                 f"(stride {self.window['stride']})",
                 f"selection: {selection.get('method', 'unresolved')} ({budget})",
                 f"execution: {self.options.get('executor', 'local')}; "
                 f"errors={self.options.get('on_error', 'skip')}; "
                 f"total-pool materialization="
                 f"{', '.join(materializes) if materializes else 'none'}",
                 f"optional extras: {', '.join(extras) if extras else 'none (NumPy core)'}"]
        if self.output:
            modalities = sorted((self.output.get("modalities") or {}).keys())
            lines.append(f"output: {self.output.get('container', 'unresolved')}; "
                         f"shard_length={self.output.get('shard_length', 'unresolved')}; "
                         f"modalities={modalities}")
        if estimate:
            qualifier = "at least " if estimate.get("lower_bound") else "about "
            lines.append(f"estimate: {qualifier}{estimate['candidate_windows_estimate']} "
                         f"candidate windows ({estimate['basis']})")
            if "output_shards_estimate" in estimate:
                lines.append(f"estimate: about {estimate['output_shards_estimate']} output "
                             f"shards; {estimate['output_estimate_assumptions']}")
        for warning in self.warnings:
            lines.append(f"warning [{warning.get('code', 'general')}]: {warning['message']}")
            if warning.get("hint"):
                lines.append(f"  fix: {warning['hint']}")
        if detailed:
            for stage in stages:
                line = (f"stage: {stage['name']} (mode={stage.get('mode', 'unresolved')}, "
                        f"passes={stage.get('passes', 1)}, streaming={stage['streaming']})")
                if stage.get("barrier_reason"):
                    line += f" — barrier: {stage['barrier_reason']}"
                lines.append(line)
        return "\n".join(lines)
