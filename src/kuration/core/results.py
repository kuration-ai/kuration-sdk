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
    def stats(self, **kwargs):
        """Summarize the written artifact identified by this result."""
        from .stats import stats

        return stats(self, **kwargs)


class ExperimentResult(TypedResult):
    pass


class DatasetReport(TypedResult):
    pass


class StatsReport(TypedResult):
    """Unified statistics over an input pipeline, selection, or written artifact."""

    _stats_context: dict[str, Any]

    def render(self) -> str:
        lines = [f"stats: {self['stage']}"]
        sections = self.get("sections") or {}
        input_summary = sections.get("input") or {}
        if input_summary:
            sources = input_summary.get("sources") or {}
            records = input_summary.get("records") or {}
            lines.append(
                f"input: {sources.get('sampled', 0)}/{sources.get('total', 0)} sources; "
                f"{records.get('ticks', 0)} ticks; "
                f"{records.get('entity_records', 0)} entity records"
            )
            for name, field in (input_summary.get("fields") or {}).items():
                detail = f"missing={field.get('missing_fraction', 0.0):.1%}"
                if field.get("kind") == "categorical":
                    detail += f"; unique={field.get('unique', 0)}"
                elif "min" in field:
                    detail += f"; range=[{field['min']:.3g}, {field['max']:.3g}]"
                lines.append(f"field {name} ({field.get('kind', 'unknown')}): {detail}")
        candidates = sections.get("candidates") or {}
        if candidates:
            lines.append(
                f"candidates: {candidates.get('n_segments', 0)} segments; "
                f"{candidates.get('n_sources', 0)} sources; "
                f"{len(candidates.get('metrics') or {})} metrics"
            )
        selection = sections.get("selection") or {}
        if selection:
            lines.append(
                f"selection: {selection.get('selected', 0)}/"
                f"{selection.get('candidates', 0)} "
                f"({selection.get('selection_rate', 0.0):.1%})"
            )
        artifact = sections.get("artifact") or {}
        if artifact:
            lines.append(
                f"artifact: {artifact.get('shards', 0)} shards; "
                f"{artifact.get('sources', 0)} sources; "
                f"container={artifact.get('container', 'unknown')}"
            )
        contents = sections.get("contents") or {}
        if contents:
            lines.append(
                f"contents: {contents.get('shards_scanned', 0)} shards; "
                f"{len(contents.get('modalities') or {})} modalities"
            )
        hours = sections.get("hours_selection") or {}
        if hours:
            lines.append(
                f"hours selection: {hours.get('admitted_items', 0)}/"
                f"{hours.get('candidate_items', 0)} items admitted"
            )
        standard_sections = {
            "input", "candidates", "selection", "budget_plan", "artifact", "contents",
            "hours_selection",
        }
        for section_name, section in sections.items():
            if section_name in standard_sections or not isinstance(section, dict):
                continue
            for item in section.get("summary") or ():
                if isinstance(item, dict):
                    detail = (
                        f" ({item['detail']})" if item.get("detail") is not None else ""
                    )
                    lines.append(
                        f"{section_name} {item.get('label', 'summary')}: "
                        f"{item.get('value', '—')}{detail}"
                    )
        for warning in self.get("warnings") or ():
            lines.append(
                f"warning [{warning.get('code', 'general')}]: {warning['message']}"
            )
        return "\n".join(lines)

    def plot(self, metric: str | None = None, *, kind: str | None = None, ax=None):
        """Return a Matplotlib figure for this report without calling ``show()``."""
        from .stats import plot_stats

        return plot_stats(self, metric=metric, kind=kind, ax=ax)

    def available(self) -> dict:
        """Describe plot and evidence choices available for this report."""
        from .stats import stats_capabilities

        return stats_capabilities(self)

    def examples(
        self,
        *,
        mode: str = "auto",
        metric: str | None = None,
        compare: str | None = None,
        n: int = 6,
        frames: int = 3,
    ):
        """Select bounded evidence and retain live media access when available."""
        from .stats import stats_examples

        return stats_examples(
            self,
            mode=mode,
            metric=metric,
            compare=compare,
            n=n,
            frames=frames,
        )

    def save(self, path, *, title: str = "kuration stats") -> str:
        """Write a dependency-free HTML report and return its path."""
        from .stats import write_stats_html

        return write_stats_html(path, self, title=title)

    def _repr_html_(self) -> str:
        """Side-effect-free rich representation for Jupyter and IPython frontends."""
        from .stats import render_stats_html

        return render_stats_html(self, full_document=False)


class StatsExamples(TypedResult):
    """Bounded examples selected from a stats report for human review."""

    _stats_context: dict[str, Any]

    def render(self) -> str:
        lines = [
            f"examples: {sum(len(items) for items in self['groups'].values())} "
            f"across {len(self['groups'])} group(s)"
        ]
        for group, items in self["groups"].items():
            lines.append(f"{group}: {len(items)}")
            for item in items:
                metric = item.get("rank_by")
                value = item.get("rank_value")
                rank = f"; {metric}={value:.4g}" if metric and value is not None else ""
                lines.append(
                    f"  {item.get('source')} [{item.get('start_idx')},"
                    f"{item.get('end_idx')}){rank}"
                )
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        from .stats import render_stats_examples_html

        return render_stats_examples_html(self)


class DatasetInspection(TypedResult):
    """Metadata and optional integrity results for one written dataset artifact."""


class DatasetPreview(TypedResult):
    """Bounded source sample describing inferred fields, quality, and examples."""

    def render(self) -> str:
        lines = [f"schema: {self['schema'].get('name', 'unnamed')}",
                 f"sources: {self['sources']['sampled']}/{self['sources']['total']} sampled",
                 f"records: {self['records']['ticks']} ticks; "
                 f"{self['records']['entity_records']} entity records"]
        for name, field in self["fields"].items():
            detail = f"missing={field['missing_fraction']:.1%}"
            if field["kind"] == "categorical":
                detail += f"; unique={field['unique']}"
            elif "min" in field:
                detail += f"; range=[{field['min']:.3g}, {field['max']:.3g}]"
            lines.append(f"field {name} ({field['kind']}): {detail}")
        for warning in self["warnings"]:
            lines.append(f"warning [{warning['code']}]: {warning['message']}")
        return "\n".join(lines)


class SelectionReview(TypedResult):
    """Candidate-versus-selected distribution, coverage, rationale, and examples."""

    def render(self) -> str:
        lines = [f"selection: {self['selected']}/{self['candidates']} "
                 f"({self['selection_rate']:.1%})",
                 f"method: {self['rationale']['method']}"]
        selected = self["coverage"]["selected"]
        lines.append(f"coverage: {selected['sources']} sources; "
                     f"{(selected.get('bins') or {}).get('occupied', 0)} occupied bins")
        def fmt(value, spec: str) -> str:
            # An all-non-finite metric reports None statistics (JSON-safe) — render as a dash.
            return "—" if value is None else format(value, spec)

        for name, metric in self["metrics"].items():
            lines.append(f"metric {name}: mean "
                         f"{fmt(metric['candidate'].get('mean', 0), '.3g')} -> "
                         f"{fmt(metric['selected'].get('mean', 0), '.3g')} "
                         f"(delta {fmt(metric['mean_delta'], '+.3g')})")
        for warning in self["warnings"]:
            lines.append(f"warning [{warning['code']}]: {warning['message']}")
        return "\n".join(lines)


class HoursSelectionReview(TypedResult):
    """Hours-budget admission, rejection, utilization, split, and priority review."""

    def render(self) -> str:
        lines = [f"hours selection: {self['admitted_items']}/{self['candidate_items']} "
                 "items admitted"]
        for split, budget in self["budget"].items():
            lines.append(
                f"split {split}: requested={budget['requested_seconds']:.3f}s; "
                f"admitted={budget['admitted_seconds']:.3f}s; "
                f"emitted={budget['emitted_seconds']:.3f}s; "
                f"utilization={budget['utilization']:.1%}")
        decisions = ", ".join(f"{name}={count}" for name, count
                              in self["decision_counts"].items())
        lines.append(f"decisions: {decisions or 'none'}")
        for warning in self["warnings"]:
            lines.append(f"warning [{warning['code']}]: {warning['message']}")
        return "\n".join(lines)


class QualityGateResult(TypedResult):
    """Pass/fail results for explicit dataset quality expectations."""

    def render(self) -> str:
        lines = [f"quality gates: {'PASS' if self['passed'] else 'FAIL'}"]
        for check in self["checks"]:
            marker = "PASS" if check["passed"] else "FAIL"
            lines.append(f"{marker} {check['name']}: {check['message']}")
        return "\n".join(lines)


class DatasetComparison(TypedResult):
    """Structured differences between two curated dataset artifacts."""

    def render(self) -> str:
        lines = [f"datasets: {self['left']['dataset']} -> {self['right']['dataset']}",
                 f"identical: {self['identical']}",
                 f"shards: {self['left']['shards']} -> {self['right']['shards']}"]
        for change in self["changes"][:20]:
            lines.append(f"change {change['path']}: {change['left']!r} -> "
                         f"{change['right']!r}")
        if len(self["changes"]) > 20:
            lines.append(f"... {len(self['changes']) - 20} more change(s)")
        return "\n".join(lines)


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
        if self.options.get("quality_expectations"):
            lines.append("quality gates: " + ", ".join(
                sorted(self.options["quality_expectations"])))
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
