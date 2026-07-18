"""Explicit, reusable quality gates over preview, review, and artifact reports."""
from __future__ import annotations

import math
from typing import Any

from .results import QualityGateResult


_EXPECTATIONS = {
    "max_missing", "min_sources", "min_tag_coverage", "max_group_fraction",
    "min_budget_utilization", "max_budget_underfill_seconds", "max_skipped_sources",
    "require_modalities", "require_verified", "min_shards",
}


class QualityGateError(ValueError):
    """Raised when one or more configured quality expectations fail."""

    def __init__(self, result: QualityGateResult):
        self.result = result
        failed = [check["name"] for check in result["checks"] if not check["passed"]]
        super().__init__("quality gates failed: " + ", ".join(failed))


def validate_expectations(expectations: dict) -> dict:
    """Validate and copy the closed quality-policy grammar."""
    unknown = sorted(set(expectations) - _EXPECTATIONS)
    if unknown:
        raise ValueError(f"unknown quality expectation(s) {unknown}; expected "
                         f"{sorted(_EXPECTATIONS)}")
    policy = dict(expectations)
    for name in ("max_missing", "min_tag_coverage"):
        if name in policy and (not isinstance(policy[name], dict)
                               or not policy[name]):
            raise ValueError(f"{name} must be a non-empty name-to-threshold mapping")
    if "max_missing" in policy and any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or not 0 <= value <= 1
            for value in policy["max_missing"].values()):
        raise ValueError("max_missing thresholds must be finite numbers in [0, 1]")
    if "min_tag_coverage" in policy and any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in policy["min_tag_coverage"].values()):
        raise ValueError("min_tag_coverage thresholds must be non-negative integers")
    for name in ("min_sources", "max_skipped_sources", "min_shards"):
        if name in policy and (not isinstance(policy[name], int)
                               or isinstance(policy[name], bool) or policy[name] < 0):
            raise ValueError(f"{name} must be a non-negative integer")
    for name in ("max_group_fraction", "min_budget_utilization"):
        if name == "min_budget_utilization" and name in policy \
                and isinstance(policy[name], dict):
            values = policy[name].values()
        elif name in policy:
            values = [policy[name]]
        else:
            continue
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError(f"{name} threshold(s) must be finite numbers in [0, 1]")
    if "max_budget_underfill_seconds" in policy:
        raw = policy["max_budget_underfill_seconds"]
        values = raw.values() if isinstance(raw, dict) else [raw]
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("max_budget_underfill_seconds threshold(s) must be non-negative")
    if "require_modalities" in policy and (
            not isinstance(policy["require_modalities"], (list, tuple))
            or not all(isinstance(name, str) and name for name in policy["require_modalities"])):
        raise ValueError("require_modalities must be a list of non-empty names")
    if "require_verified" in policy and not isinstance(policy["require_verified"], bool):
        raise ValueError("require_verified must be true or false")
    return policy


def _threshold(raw, name: str, default: float) -> float:
    return float(raw.get(name, default)) if isinstance(raw, dict) else float(raw)


def _budget_splits(data: dict) -> dict:
    budget = data.get("budget") or {}
    if "requested_seconds" not in budget:
        return {name: values for name, values in budget.items()
                if isinstance(values, dict) and "utilization" in values}
    emitted = budget.get("emitted_seconds") or budget.get("admitted_seconds") or {}
    out = {}
    for split in ("train", "test"):
        requested = float((budget.get("requested_seconds") or {}).get(split, 0.0))
        emitted_seconds = float(emitted.get(split, 0.0))
        out[split] = {
            "utilization": emitted_seconds / requested if requested > 0 else 1.0,
            "emitted_underfill_seconds": max(0.0, requested - emitted_seconds),
        }
    return out


def evaluate_quality(report, expectations: dict | None = None,
                     **kwargs) -> QualityGateResult:
    """Evaluate a closed quality policy without raising, returning every check result."""
    policy = dict(expectations or {})
    policy.update(kwargs)
    policy = validate_expectations(policy)
    data = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    checks = []
    budget_splits = _budget_splits(data)

    def add(name: str, passed: bool, actual: Any, expected: Any, message: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "actual": actual,
                       "expected": expected, "message": message})

    if "max_missing" in policy:
        available = {**(data.get("fields") or {}), **(data.get("globals") or {})}
        for field, maximum in policy["max_missing"].items():
            actual = (available.get(field) or {}).get("missing_fraction")
            passed = actual is not None and actual <= maximum
            message = (f"{field} missing fraction is {actual!r}; maximum is {maximum}"
                       if actual is not None else
                       f"missing-fraction evidence for {field!r} is unavailable")
            add(f"max_missing.{field}", passed, actual, maximum, message)

    if "min_sources" in policy:
        if isinstance(data.get("sources"), dict):
            actual = data["sources"].get("total")
        elif isinstance(data.get("sources"), int):
            actual = data["sources"]
        elif data.get("coverage"):
            actual = data["coverage"]["selected"].get("sources")
        elif data.get("source_splits"):
            actual = sum(data["source_splits"].values())
        else:
            actual = None
        minimum = policy["min_sources"]
        add("min_sources", actual is not None and actual >= minimum, actual, minimum,
            f"source count is {actual!r}; minimum is {minimum}")

    if "min_tag_coverage" in policy:
        selected_coverage = ((data.get("coverage") or {}).get("selected") or {})
        tags = selected_coverage.get("tags") if selected_coverage else None
        for tag, minimum in policy["min_tag_coverage"].items():
            actual = int(tags.get(str(tag), 0)) if tags is not None else None
            message = (f"tag {tag!r} appears {actual} time(s); minimum is {minimum}"
                       if actual is not None else "selected-tag coverage evidence is unavailable")
            add(f"min_tag_coverage.{tag}", actual is not None and actual >= minimum,
                actual, minimum, message)

    if "max_group_fraction" in policy:
        selected_coverage = ((data.get("coverage") or {}).get("selected") or {})
        groups = selected_coverage.get("groups") if selected_coverage else None
        if isinstance(groups, dict) and groups:
            total = sum(groups.values())
            actual = max(groups.values()) / total if total else None
        else:
            actual = None
        maximum = float(policy["max_group_fraction"])
        message = (f"largest group fraction is {actual!r}; maximum is {maximum}"
                   if actual is not None else "selected-group coverage evidence is unavailable")
        add("max_group_fraction", actual is not None and actual <= maximum, actual, maximum,
            message)

    if "min_budget_utilization" in policy:
        requested = policy["min_budget_utilization"]
        splits = (tuple(requested) if isinstance(requested, dict)
                  else tuple(budget_splits) or ("train", "test"))
        for split in splits:
            values = budget_splits.get(split) or {}
            minimum = _threshold(policy["min_budget_utilization"], split, 0.0)
            actual = values.get("utilization")
            message = (f"{split} utilization is {actual:.1%}; minimum is {minimum:.1%}"
                       if actual is not None else
                       f"budget-utilization evidence for {split!r} is unavailable")
            add(f"min_budget_utilization.{split}",
                actual is not None and actual >= minimum, actual, minimum, message)

    if "max_budget_underfill_seconds" in policy:
        requested = policy["max_budget_underfill_seconds"]
        splits = (tuple(requested) if isinstance(requested, dict)
                  else tuple(budget_splits) or ("train", "test"))
        for split in splits:
            values = budget_splits.get(split) or {}
            actual = values.get("emitted_underfill_seconds", values.get("underfill_seconds"))
            maximum = _threshold(policy["max_budget_underfill_seconds"], split, float("inf"))
            message = (f"{split} underfill is {actual:.3f}s; maximum is {maximum:.3f}s"
                       if actual is not None else
                       f"budget-underfill evidence for {split!r} is unavailable")
            add(f"max_budget_underfill_seconds.{split}",
                actual is not None and actual <= maximum, actual, maximum, message)

    if "max_skipped_sources" in policy:
        if "sources_skipped" in data:
            actual = data.get("sources_skipped")
        elif "errors" in data:
            actual = len(data.get("errors") or ())
        else:
            actual = None
        maximum = policy["max_skipped_sources"]
        add("max_skipped_sources", actual is not None and actual <= maximum, actual, maximum,
            (f"skipped sources: {actual}; maximum is {maximum}" if actual is not None
             else "skipped-source evidence is unavailable"))

    if "require_modalities" in policy:
        has_evidence = "modalities" in data or "fields" in data
        modality_available = set(data.get("modalities") or data.get("fields") or ())
        modality_required = set(policy["require_modalities"])
        missing = sorted(modality_required - modality_available)
        add("require_modalities", has_evidence and not missing,
            sorted(modality_available) if has_evidence else None, sorted(modality_required),
            (f"missing required modalities: {missing}" if has_evidence
             else "modality evidence is unavailable"))

    if "require_verified" in policy:
        verification = data.get("verification") or {}
        actual = bool(verification.get("requested") and verification.get("valid")) \
            if "verification" in data else None
        required = policy["require_verified"]
        passed = not required or actual is True
        add("require_verified", passed, actual, required,
            ("artifact verification is not required" if not required else
             "artifact verification completed successfully" if actual else
             "artifact has not passed full shard verification" if actual is False else
             "artifact-verification evidence is unavailable"))

    if "min_shards" in policy:
        actual = data.get("shards")
        minimum = policy["min_shards"]
        add("min_shards", actual is not None and actual >= minimum, actual, minimum,
            (f"shard count is {actual!r}; minimum is {minimum}" if actual is not None
             else "shard-count evidence is unavailable"))

    if policy and not checks:
        raise ValueError("quality policy does not apply to this report type")
    return QualityGateResult(passed=all(check["passed"] for check in checks),
                             checks=checks, expectations=policy,
                             report_type=type(report).__name__)


def check_quality(report, expectations: dict | None = None, **kwargs) -> QualityGateResult:
    """Evaluate expectations and raise ``QualityGateError`` when any check fails."""
    result = evaluate_quality(report, expectations, **kwargs)
    if not result.passed:
        raise QualityGateError(result)
    return result
