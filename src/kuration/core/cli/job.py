"""One authoritative parser/resolver for every declarative CLI job."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from ..adapters.base import validate_adapter_availability
from ..output import OutputSpec
from ..strategy import Strategy, load_strategy
from ._common import (build_adapter, build_schema, build_segmenter, load_job_plugins,
                      resolve_sources)


_COMMON = {
    "job_schema_version", "schema", "adapter", "sources", "segmenter", "strategy",
    "plugins", "include_plugin_entry_points", "seed", "workers", "on_error", "streaming",
    "executor", "mode",
}
_CURATE = {
    "output", "output_preset", "encoder", "train_hours", "test_hours", "out", "resume", "force",
    "span_steps", "span_seconds", "clip_aggregate", "memory_budget", "batch_size",
    "checkpoint_dir", "spill_dir", "approximation", "max_in_flight",
}
_EXPERIMENT = {"arms", "baseline", "target_size"}


@dataclass(frozen=True)
class ResolvedJob:
    raw: dict
    kind: str
    schema: Any
    adapter: Any
    segmenter: Any
    sources: tuple | None
    registry: Any = None
    strategy: Strategy | None = None
    arms: tuple = ()
    output: Any = None
    encoder: Any = None
    on_error: str = "skip"


def job_kind(job: dict) -> str:
    if "arms" in job:
        return "experiment"
    if set(job) & (_CURATE | {"output_preset"}):
        return "curate"
    return "report"


def _closed_keys(job: dict, kind: str) -> None:
    allowed = _COMMON | (_CURATE if kind == "curate" else _EXPERIMENT if kind == "experiment"
                         else set())
    unknown = sorted(set(job) - allowed)
    if unknown:
        raise ValueError(f"unknown {kind} job key(s) {unknown}; expected {sorted(allowed)}")


def _positive_int(job: dict, name: str, default: int) -> int:
    value = job.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _common_values(job: dict) -> str:
    _positive_int(job, "workers", 1)
    seed = job.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if job.get("on_error", "skip") not in {"skip", "raise"}:
        raise ValueError("on_error must be 'skip' or 'raise'")
    if job.get("executor", "local") not in {"local", "ray"}:
        raise ValueError("executor must be 'local' or 'ray'")
    if job.get("mode") not in {None, "explore", "strict"}:
        raise ValueError("mode must be 'explore' or 'strict'")
    for flag in ("resume", "force", "streaming"):
        if flag in job and not isinstance(job[flag], bool):
            raise ValueError(f"{flag} must be true or false")
    if job.get("resume", False) and job.get("force", False):
        raise ValueError("force=true and resume=true are mutually exclusive")
    plugins = job.get("plugins", [])
    if (not isinstance(plugins, list)
            or any(not isinstance(module, str) or not module for module in plugins)):
        raise ValueError("plugins must be a list of non-empty module names")
    if ("include_plugin_entry_points" in job
            and not isinstance(job["include_plugin_entry_points"], bool)):
        raise ValueError("include_plugin_entry_points must be true or false")
    mode = job.get("mode")
    return job.get("on_error", "raise" if mode == "strict" else "skip")


def _source_list(job: dict, adapter, *, enumerate_sources: bool) -> tuple | None:
    if "sources" in job:
        sources = job["sources"]
        if (not isinstance(sources, list) or not sources
                or any(not isinstance(source, str) or not source for source in sources)):
            raise ValueError("sources must be a non-empty list of non-empty strings")
        return tuple(sources)
    return tuple(resolve_sources(job, adapter)) if enumerate_sources else None


def _output(job: dict, schema, registry):
    supplied = [name for name in ("output", "output_preset") if name in job]
    if len(supplied) != 1:
        raise ValueError("curate job needs exactly one of 'output' or 'output_preset'")
    if "output_preset" in job:
        from ..output import output_preset
        raw = job["output_preset"]
        if isinstance(raw, str):
            raw = {"name": raw}
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
            raise ValueError("output_preset must be a name or an object containing 'name'")
        raw = dict(raw)
        return output_preset(raw.pop("name"), schema=schema, registry=registry, **raw)
    return OutputSpec.from_dict(job["output"], registry=registry)


def _encoder(job: dict, registry):
    raw = job.get("encoder")
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) - {"name", "params"} or "name" not in raw:
        raise ValueError("encoder must contain 'name' and optional 'params' only")
    if not isinstance(raw["name"], str) or not isinstance(raw.get("params", {}), dict):
        raise ValueError("encoder name must be a string and params must be an object")
    from ..output import get_encoder
    return get_encoder(raw["name"], registry=registry, **dict(raw.get("params") or {}))


def resolve_job(job: dict, *, enumerate_sources: bool = True) -> ResolvedJob:
    """Validate and resolve a job exactly once for validation and execution CLIs."""
    if not isinstance(job, dict):
        raise ValueError("job must be a JSON object")
    from ..versioning import check_version
    check_version(job.get("job_schema_version"), kind="job")
    kind = job_kind(job)
    _closed_keys(job, kind)
    on_error = _common_values(job)
    for required in ("adapter", "segmenter"):
        if required not in job:
            raise ValueError(f"job needs {required!r}")
    registry = load_job_plugins(job)
    schema = build_schema(job)
    adapter = build_adapter(job, schema, registry=registry)
    schema = schema or getattr(adapter, "schema", None)
    if schema is None:
        raise ValueError("no 'schema' in job and the adapter exposes none")
    segmenter = build_segmenter(job, schema)
    sources = _source_list(job, adapter, enumerate_sources=enumerate_sources)

    if kind == "experiment":
        if "strategy" in job:
            raise ValueError("experiment strategies belong in 'arms', not top-level 'strategy'")
        raw_arms = job.get("arms")
        if not isinstance(raw_arms, list) or not raw_arms:
            raise ValueError("experiment job needs a non-empty 'arms' list")
        arms = []
        for index, raw in enumerate(raw_arms):
            if not isinstance(raw, dict) or set(raw) != {"name", "strategy"}:
                raise ValueError(f"arm {index} needs exactly 'name' and 'strategy'")
            if not isinstance(raw["name"], str) or not raw["name"]:
                raise ValueError(f"arm {index} name must be a non-empty string")
            strategy = load_strategy(raw["strategy"], schema, registry=registry)
            validate_adapter_availability(adapter, schema, define=strategy.define,
                                          tags_spec=strategy.tags)
            arms.append((raw["name"], strategy))
        names = [name for name, _ in arms]
        if len(names) != len(set(names)):
            raise ValueError("experiment arm names must be unique")
        if "target_size" not in job:
            raise ValueError("experiment job needs 'target_size'")
        _positive_int(job, "target_size", 1)
        baseline = job.get("baseline")
        if baseline is not None and baseline not in names:
            raise ValueError(f"baseline {baseline!r} is not an arm name")
        return ResolvedJob(job, kind, schema, adapter, segmenter, sources, registry=registry,
                           arms=tuple(arms), on_error=on_error)

    if "strategy" not in job:
        raise ValueError("job needs a 'strategy'")
    strategy = load_strategy(job["strategy"], schema, registry=registry)
    if kind == "report":
        validate_adapter_availability(adapter, schema, define=strategy.define,
                                      tags_spec=strategy.tags)
        return ResolvedJob(job, kind, schema, adapter, segmenter, sources, registry=registry,
                           strategy=strategy, on_error=on_error)

    for name in ("train_hours", "test_hours"):
        value = job.get(name, 0.0)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be a finite non-negative number")
    if "train_hours" not in job:
        raise ValueError("curate job needs 'train_hours'")
    if not isinstance(job.get("out"), str) or not job["out"]:
        raise ValueError("curate job needs a non-empty 'out' path")
    if "span_steps" in job and "span_seconds" in job:
        raise ValueError("span_steps and span_seconds are mutually exclusive")
    if job.get("clip_aggregate", "max") not in {"max", "min", "mean"}:
        raise ValueError("clip_aggregate must be 'max', 'min', or 'mean'")
    for name, default in (("batch_size", 1024), ("max_in_flight", 2)):
        _positive_int(job, name, default)
    if "memory_budget" in job:
        _positive_int(job, "memory_budget", 0)
    if job.get("approximation", "exact") not in {"exact", "approximate"}:
        raise ValueError("approximation must be 'exact' or 'approximate'")
    for name in ("checkpoint_dir", "spill_dir"):
        if name in job and (not isinstance(job[name], str) or not job[name]):
            raise ValueError(f"{name} must be a non-empty path string")
    output = _output(job, schema, registry)
    output.validate_schema(schema)
    validate_adapter_availability(adapter, schema, define=strategy.define,
                                  tags_spec=strategy.tags, output_spec=output)
    return ResolvedJob(job, kind, schema, adapter, segmenter, sources, registry=registry,
                       strategy=strategy, output=output, encoder=_encoder(job, registry),
                       on_error=on_error)
