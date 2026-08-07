"""Serializable logical execution traits and physical-plan compilation.

This module is deliberately execution-backend neutral.  It describes *what* must run and
where a bounded pipeline must stop for a barrier; local and Ray runtimes consume the same
representation.  The first implementation is explanatory only -- existing terminals retain
their current execution path while the bounded runtime is introduced stage by stage.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any


_SCOPES = {"row", "window", "partition", "global"}
_SHAPES = {"map", "stateful-stream", "reduction", "multipass", "global-index"}
_PASSES = {"one", "replayable-multipass", "external-sort"}
_STATE_BOUNDS = {"constant", "per-key", "bounded-by-parameter", "proportional-to-pool"}
_DETERMINISM = {"stable-key", "canonical-order", "seeded", "backend-dependent"}
_CHECKPOINTS = {"partition-reusable", "reducer-state", "non-checkpointable"}


_METHOD_CONTRACTS = {
    "top-k": {"exact": "bounded-multipass", "approximate": "unsupported"},
    "uniform-bins": {"exact": "bounded-multipass", "approximate": "unsupported"},
    "weighted": {"exact": "bounded-hours; materialized-count", "approximate": "unsupported"},
    "filter": {"exact": "materialized", "approximate": "unsupported"},
    "target-distribution": {"exact": "materialized", "approximate": "unsupported"},
    "cluster-balanced": {"exact": "materialized", "approximate": "unsupported"},
    "k-center": {"exact": "materialized", "approximate": "unsupported"},
    "facility": {"exact": "materialized", "approximate": "unsupported"},
    "dsir": {"exact": "materialized", "approximate": "unsupported"},
    "rare": {"exact": "materialized", "approximate": "unsupported"},
    "typical": {"exact": "materialized", "approximate": "unsupported"},
}


def method_execution_contract(method: str) -> dict[str, str]:
    """Return the declared exact/approximate execution modes for a selection method."""
    return dict(_METHOD_CONTRACTS.get(
        method, {"exact": "plugin-declared", "approximate": "plugin-declared"}))


def _choice(field_name: str, value: str, choices: set[str]) -> None:
    if value not in choices:
        raise ValueError(f"execution trait {field_name} must be one of {sorted(choices)}, "
                         f"got {value!r}")


def _names(field_name: str, values) -> tuple[str, ...]:
    values = tuple(values or ())
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"execution trait {field_name} must contain non-empty strings")
    return values


@dataclass(frozen=True)
class OperatorResources:
    """Portable resource and batch bounds for one logical operator."""

    cpu: float = 1.0
    gpu: float = 0.0
    memory_bytes: int | None = None
    preferred_batch_size: int | None = None
    max_batch_size: int | None = None

    def __post_init__(self) -> None:
        if self.cpu < 0 or self.gpu < 0:
            raise ValueError("operator CPU/GPU resources must be non-negative")
        for name in ("memory_bytes", "preferred_batch_size", "max_batch_size"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)
                                      or value < 1):
                raise ValueError(f"operator resource {name} must be a positive integer")
        if (self.preferred_batch_size is not None and self.max_batch_size is not None
                and self.preferred_batch_size > self.max_batch_size):
            raise ValueError("preferred_batch_size cannot exceed max_batch_size")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "OperatorResources":
        raw = dict(raw or {})
        aliases = {"num_cpus": "cpu", "num_gpus": "gpu", "memory": "memory_bytes"}
        for old, new in aliases.items():
            if old in raw and new not in raw:
                raw[new] = raw.pop(old)
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown operator resource field(s) {sorted(unknown)}")
        return cls(**raw)


@dataclass(frozen=True)
class OperatorTraits:
    """Machine-readable execution requirements for a logical operator."""

    scope: str
    shape: str
    fusible: bool
    passes: str = "one"
    state_bound: str = "constant"
    required_columns: tuple[str, ...] = ()
    produced_columns: tuple[str, ...] = ()
    resources: OperatorResources = field(default_factory=OperatorResources)
    determinism: str = "stable-key"
    checkpoint: str = "partition-reusable"

    def __post_init__(self) -> None:
        _choice("scope", self.scope, _SCOPES)
        _choice("shape", self.shape, _SHAPES)
        _choice("passes", self.passes, _PASSES)
        _choice("state_bound", self.state_bound, _STATE_BOUNDS)
        _choice("determinism", self.determinism, _DETERMINISM)
        _choice("checkpoint", self.checkpoint, _CHECKPOINTS)
        if not isinstance(self.fusible, bool):
            raise ValueError("execution trait fusible must be true or false")
        object.__setattr__(self, "required_columns",
                           _names("required_columns", self.required_columns))
        object.__setattr__(self, "produced_columns",
                           _names("produced_columns", self.produced_columns))
        if isinstance(self.resources, Mapping):
            object.__setattr__(self, "resources", OperatorResources.from_dict(self.resources))
        elif not isinstance(self.resources, OperatorResources):
            raise ValueError("execution trait resources must be a mapping or OperatorResources")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OperatorTraits":
        if not isinstance(raw, Mapping):
            raise ValueError("execution_traits must be a mapping")
        data = dict(raw)
        columns = data.pop("columns", None)
        if columns is not None:
            if not isinstance(columns, Mapping):
                raise ValueError("execution trait columns must be a mapping")
            data.setdefault("required_columns", columns.get("required", ()))
            data.setdefault("produced_columns", columns.get("produced", ()))
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown execution trait field(s) {sorted(unknown)}")
        if "resources" in data:
            data["resources"] = OperatorResources.from_dict(data["resources"])
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PhysicalOperator:
    id: str
    name: str
    traits: OperatorTraits
    parameters: dict[str, Any] = field(default_factory=dict)
    component: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PhysicalStage:
    id: str
    mode: str
    operators: tuple[PhysicalOperator, ...]
    passes: int
    materializes_input: bool
    checkpoint: str
    barrier_reason: str | None = None
    estimated_peak_memory: int | None = None
    resources: OperatorResources = field(default_factory=OperatorResources)

    @property
    def name(self) -> str:
        return "+".join(operator.name for operator in self.operators)

    @property
    def streaming(self) -> bool:
        return (not self.materializes_input
                and all(operator.traits.scope != "global"
                        and operator.traits.passes == "one" for operator in self.operators))

    def to_dict(self) -> dict:
        out = asdict(self)
        out["name"] = self.name
        out["streaming"] = self.streaming
        out["materializes_total_pool"] = self.materializes_input
        return out


@dataclass(frozen=True)
class PhysicalPlan(Mapping[str, Any]):
    """Backend-neutral, deterministic physical plan."""

    version: int
    backend: str
    stages: tuple[PhysicalStage, ...]
    memory_budget: int | None = None
    spill_path: str | None = None
    warnings: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict:
        return {"version": self.version, "backend": self.backend,
                "memory_budget": self.memory_budget, "spill_path": self.spill_path,
                "stages": [stage.to_dict() for stage in self.stages],
                "warnings": [dict(warning) for warning in self.warnings]}

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def _traits(scope: str, shape: str, fusible: bool, **kwargs) -> OperatorTraits:
    return OperatorTraits(scope=scope, shape=shape, fusible=fusible, **kwargs)


_BASE_TRAITS = {
    "read": _traits("row", "stateful-stream", True, produced_columns=("source:*",)),
    "align": _traits("row", "stateful-stream", True, required_columns=("source:*",),
                     produced_columns=("media:*",)),
    "segment": _traits("window", "stateful-stream", True, required_columns=("source:*",),
                       produced_columns=("source_id", "start_idx", "end_idx")),
    "metrics": _traits("window", "map", True, required_columns=("source:*",),
                       produced_columns=("feature:*",)),
    "predicates": _traits("row", "map", True, required_columns=("feature:*",)),
    "transform": _traits("window", "map", True, required_columns=("source:*",),
                         produced_columns=("feature:*",),
                         resources=OperatorResources(preferred_batch_size=1)),
    "scorer": _traits("global", "reduction", False, state_bound="proportional-to-pool",
                      required_columns=("feature:*",), produced_columns=("feature:*",),
                      checkpoint="reducer-state", determinism="canonical-order"),
    "filter": _traits("global", "global-index", False, state_bound="proportional-to-pool",
                      required_columns=("feature:*",), checkpoint="reducer-state",
                      determinism="canonical-order"),
    "organize": _traits("global", "multipass", False, passes="replayable-multipass",
                        state_bound="bounded-by-parameter", required_columns=("feature:*",),
                        produced_columns=("bin:*",), checkpoint="reducer-state",
                        determinism="canonical-order"),
    "selection": _traits("global", "reduction", False,
                         state_bound="proportional-to-pool",
                         required_columns=("feature:*", "bin:*"),
                         produced_columns=("selected",), checkpoint="reducer-state",
                         determinism="seeded"),
    "output": _traits("partition", "stateful-stream", False,
                      required_columns=("selected", "source:*",),
                      checkpoint="partition-reusable", determinism="stable-key"),
}


def _descriptor_traits(registry, kind: str, name: str, fallback: OperatorTraits) -> tuple[OperatorTraits, dict | None]:
    try:
        descriptor = registry.descriptor(kind, name)
    except ValueError:
        return fallback, None
    raw = getattr(descriptor, "execution_traits", None)
    return (OperatorTraits.from_dict(raw) if raw else fallback,
            {"kind": kind, "name": name, "version": descriptor.version,
             "fingerprint": descriptor.fingerprint})


def logical_operators(pipeline) -> list[PhysicalOperator]:
    """Build normalized logical operators from a resolved ``Pipeline``."""
    specs: list[tuple[str, OperatorTraits, dict, dict | None]] = []
    specs.append(("read", _BASE_TRAITS["read"], {"sources": len(pipeline.sources)}, None))
    if type(pipeline.adapter).__name__ == "JoinedMediaAdapter":
        specs.append(("align", _BASE_TRAITS["align"], {}, None))
    specs.append(("segment", _BASE_TRAITS["segment"], dict(pipeline._window_desc), None))
    feature_names = tuple(pipeline._define) + tuple(pipeline._callables)
    if feature_names or pipeline._tags:
        traits = replace(_BASE_TRAITS["metrics"], produced_columns=tuple(
            [f"feature:{name}" for name in feature_names] +
            (["tags"] if pipeline._tags else [])))
        specs.append(("metrics", traits, {"features": list(feature_names),
                                          "tags": list(pipeline._tags)}, None))
    if pipeline._predicates:
        required = tuple(f"feature:{item['metric']}" for item in pipeline._predicates)
        specs.append(("predicates", replace(_BASE_TRAITS["predicates"],
                                             required_columns=required),
                      {"count": len(pipeline._predicates)}, None))
    for call in pipeline._transforms:
        fallback = replace(
            _BASE_TRAITS["transform"],
            required_columns=tuple(call["inputs"]),
            produced_columns=tuple(f"feature:{name}" for name in call["outputs"]),
            resources=OperatorResources.from_dict(call.get("resources")))
        traits, component = _descriptor_traits(
            pipeline._registry, "transform", call["implementation"], fallback)
        specs.append((f"transform:{call['name']}", traits,
                      {"implementation": call["implementation"],
                       "batch_size": call["batch_size"], "device": call["device"],
                       "lookbehind": call.get("lookbehind", 0),
                       "lookahead": call.get("lookahead", 0),
                       "ownership": call.get("ownership")}, component))
    for scorer in pipeline._scorers:
        traits, component = _descriptor_traits(
            pipeline._registry, "scorer", scorer["name"], _BASE_TRAITS["scorer"])
        specs.append((f"scorer:{scorer['name']}", traits,
                      {"output": scorer["as"]}, component))
    for item in pipeline._prepass:
        traits, component = _descriptor_traits(
            pipeline._registry, "filter", item["type"], _BASE_TRAITS["filter"])
        specs.append((f"filter:{item['type']}", traits,
                      {k: v for k, v in item.items() if k != "type"}, component))
    if pipeline._axes:
        required = tuple(f"feature:{name}" for name, _ in pipeline._axes)
        produced = tuple(f"bin:{name}" for name, _ in pipeline._axes)
        specs.append(("organize", replace(_BASE_TRAITS["organize"],
                                           required_columns=required,
                                           produced_columns=produced),
                      {"axes": [{"feature": n, "bins": b} for n, b in pipeline._axes]}, None))
    if pipeline._method is not None:
        fallback = _BASE_TRAITS["selection"]
        if pipeline._method == "top-k":
            fallback = replace(fallback, shape="multipass", passes="replayable-multipass",
                               state_bound="bounded-by-parameter")
        elif pipeline._method == "uniform-bins":
            fallback = replace(fallback, state_bound="bounded-by-parameter")
        traits, component = _descriptor_traits(
            pipeline._registry, "method", pipeline._method, fallback)
        specs.append((f"selection:{pipeline._method}", traits,
                      {"target_size": pipeline._n, "hours": pipeline._hours,
                       "execution_contract": method_execution_contract(pipeline._method)},
                      component))
    specs.append(("output", _BASE_TRAITS["output"], {}, None))
    return [PhysicalOperator(id=f"op{i:03d}", name=name, traits=traits,
                             parameters=params, component=component)
            for i, (name, traits, params, component) in enumerate(specs)]


def _pass_count(traits: OperatorTraits) -> int:
    return 1 if traits.passes == "one" else 2


def _barrier_reason(operator: PhysicalOperator) -> str | None:
    traits = operator.traits
    if traits.scope == "global":
        return f"{operator.name} has global scope"
    if traits.passes != "one":
        return f"{operator.name} requires {traits.passes}"
    if not traits.fusible:
        return f"{operator.name} is not fusible"
    return None


def compile_physical_plan(pipeline, *, memory_budget: int | None = None,
                          spill_path: str | None = None) -> PhysicalPlan:
    """Fuse compatible operators and isolate global/materializing barriers."""
    if memory_budget is not None and (not isinstance(memory_budget, int)
                                      or isinstance(memory_budget, bool) or memory_budget < 1):
        raise ValueError("memory_budget must be a positive integer or None")
    operators = logical_operators(pipeline)
    grouped: list[list[PhysicalOperator]] = []
    current: list[PhysicalOperator] = []
    for operator in operators:
        barrier = _barrier_reason(operator)
        if barrier:
            if current:
                grouped.append(current)
                current = []
            grouped.append([operator])
        else:
            current.append(operator)
    if current:
        grouped.append(current)

    stages = []
    warnings: list[dict[str, str]] = []
    for index, group in enumerate(grouped):
        barrier = _barrier_reason(group[0]) if len(group) == 1 else None
        materializes = any(op.traits.state_bound == "proportional-to-pool" for op in group)
        if materializes:
            warnings.append({"code": "unbounded-operator",
                             "message": f"{group[0].name} has no bounded runtime yet",
                             "hint": "For a bounded exact run, prefer top-k or uniform-bins "
                                     "without global filters/scorers; otherwise size the "
                                     "candidate pool for memory."})
        resources = OperatorResources(
            cpu=max(op.traits.resources.cpu for op in group),
            gpu=max(op.traits.resources.gpu for op in group),
            memory_bytes=max((op.traits.resources.memory_bytes or 0 for op in group)) or None,
            preferred_batch_size=min((op.traits.resources.preferred_batch_size
                                      for op in group
                                      if op.traits.resources.preferred_batch_size is not None),
                                     default=None),
            max_batch_size=min((op.traits.resources.max_batch_size for op in group
                                if op.traits.resources.max_batch_size is not None), default=None))
        if (memory_budget is not None and resources.memory_bytes is not None
                and resources.memory_bytes > memory_budget):
            raise ValueError(
                f"stage {group[0].name!r} declares a minimum of "
                f"{resources.memory_bytes} memory bytes, exceeding memory_budget={memory_budget}")
        stages.append(PhysicalStage(
            id=f"stage{index:03d}",
            mode="data_parallel" if any(op.traits.scope == "global" for op in group)
            else "pipeline_parallel",
            operators=tuple(group), passes=max(_pass_count(op.traits) for op in group),
            materializes_input=materializes,
            checkpoint=("reducer-state" if any(op.traits.checkpoint == "reducer-state"
                                                for op in group)
                        else "non-checkpointable" if any(
                            op.traits.checkpoint == "non-checkpointable" for op in group)
                        else "partition-reusable"),
            barrier_reason=barrier, estimated_peak_memory=resources.memory_bytes,
            resources=resources))
    return PhysicalPlan(version=1, backend=pipeline._options["executor"], stages=tuple(stages),
                        memory_budget=memory_budget, spill_path=spill_path,
                        warnings=tuple(warnings))
