"""Operation protocol + runner for chaining / config-driven use.

An Operation is any object with `.apply(pipeline) -> pipeline`. Every recipe in
:mod:`kuration.selection.operations` follows that one shape, so a client can:

    op.apply(pipeline)                          # one operation, standalone
    run(pipeline, [op_a, op_b, op_c])           # chain a few
    run(pipeline, config["operations"])         # drive from a dict / YAML / JSON

Also here:

    plan(config["operations"])                  # resolved-but-unrun Operation list
    describe(config["operations"])               # multi-line human plan summary
    register_operation_type("MyOp", MyOpClass)   # add a user Operation for config-dict use
"""
from __future__ import annotations

import dataclasses
import difflib
from typing import Any, Protocol


class Operation(Protocol):
    """Marker protocol: an operation takes a Pipeline in and returns a Pipeline."""

    def apply(self, pipeline: Any) -> Any: ...


# Runtime registry of user-registered Operations. Keyed by the string name a
# user picks in a config dict (`{"type": name}`). Populated via `register_operation_type()`.
_USER_REGISTRY: dict[str, type] = {}


def register_operation_type(name: str, operation_class: type) -> None:
    """Register a user Operation class under a name for config-dict use.

    Once registered, `{"type": name, ...}` in a config list resolves to
    `operation_class(**kwargs)` inside `run()` or `plan()`. This registry is
    independent of the SDK's operator ``Registry``: it only affects how
    dict-shaped config entries resolve in this module's ``run``/``plan``/``describe``.

    Args:
        name: the string key used in `{"type": name}` config dicts.
        operation_class: any class satisfying the Operation protocol
            (having `.apply(pipeline) -> pipeline`).

    Raises:
        ValueError: if `name` is empty, if `operation_class` has no `.apply`
            method, or if `name` is already registered with a different class.

    Idempotent for the same (name, class) pair: safe to re-run on module
    reload.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("register_operation_type: 'name' must be a non-empty string")
    if not hasattr(operation_class, "apply"):
        raise ValueError(
            f"register_operation_type: '{name}' -> {operation_class!r} has no 'apply' "
            "method: cannot register as an Operation"
        )
    existing = _USER_REGISTRY.get(name)
    if existing is not None and existing is not operation_class:
        raise ValueError(
            f"register_operation_type: '{name}' is already registered with a different "
            f"class ({existing!r}). Use a different name or reload the module."
        )
    _USER_REGISTRY[name] = operation_class


def run(pipeline: Any, operations: list[Any]) -> Any:
    """Chain operations over a pipeline.

    Each item in `operations` is either:
      - an Operation instance already configured, or
      - a dict `{"type": "ClassName", ...params}` that gets resolved from the
        built-in catalog (or the user registry populated by
        `register_operation_type()`) and instantiated.

    Raises:
        ValueError: on any bad config dict, with a message that names the
            index in the list and the fix (missing type, unknown type with
            "did you mean X?" hint, unexpected keyword, etc.).
    """
    for index, op in enumerate(operations):
        if isinstance(op, dict):
            op = _resolve(op, index=index)
        pipeline = op.apply(pipeline)
    return pipeline


def plan(operations: list[Any]) -> list[Any]:
    """Resolve a list of Operations or config dicts into configured Operation
    instances, without applying them.

    Same resolution logic as `run()`, but returns the instantiated list rather
    than executing it. Useful for inspecting or logging the pipeline before
    running it.
    """
    return [
        _resolve(op, index=i) if isinstance(op, dict) else op
        for i, op in enumerate(operations)
    ]


def describe(operations: list[Any]) -> str:
    """Return a multi-line human description of a resolved plan.

    Each line names an Operation and, if it has one, its `.describe()` output
    naming the SDK primitives it compiles to.
    """
    resolved = plan(operations)
    if not resolved:
        return "selection plan: (empty)"
    lines = ["selection plan:"]
    for i, op in enumerate(resolved, start=1):
        if hasattr(op, "describe"):
            lines.append(f"  {i}. {op.describe()}")
        else:
            lines.append(f"  {i}. {op!r}")
    return "\n".join(lines)


def _available_operations() -> list[str]:
    """All Operation class names: built-in catalog plus user registry."""
    from kuration.selection import operations as _catalog
    builtin = [
        n for n in dir(_catalog)
        if not n.startswith("_") and n[0].isupper()
    ]
    return sorted(set(builtin) | set(_USER_REGISTRY.keys()))


def _lookup_class(name: str):
    """Look up an Operation class by name (built-in catalog then user registry)."""
    from kuration.selection import operations as _catalog
    return getattr(_catalog, name, None) or _USER_REGISTRY.get(name)


def _resolve(spec: dict, *, index: int) -> Any:
    """Resolve a `{"type": "ClassName", ...}` dict into an Operation instance."""
    spec = dict(spec)  # do not mutate caller
    cls_name = spec.pop("type", None)
    available = _available_operations()

    if cls_name is None:
        raise ValueError(
            f"config item {index}: missing 'type' key. "
            f"Expected one of: {', '.join(available)}."
        )

    if not isinstance(cls_name, str):
        raise ValueError(
            f"config item {index}: 'type' must be a string "
            f"(got {type(cls_name).__name__}: {cls_name!r})."
        )

    cls = _lookup_class(cls_name)
    if cls is None:
        hint = difflib.get_close_matches(cls_name, available, n=1, cutoff=0.6)
        hint_str = f" Did you mean '{hint[0]}'? (names are case-sensitive.)" if hint else ""
        raise ValueError(
            f"config item {index}: unknown 'type' value '{cls_name}'.{hint_str} "
            f"Available: {', '.join(available)}."
        )

    try:
        return cls(**spec)
    except TypeError as e:
        msg = str(e)
        if "unexpected keyword argument" in msg:
            bad_kwarg = msg.split("'")[1] if "'" in msg else None
            valid_fields = (
                [f.name for f in dataclasses.fields(cls)]
                if dataclasses.is_dataclass(cls) else []
            )
            hint = (
                difflib.get_close_matches(bad_kwarg, valid_fields, n=1, cutoff=0.6)
                if bad_kwarg else []
            )
            hint_str = f" Did you mean '{hint[0]}'?" if hint else ""
            raise ValueError(
                f"config item {index} ({cls_name}): unexpected keyword '{bad_kwarg}'."
                f"{hint_str} Valid fields: {', '.join(valid_fields)}."
            ) from e
        # missing required argument, wrong type, etc: surface as-is
        raise
