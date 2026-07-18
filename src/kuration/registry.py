"""Typed, discoverable component registry with parent/overlay isolation."""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import hashlib
import inspect
import math
import re
from typing import Any, Callable, Optional

_KINDS = {"adapter", "metric", "scorer", "encoder", "method", "filter", "container",
          "preset", "operator", "transform"}
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?$")


def _freeze(value):
    from .schema import FrozenDict
    if isinstance(value, dict):
        return FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _validate_json(value, *, label: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"component {label} must not contain non-finite numbers")
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"component {label} object keys must be strings")
        for item in value.values():
            _validate_json(item, label=label)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json(item, label=label)
        return
    raise ValueError(f"component {label} must contain JSON-compatible values, got "
                     f"{type(value).__name__}")


@dataclass(frozen=True)
class ComponentDescriptor:
    name: str
    kind: str
    description: str = ""
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    capabilities: dict[str, bool] = field(default_factory=dict)
    execution_traits: Optional[dict] = None
    extra: Optional[str] = None
    install_hint: Optional[str] = None
    package: Optional[str] = None
    version: Optional[str] = None
    schema_version: int = 1
    documentation: Optional[str] = None
    stability: str = "stable"
    module: Optional[str] = None
    fingerprint: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unknown component kind {self.kind!r}; expected {sorted(_KINDS)}")
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            raise ValueError(f"invalid component name {self.name!r}; use 'name' or "
                             "'namespace/name'")
        if not isinstance(self.description, str):
            raise ValueError("component description must be a string")
        if not isinstance(self.parameters, dict):
            raise ValueError("component parameters must be a JSON Schema object")
        if not isinstance(self.capabilities, dict) or any(
                not isinstance(name, str) or not isinstance(enabled, bool)
                for name, enabled in self.capabilities.items()):
            raise ValueError("component capabilities must map names to booleans")
        if self.execution_traits is not None and not isinstance(self.execution_traits, dict):
            raise ValueError("component execution_traits must be an object or None")
        if (not isinstance(self.schema_version, int) or isinstance(self.schema_version, bool)
                or self.schema_version < 1):
            raise ValueError("component schema_version must be a positive integer")
        for label in ("extra", "install_hint", "package", "version", "documentation",
                      "module", "fingerprint"):
            value = getattr(self, label)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"component {label} must be a string or None")
        if not isinstance(self.stability, str) or not self.stability:
            raise ValueError("component stability must be a non-empty string")
        _validate_json(self.parameters, label="parameters")
        if self.execution_traits is not None:
            _validate_json(self.execution_traits, label="execution_traits")
        object.__setattr__(self, "parameters", _freeze(copy.deepcopy(self.parameters)))
        object.__setattr__(self, "capabilities", _freeze(dict(self.capabilities)))
        if self.execution_traits is not None:
            object.__setattr__(self, "execution_traits",
                               _freeze(copy.deepcopy(self.execution_traits)))

    def to_dict(self) -> dict:
        return {name: _plain(getattr(self, name)) for name in self.__dataclass_fields__
                if getattr(self, name) is not None}


@dataclass(frozen=True)
class _Entry:
    descriptor: ComponentDescriptor
    component: Any


def _identity(component: Any) -> tuple[Optional[str], Optional[str]]:
    module = getattr(component, "__module__", None)
    try:
        parts = [inspect.getsource(component)]
        if inspect.isfunction(component):
            parts.append(f"defaults={getattr(component, '__defaults__', None)!r}")
            parts.append(f"kwdefaults={getattr(component, '__kwdefaults__', None)!r}")
            closure = getattr(component, "__closure__", None) or ()
            parts.append("closure=" + repr(tuple(
                repr(cell.cell_contents) for cell in closure)))
        source = "\n".join(parts).encode()
    except (OSError, TypeError):
        target = type(component)
        module = module or getattr(target, "__module__", None)
        parts = [f"{target.__module__}.{target.__qualname__}"]
        try:
            parts.append(inspect.getsource(target))
        except (OSError, TypeError):
            pass
        for name, value in sorted(getattr(component, "__dict__", {}).items()):
            if callable(value):
                try:
                    rendered = inspect.getsource(value)
                except (OSError, TypeError):
                    rendered = f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', name)}"
            else:
                rendered = repr(value)
            parts.append(f"{name}={rendered}")
        source = "\n".join(parts).encode()
    return module, hashlib.sha256(source).hexdigest()


class Registry:
    """A deterministic component registry. Child entries shadow a parent without mutation."""

    def __init__(self, parent: Optional["Registry"] = None):
        self.parent = parent
        self._entries: dict[str, dict[str, _Entry]] = {k: {} for k in _KINDS}

    def register(self, kind: str, name: str, component: Any, *, replace: bool = False,
                 description: str = "", parameters: Optional[dict] = None,
                 capabilities: Optional[dict[str, bool]] = None, extra: Optional[str] = None,
                 execution_traits: Optional[dict] = None,
                 install_hint: Optional[str] = None, package: Optional[str] = None,
                 version: Optional[str] = None, schema_version: int = 1,
                 documentation: Optional[str] = None, stability: str = "stable") -> None:
        if kind not in _KINDS:
            raise ValueError(f"unknown component kind {kind!r}; expected {sorted(_KINDS)}")
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError(f"invalid component name {name!r}; use 'name' or 'namespace/name'")
        if name in self._entries[kind] and not replace:
            raise ValueError(f"{kind} {name!r} already registered in this registry")
        if execution_traits is not None:
            from .planning import OperatorTraits
            execution_traits = OperatorTraits.from_dict(execution_traits).to_dict()
        module, fingerprint = _identity(component)
        desc = ComponentDescriptor(
            name=name, kind=kind, description=description,
            parameters=parameters or {"type": "object", "properties": {},
                                      "additionalProperties": False},
            capabilities=dict(capabilities or {}), execution_traits=execution_traits,
            extra=extra, install_hint=install_hint,
            package=package, version=version, schema_version=schema_version,
            documentation=documentation, stability=stability, module=module,
            fingerprint=fingerprint)
        self._entries[kind][name] = _Entry(desc, component)

    def get(self, kind: str, name: str) -> Any:
        entry = self._entries.get(kind, {}).get(name)
        if entry is not None:
            return entry.component
        if self.parent is not None:
            return self.parent.get(kind, name)
        raise ValueError(f"unknown {kind} {name!r}; registered: {self.names(kind)}")

    def descriptor(self, kind: str, name: str) -> ComponentDescriptor:
        entry = self._entries.get(kind, {}).get(name)
        if entry is not None:
            return entry.descriptor
        if self.parent is not None:
            return self.parent.descriptor(kind, name)
        raise ValueError(f"unknown {kind} {name!r}; registered: {self.names(kind)}")

    def names(self, kind: str) -> list[str]:
        if kind not in _KINDS:
            raise ValueError(f"unknown component kind {kind!r}; expected {sorted(_KINDS)}")
        names = set(self._entries[kind])
        if self.parent is not None:
            names.update(self.parent.names(kind))
        return sorted(names)

    def descriptors(self, kind: Optional[str] = None) -> list[ComponentDescriptor]:
        kinds = [kind] if kind is not None else sorted(_KINDS)
        out = []
        for k in kinds:
            for name in self.names(k):
                out.append(self.descriptor(k, name))
        return out

    def child(self) -> "Registry":
        return Registry(parent=self)

    def snapshot(self) -> "Registry":
        """Return an isolated, flattened snapshot of the currently visible components.

        A registry is intentionally mutable while an application is configuring plugins.  A
        resolved pipeline, however, must not change because some other part of the process later
        registers or replaces a component.  Snapshots retain component objects but defensively
        copy their descriptors and have no mutable parent link.
        """
        out = Registry()
        for kind in sorted(_KINDS):
            for name in self.names(kind):
                out._entries[kind][name] = _Entry(
                    copy.deepcopy(self.descriptor(kind, name)), self.get(kind, name))
        return out

    @classmethod
    def merged(cls, *registries: "Registry") -> "Registry":
        """Merge visible registry snapshots, rejecting semantic name collisions.

        This is primarily useful for shared execution such as experiments: two arms may add
        disjoint overlay components, but the same ``(kind, name)`` must still identify the same
        descriptor and implementation in every arm.
        """
        if not registries:
            return cls()
        if any(not isinstance(registry, Registry) for registry in registries):
            raise ValueError("Registry.merged() accepts only Registry instances")
        out = cls()
        for registry in registries:
            for kind in sorted(_KINDS):
                for name in registry.names(kind):
                    descriptor = registry.descriptor(kind, name)
                    existing = out._entries[kind].get(name)
                    if existing is not None:
                        if existing.descriptor != descriptor:
                            raise ValueError(
                                f"incompatible {kind} {name!r} across registry overlays; "
                                "experiment arms must resolve a shared name identically")
                        continue
                    out._entries[kind][name] = _Entry(
                        copy.deepcopy(descriptor), registry.get(kind, name))
        return out

    def register_adapter(self, name: str, component: Any, **metadata) -> None:
        self.register("adapter", name, component, **metadata)

    def register_metric(self, name: str, component: Any, **metadata) -> None:
        self.register("metric", name, component, **metadata)

    def register_scorer(self, name: str, component: Any, **metadata) -> None:
        self.register("scorer", name, component, **metadata)

    def register_encoder(self, name: str, component: Any, **metadata) -> None:
        self.register("encoder", name, component, **metadata)

    def register_method(self, name: str, component: Any, **metadata) -> None:
        self.register("method", name, component, **metadata)

    def register_filter(self, name: str, component: Any, **metadata) -> None:
        self.register("filter", name, component, **metadata)

    def register_container(self, name: str, component: Any, **metadata) -> None:
        self.register("container", name, component, **metadata)

    def register_preset(self, name: str, component: Any, **metadata) -> None:
        self.register("preset", name, component, **metadata)

    def register_operator(self, name: str, component: Any, **metadata) -> None:
        """Register a compositional ``Pipeline.apply`` operator in this registry."""
        if not callable(component):
            raise ValueError("operator component must be callable")
        self.register("operator", name, component, **metadata)

    def register_transform(self, name: str, component: Any, **metadata) -> None:
        self.register("transform", name, component, **metadata)


default_registry = Registry()


def list_components(kind: Optional[str] = None, *, registry: Registry = default_registry) -> list[dict]:
    """Return JSON-safe descriptors for installed components without reading datasets."""
    from .plugins import load_plugins
    load_plugins()
    return [d.to_dict() for d in registry.descriptors(kind)]


def describe_component(kind: str, name: str, *,
                       registry: Registry = default_registry) -> dict:
    """Return one installed component descriptor as a JSON-safe dictionary.

    Plugins are loaded before lookup; an unknown kind or name raises ``ValueError``.
    """
    from .plugins import load_plugins
    load_plugins()
    return registry.descriptor(kind, name).to_dict()


def validate_parameters(descriptor: ComponentDescriptor, params: dict) -> None:
    """Validate the small JSON-Schema subset used by component descriptors."""
    schema = descriptor.parameters or {}
    props = schema.get("properties", {})
    required = set(schema.get("required", ()))
    missing = required - set(params)
    if missing:
        raise ValueError(f"{descriptor.kind} {descriptor.name!r} missing parameter(s) "
                         f"{sorted(missing)}")
    if schema.get("additionalProperties") is False:
        unknown = set(params) - set(props)
        if unknown:
            raise ValueError(f"{descriptor.kind} {descriptor.name!r}: unknown parameter(s) "
                             f"{sorted(unknown)}; expected {sorted(props)}")
    json_types = {
        "string": lambda value: isinstance(value, str),
        "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda value: (isinstance(value, (int, float))
                                 and not isinstance(value, bool)
                                 and math.isfinite(float(value))),
        "boolean": lambda value: isinstance(value, bool),
        "object": lambda value: isinstance(value, dict),
        "array": lambda value: isinstance(value, (list, tuple)),
        "null": lambda value: value is None,
    }
    for name in set(params) & set(props):
        value, rule = params[name], props[name]
        expected = rule.get("type")
        allowed = [expected] if isinstance(expected, str) else list(expected or [])
        if allowed and not any(kind in json_types and json_types[kind](value) for kind in allowed):
            raise ValueError(
                f"{descriptor.kind} {descriptor.name!r}: parameter {name!r} must have "
                f"JSON type {' or '.join(allowed)}, got {type(value).__name__}")
        if "enum" in rule and value not in rule["enum"]:
            raise ValueError(
                f"{descriptor.kind} {descriptor.name!r}: parameter {name!r} must be one of "
                f"{rule['enum']}, got {value!r}")
        if "minimum" in rule and value < rule["minimum"]:
            raise ValueError(
                f"{descriptor.kind} {descriptor.name!r}: parameter {name!r} must be >= "
                f"{rule['minimum']}, got {value!r}")
        if "maximum" in rule and value > rule["maximum"]:
            raise ValueError(
                f"{descriptor.kind} {descriptor.name!r}: parameter {name!r} must be <= "
                f"{rule['maximum']}, got {value!r}")


def register_method(name: str, fn: Callable, **metadata) -> None:
    """Register a selection-priority method in the default registry."""
    default_registry.register("method", name, fn, **metadata)


def register_filter(name: str, fn: Callable, **metadata) -> None:
    """Register a pre-pass filter in the default registry."""
    default_registry.register("filter", name, fn, **metadata)


def register_container(name: str, handler: Any, **metadata) -> None:
    default_registry.register("container", name, handler, **metadata)


def register_preset(name: str, factory: Callable, **metadata) -> None:
    """Register an output-preset factory in the default registry."""
    default_registry.register("preset", name, factory, **metadata)


def register_operator(name: str, fn: Optional[Callable] = None, **metadata):
    """Register a compositional pipeline operator, directly or as a decorator.

    The callable receives ``(pipeline, **params)`` and must return a derived ``Pipeline`` that
    adds at least one ordinary operator. Parameters are validated against descriptor metadata
    before it runs. Duplicate names and non-callable components raise ``ValueError``.
    """
    def decorate(component: Callable):
        default_registry.register_operator(name, component, **metadata)
        return component

    if fn is None:
        return decorate
    decorate(fn)
    return None
