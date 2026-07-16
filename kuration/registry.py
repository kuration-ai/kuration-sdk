"""Typed, discoverable component registry with parent/overlay isolation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import inspect
import re
from typing import Any, Callable, Optional

_KINDS = {"adapter", "metric", "scorer", "encoder", "method", "filter", "container",
          "preset", "operator", "transform"}
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?$")


@dataclass(frozen=True)
class ComponentDescriptor:
    name: str
    kind: str
    description: str = ""
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    capabilities: dict[str, bool] = field(default_factory=dict)
    extra: Optional[str] = None
    install_hint: Optional[str] = None
    package: Optional[str] = None
    version: Optional[str] = None
    schema_version: int = 1
    documentation: Optional[str] = None
    stability: str = "stable"
    module: Optional[str] = None
    fingerprint: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class _Entry:
    descriptor: ComponentDescriptor
    component: Any


def _identity(component: Any) -> tuple[Optional[str], Optional[str]]:
    module = getattr(component, "__module__", None)
    try:
        source = inspect.getsource(component).encode()
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
                 install_hint: Optional[str] = None, package: Optional[str] = None,
                 version: Optional[str] = None, schema_version: int = 1,
                 documentation: Optional[str] = None, stability: str = "stable") -> None:
        if kind not in _KINDS:
            raise ValueError(f"unknown component kind {kind!r}; expected {sorted(_KINDS)}")
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError(f"invalid component name {name!r}; use 'name' or 'namespace/name'")
        if name in self._entries[kind] and not replace:
            raise ValueError(f"{kind} {name!r} already registered in this registry")
        module, fingerprint = _identity(component)
        desc = ComponentDescriptor(
            name=name, kind=kind, description=description,
            parameters=parameters or {"type": "object", "properties": {},
                                      "additionalProperties": False},
            capabilities=dict(capabilities or {}), extra=extra, install_hint=install_hint,
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
        "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
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
