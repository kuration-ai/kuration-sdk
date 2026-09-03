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
          "preset", "operator", "transform", "analyzer"}
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?$")
_JSON_SCHEMA_TYPES = {"string", "integer", "number", "boolean", "object", "array", "null"}
_PARAMETER_SCHEMA_KEYS = {
    "type", "enum", "const", "allOf", "anyOf", "oneOf",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minLength", "maxLength", "pattern",
    "minItems", "maxItems", "uniqueItems", "items",
    "properties", "required", "additionalProperties",
    # JSON-Schema annotations used by built-in and repository-plugin descriptors. They do not
    # affect validation, but retaining them makes component discovery useful to UIs.
    "title", "description", "default", "examples", "deprecated", "readOnly", "writeOnly",
}


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


def validate_parameter_schema(schema: dict, *, label: str = "parameters") -> None:
    """Validate the recursively supported descriptor JSON-Schema subset.

    Runtime parameter checking deliberately has no ``jsonschema`` dependency. Consequently a
    descriptor must not advertise keywords or malformed rules that the runtime cannot honor.
    This validator is shared by registration and ``kur-plugin-check`` so a bad regex or nested
    rule fails while the plugin is loaded, not when a long-running job first uses it.
    """
    if not isinstance(schema, dict):
        raise ValueError(f"component {label} must be a JSON Schema object")

    def fail(path: str, detail: str) -> None:
        raise ValueError(f"component {label} schema {path} {detail}")

    def bounded_integer(rule: dict, name: str, path: str) -> None:
        if name not in rule:
            return
        value = rule[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            fail(path, f"keyword {name!r} must be a non-negative integer")

    def visit(rule, path: str) -> None:
        if isinstance(rule, bool):
            return
        if not isinstance(rule, dict):
            fail(path, f"must be an object or boolean, got {type(rule).__name__}")
        unknown = sorted(set(rule) - _PARAMETER_SCHEMA_KEYS)
        if unknown:
            fail(path, f"contains unsupported keyword(s) {unknown}")

        expected = rule.get("type")
        if expected is not None:
            allowed = [expected] if isinstance(expected, str) else expected
            if (not isinstance(allowed, (list, tuple)) or not allowed
                    or any(not isinstance(kind, str) or kind not in _JSON_SCHEMA_TYPES
                           for kind in allowed)
                    or len(set(allowed)) != len(allowed)):
                fail(path, f"has invalid type declaration {expected!r}")

        if "enum" in rule:
            enum = rule["enum"]
            if not isinstance(enum, (list, tuple)) or not enum:
                fail(path, "keyword 'enum' must be a non-empty array")

        for keyword in ("allOf", "anyOf", "oneOf"):
            if keyword not in rule:
                continue
            branches = rule[keyword]
            if not isinstance(branches, (list, tuple)) or not branches:
                fail(path, f"keyword {keyword!r} must be a non-empty array of schemas")
            for index, branch in enumerate(branches):
                visit(branch, f"{path}.{keyword}[{index}]")

        for keyword in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            if keyword not in rule:
                continue
            value = rule[keyword]
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(float(value))):
                fail(path, f"keyword {keyword!r} must be a finite number")
        if "minimum" in rule and "maximum" in rule and rule["minimum"] > rule["maximum"]:
            fail(path, "has minimum greater than maximum")

        for keyword in ("minLength", "maxLength", "minItems", "maxItems"):
            bounded_integer(rule, keyword, path)
        for low, high in (("minLength", "maxLength"), ("minItems", "maxItems")):
            if low in rule and high in rule and rule[low] > rule[high]:
                fail(path, f"has {low} greater than {high}")

        if "pattern" in rule:
            pattern = rule["pattern"]
            if not isinstance(pattern, str):
                fail(path, "keyword 'pattern' must be a string")
            try:
                re.compile(pattern)
            except re.error as error:
                fail(path, f"contains invalid regex pattern {pattern!r}: {error}")
        if "uniqueItems" in rule and not isinstance(rule["uniqueItems"], bool):
            fail(path, "keyword 'uniqueItems' must be a boolean")
        if "items" in rule:
            visit(rule["items"], f"{path}.items")

        properties = rule.get("properties", {})
        if not isinstance(properties, dict) or any(not isinstance(name, str)
                                                   for name in properties):
            fail(path, "keyword 'properties' must map string names to schemas")
        for name, child in properties.items():
            visit(child, f"{path}.properties[{name!r}]")

        required = rule.get("required", ())
        if (not isinstance(required, (list, tuple))
                or any(not isinstance(name, str) for name in required)
                or len(set(required)) != len(required)):
            fail(path, "keyword 'required' must contain unique string property names")

        if "additionalProperties" in rule:
            additional = rule["additionalProperties"]
            if not isinstance(additional, (bool, dict)):
                fail(path, "keyword 'additionalProperties' must be a boolean or schema")
            if isinstance(additional, dict):
                visit(additional, f"{path}.additionalProperties")

        for keyword in ("title", "description"):
            if keyword in rule and not isinstance(rule[keyword], str):
                fail(path, f"annotation {keyword!r} must be a string")
        for keyword in ("deprecated", "readOnly", "writeOnly"):
            if keyword in rule and not isinstance(rule[keyword], bool):
                fail(path, f"annotation {keyword!r} must be a boolean")
        if "examples" in rule and not isinstance(rule["examples"], (list, tuple)):
            fail(path, "annotation 'examples' must be an array")

    _validate_json(schema, label=label)
    visit(schema, "root")


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
        validate_parameter_schema(self.parameters)
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
        if kind == "operator":
            # Operators become Pipeline methods; a name whose method spelling is already a
            # real Pipeline attribute would be silently shadowed, so reject it at registration.
            from .pipeline import operator_name_conflict
            conflict = operator_name_conflict(name)
            if conflict is not None:
                raise ValueError(conflict)
        if execution_traits is not None:
            from .planning import OperatorTraits
            execution_traits = OperatorTraits.from_dict(execution_traits).to_dict()
        module, fingerprint = _identity(component)
        desc = ComponentDescriptor(
            name=name, kind=kind, description=description,
            parameters=({"type": "object", "properties": {}, "additionalProperties": False}
                        if parameters is None else parameters),
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
        """Register a compositional operator exposed as a ``Pipeline`` method."""
        if not callable(component):
            raise ValueError("operator component must be callable")
        self.register("operator", name, component, **metadata)

    def register_transform(self, name: str, component: Any, **metadata) -> None:
        self.register("transform", name, component, **metadata)

    def register_analyzer(self, name: str, component: Any, **metadata) -> None:
        """Register a stateful, explicitly driven analyzer factory.

        Analyzers are discoverable direct APIs, not implicit pipeline stages. The factory
        receives validated keyword parameters and returns a user-owned object whose update
        lifecycle is capability-specific and whose ``summary()`` returns the final result.
        """
        if not callable(component):
            raise ValueError("analyzer component must be a callable factory")
        self.register("analyzer", name, component, **metadata)


default_registry = Registry()


def list_components(kind: Optional[str] = None, *, registry: Registry = default_registry) -> list[dict]:
    """Return JSON-safe descriptors for installed components without reading datasets."""
    from .plugins import load_plugins
    if registry is default_registry:
        load_plugins()
    return [d.to_dict() for d in registry.descriptors(kind)]


def describe_component(kind: str, name: str, *,
                       registry: Registry = default_registry) -> dict:
    """Return one installed component descriptor as a JSON-safe dictionary.

    Plugins are loaded before lookup; an unknown kind or name raises ``ValueError``.
    """
    from .plugins import load_plugins
    if registry is default_registry:
        load_plugins()
    return registry.descriptor(kind, name).to_dict()


def validate_parameters(descriptor: ComponentDescriptor, params: dict) -> None:
    """Recursively validate the JSON-Schema subset supported by component descriptors.

    The core intentionally avoids a mandatory ``jsonschema`` dependency, but plugin contracts
    still need nested objects and arrays to fail during preflight rather than inside execution.
    Supported keywords are documented in ``docs/customization.md``.
    """
    if not isinstance(params, dict):
        raise ValueError(f"{descriptor.kind} {descriptor.name!r}: parameters must be an object")
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

    def fail(path: str, detail: str) -> None:
        raise ValueError(f"{descriptor.kind} {descriptor.name!r}: {path} {detail}")

    def matches(value, rule) -> bool:
        try:
            check(value, rule, "value")
            return True
        except ValueError:
            return False

    def json_equal(left, right) -> bool:
        """Compare JSON values without Python's bool/int equality aliasing."""
        if left is None or right is None:
            return left is None and right is None
        if isinstance(left, bool) or isinstance(right, bool):
            return isinstance(left, bool) and isinstance(right, bool) and left == right
        if (isinstance(left, (int, float)) and not isinstance(left, bool)
                and isinstance(right, (int, float)) and not isinstance(right, bool)):
            return left == right
        if isinstance(left, str) or isinstance(right, str):
            return isinstance(left, str) and isinstance(right, str) and left == right
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            return (isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))
                    and len(left) == len(right)
                    and all(json_equal(a, b) for a, b in zip(left, right)))
        if isinstance(left, dict) or isinstance(right, dict):
            return (isinstance(left, dict) and isinstance(right, dict)
                    and set(left) == set(right)
                    and all(json_equal(left[key], right[key]) for key in left))
        return left is right

    def check(value, rule, path: str) -> None:
        if isinstance(rule, bool):
            if not rule:
                fail(path, "is forbidden by its schema")
            return
        if not isinstance(rule, dict):
            fail(path, "has an invalid schema rule")
        if "allOf" in rule:
            for branch in rule["allOf"]:
                check(value, branch, path)
        if "anyOf" in rule and not any(matches(value, branch) for branch in rule["anyOf"]):
            fail(path, "does not match any allowed schema")
        if "oneOf" in rule and sum(matches(value, branch) for branch in rule["oneOf"]) != 1:
            fail(path, "must match exactly one allowed schema")
        expected = rule.get("type")
        allowed = [expected] if isinstance(expected, str) else list(expected or [])
        if allowed and not any(kind in json_types and json_types[kind](value) for kind in allowed):
            fail(path, f"must have JSON type {' or '.join(allowed)}, got {type(value).__name__}")
        if "enum" in rule and not any(json_equal(value, item) for item in rule["enum"]):
            fail(path, f"must be one of {rule['enum']}, got {value!r}")
        if "const" in rule and not json_equal(value, rule["const"]):
            fail(path, f"must equal {rule['const']!r}, got {value!r}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in rule and value < rule["minimum"]:
                fail(path, f"must be >= {rule['minimum']}, got {value!r}")
            if "maximum" in rule and value > rule["maximum"]:
                fail(path, f"must be <= {rule['maximum']}, got {value!r}")
            if "exclusiveMinimum" in rule and value <= rule["exclusiveMinimum"]:
                fail(path, f"must be > {rule['exclusiveMinimum']}, got {value!r}")
            if "exclusiveMaximum" in rule and value >= rule["exclusiveMaximum"]:
                fail(path, f"must be < {rule['exclusiveMaximum']}, got {value!r}")
        if isinstance(value, str):
            if "minLength" in rule and len(value) < rule["minLength"]:
                fail(path, f"must have at least {rule['minLength']} characters")
            if "maxLength" in rule and len(value) > rule["maxLength"]:
                fail(path, f"must have at most {rule['maxLength']} characters")
            if "pattern" in rule and re.search(rule["pattern"], value) is None:
                fail(path, f"must match pattern {rule['pattern']!r}")
        if isinstance(value, (list, tuple)):
            if "minItems" in rule and len(value) < rule["minItems"]:
                fail(path, f"must contain at least {rule['minItems']} items")
            if "maxItems" in rule and len(value) > rule["maxItems"]:
                fail(path, f"must contain at most {rule['maxItems']} items")
            if rule.get("uniqueItems") and any(
                    json_equal(value[left], value[right]) for left in range(len(value))
                    for right in range(left + 1, len(value))):
                fail(path, "must contain unique items")
            items = rule.get("items")
            if items is not None:
                for index, item in enumerate(value):
                    check(item, items, f"{path}[{index}]")
        if isinstance(value, dict):
            properties = rule.get("properties", {})
            required = set(rule.get("required", ()))
            missing = required - set(value)
            if missing:
                fail(path, f"missing parameter(s) {sorted(missing)}")
            additional = rule.get("additionalProperties", True)
            unknown = set(value) - set(properties)
            if additional is False and unknown:
                fail(path, f"has unknown parameter(s) {sorted(unknown)}; expected "
                     f"{sorted(properties)}")
            for name in set(value) & set(properties):
                check(value[name], properties[name], f"parameter {name!r}" if path == "parameters"
                      else f"{path}.{name}")
            if isinstance(additional, dict):
                for name in unknown:
                    check(value[name], additional, f"{path}.{name}")

    check(params, descriptor.parameters, "parameters")


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
