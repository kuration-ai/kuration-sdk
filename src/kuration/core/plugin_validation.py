"""Contract validation for installed or explicitly importable Kuration plugins."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import re
from typing import Any


def _canonical_package(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


@dataclass(frozen=True)
class PluginCheckResult:
    """One independently loaded plugin's descriptor and discovery diagnostics."""

    name: str
    value: str
    descriptors: tuple[dict[str, Any], ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "passed": self.passed,
            "descriptors": [dict(descriptor) for descriptor in self.descriptors],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }

    def render(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"{status} {self.name} ({len(self.descriptors)} component(s))"]
        lines.extend(f"  component: {item['kind']} {item['name']}" for item in self.descriptors)
        lines.extend(f"  warning: {message}" for message in self.warnings)
        lines.extend(f"  error: {message}" for message in self.errors)
        return "\n".join(lines)


def _available_entry_points():
    from importlib.metadata import entry_points

    points = entry_points()
    selected = points.select(group="kuration.plugins") if hasattr(points, "select") \
        else points.get("kuration.plugins", ())
    return sorted(selected, key=lambda point: (point.name, point.value))


def _descriptor_errors(descriptor, *, expected_package: str | None) -> list[str]:
    errors = []
    prefix = f"{descriptor.kind} {descriptor.name!r}"
    if "/" not in descriptor.name:
        errors.append(f"{prefix} must use a namespaced component name")
    if not descriptor.description.strip():
        errors.append(f"{prefix} needs a non-empty description")
    if not descriptor.package:
        errors.append(f"{prefix} needs descriptor package metadata")
    elif (expected_package is not None
          and _canonical_package(descriptor.package) != _canonical_package(expected_package)):
        errors.append(
            f"{prefix} declares package {descriptor.package!r}, expected {expected_package!r}")
    if not descriptor.install_hint:
        errors.append(f"{prefix} needs an actionable install_hint")
    if not descriptor.version:
        errors.append(f"{prefix} needs a component version")
    schema = descriptor.to_dict().get("parameters", {})
    from .registry import validate_parameter_schema
    try:
        validate_parameter_schema(schema)
    except ValueError as error:
        errors.append(f"{prefix} has an invalid parameter schema: {error}")
    if schema.get("type") != "object":
        errors.append(f"{prefix} parameters must have type='object'")
    if not isinstance(schema.get("properties"), dict):
        errors.append(f"{prefix} parameters must declare an object properties mapping")
    if schema.get("additionalProperties") is not False:
        errors.append(f"{prefix} parameters must set additionalProperties=false")
    required = schema.get("required", [])
    if (not isinstance(required, list)
            or any(not isinstance(name, str) for name in required)
            or not set(required).issubset(schema.get("properties", {}))):
        errors.append(f"{prefix} parameters contain an invalid required list")
    if descriptor.kind == "operator":
        # Registration rejects this too; checking the descriptor keeps `kur-plugin-check` able to
        # name the collision for entries that reached a registry by any other route.
        from .pipeline import operator_name_conflict
        conflict = operator_name_conflict(descriptor.name)
        if conflict is not None:
            errors.append(conflict)
    if descriptor.kind == "transform":
        from .transforms import _metadata

        from .registry import default_registry
        metadata = _metadata(descriptor.name,
                             default_registry.get("transform", descriptor.name))
        if not metadata["outputs"]:
            errors.append(f"{prefix} must declare scalar outputs")
        if not metadata["inputs"]:
            errors.append(f"{prefix} must declare field:/feature:/media: inputs")
        elif any(not token.startswith(("field:", "feature:", "media:"))
                 for token in metadata["inputs"]):
            errors.append(f"{prefix} declares an invalid transform input token")
    if descriptor.kind == "analyzer":
        # Registration rejects these too; checking the descriptor keeps `kur-plugin-check` able
        # to name the problem for entries that reached a registry by any other route. Inputs are
        # optional for an analyzer (a dataset-only analyzer declares none), but a declared input
        # must be drivable by the tick-stream driver.
        from .analyzers import analyzer_metadata

        from .registry import default_registry
        metadata = analyzer_metadata(descriptor.name,
                                     default_registry.get("analyzer", descriptor.name))
        if any(not token.startswith(("field:", "media:")) for token in metadata["inputs"]):
            errors.append(f"{prefix} declares an invalid analyzer input token "
                          f"(field:/media: only)")
        if metadata["ownership"] not in {None, "rightmost-start"}:
            errors.append(f"{prefix} declares an unsupported ownership policy")
    return errors


def _owned_entries(module_names, previous: dict) -> dict:
    from .plugins import _owned_plugin_entries

    return _owned_plugin_entries(module_names, previous)


def _check_one(*, name: str, value: str, point=None) -> PluginCheckResult:
    from .plugins import (_LOCK, _plugin_state_snapshot, _registry_entries,
                          _restore_plugin_state)

    # Plugin imports execute arbitrary registration code against process-global registries.
    # Serialize the whole snapshot/load/validate/restore transaction with normal discovery.
    with _LOCK:
        state = _plugin_state_snapshot()
        previous = _registry_entries()
        module_name = value.split(":", 1)[0]
        expected_package = None
        errors: list[str] = []
        warnings: list[str] = []
        descriptors: list[dict[str, Any]] = []
        target_module: str | None = module_name
        try:
            if point is None:
                importlib.import_module(value)
            else:
                distribution = getattr(point, "dist", None)
                expected_package = getattr(distribution, "name", None)
                target = point.load()
                if inspect_ismodule(target):
                    pass
                elif callable(target):
                    target()
                else:
                    raise TypeError(
                        "entry point must resolve to a module or zero-argument initializer, "
                        f"got {type(target).__name__}")
                target_module = target.__name__ if inspect_ismodule(target) \
                    else getattr(target, "__module__", None)
            entries = _owned_entries((module_name, target_module), previous)
            if not entries:
                errors.append("plugin loaded but registered no components")
            for key in sorted(entries):
                descriptor = entries[key].descriptor
                descriptors.append(descriptor.to_dict())
                errors.extend(_descriptor_errors(descriptor, expected_package=expected_package))
            packages = {_canonical_package(item["package"]) for item in descriptors
                        if item.get("package")}
            if len(packages) > 1:
                errors.append(f"plugin components declare multiple distribution packages: "
                              f"{sorted(packages)}")
        except Exception as error:
            errors.append(f"load failed: {type(error).__name__}: {error}")
        finally:
            _restore_plugin_state(state, module_name)
    return PluginCheckResult(name, value, tuple(descriptors), tuple(errors), tuple(warnings))


def inspect_ismodule(value) -> bool:
    """Small wrapper kept separate so checker tests can use lightweight fake entry points."""
    import inspect

    return inspect.ismodule(value)


def check_plugins(target: str | None = None) -> list[PluginCheckResult]:
    """Validate installed plugin entry points, or one entry-point name/importable module.

    Each plugin is loaded transactionally and registry state is restored afterward. A target that
    matches an installed ``kuration.plugins`` entry-point name checks that entry point; otherwise
    it is treated as an explicit registration module.
    """
    if target is not None and (not isinstance(target, str) or not target):
        raise ValueError("plugin target must be a non-empty string or None")
    points = _available_entry_points()
    if target is None:
        selected = points
    else:
        selected = [point for point in points if point.name == target or point.value == target]
    if selected:
        return [_check_one(name=point.name, value=point.value, point=point)
                for point in selected]
    if target is None:
        return []
    return [_check_one(name=target, value=target)]
