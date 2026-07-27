"""Plugin discovery for registry extensions.

Plugins either register on module import or expose a zero-argument initializer through the
`kuration.plugins` package entry-point group.
"""
from __future__ import annotations

import importlib
import hashlib
import inspect
import sys
import warnings
from importlib.metadata import PackageNotFoundError, entry_points, packages_distributions, version
from threading import RLock
from typing import Any

ENTRY_POINT_GROUP = "kuration.plugins"
_LOADED_MODULES: set[str] = set()
_LOADED_ENTRY_POINTS: set[str] = set()
_ENTRY_POINTS_SCANNED = False
# Original entries replaced (or ``None`` for names first added) by installed entry points.
# They let an explicit job obtain a clean registry even if ambient discovery happened earlier
# in this Python process.
_ENTRY_POINT_BASELINES: dict[tuple[str, str], Any | None] = {}
_MODULE_COMPONENTS: dict[str, dict[tuple[str, str], Any]] = {}
_PLUGIN_FAILURES: dict[str, dict[str, str]] = {}
_LOCK = RLock()


def _plugin_state_snapshot() -> dict[str, Any]:
    """Capture registries that public plugin registration helpers can mutate."""
    from .registry import default_registry
    from .adapters.base import _ADAPTERS
    from .dsl.registry import METRICS
    from .output.base import _ENCODERS
    from .scorers import _SCORERS
    from .transforms import _TRANSFORM_META
    from .analyzers import _ANALYZER_META

    return {
        "registry": {kind: dict(entries) for kind, entries in default_registry._entries.items()},
        "legacy": [
            (_ADAPTERS, dict(_ADAPTERS)),
            (METRICS, dict(METRICS)),
            (_ENCODERS, dict(_ENCODERS)),
            (_SCORERS, dict(_SCORERS)),
            (_TRANSFORM_META, dict(_TRANSFORM_META)),
            (_ANALYZER_META, dict(_ANALYZER_META)),
        ],
        "loaded_modules": set(_LOADED_MODULES),
        "loaded_entry_points": set(_LOADED_ENTRY_POINTS),
        "entry_point_baselines": dict(_ENTRY_POINT_BASELINES),
        "module_components": {name: dict(entries)
                              for name, entries in _MODULE_COMPONENTS.items()},
        "plugin_failures": {key: dict(entry) for key, entry in _PLUGIN_FAILURES.items()},
        "sys_modules": set(sys.modules),
    }


def _restore_plugin_state(snapshot: dict[str, Any], module_name: str | None) -> None:
    """Roll back a failed plugin initializer, including legacy compatibility registries."""
    from .registry import default_registry

    for kind, entries in snapshot["registry"].items():
        default_registry._entries[kind].clear()
        default_registry._entries[kind].update(entries)
    for registry, entries in snapshot["legacy"]:
        registry.clear()
        registry.update(entries)
    _LOADED_MODULES.clear()
    _LOADED_MODULES.update(snapshot["loaded_modules"])
    _LOADED_ENTRY_POINTS.clear()
    _LOADED_ENTRY_POINTS.update(snapshot["loaded_entry_points"])
    _ENTRY_POINT_BASELINES.clear()
    _ENTRY_POINT_BASELINES.update(snapshot["entry_point_baselines"])
    _MODULE_COMPONENTS.clear()
    _MODULE_COMPONENTS.update(snapshot["module_components"])
    _PLUGIN_FAILURES.clear()
    _PLUGIN_FAILURES.update(snapshot["plugin_failures"])

    if module_name:
        root = module_name.split(".", 1)[0]
        for name in set(sys.modules) - snapshot["sys_modules"]:
            if name == root or name.startswith(root + "."):
                sys.modules.pop(name, None)


def _component_keys() -> set[tuple[str, str, str | None]]:
    """Snapshot registered component identities without importing plugin machinery."""
    from .registry import default_registry
    return {(descriptor.kind, descriptor.name, descriptor.fingerprint)
            for descriptor in default_registry.descriptors()}


def _registry_entries() -> dict[tuple[str, str], Any]:
    from .registry import default_registry
    return {(kind, name): entry for kind, entries in default_registry._entries.items()
            for name, entry in entries.items()}


def _changed_entries(previous: dict[tuple[str, str], Any]) -> dict[tuple[str, str], Any]:
    current = _registry_entries()
    return {key: entry for key, entry in current.items()
            if key not in previous or entry != previous[key]}


def _owned_plugin_entries(module_names, previous: dict[tuple[str, str], Any]) \
        -> dict[tuple[str, str], Any]:
    """Return components registered by this load or already owned by its target modules.

    Entry-point targets may already be imported (for example, after an explicit plugin load),
    making a second import or idempotent initializer leave the registry unchanged.  Recorded
    ownership and descriptor implementation modules distinguish that case from a true no-op.
    """
    changed = _changed_entries(previous)
    if changed:
        return changed
    if isinstance(module_names, str):
        module_names = (module_names,)
    candidates = {name for name in module_names if isinstance(name, str) and name}
    current = _registry_entries()
    owned: dict[tuple[str, str], Any] = {}
    for name in candidates:
        owned.update({key: entry for key, entry in _MODULE_COMPONENTS.get(name, {}).items()
                      if current.get(key) == entry})
    if owned:
        return owned
    return {
        key: entry for key, entry in current.items()
        if any(
            getattr(entry.descriptor, "module", None) == candidate
            or str(getattr(entry.descriptor, "module", "")).startswith(candidate + ".")
            for candidate in candidates
        )
    }


def _record_module_components(module: str | None,
                              changed: dict[tuple[str, str], Any]) -> None:
    if module and changed:
        _MODULE_COMPONENTS.setdefault(module, {}).update(changed)
    for key, entry in changed.items():
        implementation_module = getattr(entry.descriptor, "module", None)
        if implementation_module:
            _MODULE_COMPONENTS.setdefault(implementation_module, {})[key] = entry


def _record_component_modules(previous: set[tuple[str, str, str | None]]) -> None:
    """Track implementation modules registered by a plugin initializer.

    An entry point commonly lives in ``package.plugin`` while the functions it registers live
    in sibling modules.  Recording only the initializer would make provenance blind to changes
    in those implementations and could incorrectly approve checkpoint reuse.
    """
    from .registry import default_registry
    for descriptor in default_registry.descriptors():
        key = (descriptor.kind, descriptor.name, descriptor.fingerprint)
        if key not in previous and descriptor.module:
            _LOADED_MODULES.add(descriptor.module)


def _available_entry_points():
    eps = entry_points()
    return eps.select(group=ENTRY_POINT_GROUP) if hasattr(eps, "select") \
        else eps.get(ENTRY_POINT_GROUP, ())


def load_plugins(modules=(), *, include_entry_points: bool = True) -> list[dict]:
    """Load explicit registration modules and installed `kuration.plugins` entry points.

    Explicit modules register through import side effects and fail fast. Entry points must resolve
    to either a module or a zero-argument initializer. A failing ambient entry point is rolled back,
    warned, and returned as a ``kind='error'`` record so later healthy plugins remain discoverable.
    Repeated calls are safe.
    """
    if isinstance(modules, str):
        modules = [modules]
    if not isinstance(modules, (list, tuple)) or not all(isinstance(m, str) and m for m in modules):
        raise ValueError("plugins must be a module name or a list of non-empty module names")

    global _ENTRY_POINTS_SCANNED
    loaded: list[dict] = []
    with _LOCK:
        for name in modules:
            if name in _LOADED_MODULES:
                continue
            state = _plugin_state_snapshot()
            previous_components = _component_keys()
            previous_entries = _registry_entries()
            _LOADED_MODULES.add(name)
            try:
                importlib.import_module(name)
            except Exception as e:
                _restore_plugin_state(state, name)
                raise ImportError(f"failed to load kuration plugin module {name!r}: {e}") from e
            changed = _changed_entries(previous_entries)
            _record_module_components(name, changed)
            _record_component_modules(previous_components)
            loaded.append({"kind": "module", "name": name})

        if include_entry_points and not _ENTRY_POINTS_SCANNED:
            _ENTRY_POINTS_SCANNED = True
            for ep in _available_entry_points():
                key = f"{ep.name}={ep.value}"
                if key in _LOADED_ENTRY_POINTS:
                    continue
                state = _plugin_state_snapshot()
                _LOADED_ENTRY_POINTS.add(key)
                previous_components = _component_keys()
                previous_entries = _registry_entries()
                try:
                    target = ep.load()
                    if inspect.ismodule(target):
                        pass
                    elif callable(target):
                        target()
                    else:
                        raise TypeError(
                            "entry point must resolve to a module or zero-argument initializer, "
                            f"got {type(target).__name__}")
                    module = target.__name__ if inspect.ismodule(target) \
                        else getattr(target, "__module__", None)
                    value_module = ep.value.split(":", 1)[0] \
                        if isinstance(ep.value, str) else None
                    changed = _changed_entries(previous_entries)
                    if not _owned_plugin_entries((value_module, module), previous_entries):
                        raise ValueError("plugin loaded but registered no components")
                except Exception as e:
                    module_name = ep.value.split(":", 1)[0] if isinstance(ep.value, str) else None
                    _restore_plugin_state(state, module_name)
                    message = f"failed to load kuration plugin entry point {key!r}: {e}"
                    failure = {"kind": "error", "name": ep.name, "value": ep.value,
                               "error_type": f"{type(e).__module__}.{type(e).__qualname__}",
                               "error": message}
                    _PLUGIN_FAILURES[key] = failure
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
                    loaded.append(dict(failure))
                    continue
                _PLUGIN_FAILURES.pop(key, None)
                if module:
                    _LOADED_MODULES.add(module)
                for component_key in changed:
                    _ENTRY_POINT_BASELINES.setdefault(
                        component_key, previous_entries.get(component_key))
                if value_module != module:
                    _record_module_components(value_module, changed)
                _record_module_components(module, changed)
                _record_component_modules(previous_components)
                loaded.append({"kind": "entry_point", "name": ep.name, "value": ep.value})
    return loaded


def plugin_registry_snapshot(modules=(), *, include_entry_points: bool = True):
    """Return a flattened registry honoring a job's plugin-discovery boundary."""
    from .registry import default_registry
    with _LOCK:
        registry = default_registry.snapshot()
        if include_entry_points:
            return registry

        for (kind, name), baseline in _ENTRY_POINT_BASELINES.items():
            if baseline is None:
                registry._entries[kind].pop(name, None)
            else:
                registry._entries[kind][name] = baseline

        # Explicit modules win over the ambient exclusion, including when an entry point
        # imported the same module earlier in the process.
        for module in modules:
            owned = _MODULE_COMPONENTS.get(module, {})
            if not owned:
                owned = {key: entry for key, entry in _registry_entries().items()
                         if getattr(entry.descriptor, "module", None) == module}
            for (kind, name), entry in owned.items():
                registry._entries[kind][name] = entry
        return registry


def loaded_plugin_modules() -> tuple[str, ...]:
    """Return loaded plugin module names in deterministic sorted order."""
    return tuple(sorted(_LOADED_MODULES))


def plugin_failures() -> tuple[dict[str, str], ...]:
    """Return persistent diagnostics for ambient entry points that failed discovery."""
    with _LOCK:
        return tuple(dict(_PLUGIN_FAILURES[key]) for key in sorted(_PLUGIN_FAILURES))


def plugin_provenance(*, registry=None) -> list[dict]:
    """Host-independent plugin fingerprints plus ambient discovery failure records."""
    out = []
    distributions = packages_distributions()
    names = set(loaded_plugin_modules())
    if registry is not None:
        implementation_modules = {descriptor.module for descriptor in registry.descriptors()
                                  if descriptor.module}
        names &= implementation_modules
    for name in sorted(names):
        module = sys.modules.get(name)
        path = getattr(module, "__file__", None)
        digest = None
        if path:
            try:
                module_digest = hashlib.sha256()
                with open(path, "rb") as f:
                    while chunk := f.read(1024 * 1024):
                        module_digest.update(chunk)
                digest = module_digest.hexdigest()
            except OSError:
                pass
        entry = {"module": name, "sha256": digest}
        candidates = sorted(distributions.get(name.split(".", 1)[0], ()))
        if candidates:
            distribution = candidates[0]
            entry["distribution"] = distribution
            try:
                entry["version"] = version(distribution)
            except PackageNotFoundError:
                pass
        out.append(entry)
    for failure in plugin_failures():
        # Exception text may contain credentials or host paths. Provenance records a stable,
        # actionable failure marker; plugin_failures() and kur-plugin-check retain full detail.
        out.append({
            "kind": "error",
            "name": failure["name"],
            "value": failure["value"],
            "error_type": failure["error_type"],
            "error": f"plugin entry point failed; run kur-plugin-check {failure['name']}",
        })
    return out
