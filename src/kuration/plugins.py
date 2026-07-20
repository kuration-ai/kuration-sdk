"""Plugin discovery for registry extensions.

Plugins either register on module import or expose a zero-argument initializer through the
`kuration.plugins` package entry-point group.
"""
from __future__ import annotations

import importlib
import hashlib
import inspect
import sys
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
_LOCK = RLock()


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

    Explicit modules register through import side effects. Entry points must resolve to either
    a module or a zero-argument function that performs registrations. Repeated calls are safe.
    Returns descriptors for plugins loaded by this call.
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
            previous_components = _component_keys()
            previous_entries = _registry_entries()
            _LOADED_MODULES.add(name)
            try:
                importlib.import_module(name)
            except Exception as e:
                _LOADED_MODULES.discard(name)
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
                _LOADED_ENTRY_POINTS.add(key)
                previous_components = _component_keys()
                previous_entries = _registry_entries()
                try:
                    target = ep.load()
                    if callable(target) and not inspect.ismodule(target):
                        target()
                except Exception as e:
                    _LOADED_ENTRY_POINTS.discard(key)
                    _ENTRY_POINTS_SCANNED = False
                    raise ImportError(f"failed to load kuration plugin entry point {key!r}: {e}") from e
                module = target.__name__ if inspect.ismodule(target) \
                    else getattr(target, "__module__", None)
                if module:
                    _LOADED_MODULES.add(module)
                changed = _changed_entries(previous_entries)
                for component_key in changed:
                    _ENTRY_POINT_BASELINES.setdefault(
                        component_key, previous_entries.get(component_key))
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


def plugin_provenance(*, registry=None) -> list[dict]:
    """Host-independent code fingerprints for loaded plugin implementation modules."""
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
    return out
