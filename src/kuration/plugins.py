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

ENTRY_POINT_GROUP = "kuration.plugins"
_LOADED_MODULES: set[str] = set()
_LOADED_ENTRY_POINTS: set[str] = set()
_ENTRY_POINTS_SCANNED = False
_LOCK = RLock()


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
            _LOADED_MODULES.add(name)
            try:
                importlib.import_module(name)
            except Exception as e:
                _LOADED_MODULES.discard(name)
                raise ImportError(f"failed to load kuration plugin module {name!r}: {e}") from e
            loaded.append({"kind": "module", "name": name})

        if include_entry_points and not _ENTRY_POINTS_SCANNED:
            _ENTRY_POINTS_SCANNED = True
            for ep in _available_entry_points():
                key = f"{ep.name}={ep.value}"
                if key in _LOADED_ENTRY_POINTS:
                    continue
                _LOADED_ENTRY_POINTS.add(key)
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
                loaded.append({"kind": "entry_point", "name": ep.name, "value": ep.value})
    return loaded


def loaded_plugin_modules() -> tuple[str, ...]:
    """Return loaded plugin module names in deterministic sorted order."""
    return tuple(sorted(_LOADED_MODULES))


def plugin_provenance() -> list[dict]:
    """Host-independent code fingerprints for loaded plugin modules."""
    out = []
    distributions = packages_distributions()
    for name in loaded_plugin_modules():
        module = sys.modules.get(name)
        path = getattr(module, "__file__", None)
        digest = None
        if path:
            try:
                with open(path, "rb") as f:
                    digest = hashlib.sha256(f.read()).hexdigest()
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
