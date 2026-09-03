"""Shared collision-safe idempotence for built-in component registration."""
from __future__ import annotations


def needs_registration(registry, kind: str, name: str, component) -> bool:
    """Return whether a component is absent, accepting only exact repeat registration."""
    if name not in registry.names(kind):
        return True
    if registry.get(kind, name) is component:
        return False
    descriptor = registry.descriptor(kind, name)
    owner = descriptor.module or descriptor.package or "an unknown implementation"
    raise ValueError(
        f"cannot register {kind} {name!r}: the name is already owned by {owner!r}"
    )
