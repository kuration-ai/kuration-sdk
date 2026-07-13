"""`InputAdapter` — the input seam. The one place domain knowledge about a raw format
enters the SDK.

An adapter's contract is deliberately tiny: turn a raw source into an ordered stream of
canonical `Tick`s (parse + group-by-timestep). Everything after that — building the
`(T,E,D)` buffer, segmentation, metrics, selection — is generic SDK code that never knows
which dataset it's looking at. Streaming-capable adapters expose `iter_ticks`; the compatible
`ticks` path materializes a list for callers that need random access.

Adapters are registered by name for declarative job construction. The parallel feature pass
pickles the configured adapter into spawned workers.
"""
from __future__ import annotations

from typing import Callable, Iterable, Protocol, runtime_checkable

import numpy as np

from ..buffer import Tick
from ..schema import SchemaSpec


@runtime_checkable
class InputAdapter(Protocol):
    schema: SchemaSpec
    def ticks(self, source_uri: str) -> Iterable[Tick]: ...


def iter_arrays_to_ticks(schema: SchemaSpec, arrs: dict, gl: dict = None):
    """Lazily build single-entity ticks from indexable per-field arrays."""
    gl = gl or {}
    n = len(next(iter(arrs.values()))) if arrs else (len(next(iter(gl.values()))) if gl else 0)
    mismatched = {k: len(a) for k, a in {**arrs, **gl}.items() if len(a) != n}
    if mismatched:                       # silent truncation to the first field's length is worse
        raise ValueError(f"arrays_to_ticks: length mismatch — expected {n} steps, got "
                         f"{mismatched}")
    for t in range(n):
        ent: dict[str, object] = {}
        for fn, arr in arrs.items():
            fs = schema.fields[fn]
            x = arr[t]
            if fs.kind == "vector":
                ent[fn] = tuple(float(v) for v in np.ravel(x))
            elif fs.kind == "categorical":
                ent[fn] = x.decode() if isinstance(x, bytes) else str(x)
            else:
                ent[fn] = float(np.ravel(x)[0]) if np.ndim(x) else float(x)
        g = {gn: float(np.ravel(arr[t])[0]) for gn, arr in gl.items()}
        yield Tick(time=t, entities={0: ent}, globals=g)


def arrays_to_ticks(schema: SchemaSpec, arrs: dict, gl: dict = None) -> list:
    """Materialized counterpart to `iter_arrays_to_ticks` for existing callers."""
    return list(iter_arrays_to_ticks(schema, arrs, gl))


def check_mapped_arrays(schema: SchemaSpec, fields: dict, globals_: dict, source_uri: str,
                        shape_of: Callable) -> None:
    """Metadata-level mirror of `arrays_to_ticks`'s run-time checks, shared by the array-store
    adapters' `check_source` hooks. `shape_of(key)` returns the stored array's shape, or None
    if the key doesn't exist — shapes are store metadata, so no array data is read. Checks:
    every mapped key exists, all first dims agree (step count), and each field's per-step
    width matches the schema (a wide array mapped to a scalar would silently truncate to its
    first component at run time — loud beats masked)."""
    import math
    keys = {**fields, **globals_}                # mapped name -> store key
    shapes, missing = {}, []
    for key in keys.values():
        shp = shape_of(key)
        if shp is None:
            missing.append(key)
        else:
            shapes[key] = tuple(shp)
    if missing:
        raise ValueError(f"source {source_uri!r} is missing mapped key(s) {sorted(missing)}")
    lens = {k: (s[0] if s else 0) for k, s in shapes.items()}
    if len(set(lens.values())) > 1:
        raise ValueError(f"source {source_uri!r}: mapped arrays disagree on step count: {lens}")
    for f, key in fields.items():
        fs = schema.fields[f]
        if fs.kind == "categorical":             # string/object arrays — width has no meaning
            continue
        width = math.prod(shapes[key][1:])
        if width != fs.width:
            raise ValueError(f"source {source_uri!r}: array {key!r} has per-step width {width}, "
                             f"but field {f!r} ({fs.kind}) expects {fs.width}")


_ADAPTERS: dict[str, Callable[..., InputAdapter]] = {}


def register_adapter(name: str, factory: Callable[..., InputAdapter], **metadata) -> None:
    if name in _ADAPTERS:
        raise ValueError(f"adapter {name!r} already registered")
    _ADAPTERS[name] = factory
    from ..registry import default_registry
    default_registry.register("adapter", name, factory, **metadata)


def get_adapter(name: str, *, registry=None, **kwargs) -> InputAdapter:
    if registry is not None:
        return registry.get("adapter", name)(**kwargs)
    if name not in _ADAPTERS:
        from ..plugins import load_plugins
        load_plugins()
    if name not in _ADAPTERS:
        raise ValueError(f"unknown adapter {name!r}; registered: {sorted(_ADAPTERS)}")
    return _ADAPTERS[name](**kwargs)
