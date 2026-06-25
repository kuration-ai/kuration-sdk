"""`InputAdapter` — the input seam. The one place domain knowledge about a raw format
enters the SDK.

An adapter's contract is deliberately tiny: turn a raw source into an ordered stream of
canonical `Tick`s (parse + group-by-timestep). Everything after that — building the
`(T,E,D)` buffer, segmentation, metrics, selection — is generic SDK code that never knows
which dataset it's looking at. The stream is an `Iterable`, so a future streaming segmenter
can consume it lazily; the in-memory path materializes it.

Adapters are registered by name so worker processes can reconstruct them (the SDK's
parallel feature pass rebuilds the adapter from its registered name + config rather than
pickling a live object — the same pattern the GCS client uses).
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


def arrays_to_ticks(schema: SchemaSpec, arrs: dict, gl: dict = None) -> list:
    """Build single-entity ticks from per-field `(T, *shape)` arrays — shared by the HDF5 and
    NPZ adapters. Scalar -> float, vector -> tuple, categorical -> str (bytes decoded)."""
    gl = gl or {}
    n = len(next(iter(arrs.values()))) if arrs else (len(next(iter(gl.values()))) if gl else 0)
    out = []
    for t in range(n):
        ent = {}
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
        out.append(Tick(time=t, entities={0: ent}, globals=g))
    return out


_ADAPTERS: dict[str, Callable[..., InputAdapter]] = {}


def register_adapter(name: str, factory: Callable[..., InputAdapter]) -> None:
    if name in _ADAPTERS:
        raise ValueError(f"adapter {name!r} already registered")
    _ADAPTERS[name] = factory


def get_adapter(name: str, **kwargs) -> InputAdapter:
    if name not in _ADAPTERS:
        raise ValueError(f"unknown adapter {name!r}; registered: {sorted(_ADAPTERS)}")
    return _ADAPTERS[name](**kwargs)
