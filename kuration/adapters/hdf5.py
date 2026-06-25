"""HDF5 input adapter — read an HDF5 file as a canonical Tick stream.

Convention: one file = one source, one entity. Each mapped dataset holds `(T, *shape)`;
timestep t's field value is row t (scalar -> float, vector -> tuple, categorical -> str).
Needs h5py (the optional `[array]` extra).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..buffer import Tick
from ..schema import SchemaSpec
from .base import register_adapter


class HDF5Adapter:
    def __init__(self, schema: SchemaSpec, *, fields: dict, globals: Optional[dict] = None):
        self.schema = schema
        self.fields = fields                 # schema_field -> hdf5 dataset name
        self.globals = globals or {}
        for f in fields:
            if f not in schema.fields:
                raise ValueError(f"field map references undeclared schema field {f!r}")

    def _value(self, fname: str, x):
        fs = self.schema.fields[fname]
        if fs.kind == "vector":
            return tuple(float(v) for v in np.ravel(x))
        if fs.kind == "categorical":
            return x.decode() if isinstance(x, bytes) else str(x)
        return float(np.ravel(x)[0]) if np.ndim(x) else float(x)

    def ticks(self, source_uri: str) -> list[Tick]:
        try:
            import h5py
        except ImportError as e:
            raise ImportError("HDF5Adapter needs h5py: pip install -e '.[array]'") from e
        from ..io import local_copy
        with local_copy(source_uri) as p, h5py.File(p, "r") as f:   # remote staged to tempfile
            arrs = {fn: f[ds][:] for fn, ds in self.fields.items()}
            gl = {gn: f[ds][:] for gn, ds in self.globals.items()}
        T = len(next(iter(arrs.values()))) if arrs else (len(next(iter(gl.values()))) if gl else 0)
        out: list[Tick] = []
        for t in range(T):
            entities = {0: {fn: self._value(fn, arr[t]) for fn, arr in arrs.items()}}
            g = {gn: float(np.ravel(arr[t])[0]) for gn, arr in gl.items()}
            out.append(Tick(time=t, entities=entities, globals=g))
        return out


register_adapter("hdf5", HDF5Adapter)
