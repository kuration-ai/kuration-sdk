"""NPZ input adapter — read a numpy `.npz` file as a Tick stream (no extra deps).

One file = one source, one entity; each mapped key holds `(T, *shape)`. The dependency-free
counterpart to the HDF5 adapter.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..buffer import Tick
from ..schema import SchemaSpec
from .base import arrays_to_ticks, register_adapter


class NpzAdapter:
    def __init__(self, schema: SchemaSpec, *, fields: dict, globals: Optional[dict] = None):
        self.schema = schema
        self.fields = fields                 # schema_field -> npz key
        self.globals = globals or {}
        for f in fields:
            if f not in schema.fields:
                raise ValueError(f"field map references undeclared schema field {f!r}")

    def ticks(self, source_uri: str) -> list[Tick]:
        from ..io import local_copy
        with local_copy(source_uri) as p:
            z = np.load(p, allow_pickle=False)
            arrs = {fn: z[key] for fn, key in self.fields.items()}
            gl = {gn: z[key] for gn, key in self.globals.items()}
        return arrays_to_ticks(self.schema, arrs, gl)


register_adapter("npz", NpzAdapter)
