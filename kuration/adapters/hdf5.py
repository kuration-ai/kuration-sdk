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
from .base import arrays_to_ticks, check_mapped_arrays, register_adapter


class HDF5Adapter:
    def __init__(self, schema: SchemaSpec, *, fields: dict, globals: Optional[dict] = None):
        self.schema = schema
        self.fields = fields                 # schema_field -> hdf5 dataset name
        self.globals = globals or {}
        for f in fields:
            if f not in schema.fields:
                raise ValueError(f"field map references undeclared schema field {f!r}")

    def ticks(self, source_uri: str) -> list[Tick]:
        try:
            import h5py
        except ImportError as e:
            raise ImportError("HDF5Adapter needs h5py: pip install -e '.[array]'") from e
        from ..io import local_copy
        with local_copy(source_uri) as p, h5py.File(p, "r") as f:   # remote staged to tempfile
            arrs = {fn: f[ds][:] for fn, ds in self.fields.items()}
            gl = {gn: f[ds][:] for gn, ds in self.globals.items()}
        return arrays_to_ticks(self.schema, arrs, gl)

    def check_source(self, source_uri: str) -> None:
        """Preflight one file from metadata only (dataset names + shapes, no array reads):
        every mapped dataset exists, first dims agree, and vector datasets match their
        declared dims — so a key typo fails at `kur-validate --check-sources` time."""
        try:
            import h5py
        except ImportError as e:
            raise ImportError("HDF5Adapter needs h5py: pip install -e '.[array]'") from e
        from ..io import local_copy
        with local_copy(source_uri) as p, h5py.File(p, "r") as f:
            check_mapped_arrays(self.schema, self.fields, self.globals, source_uri,
                                lambda ds: f[ds].shape if ds in f else None)


register_adapter("hdf5", HDF5Adapter)
