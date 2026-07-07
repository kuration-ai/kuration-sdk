"""NPZ input adapter — read a numpy `.npz` file as a Tick stream (no extra deps).

One file = one source, one entity; each mapped key holds `(T, *shape)`. The dependency-free
counterpart to the HDF5 adapter.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..buffer import Tick
from ..schema import SchemaSpec
from .base import arrays_to_ticks, check_mapped_arrays, register_adapter


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

    def check_source(self, source_uri: str) -> None:
        """Preflight one .npz from metadata only: member names + `.npy` headers (shape lives
        in the header — no array data decompressed), so a key typo or shape mismatch fails at
        `kur-validate --check-sources` time."""
        import zipfile
        from numpy.lib import format as npf
        from ..io import local_copy
        with local_copy(source_uri) as p, zipfile.ZipFile(p) as z:
            members = {n[:-4] if n.endswith(".npy") else n: n for n in z.namelist()}

            def shape_of(key):
                if key not in members:
                    return None
                with z.open(members[key]) as fh:
                    version = npf.read_magic(fh)
                    if version == (1, 0):
                        shape, _, _ = npf.read_array_header_1_0(fh)
                    elif version == (2, 0):
                        shape, _, _ = npf.read_array_header_2_0(fh)
                    else:
                        raise ValueError(f"source {source_uri!r}: member {key!r} uses .npy "
                                         f"format version {version} — cannot read its header")
                return shape

            check_mapped_arrays(self.schema, self.fields, self.globals, source_uri, shape_of)


register_adapter("npz", NpzAdapter)
