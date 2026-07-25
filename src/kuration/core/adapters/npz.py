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
    """Map arrays in one NumPy NPZ archive to a canonical single-entity tick stream.

    ``fields`` and ``globals`` map schema names to archive keys. Undeclared schema names,
    missing keys, inconsistent time dimensions, and incompatible shapes raise ``ValueError``.
    NPZ input does not support the streaming execution path.
    """

    def __init__(self, schema: SchemaSpec, *, fields: dict, globals: Optional[dict] = None):
        self.schema = schema
        self.fields = fields                 # schema_field -> npz key
        self.globals = globals or {}
        for f in fields:
            if f not in schema.fields:
                raise ValueError(f"field map references undeclared schema field {f!r}")
        for g in self.globals:
            if g not in schema.globals:
                raise ValueError(f"global map references undeclared schema global {g!r}")
        self.available_fields = frozenset(fields)
        self.available_globals = frozenset(self.globals)

    supports_streaming = False

    @staticmethod
    def detect_source(source_uri: str) -> int:
        return 100 if str(source_uri).lower().endswith(".npz") else 0

    @staticmethod
    def import_arrays(source_uri: str) -> dict[str, np.ndarray]:
        """Return every archive array for lossless artifact import."""
        from ..io import local_copy

        with local_copy(source_uri) as path:
            with np.load(path, allow_pickle=False) as archive:
                return {name: np.asarray(archive[name]) for name in archive.files}

    def ticks(self, source_uri: str) -> list[Tick]:
        from ..io import local_copy
        with local_copy(source_uri) as p:
            with np.load(p, allow_pickle=False) as z:
                # Materialize while the archive and any downloaded temporary file are open.
                arrs = {fn: np.asarray(z[key]) for fn, key in self.fields.items()}
                gl = {gn: np.asarray(z[key]) for gn, key in self.globals.items()}
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
