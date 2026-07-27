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
from .base import check_mapped_arrays, iter_arrays_to_ticks, register_adapter


class HDF5Adapter:
    """Map datasets in one HDF5 file to a canonical single-entity tick stream.

    Requires the ``array`` extra. ``fields`` and ``globals`` map schema names to HDF5
    dataset paths; undeclared schema names and incompatible source shapes raise ``ValueError``.
    """

    def __init__(self, schema: SchemaSpec, *, fields: dict, globals: Optional[dict] = None):
        self.schema = schema
        self.fields = fields                 # schema_field -> hdf5 dataset name
        self.globals = globals or {}
        for f in fields:
            if f not in schema.fields:
                raise ValueError(f"field map references undeclared schema field {f!r}")
        for g in self.globals:
            if g not in schema.globals:
                raise ValueError(f"global map references undeclared schema global {g!r}")
        self.available_fields = frozenset(fields)
        self.available_globals = frozenset(self.globals)

    @staticmethod
    def detect_source(source_uri: str) -> int:
        return 100 if str(source_uri).lower().endswith((".h5", ".hdf5")) else 0

    @staticmethod
    def import_arrays(source_uri: str) -> dict[str, np.ndarray]:
        """Return top-level datasets for lossless artifact import.

        A flat array import has no unambiguous modality name for a grouped dataset path, so a
        file holding any nested dataset raises rather than importing a silent subset of it.
        """
        try:
            import h5py
        except ImportError as error:
            raise ImportError(
                "importing HDF5 needs h5py: pip install 'kuration-sdk[array]'"
            ) from error
        from ..io import local_copy

        with local_copy(source_uri) as path, h5py.File(path, "r") as handle:
            nested: list[str] = []

            def _note_nested(name, value):
                if isinstance(value, h5py.Dataset) and "/" in name:
                    nested.append(name)

            handle.visititems(_note_nested)
            if nested:
                listed = ", ".join(sorted(nested)[:8])
                if len(nested) > 8:
                    listed += f", … (+{len(nested) - 8} more)"
                raise ValueError(
                    f"HDF5 source {source_uri!r} holds {len(nested)} nested dataset(s) that a "
                    f"flat array import cannot name: {listed}. Import it with an explicit "
                    "configured adapter (adapter='hdf5' with schema= and a fields map of "
                    "dataset paths), or write the arrays as top-level datasets."
                )
            arrays = {
                name: np.asarray(value)
                for name, value in handle.items()
                if isinstance(value, h5py.Dataset)
            }
        if not arrays:
            raise ValueError(
                f"HDF5 source {source_uri!r} contains no top-level datasets; "
                "use an explicit configured adapter for nested dataset paths"
            )
        return arrays

    def ticks(self, source_uri: str) -> list[Tick]:
        return list(self.iter_ticks(source_uri))

    def iter_ticks(self, source_uri: str):
        try:
            import h5py
        except ImportError as e:
            raise ImportError("HDF5Adapter needs h5py: pip install 'kuration-sdk[array]'") from e
        from ..io import local_copy
        with local_copy(source_uri) as p, h5py.File(p, "r") as f:   # remote staged to tempfile
            arrs = {fn: f[ds] for fn, ds in self.fields.items()}
            gl = {gn: f[ds] for gn, ds in self.globals.items()}
            yield from iter_arrays_to_ticks(self.schema, arrs, gl)

    def check_source(self, source_uri: str) -> None:
        """Preflight one file from metadata only (dataset names + shapes, no array reads):
        every mapped dataset exists, first dims agree, and vector datasets match their
        declared dims — so a key typo fails at `kur-validate --check-sources` time."""
        try:
            import h5py
        except ImportError as e:
            raise ImportError("HDF5Adapter needs h5py: pip install 'kuration-sdk[array]'") from e
        from ..io import local_copy
        with local_copy(source_uri) as p, h5py.File(p, "r") as f:
            check_mapped_arrays(self.schema, self.fields, self.globals, source_uri,
                                lambda ds: f[ds].shape if ds in f else None)


register_adapter("hdf5", HDF5Adapter)
