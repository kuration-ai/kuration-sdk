"""Zarr input adapter — read a Zarr array store as a canonical Tick stream.

Convention mirrors the HDF5/NPZ adapters: one store = one source, one entity. Each mapped array
holds `(T, *shape)`; timestep t's field value is row t (scalar -> float, vector -> tuple,
categorical -> str). Needs the `[zarr]` extra. Remote stores (`gs://`, `s3://`, …) open via
fsspec (`[cloud]` extra + a backend).
"""
from __future__ import annotations

from typing import Optional

from ..buffer import Tick
from ..schema import SchemaSpec
from .base import check_mapped_arrays, iter_arrays_to_ticks, register_adapter


class ZarrAdapter:
    """Map arrays in one local or remote Zarr store to a canonical tick stream.

    Requires the ``zarr`` extra; remote URLs also require ``cloud`` and the matching fsspec
    backend. ``fields`` and ``globals`` map schema names to store keys.
    """

    def __init__(self, schema: SchemaSpec, *, fields: dict, globals: Optional[dict] = None):
        self.schema = schema
        self.fields = fields                 # schema_field -> zarr array key
        self.globals = globals or {}
        for f in fields:
            if f not in schema.fields:
                raise ValueError(f"field map references undeclared schema field {f!r}")
        for g in self.globals:
            if g not in schema.globals:
                raise ValueError(f"global map references undeclared schema global {g!r}")

    def _open_group(self, source_uri: str):
        try:
            import zarr
        except ImportError as e:
            raise ImportError("ZarrAdapter needs zarr: pip install -e '.[zarr]'") from e
        from ..io import is_remote
        if is_remote(source_uri):
            try:
                import fsspec
            except ImportError as e:
                raise ImportError(
                    f"remote zarr store {source_uri!r} needs fsspec: pip install -e '.[cloud]' "
                    f"(plus the backend, e.g. gcsfs for gs://)") from e
            store = fsspec.get_mapper(str(source_uri))   # remote store via the fsspec backend
        else:
            store = str(source_uri)
        return zarr.open(store, mode="r")

    def ticks(self, source_uri: str) -> list[Tick]:
        return list(self.iter_ticks(source_uri))

    def iter_ticks(self, source_uri: str):
        g = self._open_group(source_uri)
        arrs = {fn: g[key] for fn, key in self.fields.items()}
        gl = {gn: g[key] for gn, key in self.globals.items()}
        yield from iter_arrays_to_ticks(self.schema, arrs, gl)

    def check_source(self, source_uri: str) -> None:
        """Preflight one store from metadata only (array keys + shapes are store metadata —
        no chunk data read): every mapped key exists, first dims agree, widths match the
        schema. Works for local and remote (fsspec) stores alike."""
        g = self._open_group(source_uri)
        check_mapped_arrays(self.schema, self.fields, self.globals, source_uri,
                            lambda key: g[key].shape if key in g else None)


register_adapter("zarr", ZarrAdapter)
