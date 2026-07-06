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
from .base import arrays_to_ticks, register_adapter


class ZarrAdapter:
    def __init__(self, schema: SchemaSpec, *, fields: dict, globals: Optional[dict] = None):
        self.schema = schema
        self.fields = fields                 # schema_field -> zarr array key
        self.globals = globals or {}
        for f in fields:
            if f not in schema.fields:
                raise ValueError(f"field map references undeclared schema field {f!r}")

    def ticks(self, source_uri: str) -> list[Tick]:
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
        g = zarr.open(store, mode="r")
        arrs = {fn: g[key][:] for fn, key in self.fields.items()}
        gl = {gn: g[key][:] for gn, key in self.globals.items()}
        return arrays_to_ticks(self.schema, arrs, gl)


register_adapter("zarr", ZarrAdapter)
