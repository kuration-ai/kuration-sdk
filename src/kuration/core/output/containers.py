"""Shard containers — how a shard's modality arrays land on disk.

Four formats, same semantics: arrays are written as-shaped `(L, *modality_shape)` plus the
real (pre-pad) `n_steps` in-band, so a variable-length loader needs no manifest. `npz`/`json`
are dependency-free; `hdf5` needs the `[array]` extra, `parquet` the `[parquet]` extra (both
lazily imported, loud if missing). Plugging in a new container is one branch here.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from ..io import staged_write
from ..registry import default_registry as _default_registry

# Final extension per container — lets the writer check a shard's existence for resume.
CONTAINER_EXT = {"npz": ".npz", "json": ".json", "hdf5": ".h5", "parquet": ".parquet"}


class Container:
    """Third-party container contract: extension plus write/read callables."""
    def __init__(self, extension, writer, reader):
        if (not isinstance(extension, str)
                or re.fullmatch(r"\.[A-Za-z0-9][A-Za-z0-9._-]*", extension) is None):
            raise ValueError(
                "container extension must be one filename suffix such as '.bin' or "
                "'.array.v1' (letters, digits, dot, underscore, and hyphen only)"
            )
        self.extension, self.writer, self.reader = extension, writer, reader


def register_container(name, container=None, *, extension=None, writer=None, reader=None,
                       **metadata):
    """Register a namespaced shard container without editing SDK dispatch code."""
    from ..registry import register_container as _register
    obj = container or Container(extension, writer, reader)
    if not all(callable(getattr(obj, x, None)) for x in ("writer", "reader")):
        raise ValueError("container needs callable writer and reader")
    _register(name, obj, **metadata)


def _container_handler(name: str, registry=None):
    from ..registry import default_registry
    registry = registry or default_registry
    handler = registry.get("container", name)
    extension = getattr(handler, "extension", None)
    if not (isinstance(extension, str)
            and re.fullmatch(r"\.[A-Za-z0-9][A-Za-z0-9._-]*", extension) is not None
            and callable(getattr(handler, "writer", None))
            and callable(getattr(handler, "reader", None))):
        raise ValueError(
            f"registered container {name!r} must be a Container-like object with "
            "one safe filename extension plus callable writer and reader")
    return handler


def container_extension(name: str, *, registry=None) -> str:
    if name in CONTAINER_EXT:
        return CONTAINER_EXT[name]
    return _container_handler(name, registry).extension


def validate_container_availability(name: str, *, registry=None) -> None:
    """Fail before publication when a container implementation cannot run."""
    if name == "hdf5":
        try:
            import h5py  # noqa: F401
        except ImportError as error:
            raise ImportError("container 'hdf5' needs h5py: "
                              "pip install 'kuration-sdk[array]'") from error
    elif name == "parquet":
        try:
            import pyarrow  # noqa: F401
            import pyarrow.parquet  # noqa: F401
        except ImportError as error:
            raise ImportError("container 'parquet' needs pyarrow: "
                              "pip install 'kuration-sdk[parquet]'") from error
    elif name not in CONTAINER_EXT:
        _container_handler(name, registry)


def write_shard(path_no_ext: str, container: str, arrays: dict[str, np.ndarray],
                n_steps: int, *, registry=None) -> str:
    """Write one shard's modality arrays; returns the written path (local or remote URL).
    All formats stage through `staged_write`, so a `gs://`/`s3://`/… destination just works."""
    if container not in CONTAINER_EXT:
        handler = _container_handler(container, registry)
        path = path_no_ext + handler.extension
        with staged_write(path) as local:
            handler.writer(local, arrays, n_steps)
        return path
    path = path_no_ext + CONTAINER_EXT[container]
    with staged_write(path) as local:    # real local path; uploaded on exit if `path` is remote
        if container == "npz":
            np.savez(local, _n_steps=np.asarray(n_steps), **arrays)  # type: ignore[arg-type]
        elif container == "json":
            Path(local).write_text(json.dumps(
                {"_n_steps": int(n_steps),
                 "modalities": {k: v.tolist() for k, v in arrays.items()}}))
        elif container == "hdf5":
            try:
                import h5py
            except ImportError as e:
                raise ImportError("container 'hdf5' needs h5py: "
                                  "pip install 'kuration-sdk[array]'") from e
            with h5py.File(local, "w") as f:
                for k, v in arrays.items():
                    f.create_dataset(k, data=v, compression="gzip")
                f.attrs["n_steps"] = int(n_steps)
        else:  # parquet
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as e:
                raise ImportError("container 'parquet' needs pyarrow: "
                                  "pip install 'kuration-sdk[parquet]'") from e
            # scalar modality -> column of values; tensor modality -> column of flattened lists.
            cols = {k: (pa.array(v) if v.ndim == 1
                        else pa.array([row.reshape(-1).tolist() for row in v]))
                    for k, v in arrays.items()}
            table = pa.table(cols).replace_schema_metadata({"_n_steps": str(int(n_steps))})
            pq.write_table(table, local)
    return path


def read_shard(path: str, container: str, *, registry=None) -> tuple[dict[str, np.ndarray], int]:
    """Inverse of `write_shard`: load a shard's modality arrays + real `n_steps` from any
    container. Remote shards (gs://, s3://, …) are staged to a tempfile. Arrays come back
    as stored — a tensor modality written to Parquet is flat `(L, prod(shape))`; reshape it
    with the manifest's per-modality `shape` if you need the original rank (the loaders do)."""
    if container not in CONTAINER_EXT:
        from ..io import local_copy
        with local_copy(path) as local:
            return _container_handler(container, registry).reader(local)
    from ..io import local_copy, read_text
    if container == "json":
        data = json.loads(read_text(path))
        return {k: np.asarray(v) for k, v in data["modalities"].items()}, int(data["_n_steps"])
    with local_copy(path) as p:
        if container == "npz":
            with np.load(p, allow_pickle=False) as z:
                # Materialize before the archive (and a possible remote tempfile) is closed.
                arrays = {k: np.asarray(z[k]) for k in z.files if k != "_n_steps"}
                n_steps = int(z["_n_steps"])
            return arrays, n_steps
        if container == "hdf5":
            try:
                import h5py
            except ImportError as e:
                raise ImportError("reading hdf5 shards needs h5py: "
                                  "pip install 'kuration-sdk[array]'") from e
            with h5py.File(p, "r") as f:
                return {k: f[k][:] for k in f.keys()}, int(f.attrs["n_steps"])
        # parquet
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError("reading parquet shards needs pyarrow: "
                              "pip install 'kuration-sdk[parquet]'") from e
        t = pq.read_table(p)
        meta = t.schema.metadata or {}
        if b"_n_steps" not in meta:
            raise ValueError(
                f"parquet shard {path!r} has no '_n_steps' schema metadata — not written by "
                f"kuration, or rewritten by a tool that drops custom metadata")
        n = int(meta[b"_n_steps"])
        return {name: np.asarray(t.column(name).to_pylist()) for name in t.column_names}, n


for _name in CONTAINER_EXT:
    if _name not in _default_registry.names("container"):
        _default_registry.register("container", _name, write_shard,
                                   capabilities={"remote_paths": True, "streaming": True})
