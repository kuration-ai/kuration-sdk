"""Shard containers — how a shard's modality arrays land on disk.

Four formats, same semantics: arrays are written as-shaped `(L, *modality_shape)` plus the
real (pre-pad) `n_steps` in-band, so a variable-length loader needs no manifest. `npz`/`json`
are dependency-free; `hdf5` needs the `[array]` extra, `parquet` the `[parquet]` extra (both
lazily imported, loud if missing). Plugging in a new container is one branch here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..io import staged_write

# Final extension per container — lets the writer check a shard's existence for resume.
CONTAINER_EXT = {"npz": ".npz", "json": ".json", "hdf5": ".h5", "parquet": ".parquet"}


def write_shard(path_no_ext: str, container: str, arrays: dict[str, np.ndarray],
                n_steps: int) -> str:
    """Write one shard's modality arrays; returns the written path (local or remote URL).
    All formats stage through `staged_write`, so a `gs://`/`s3://`/… destination just works."""
    if container not in CONTAINER_EXT:
        raise ValueError(f"unknown container {container!r}")
    path = path_no_ext + CONTAINER_EXT[container]
    with staged_write(path) as local:    # real local path; uploaded on exit if `path` is remote
        if container == "npz":
            np.savez(local, _n_steps=np.asarray(n_steps), **arrays)
        elif container == "json":
            Path(local).write_text(json.dumps(
                {"_n_steps": int(n_steps),
                 "modalities": {k: v.tolist() for k, v in arrays.items()}}))
        elif container == "hdf5":
            try:
                import h5py
            except ImportError as e:
                raise ImportError("container 'hdf5' needs h5py: pip install -e '.[array]'") from e
            with h5py.File(local, "w") as f:
                for k, v in arrays.items():
                    f.create_dataset(k, data=v, compression="gzip")
                f.attrs["n_steps"] = int(n_steps)
        else:  # parquet
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as e:
                raise ImportError("container 'parquet' needs pyarrow: pip install -e '.[parquet]'") from e
            # scalar modality -> column of values; tensor modality -> column of flattened lists.
            cols = {k: (pa.array(v) if v.ndim == 1
                        else pa.array([row.reshape(-1).tolist() for row in v]))
                    for k, v in arrays.items()}
            table = pa.table(cols).replace_schema_metadata({"_n_steps": str(int(n_steps))})
            pq.write_table(table, local)
    return path


def read_shard(path: str, container: str) -> tuple[dict[str, np.ndarray], int]:
    """Inverse of `write_shard`: load a shard's modality arrays + real `n_steps` from any
    container. Remote shards (gs://, s3://, …) are staged to a tempfile. Arrays come back
    as stored — a tensor modality written to Parquet is flat `(L, prod(shape))`; reshape it
    with the manifest's per-modality `shape` if you need the original rank (the loaders do)."""
    if container not in CONTAINER_EXT:
        raise ValueError(f"unknown container {container!r}")
    from ..io import local_copy, read_text
    if container == "json":
        data = json.loads(read_text(path))
        return {k: np.asarray(v) for k, v in data["modalities"].items()}, int(data["_n_steps"])
    with local_copy(path) as p:
        if container == "npz":
            z = np.load(p)
            # exclude exactly the in-band length key — a modality may legitimately start with '_'
            return {k: z[k] for k in z.files if k != "_n_steps"}, int(z["_n_steps"])
        if container == "hdf5":
            try:
                import h5py
            except ImportError as e:
                raise ImportError("reading hdf5 shards needs h5py: pip install -e '.[array]'") from e
            with h5py.File(p, "r") as f:
                return {k: f[k][:] for k in f.keys()}, int(f.attrs["n_steps"])
        # parquet
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError("reading parquet shards needs pyarrow: pip install -e '.[parquet]'") from e
        t = pq.read_table(p)
        meta = t.schema.metadata or {}
        if b"_n_steps" not in meta:
            raise ValueError(
                f"parquet shard {path!r} has no '_n_steps' schema metadata — not written by "
                f"kuration, or rewritten by a tool that drops custom metadata")
        n = int(meta[b"_n_steps"])
        return {name: np.asarray(t.column(name).to_pylist()) for name in t.column_names}, n
