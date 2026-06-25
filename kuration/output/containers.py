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


def write_shard(path_no_ext: str, container: str, arrays: dict[str, np.ndarray],
                n_steps: int) -> str:
    """Write one shard's modality arrays; returns the written path."""
    Path(path_no_ext).parent.mkdir(parents=True, exist_ok=True)
    if container == "npz":
        path = path_no_ext + ".npz"
        np.savez(path, _n_steps=np.asarray(n_steps), **arrays)
        return path
    if container == "json":
        path = path_no_ext + ".json"
        Path(path).write_text(json.dumps(
            {"_n_steps": int(n_steps), "modalities": {k: v.tolist() for k, v in arrays.items()}}))
        return path
    if container == "hdf5":
        try:
            import h5py
        except ImportError as e:
            raise ImportError("container 'hdf5' needs h5py: pip install -e '.[array]'") from e
        path = path_no_ext + ".h5"
        with h5py.File(path, "w") as f:
            for k, v in arrays.items():
                f.create_dataset(k, data=v, compression="gzip")
            f.attrs["n_steps"] = int(n_steps)
        return path
    if container == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError("container 'parquet' needs pyarrow: pip install -e '.[parquet]'") from e
        path = path_no_ext + ".parquet"
        # scalar modality -> a column of values; tensor modality -> a column of flattened lists
        # (reshape on read using the manifest's per-modality shape).
        cols = {k: (pa.array(v) if v.ndim == 1
                    else pa.array([row.reshape(-1).tolist() for row in v]))
                for k, v in arrays.items()}
        table = pa.table(cols).replace_schema_metadata({"_n_steps": str(int(n_steps))})
        pq.write_table(table, path)
        return path
    raise ValueError(f"unknown container {container!r}")
