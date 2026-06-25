"""Curated-dataset-as-source — read kuration's own output back in as a source.

This closes the loop: a dataset written by `write_dataset` becomes a source you can re-curate.
That enables composition (mix/split several curated datasets) and DSL-driven introspection
(segment the shards, compute metrics, and `filter` to clips matching criteria).

Each shard is treated as one source (so segmentation looks *within* a clip), and each output
modality becomes a schema field — so the DSL runs over the modalities. Higher-rank tensor
modalities (e.g. images) aren't representable as DSL fields and are skipped; pass an explicit
`schema` to override the derived one. Containers: npz / json (HDF5 shard containers later).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..buffer import Tick
from ..schema import SchemaSpec
from .base import register_adapter


def schema_from_output(output_spec: dict, *, name: str = "curated",
                       sample_hz: Optional[float] = None, entity_axis: str = "e") -> SchemaSpec:
    """Derive a SchemaSpec whose fields are the dataset's scalar/1-D-vector modalities."""
    fields: dict = {}
    for mname, m in output_spec["modalities"].items():
        shape = list(m.get("shape") or [])
        if len(shape) == 0:
            fields[mname] = {"kind": "scalar"}
        elif len(shape) == 1:
            fields[mname] = {"kind": "vector", "dims": [f"{mname}{i}" for i in range(shape[0])]}
        # higher-rank tensors are not DSL fields — skipped
    if not fields:
        raise ValueError("no scalar/1-D modalities to derive a schema from; pass schema=...")
    return SchemaSpec.from_dict({"name": name, "entity_axis": entity_axis,
                                 "sample_hz": sample_hz, "fields": fields})


def _load_shard(path: str, container: str):
    """Return (arrays: {modality -> (T, *shape)}, n_steps) for any output container."""
    if container == "npz":
        z = np.load(path)
        return {k: z[k] for k in z.files if not k.startswith("_")}, int(z["_n_steps"])
    if container == "json":
        data = json.loads(Path(path).read_text())
        return {k: np.asarray(v) for k, v in data["modalities"].items()}, int(data["_n_steps"])
    if container == "hdf5":
        import h5py
        with h5py.File(path, "r") as f:
            return {k: f[k][:] for k in f.keys()}, int(f.attrs["n_steps"])
    if container == "parquet":
        import pyarrow.parquet as pq
        t = pq.read_table(path)
        n = int((t.schema.metadata or {}).get(b"_n_steps", b"0"))
        return {name: np.asarray(t.column(name).to_pylist()) for name in t.column_names}, n
    raise ValueError(f"unknown container {container!r}")


class CuratedDatasetAdapter:
    """Wrap one or more kuration dataset directories as a re-curatable source."""

    def __init__(self, dataset_dir: Union[str, list], schema: Optional[SchemaSpec] = None):
        dirs = [dataset_dir] if isinstance(dataset_dir, str) else list(dataset_dir)
        self._container: dict = {}     # shard path -> container
        self.dataset_of: dict = {}     # shard path -> originating dataset dir
        self._shards: list = []
        first_cfg = None
        for d in dirs:
            cfg = json.loads((Path(d) / "config.json").read_text())
            first_cfg = first_cfg or cfg
            container = cfg["output"]["container"]
            man = json.loads((Path(d) / "dataset_manifest.json").read_text())
            for e in man["shards"]:
                self._shards.append(e["path"])
                self._container[e["path"]] = container
                self.dataset_of[e["path"]] = d
        self.schema = schema or schema_from_output(
            first_cfg["output"], sample_hz=(first_cfg.get("schema") or {}).get("sample_hz"))
        self._fields = list(self.schema.fields)

    def sources(self) -> list:
        return list(self._shards)

    def ticks(self, shard_path: str) -> list[Tick]:
        arrays, n = _load_shard(shard_path, self._container.get(shard_path, "npz"))
        out: list[Tick] = []
        for t in range(n):
            ent: dict = {}
            for fn in self._fields:
                if fn in arrays:
                    fs = self.schema.fields[fn]
                    x = arrays[fn][t]
                    ent[fn] = (tuple(float(v) for v in np.ravel(x)) if fs.kind == "vector"
                               else float(np.ravel(x)[0]) if np.ndim(x) else float(x))
            out.append(Tick(time=t, entities={0: ent}))
        return out


register_adapter("curated", CuratedDatasetAdapter)
