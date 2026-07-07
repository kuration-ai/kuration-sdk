"""Curated-dataset-as-source — read kuration's own output back in as a source.

This closes the loop: a dataset written by `write_dataset` becomes a source you can re-curate.
That enables composition (mix/split several curated datasets) and DSL-driven introspection
(segment the shards, compute metrics, and `filter` to clips matching criteria).

Each shard is treated as one source (so segmentation looks *within* a clip), and each output
modality becomes a schema field — so the DSL runs over the modalities. Higher-rank tensor
modalities (e.g. images) aren't representable as DSL fields and are skipped; pass an explicit
`schema` to override the derived one. All four containers (npz/json/hdf5/parquet) re-ingest;
a moved/copied dataset dir works (stale manifest paths are rebased onto the given dir).
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
    """Return (arrays: {modality -> (T, *shape)}, n_steps) for any output container; remote
    shards (gs://, s3://, …) are staged to a tempfile. Delegates to the canonical reader so
    the read path stays symmetric with `write_shard` (one source of truth)."""
    from ..output.containers import read_shard   # lazy: avoids the adapters<->output cycle
    return read_shard(path, container)


class CuratedDatasetAdapter:
    """Wrap one or more kuration dataset directories as a re-curatable source."""

    def __init__(self, dataset_dir: Union[str, list], schema: Optional[SchemaSpec] = None):
        dirs = [dataset_dir] if isinstance(dataset_dir, str) else list(dataset_dir)
        if not dirs:
            raise ValueError("CuratedDatasetAdapter needs at least one dataset dir")
        self._container: dict = {}     # shard path -> container
        self.dataset_of: dict = {}     # shard path -> originating dataset dir
        self._shards: list = []
        from ..io import read_text
        from ..loaders import rebase_entries
        loaded = [(d, json.loads(read_text(f"{d}/config.json")),
                   json.loads(read_text(f"{d}/dataset_manifest.json"))) for d in dirs]
        first_cfg = loaded[0][1]
        self.schema = schema or schema_from_output(
            first_cfg["output"], sample_hz=(first_cfg.get("schema") or {}).get("sample_hz"))
        self._fields = list(self.schema.fields)
        for d, cfg, man in loaded:
            missing = [f for f in self._fields
                       if f not in (cfg["output"].get("modalities") or {})]
            if missing:
                raise ValueError(
                    f"dataset {d!r} lacks modalities {missing} required by the schema "
                    f"(derived from {dirs[0]!r}); pass an explicit common schema=")
            # Rebase manifest paths onto the dir the caller gave us — a moved/copied
            # dataset's recorded (write-time) paths would otherwise dangle.
            for e in rebase_entries(man["shards"], d):
                self._shards.append(e["path"])
                self._container[e["path"]] = cfg["output"]["container"]
                self.dataset_of[e["path"]] = d

    def sources(self) -> list:
        return list(self._shards)

    def check_source(self, shard_path: str) -> None:
        """Preflight one shard: it must come from this adapter's manifests and still exist on
        disk (catches a dataset dir that moved after the manifest was read). Modalities were
        already validated against each dataset's config at construction; shard contents are
        kuration's own writes, so no data read here."""
        if shard_path not in self._container:
            raise KeyError(f"unknown shard {shard_path!r} — not in this adapter's manifests")
        from ..io import exists
        if not exists(shard_path):
            raise ValueError(f"manifest shard {shard_path!r} does not exist — was the dataset "
                             f"dir moved or partially deleted?")

    def ticks(self, shard_path: str) -> list[Tick]:
        if shard_path not in self._container:
            raise KeyError(f"unknown shard {shard_path!r} — not in this adapter's manifests")
        arrays, n = _load_shard(shard_path, self._container[shard_path])
        missing = [f for f in self._fields if f not in arrays]
        if missing:
            raise KeyError(f"shard {shard_path!r} lacks modalities {missing} required by "
                           f"the schema")
        out: list[Tick] = []
        for t in range(n):
            ent: dict = {}
            for fn in self._fields:
                fs = self.schema.fields[fn]
                x = arrays[fn][t]
                ent[fn] = (tuple(float(v) for v in np.ravel(x)) if fs.kind == "vector"
                           else float(np.ravel(x)[0]) if np.ndim(x) else float(x))
            out.append(Tick(time=t, entities={0: ent}))
        return out


register_adapter("curated", CuratedDatasetAdapter)
