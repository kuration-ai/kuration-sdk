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
        if str(m.get("source", "")).startswith("vocab:"):
            fields[mname] = {"kind": "categorical"}
        elif len(shape) == 0:
            fields[mname] = {"kind": "scalar"}
        elif len(shape) == 1:
            fields[mname] = {"kind": "vector", "dims": [f"{mname}{i}" for i in range(shape[0])]}
        # higher-rank tensors are not DSL fields — skipped
    if not fields:
        raise ValueError("no scalar/1-D modalities to derive a schema from; pass schema=...")
    return SchemaSpec.from_dict({"name": name, "entity_axis": entity_axis,
                                 "sample_hz": sample_hz, "fields": fields})


def _load_shard(path: str, container: str, *, registry=None):
    """Return (arrays: {modality -> (T, *shape)}, n_steps) for any output container; remote
    shards (gs://, s3://, …) are staged to a tempfile. Delegates to the canonical reader so
    the read path stays symmetric with `write_shard` (one source of truth)."""
    from ..output.containers import read_shard   # lazy: avoids the adapters<->output cycle
    return read_shard(path, container, registry=registry)


class CuratedDatasetAdapter:
    """Wrap one or more kuration dataset directories as a re-curatable source."""

    @staticmethod
    def detect_source(source_uri: str) -> int:
        from ..io import exists

        root = str(source_uri).rstrip("/")
        return 200 if (
            exists(f"{root}/dataset_manifest.json") and exists(f"{root}/config.json")
        ) else 0

    def __init__(self, dataset_dir: Union[str, list], schema: Optional[SchemaSpec] = None,
                 registry=None):
        dirs = [dataset_dir] if isinstance(dataset_dir, str) else list(dataset_dir)
        if not dirs:
            raise ValueError("CuratedDatasetAdapter needs at least one dataset dir")
        self._container: dict = {}     # shard path -> container
        self._registry = registry
        self._config: dict = {}        # shard path -> originating resolved config
        self._entries: dict = {}       # shard path -> rebased manifest entry
        self._snapshot: dict = {}      # shard path -> commit-pinned originating generation
        self.dataset_of: dict = {}     # shard path -> originating dataset dir
        self._shards: list = []
        from ..loaders import _resolve_operation_snapshot
        loaded = []
        for directory in dirs:
            snapshot = _resolve_operation_snapshot(str(directory).rstrip("/"), None)
            if snapshot.config is None:
                raise ValueError(f"curated dataset {directory!r} has no config.json")
            loaded.append((directory, snapshot))
        first_cfg = loaded[0][1].config
        assert first_cfg is not None
        self.schema = schema or schema_from_output(
            first_cfg["output"], sample_hz=(first_cfg.get("schema") or {}).get("sample_hz"))
        self._fields = list(self.schema.fields)
        reference_modalities = first_cfg["output"].get("modalities") or {}
        from ..loaders import _assert_snapshot_current, _snapshot_entries
        for d, snapshot in loaded:
            cfg = snapshot.config
            assert cfg is not None
            missing = [f for f in self._fields
                       if f not in (cfg["output"].get("modalities") or {})]
            if missing:
                raise ValueError(
                    f"dataset {d!r} lacks modalities {missing} required by the schema "
                    f"(derived from {dirs[0]!r}); pass an explicit common schema=")
            incompatible = []
            current_modalities = cfg["output"].get("modalities") or {}
            for field_name in self._fields:
                expected = reference_modalities.get(field_name) or {}
                actual = current_modalities.get(field_name) or {}
                for key in ("shape", "dtype", "source"):
                    if actual.get(key) != expected.get(key):
                        incompatible.append((field_name, key, expected.get(key), actual.get(key)))
            if incompatible:
                raise ValueError(f"dataset {d!r} has incompatible modality contracts: "
                                 f"{incompatible}")
            # Rebase manifest paths onto the dir the caller gave us — a moved/copied
            # dataset's recorded (write-time) paths would otherwise dangle.
            for e in _snapshot_entries(snapshot):
                self._shards.append(e["path"])
                self._container[e["path"]] = snapshot.container
                self._config[e["path"]] = cfg
                self._entries[e["path"]] = e
                self._snapshot[e["path"]] = snapshot
                self.dataset_of[e["path"]] = d
            _assert_snapshot_current(snapshot, operation="curated adapter construction")
        for _directory, snapshot in loaded:
            _assert_snapshot_current(snapshot, operation="curated adapter construction")
        self._datasets = [str(d) for d in dirs]

    def sources(self) -> list:
        return list(self._shards)

    def datasets(self) -> list:
        """The written dataset directories behind this adapter's shard sources.

        Sources are shards, so an artifact-level consumer (a quality policy, an analyzer's
        ``check_dataset`` hook) cannot recover the dataset from them. Declaring the artifacts
        here is the seam ``Pipeline.analyze()``/``Pipeline.check()`` use to reach those hooks.
        """
        return list(self._datasets)

    def check_source(self, shard_path: str) -> None:
        """Preflight one shard: it must come from this adapter's manifests and still exist on
        disk (catches a dataset dir that moved after the manifest was read). Modalities were
        already validated against each dataset's config at construction; shard contents are
        kuration's own writes, so no data read here."""
        if shard_path not in self._container:
            raise KeyError(f"unknown shard {shard_path!r} — not in this adapter's manifests")
        from ..loaders import _assert_snapshot_current, _validate_snapshot_shard
        snapshot = self._snapshot[shard_path]
        _assert_snapshot_current(snapshot, operation="curated source preflight")
        _validate_snapshot_shard(snapshot, self._entries[shard_path])
        from ..io import exists
        if not exists(shard_path):
            raise ValueError(f"manifest shard {shard_path!r} does not exist — was the dataset "
                             f"dir moved or partially deleted?")
        _assert_snapshot_current(snapshot, operation="curated source preflight")

    def ticks(self, shard_path: str) -> list[Tick]:
        if shard_path not in self._container:
            raise KeyError(f"unknown shard {shard_path!r} — not in this adapter's manifests")
        from ..loaders import _assert_snapshot_current, _validate_snapshot_shard
        snapshot = self._snapshot[shard_path]
        _assert_snapshot_current(snapshot, operation="curated shard read")
        entry = self._entries[shard_path]
        _validate_snapshot_shard(snapshot, entry)
        if entry.get("n_steps") is None:             # legacy manifest without in-band contract
            arrays, n = _load_shard(shard_path, self._container[shard_path],
                                    registry=self._registry)
        else:
            from ..loaders import read_shard_arrays
            arrays = read_shard_arrays(entry, self._container[shard_path], trim=True,
                                       registry=self._registry, verify=True)
            n = int(entry["n_steps"])
        _validate_snapshot_shard(snapshot, entry)
        _assert_snapshot_current(snapshot, operation="curated shard read")
        missing = [f for f in self._fields if f not in arrays]
        if missing:
            raise KeyError(f"shard {shard_path!r} lacks modalities {missing} required by "
                           f"the schema")
        out: list[Tick] = []
        present = np.asarray(arrays.get("_present", np.ones(n, dtype=np.bool_)))[:n]
        if present.shape != (n,):
            raise ValueError(f"shard {shard_path!r} has invalid _present shape {present.shape}")
        output = self._config[shard_path]["output"]
        modality_specs = output.get("modalities") or {}
        vocabs = output.get("vocabs") or {}
        for t in range(n):
            if not bool(present[t]):
                out.append(Tick(time=t, entities={}))
                continue
            ent: dict = {}
            for fn in self._fields:
                fs = self.schema.fields[fn]
                x = arrays[fn][t]
                if fs.kind == "vector":
                    ent[fn] = tuple(float(v) for v in np.ravel(x))
                elif fs.kind == "categorical":
                    source = str((modality_specs.get(fn) or {}).get("source", ""))
                    vocab_name = source.split(":", 1)[1] if source.startswith("vocab:") else fn
                    inverse = {int(code): str(label) for label, code in
                               (vocabs.get(vocab_name) or {}).items()}
                    code = int(np.ravel(x)[0])
                    if code not in inverse:
                        raise ValueError(f"shard {shard_path!r} categorical {fn!r} has unknown "
                                         f"vocabulary code {code}")
                    ent[fn] = inverse[code]
                else:
                    ent[fn] = float(np.ravel(x)[0]) if np.ndim(x) else float(x)
            out.append(Tick(time=t, entities={0: ent}))
        _assert_snapshot_current(snapshot, operation="curated shard read")
        return out


register_adapter("curated", CuratedDatasetAdapter)
