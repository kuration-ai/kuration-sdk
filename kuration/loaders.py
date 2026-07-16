"""Read a curated dataset back as arrays — the consumer side of `write_dataset`.

`write_dataset` produces shards + a `dataset_manifest.json` + a reproducible `config.json`; this
module closes the loop on the *read* side. It is numpy-only on purpose: a JAX/TF/plain-numpy
user iterates here, and the PyTorch `Dataset` (`kuration.torch`) is a thin wrapper over it.

    iter_shards(dataset, split="train") -> yields {modality -> (T, *shape) np.ndarray}

`dataset` may be a dataset directory, a path to its `dataset_manifest.json`, or an already-loaded
manifest dict. The container (npz/json/hdf5/parquet) is read from the sibling `config.json` and
can be overridden. Each shard's arrays are trimmed to its real `n_steps` (padding dropped) and
reshaped to the modality's declared rank — Parquet stores tensor modalities flat, so the
manifest's per-modality `shape` is what restores `(T, *shape)`.
"""
from __future__ import annotations

import json
from typing import Any, Iterator, Optional, Sequence

import numpy as np

from .io import read_text


def _resolve(dataset, container: Optional[str]) -> tuple[dict, str, Optional[str]]:
    """Return (manifest_dict, container, dataset_dir). Accepts a dir, a manifest .json path,
    or a dict (dataset_dir is None for a dict)."""
    if isinstance(dataset, dict):
        man, base = dataset, None
    else:
        s = str(dataset).rstrip("/")
        if s.endswith(".json"):
            man = json.loads(read_text(s))
            base = s.rsplit("/", 1)[0] if "/" in s else "."
        else:
            base = s
            man = json.loads(read_text(f"{base}/dataset_manifest.json"))
    from .versioning import check_version
    check_version(man.get("artifact_format_version"), kind="artifact")
    if container is None and base is not None:
        try:                                     # container lives in the run's config.json
            config_text = read_text(f"{base}/config.json")
        except FileNotFoundError:
            config_text = None                   # legacy manifest with no sibling config
        if config_text is not None:
            cfg = json.loads(config_text)
            check_version(cfg.get("artifact_format_version"), kind="artifact")
            container = (cfg.get("output") or {}).get("container")
    return man, (container or "npz"), base


def rebase_entries(entries: list[dict], base: Optional[str]) -> list[dict]:
    """Manifest entries record shard paths as written (out_dir-prefixed, possibly relative to
    the writing machine's cwd). If the first recorded path is gone but the dataset dir is
    known, rebase every path onto it via the fixed '{split}/{name}' layout — so a moved or
    copied dataset still loads."""
    if not entries or base is None:
        return entries
    from .io import exists
    if exists(entries[0]["path"]):
        return entries
    def _rebase(p: str) -> str:
        tail = "/".join(str(p).replace("\\", "/").split("/")[-2:])
        return f"{base}/{tail}"
    return [{**e, "path": _rebase(e["path"])} for e in entries]


def shard_entries(dataset, *, split: Optional[str] = None,
                  container: Optional[str] = None) -> tuple[list[dict], str]:
    """Return (manifest entries, container), optionally filtered to one split. Shard paths
    are rebased onto the dataset dir when the recorded ones no longer exist."""
    man, cont, base = _resolve(dataset, container)
    entries = rebase_entries(list(man.get("shards", [])), base)
    if split is not None:
        entries = [e for e in entries if e.get("split") == split]
    return entries, cont


def read_shard_arrays(entry: dict, container: str, *,
                      modalities: Optional[Sequence[str]] = None,
                      trim: bool = True) -> dict[str, np.ndarray]:
    """Load one manifest entry's shard into `{modality -> (T, *shape)}`. Reshapes tensor
    modalities to their declared rank (restores Parquet-flattened columns) and, when `trim`,
    drops trailing pad rows so `T == n_steps`."""
    from .output.containers import read_shard
    arrays, n = read_shard(entry["path"], container)
    shapes = entry.get("shape", {})
    dtypes = entry.get("dtype", {})
    names = modalities if modalities is not None else list(arrays)
    out: dict[str, np.ndarray] = {}
    for name in names:
        if name not in arrays:
            continue
        a = np.asarray(arrays[name])
        shp = tuple(shapes.get(name, ()))
        if shp:                                  # restore (L, *shape) — a no-op unless flattened
            a = a.reshape((a.shape[0], *shp))
        dt = dtypes.get(name)
        if dt is not None and a.dtype != np.dtype(dt):
            a = a.astype(dt)                     # json/parquet round-trips widen to float64/int64
        out[name] = a[:n] if trim else a
    return out


def iter_shards(dataset, *, split: Optional[str] = None,
                modalities: Optional[Sequence[str]] = None, trim: bool = True,
                container: Optional[str] = None,
                with_meta: bool = False) -> Iterator[Any]:
    """Iterate a written dataset's shards as numpy arrays.

    Yields `{modality -> (T, *shape)}` per shard (or `(arrays, entry)` when `with_meta=True`,
    where `entry` is the manifest record: path/source_id/entity_id/split/n_steps/padded/…).
    `split` filters to "train"/"test"; `modalities` selects a subset; `trim` drops pad rows."""
    entries, cont = shard_entries(dataset, split=split, container=container)
    for e in entries:
        arrays = read_shard_arrays(e, cont, modalities=modalities, trim=trim)
        yield (arrays, e) if with_meta else arrays
