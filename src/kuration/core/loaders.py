"""Read a curated dataset back as arrays — the consumer side of `write_dataset`.

`write_dataset` produces shards + a `dataset_manifest.json` + a reproducible `config.json`; this
module closes the loop on the *read* side. It is numpy-only on purpose: a JAX/TF/plain-numpy
user iterates here, and the PyTorch `Dataset` (`kuration.core.torch`) is a thin wrapper over it.

    iter_shards(dataset, split="train") -> yields {modality -> (T, *shape) np.ndarray}

`dataset` may be a dataset directory, a path to its `dataset_manifest.json`, or a legacy loaded
manifest dict with self-contained shard paths. The container (npz/json/hdf5/parquet) is read from the sibling `config.json` and
can be overridden. Each shard's arrays are trimmed to its real `n_steps` (padding dropped) and
reshaped to the modality's declared rank — Parquet stores tensor modalities flat, so the
manifest's per-modality `shape` is what restores `(T, *shape)`.
"""
from __future__ import annotations

import json
from typing import Any, Iterator, Optional, Sequence

import numpy as np

from .io import read_text


def _verify_commit(base: str, manifest: dict) -> None:
    protocol = manifest.get("commit_protocol")
    if protocol is None:                         # legacy artifact predating transactional sidecar
        return
    if protocol != "sidecar-v1":
        raise ValueError(f"unsupported dataset commit protocol {protocol!r}")
    from .io import exists, file_integrity
    commit_path = f"{base}/dataset.commit.json"
    if not exists(commit_path):
        raise ValueError(f"dataset {base!r} is incomplete: missing final dataset.commit.json")
    commit = json.loads(read_text(commit_path))
    if commit.get("protocol") != protocol:
        raise ValueError(f"dataset {base!r} has a mismatched commit protocol")
    if commit.get("state") != "committed":
        raise ValueError(f"dataset {base!r} is incomplete: commit state is "
                         f"{commit.get('state')!r}, expected 'committed'")
    for name in ("dataset_manifest.json", "config.json"):
        expected = (commit.get("files") or {}).get(name)
        if not isinstance(expected, dict):
            raise ValueError(f"dataset commit is missing integrity for {name}")
        actual = file_integrity(f"{base}/{name}")
        if actual != expected:
            raise ValueError(f"dataset {base!r} is not atomically committed: {name} changed "
                             "after the commit record was published")


def _resolve(dataset, container: Optional[str]) -> tuple[dict, str, Optional[str]]:
    """Return (manifest_dict, container, dataset_dir). Accepts a dir, a manifest .json path,
    or a dict (dataset_dir is None for a dict)."""
    if isinstance(dataset, dict):
        man, base = dataset, None
        paths = [str(entry.get("path", "")) for entry in (man.get("shards") or [])
                 if isinstance(entry, dict)]
        has_relative = any(path and "://" not in path and not path.startswith("/")
                           for path in paths)
        if man.get("commit_protocol") is not None or has_relative:
            raise ValueError(
                "a portable manifest dict has relative shard paths and/or a commit sidecar; "
                "pass the dataset directory or dataset_manifest.json path so shard paths and "
                "dataset.commit.json can be resolved and verified")
    else:
        s = str(dataset).rstrip("/")
        if s.endswith(".json"):
            man = json.loads(read_text(s))
            base = s.rsplit("/", 1)[0] if "/" in s else "."
        else:
            base = s
            man = json.loads(read_text(f"{base}/dataset_manifest.json"))
        _verify_commit(base, man)
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
    """Resolve manifest shard paths relative to the dataset being loaded.

    Current manifests store ``split/name`` paths.  Legacy manifests stored the writer's full or
    cwd-relative output path.  In either case the dataset directory is authoritative: preferring
    an old path merely because it still exists can make a copied artifact read and even verify the
    original dataset.  Resolve every entry independently so one surviving legacy path cannot mask
    missing siblings.
    """
    if not entries or base is None:
        return entries

    def _rebase(entry: dict) -> dict:
        p = str(entry["path"]).replace("\\", "/")
        split = entry.get("split") or ""
        if not isinstance(split, str):
            raise ValueError(f"manifest shard split must be a string, got {split!r}")
        split = split.replace("\\", "/")
        if split and ("/" in split or split in {".", ".."} or "\x00" in split):
            raise ValueError(f"manifest shard split {split!r} is not a safe path component")
        name = p.rsplit("/", 1)[-1]
        if not name or name in {".", ".."} or "\x00" in name:
            raise ValueError(f"manifest shard path {entry['path']!r} has no safe filename")
        # The split is part of the manifest contract.  Fall back to the legacy final two path
        # components only for manifests that predate the split field.
        if split:
            relative = f"{split}/{name}"
        else:
            legacy_parts = [part for part in p.split("/") if part]
            relative_parts = legacy_parts[-2:]
            if (not relative_parts
                    or any(part in {".", ".."} or "\x00" in part
                           for part in relative_parts)):
                raise ValueError(f"legacy manifest shard path {entry['path']!r} is unsafe")
            relative = "/".join(relative_parts)
        return {**entry, "path": f"{str(base).rstrip('/')}/{relative}"}

    return [_rebase(entry) for entry in entries]


def shard_entries(dataset, *, split: Optional[str] = None,
                  container: Optional[str] = None) -> tuple[list[dict], str]:
    """Return (manifest entries, container), optionally filtered to one split. Shard paths
    are always resolved against the dataset directory for artifact self-containment."""
    man, cont, base = _resolve(dataset, container)
    entries = rebase_entries(list(man.get("shards", [])), base)
    if split is not None:
        entries = [e for e in entries if e.get("split") == split]
    return entries, cont


def read_shard_arrays(entry: dict, container: str, *,
                      modalities: Optional[Sequence[str]] = None,
                      trim: bool = True, registry=None,
                      verify: bool = True) -> dict[str, np.ndarray]:
    """Load one manifest entry's shard into `{modality -> (T, *shape)}`. Reshapes tensor
    modalities to their declared rank (restores Parquet-flattened columns) and, when `trim`,
    drops trailing pad rows so `T == n_steps`. By default, manifest checksums and sizes are
    verified before decoding; pass ``verify=False`` only when that integrity cost is unwanted."""
    if not isinstance(verify, bool):
        raise ValueError("verify must be true or false")
    if verify and (entry.get("sha256") is not None or entry.get("size_bytes") is not None):
        from .io import file_integrity
        actual = file_integrity(entry["path"])
        expected_sha = entry.get("sha256")
        expected_size = entry.get("size_bytes")
        if expected_sha is not None and actual["sha256"] != expected_sha:
            raise ValueError(f"shard {entry['path']!r} checksum mismatch: manifest says "
                             f"{expected_sha}, payload is {actual['sha256']}")
        if expected_size is not None and actual["size_bytes"] != expected_size:
            raise ValueError(f"shard {entry['path']!r} size mismatch: manifest says "
                             f"{expected_size}, payload is {actual['size_bytes']}")
    from .output.containers import read_shard
    arrays, n = read_shard(entry["path"], container, registry=registry)
    shapes = entry.get("shape", {})
    dtypes = entry.get("dtype", {})
    names = modalities if modalities is not None else list(arrays)
    missing = sorted(set(names) - set(arrays))
    if missing:
        raise KeyError(f"shard {entry['path']!r} is missing requested modalities {missing}; "
                       f"available: {sorted(arrays)}")
    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < 0:
        raise ValueError(f"shard {entry['path']!r} has invalid in-band n_steps {n!r}")
    manifest_n = entry.get("n_steps")
    if manifest_n is not None and int(manifest_n) != int(n):
        raise ValueError(f"shard {entry['path']!r} n_steps mismatch: manifest says "
                         f"{manifest_n}, payload says {n}")
    out: dict[str, np.ndarray] = {}
    for name in names:
        a = np.asarray(arrays[name])
        if a.ndim < 1:
            raise ValueError(f"shard {entry['path']!r} modality {name!r} must have a leading "
                             f"time dimension, got shape {a.shape}")
        if int(n) > a.shape[0]:
            raise ValueError(f"shard {entry['path']!r} modality {name!r} contains "
                             f"{a.shape[0]} rows but n_steps is {n}")
        shp = tuple(shapes.get(name, ()))
        expected_width = int(np.prod(shp, dtype=np.int64)) if shp else 1
        actual_width = int(np.prod(a.shape[1:], dtype=np.int64)) if a.ndim > 1 else 1
        if actual_width != expected_width:
            raise ValueError(f"shard {entry['path']!r} modality {name!r} has per-row width "
                             f"{actual_width}, manifest shape {shp} requires {expected_width}")
        if shp:                                  # restore (L, *shape) — needed for flat Parquet
            a = a.reshape((a.shape[0], *shp))
        dt = dtypes.get(name)
        if dt is not None:
            expected_dtype = np.dtype(dt)
            if expected_dtype.hasobject:
                raise ValueError(f"manifest modality {name!r} uses unsafe object dtype")
            if a.dtype != expected_dtype:
                a = a.astype(expected_dtype)     # json/parquet round-trips widen float/int values
        out[name] = a[:n] if trim else a
    return out


def iter_shards(dataset, *, split: Optional[str] = None,
                modalities: Optional[Sequence[str]] = None, trim: bool = True,
                container: Optional[str] = None,
                with_meta: bool = False, registry=None,
                verify: bool = True) -> Iterator[Any]:
    """Iterate a written dataset's shards as numpy arrays.

    Yields `{modality -> (T, *shape)}` per shard (or `(arrays, entry)` when `with_meta=True`,
    where `entry` is the manifest record: path/source_id/entity_id/split/n_steps/padded/…).
    `split` filters to "train"/"test"; `modalities` selects a subset; `trim` drops pad rows.
    Manifest integrity is checked by default; ``verify=False`` explicitly opts out."""
    entries, cont = shard_entries(dataset, split=split, container=container)
    for e in entries:
        arrays = read_shard_arrays(e, cont, modalities=modalities, trim=trim,
                                   registry=registry, verify=verify)
        yield (arrays, e) if with_meta else arrays
