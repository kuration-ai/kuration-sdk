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

from dataclasses import dataclass
import errno
import hashlib
import os
import re
import stat
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np

from .io import open_file
from .jsonutil import parse_json_object
from .output.spec import validate_modality_name


@dataclass(frozen=True)
class _ResolvedDatasetSnapshot:
    """Commit-pinned metadata used by one logical reader operation."""

    manifest: dict
    container: str
    base: str | None
    config: dict | None
    commit_token: bytes | None


_HAS_POSIX_DESCRIPTOR_READ = (
    os.name == "posix"
    and all(
        hasattr(os, name)
        for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    )
    and os.open in getattr(os, "supports_dir_fd", ())
)


def _read_descriptor(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _open_posix_directory(path: str, flags: int) -> int:
    """Walk an absolute directory path without following any component."""
    from .io import _trusted_root_alias

    path = _trusted_root_alias(path)
    descriptor = os.open(os.path.sep, flags)
    cursor = os.path.sep
    try:
        for component in Path(path).parts[1:]:
            cursor = os.path.join(cursor, component)
            try:
                next_descriptor = os.open(
                    component, flags, dir_fd=descriptor
                )
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ValueError(
                        "local dataset metadata path crosses symlink or "
                        f"non-directory ancestor {cursor!r}"
                    ) from error
                raise
            previous_descriptor = descriptor
            descriptor = next_descriptor
            os.close(previous_descriptor)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_posix_local_bytes(path: str) -> bytes:
    """Open one local metadata file without following a swapped pathname."""
    from .io import _local

    candidate = os.path.abspath(_local(path))
    parent, name = os.path.split(candidate)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOCTTY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    directory_descriptor = _open_posix_directory(parent, directory_flags)
    try:
        try:
            descriptor = os.open(name, file_flags, dir_fd=directory_descriptor)
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise ValueError(
                    f"local dataset metadata {path!r} must not be a symlink"
                ) from error
            if error.errno in {errno.ENODEV, errno.ENXIO}:
                raise ValueError(
                    f"local dataset metadata {path!r} must be a regular file"
                ) from error
            raise
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(
                    f"local dataset metadata {path!r} must be a regular file"
                )
            return _read_descriptor(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory_descriptor)


def _read_fallback_local_bytes(path: str) -> bytes:
    """Best-effort identity-pinned fallback for platforms without POSIX openat."""
    from .io import _local

    candidate = _local(path)
    before = os.lstat(candidate)
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"local dataset metadata {path!r} must not be a symlink")
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"local dataset metadata {path!r} must be a regular file")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOCTTY", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(candidate, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"local dataset metadata {path!r} must be a regular file")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(
                f"local dataset metadata {path!r} changed while it was being opened"
            )
        return _read_descriptor(descriptor)
    finally:
        os.close(descriptor)


def _read_bytes(path: str) -> bytes:
    from .io import is_remote

    if not is_remote(path):
        if _HAS_POSIX_DESCRIPTOR_READ:
            return _read_posix_local_bytes(path)
        return _read_fallback_local_bytes(path)
    with open_file(path, "rb") as stream:
        return stream.read()


def _validate_local_dataset_file(
    path: str,
    base: str,
    *,
    label: str,
    required: bool = True,
) -> bool:
    """Reject local metadata/shard symlinks and special files before opening them."""
    from .io import _local, is_remote

    if is_remote(base) or is_remote(path):
        return True
    root = Path(os.path.abspath(_local(base)))
    candidate = Path(os.path.abspath(_local(path)))
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} {path!r} escapes dataset root {base!r}") from error

    cursor = root
    parent_paths = [root]
    for part in relative.parts[:-1]:
        cursor /= part
        parent_paths.append(cursor)
    for cursor in parent_paths:
        try:
            status = os.lstat(cursor)
        except FileNotFoundError:
            if required:
                raise FileNotFoundError(f"{label} parent not found: {cursor}")
            return False
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"{label} {path!r} crosses symlink {str(cursor)!r}")
        if not stat.S_ISDIR(status.st_mode):
            raise ValueError(f"{label} {path!r} crosses non-directory {str(cursor)!r}")

    try:
        status = os.lstat(candidate)
    except FileNotFoundError:
        if required:
            raise
        return False
    if stat.S_ISLNK(status.st_mode):
        raise ValueError(f"{label} {path!r} must not be a symlink")
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{label} {path!r} must be a regular file")
    return True


def _json_object(payload: bytes, *, label: str) -> dict:
    return parse_json_object(payload, label=label)


def _payload_integrity(payload: bytes) -> dict[str, int | str]:
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _validate_committed_shards(manifest: dict) -> None:
    entries = manifest.get("shards")
    if not isinstance(entries, list):
        raise ValueError("committed dataset manifest must contain a shard list")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"committed dataset shard entry {index} must be an object")
        digest = entry.get("sha256")
        size = entry.get("size_bytes")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(
                f"committed dataset shard entry {index} is missing a lowercase SHA-256 digest"
            )
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(
                f"committed dataset shard entry {index} is missing a non-negative byte size"
            )


def _validate_declared_modality_names(config: dict) -> None:
    """Apply the portable name baseline without imposing current structure on legacy configs."""
    output = config.get("output")
    if not isinstance(output, dict):
        return
    modalities = output.get("modalities")
    if not isinstance(modalities, dict):
        return
    for name in modalities:
        validate_modality_name(name)


def _validate_snapshot_semantics(manifest: dict, config: dict) -> None:
    """Validate current-format manifest structure and manifest/config agreement."""
    _validate_declared_modality_names(config)
    if manifest.get("commit_protocol") != "sidecar-v1":
        return
    if "schema" in config:
        from .schema import SchemaSpec
        try:
            SchemaSpec.from_dict(config["schema"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"committed dataset config has an invalid embedded schema: {error}"
            ) from error
    entries = manifest.get("shards")
    if not isinstance(entries, list):  # also checked by _validate_committed_shards
        raise ValueError("committed dataset manifest must contain a shard list")
    count = manifest.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("committed dataset manifest count must be a non-negative integer")
    if count != len(entries):
        raise ValueError(
            f"committed dataset manifest count={count} does not match "
            f"its {len(entries)} shard entries"
        )

    path_pattern = re.compile(
        r"(?P<track>.+)_shard(?P<index>[0-9]{4,})(?P<padded>_padded)?"
        r"(?P<extension>\.[^/]+)\Z"
    )
    paths: set[str] = set()
    track_indices: dict[tuple[str, str], list[int]] = {}
    semantic_keys: set[tuple[str, str, str, str, int]] = set()
    for position, entry in enumerate(entries):
        path = entry.get("path")
        split = entry.get("split")
        shard_index = entry.get("shard_index")
        n_steps = entry.get("n_steps")
        if not isinstance(path, str) or not path:
            raise ValueError(f"committed dataset shard entry {position} has no path")
        portable_path = path.replace("\\", "/")
        if portable_path in paths:
            raise ValueError(f"committed dataset manifest repeats shard path {path!r}")
        paths.add(portable_path)
        if (not isinstance(split, str) or not split or "/" in split
                or split in {".", ".."} or "\x00" in split):
            raise ValueError(
                f"committed dataset shard entry {position} has unsafe split {split!r}"
            )
        filename = portable_path.rsplit("/", 1)[-1]
        if portable_path != f"{split}/{filename}":
            raise ValueError(
                f"committed dataset shard path {path!r} does not agree with split {split!r}"
            )
        match = path_pattern.fullmatch(filename)
        if match is None:
            raise ValueError(
                f"committed dataset shard path {path!r} does not use the canonical "
                "'*_shardNNNN.ext' shape"
            )
        if (not isinstance(shard_index, int) or isinstance(shard_index, bool)
                or shard_index < 0):
            raise ValueError(
                f"committed dataset shard entry {position} has invalid shard_index "
                f"{shard_index!r}"
            )
        filename_index = int(match.group("index"))
        if shard_index != filename_index:
            raise ValueError(
                f"committed dataset shard entry {position} says shard_index={shard_index}, "
                f"but its filename says {filename_index}"
            )
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
            raise ValueError(
                f"committed dataset shard entry {position} has invalid n_steps {n_steps!r}"
            )
        if not isinstance(entry.get("source_id"), str):
            raise ValueError(
                f"committed dataset shard entry {position} source_id must be a string"
            )
        if not isinstance(entry.get("entity_id"), str):
            raise ValueError(
                f"committed dataset shard entry {position} entity_id must be a string"
            )
        track_id = entry.get("track_id")
        filename_track = match.group("track")
        if not isinstance(track_id, str) or track_id != filename_track:
            raise ValueError(
                f"committed dataset shard entry {position} track_id {track_id!r} does not "
                f"match its filename track {filename_track!r}"
            )
        semantic_key = (
            entry["source_id"], entry["entity_id"], split, track_id, shard_index)
        if semantic_key in semantic_keys:
            raise ValueError(
                f"committed dataset manifest repeats semantic shard identity "
                f"{semantic_key!r}"
            )
        semantic_keys.add(semantic_key)
        track_indices.setdefault((split, track_id), []).append(shard_index)

    for (split, track), indices in track_indices.items():
        ordered = sorted(indices)
        if ordered != list(range(len(ordered))):
            raise ValueError(
                f"committed dataset track {split}/{track!s} has repeated or gapped shard "
                f"indices {ordered}; expected {list(range(len(ordered)))}"
            )

    results = config.get("results")
    if isinstance(results, dict) and "shards" in results:
        configured = results["shards"]
        if (not isinstance(configured, int) or isinstance(configured, bool)
                or configured != len(entries)):
            raise ValueError(
                f"dataset config results.shards={configured!r} does not match "
                f"the committed manifest's {len(entries)} entries"
            )

    output = config.get("output")
    if not isinstance(output, dict):
        raise ValueError("committed dataset config output must be an object")
    configured_modalities = output.get("modalities")
    if not isinstance(configured_modalities, dict) or not configured_modalities:
        raise ValueError("committed dataset config output.modalities must be a non-empty object")
    expected_contract: dict[str, tuple[list[int], np.dtype]] = {}
    for name, modality in configured_modalities.items():
        validate_modality_name(name)
        if not isinstance(modality, dict):
            raise ValueError("committed dataset config has an invalid output modality")
        raw_shape = modality.get("shape") or []
        if (not isinstance(raw_shape, (list, tuple))
                or any(not isinstance(size, int) or isinstance(size, bool) or size < 1
                       for size in raw_shape)):
            raise ValueError(
                f"committed dataset config modality {name!r} has invalid shape {raw_shape!r}"
            )
        try:
            dtype = np.dtype(modality.get("dtype"))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"committed dataset config modality {name!r} has invalid dtype "
                f"{modality.get('dtype')!r}"
            ) from error
        if dtype.hasobject:
            raise ValueError(
                f"committed dataset config modality {name!r} uses unsafe object dtype"
            )
        expected_contract[name] = (list(raw_shape), dtype)
    presence_mask = output.get("presence_mask", True)
    if not isinstance(presence_mask, bool):
        raise ValueError("committed dataset config output.presence_mask must be true or false")
    if presence_mask:
        expected_contract["_present"] = ([], np.dtype("bool"))

    for position, entry in enumerate(entries):
        shapes = entry.get("shape")
        dtypes = entry.get("dtype")
        if not isinstance(shapes, dict) or not isinstance(dtypes, dict):
            raise ValueError(
                f"committed dataset shard entry {position} must declare shape and dtype maps"
            )
        expected_names = set(expected_contract)
        if set(shapes) != expected_names or set(dtypes) != expected_names:
            raise ValueError(
                f"committed dataset shard entry {position} modality names do not match "
                f"config.output: expected {sorted(expected_names)}, "
                f"shape has {sorted(map(str, shapes))}, dtype has "
                f"{sorted(map(str, dtypes))}"
            )
        for name, (expected_shape, expected_dtype) in expected_contract.items():
            shape = shapes[name]
            if not isinstance(shape, (list, tuple)) or list(shape) != expected_shape:
                raise ValueError(
                    f"committed dataset shard entry {position} modality {name!r} shape "
                    f"{shape!r} does not match config.output shape {expected_shape!r}"
                )
            try:
                actual_dtype = np.dtype(dtypes[name])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"committed dataset shard entry {position} modality {name!r} has "
                    f"invalid dtype {dtypes[name]!r}"
                ) from error
            if actual_dtype != expected_dtype:
                raise ValueError(
                    f"committed dataset shard entry {position} modality {name!r} dtype "
                    f"{dtypes[name]!r} does not match config.output dtype "
                    f"{expected_dtype.name!r}"
                )


def _verified_commit_metadata(
    base: str,
    manifest: dict,
    *,
    manifest_payload: bytes | None = None,
) -> tuple[bytes | None, bytes | None]:
    protocol = manifest.get("commit_protocol")
    if protocol is None:                         # legacy artifact predating transactional sidecar
        return None, None
    if protocol != "sidecar-v1":
        raise ValueError(f"unsupported dataset commit protocol {protocol!r}")
    from .io import exists
    commit_path = f"{base}/dataset.commit.json"
    if not exists(commit_path):
        raise ValueError(f"dataset {base!r} is incomplete: missing final dataset.commit.json")
    _validate_local_dataset_file(
        commit_path, base, label="dataset commit")
    commit_payload = _read_bytes(commit_path)
    commit = _json_object(commit_payload, label=f"dataset commit {commit_path!r}")
    if commit.get("protocol") != protocol:
        raise ValueError(f"dataset {base!r} has a mismatched commit protocol")
    if commit.get("state") != "committed":
        raise ValueError(f"dataset {base!r} is incomplete: commit state is "
                         f"{commit.get('state')!r}, expected 'committed'")
    files = commit.get("files") or {}
    if not isinstance(files, dict):
        raise ValueError(f"dataset commit {commit_path!r} has an invalid files map")
    names = ["dataset_manifest.json", "config.json"]
    if ".kuration_output_spec.json" in files:
        names.append(".kuration_output_spec.json")
    if manifest_payload is None:
        _validate_local_dataset_file(
            f"{base}/dataset_manifest.json", base, label="dataset manifest")
        manifest_payload = _read_bytes(f"{base}/dataset_manifest.json")
        if _json_object(
            manifest_payload, label=f"dataset manifest {base!r}"
        ) != manifest:
            raise ValueError(
                f"dataset {base!r} changed while its manifest was being verified"
            )
    config_path = f"{base}/config.json"
    _validate_local_dataset_file(config_path, base, label="dataset config")
    payloads = {
        "dataset_manifest.json": manifest_payload,
        "config.json": _read_bytes(config_path),
    }
    if ".kuration_output_spec.json" in names:
        marker_path = f"{base}/.kuration_output_spec.json"
        _validate_local_dataset_file(marker_path, base, label="dataset resume marker")
        payloads[".kuration_output_spec.json"] = _read_bytes(
            marker_path
        )
    for name in names:
        expected = files.get(name)
        if not isinstance(expected, dict):
            raise ValueError(f"dataset commit is missing integrity for {name}")
        actual = _payload_integrity(payloads[name])
        if actual != expected:
            raise ValueError(f"dataset {base!r} is not atomically committed: {name} changed "
                             "after the commit record was published")
    if _read_bytes(commit_path) != commit_payload:
        raise ValueError(f"dataset {base!r} commit changed while metadata was being read")
    _validate_committed_shards(manifest)
    return payloads["config.json"], commit_payload


def _verify_commit(
    base: str,
    manifest: dict,
    *,
    manifest_payload: bytes | None = None,
) -> bytes | None:
    """Compatibility wrapper returning the commit-pinned config payload."""
    config_payload, _commit_token = _verified_commit_metadata(
        base, manifest, manifest_payload=manifest_payload
    )
    return config_payload


def _assert_commit_token_current(
    base: str | None,
    commit_token: bytes | None,
    *,
    operation: str,
) -> None:
    """Reject a reader operation that crossed a force-replacement generation."""
    if base is None or commit_token is None:
        return
    commit_path = f"{base}/dataset.commit.json"
    try:
        _validate_local_dataset_file(
            commit_path, base, label="dataset commit")
        current = _read_bytes(commit_path)
    except (FileNotFoundError, NotADirectoryError) as error:
        raise ValueError(
            f"dataset {base!r} changed generation during {operation}: "
            "the commit record disappeared"
        ) from error
    if current != commit_token:
        raise ValueError(
            f"dataset {base!r} changed generation during {operation}; retry against "
            "one stable committed artifact"
        )


def _assert_snapshot_current(
    snapshot: _ResolvedDatasetSnapshot,
    *,
    operation: str,
) -> None:
    _assert_commit_token_current(
        snapshot.base, snapshot.commit_token, operation=operation
    )


def _load_metadata_snapshot(
    base: str,
    *,
    manifest_path: str | None = None,
) -> tuple[dict, dict | None]:
    manifest, config, _commit_token = _load_metadata_operation_snapshot(
        base, manifest_path=manifest_path
    )
    return manifest, config


def _load_metadata_operation_snapshot(
    base: str,
    *,
    manifest_path: str | None = None,
) -> tuple[dict, dict | None, bytes | None]:
    path = manifest_path or f"{base}/dataset_manifest.json"
    _validate_local_dataset_file(path, base, label="dataset manifest")
    manifest_payload = _read_bytes(path)
    manifest = _json_object(manifest_payload, label=f"dataset manifest {path!r}")
    config_payload, commit_token = _verified_commit_metadata(
        base, manifest, manifest_payload=manifest_payload
    )
    if config_payload is None:
        config_path = f"{base}/config.json"
        try:
            if not _validate_local_dataset_file(
                    config_path, base, label="dataset config", required=False):
                return manifest, None, commit_token
            config_payload = _read_bytes(config_path)
        except FileNotFoundError:
            return manifest, None, commit_token
    config = _json_object(config_payload, label=f"dataset config {base!r}")
    _validate_snapshot_semantics(manifest, config)
    return manifest, config, commit_token


def _resolve_operation_snapshot(
    dataset, container: Optional[str]
) -> _ResolvedDatasetSnapshot:
    """Resolve one commit-pinned metadata snapshot for a logical reader operation."""
    if isinstance(dataset, dict):
        man, base, config, commit_token = dataset, None, None, None
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
            base = s.rsplit("/", 1)[0] if "/" in s else "."
            man, config, commit_token = _load_metadata_operation_snapshot(
                base, manifest_path=s
            )
        else:
            base = s
            man, config, commit_token = _load_metadata_operation_snapshot(base)
    from .versioning import check_version
    check_version(man.get("artifact_format_version"), kind="artifact")
    if config is not None:
        check_version(config.get("artifact_format_version"), kind="artifact")
        if container is None:
            container = (config.get("output") or {}).get("container")
    return _ResolvedDatasetSnapshot(
        manifest=man,
        container=container or "npz",
        base=base,
        config=config,
        commit_token=commit_token,
    )


def _resolve_snapshot(
    dataset, container: Optional[str]
) -> tuple[dict, str, Optional[str], dict | None]:
    """Return (manifest_dict, container, dataset_dir, config) from one metadata snapshot."""
    snapshot = _resolve_operation_snapshot(dataset, container)
    return (
        snapshot.manifest,
        snapshot.container,
        snapshot.base,
        snapshot.config,
    )


def _resolve(dataset, container: Optional[str]) -> tuple[dict, str, Optional[str]]:
    man, resolved, base, _config = _resolve_snapshot(dataset, container)
    return man, resolved, base


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


def _validate_local_snapshot_shards(entries: list[dict], base: str | None) -> None:
    """Reject redirects and special files below a current-format local dataset root."""
    if base is None:
        return
    from .io import _local, is_remote
    if is_remote(base):
        return
    root = Path(os.path.abspath(_local(base)))
    resolved_root = root.resolve(strict=False)
    for entry in entries:
        raw_path = entry.get("path")
        if not isinstance(raw_path, str):
            raise ValueError("manifest shard path must be a string")
        candidate = Path(os.path.abspath(_local(raw_path)))
        try:
            relative = candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"current-format shard path {raw_path!r} escapes dataset root {base!r}"
            ) from error

        cursor = root
        for part in relative.parts[:-1]:
            cursor /= part
            try:
                status = os.lstat(cursor)
            except FileNotFoundError:
                break
            if stat.S_ISLNK(status.st_mode):
                raise ValueError(
                    f"current-format shard path {raw_path!r} crosses symlink "
                    f"{str(cursor)!r}"
                )
            if not stat.S_ISDIR(status.st_mode):
                raise ValueError(
                    f"current-format shard path {raw_path!r} crosses non-directory "
                    f"{str(cursor)!r}"
                )

        try:
            status = os.lstat(candidate)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(
                f"current-format shard path {raw_path!r} is a symlink; shards must be "
                "regular files contained by the dataset"
            )
        if not stat.S_ISREG(status.st_mode):
            raise ValueError(
                f"current-format shard path {raw_path!r} is not a regular file"
            )
        try:
            candidate.resolve(strict=True).relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(
                f"current-format shard path {raw_path!r} resolves outside dataset root "
                f"{base!r}"
            ) from error


def _snapshot_entries(snapshot: _ResolvedDatasetSnapshot) -> list[dict]:
    entries = rebase_entries(
        list(snapshot.manifest.get("shards", [])), snapshot.base
    )
    if snapshot.base is not None:
        _validate_local_snapshot_shards(entries, snapshot.base)
    return entries


def _validate_snapshot_shard(
    snapshot: _ResolvedDatasetSnapshot,
    entry: dict,
) -> None:
    """Recheck a current-format local shard immediately before a delayed read."""
    if snapshot.base is not None:
        _validate_local_snapshot_shards([entry], snapshot.base)


def shard_entries(dataset, *, split: Optional[str] = None,
                  container: Optional[str] = None) -> tuple[list[dict], str]:
    """Return (manifest entries, container), optionally filtered to one split. Shard paths
    are always resolved against the dataset directory for artifact self-containment."""
    snapshot = _resolve_operation_snapshot(dataset, container)
    entries = _snapshot_entries(snapshot)
    if split is not None:
        entries = [e for e in entries if e.get("split") == split]
    _assert_snapshot_current(snapshot, operation="shard discovery")
    return entries, snapshot.container


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
    from .output.containers import read_shard
    expected_sha = entry.get("sha256")
    expected_size = entry.get("size_bytes")
    if verify and (expected_sha is None or expected_size is None):
        raise ValueError(
            f"shard {entry.get('path')!r} has no complete sha256/size_bytes integrity "
            "metadata; pass verify=False only if this legacy artifact is trusted"
        )
    if verify:
        if (not isinstance(expected_sha, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None):
            raise ValueError(
                f"shard {entry.get('path')!r} has invalid sha256 integrity metadata"
            )
        if (not isinstance(expected_size, int) or isinstance(expected_size, bool)
                or expected_size < 0):
            raise ValueError(
                f"shard {entry.get('path')!r} has invalid size_bytes integrity metadata"
            )
        from .io import file_integrity, local_snapshot
        with local_snapshot(entry["path"]) as snapshot:
            actual = file_integrity(snapshot)
            if actual["sha256"] != expected_sha:
                raise ValueError(f"shard {entry['path']!r} checksum mismatch: manifest says "
                                 f"{expected_sha}, payload is {actual['sha256']}")
            if actual["size_bytes"] != expected_size:
                raise ValueError(f"shard {entry['path']!r} size mismatch: manifest says "
                                 f"{expected_size}, payload is {actual['size_bytes']}")
            arrays, n = read_shard(snapshot, container, registry=registry)
    else:
        arrays, n = read_shard(entry["path"], container, registry=registry)
    shapes = entry.get("shape", {})
    dtypes = entry.get("dtype", {})
    if not isinstance(arrays, dict):
        raise ValueError(
            f"shard {entry['path']!r} container reader must return a modality mapping"
        )
    for label, mapping in (("arrays", arrays), ("shape", shapes), ("dtype", dtypes)):
        if not isinstance(mapping, dict):
            raise ValueError(
                f"shard {entry['path']!r} {label} metadata must be a modality mapping"
            )
        for name in mapping:
            validate_modality_name(name, allow_presence_mask=name == "_present")
    names = modalities if modalities is not None else list(arrays)
    for name in names:
        validate_modality_name(name, allow_presence_mask=name == "_present")
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
    snapshot = _resolve_operation_snapshot(dataset, container)
    entries = _snapshot_entries(snapshot)
    if split is not None:
        entries = [e for e in entries if e.get("split") == split]
    for e in entries:
        _assert_snapshot_current(snapshot, operation="shard iteration")
        _validate_snapshot_shard(snapshot, e)
        arrays = read_shard_arrays(e, snapshot.container, modalities=modalities, trim=trim,
                                   registry=registry, verify=verify)
        _validate_snapshot_shard(snapshot, e)
        _assert_snapshot_current(snapshot, operation="shard iteration")
        yield (arrays, e) if with_meta else arrays
    _assert_snapshot_current(snapshot, operation="shard iteration")
