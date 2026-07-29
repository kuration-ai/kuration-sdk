"""Checksummed manifests, resolved run contracts, and dataset publication leases.

``config.json`` makes an artifact self-describing for audit and safe-reuse decisions. Replaying
the run additionally requires the same immutable or content-addressed sources and compatible
component implementations.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
import posixpath
import re
import tempfile
import threading
import uuid
from typing import Optional

from .io import write_text
from .jsonutil import parse_json_object
from .schema import SchemaSpec
from .segment import Segment
from .strategy import Strategy
from .versioning import ARTIFACT_FORMAT_VERSION


_SHARD_NAME = re.compile(
    r".+_shard[0-9]{4,}(?:_padded)?"
    r"\.[A-Za-z0-9][A-Za-z0-9._-]*\Z"
)
_MEMORY_LEASE_GUARD = threading.Lock()
_MEMORY_LEASES: set[str] = set()


def _validate_local_destination_symlinks(out_dir: str) -> None:
    """Refuse local output paths whose existing tree can redirect writes outside its root."""
    from pathlib import Path
    from .io import _local, is_remote

    if is_remote(out_dir):
        return
    destination = Path(os.path.abspath(_local(out_dir)))
    if destination.is_dir():
        for child in destination.iterdir():
            if child.is_symlink():
                raise ValueError(
                    f"output {out_dir!r} contains symlink child {child.name!r}; remove the "
                    "symlink or choose a fresh real directory before publishing"
                )


@contextmanager
def dataset_writer_lease(out_dir: str):
    """Hold one exclusive writer lease for a dataset destination.

    Local leases use an OS advisory lock outside the artifact, so process exit releases them
    automatically. Remote leases use the backend's exclusive-create mode and are removed only
    by their owning token. The process-local ``memory://`` backend uses a process-local guard
    because older supported fsspec releases lack exclusive-create mode for it. A persistent
    backend that cannot provide exclusive creation fails closed.
    """
    from .io import _fs, _local, is_remote

    from .io import _normalize_explicit_path
    destination = _normalize_explicit_path(out_dir, label="output path")
    _validate_local_destination_symlinks(destination)
    remote = is_remote(destination)
    if remote:
        from .io import _ensure_remote_mutation_supported
        _ensure_remote_mutation_supported(destination)
        if any(character in destination for character in "*?["):
            raise ValueError(
                f"remote output {destination!r} contains a glob metacharacter; fsspec "
                "backends do not consistently treat those names literally, so choose a "
                "destination without '*', '?', or '['"
            )
        lock_path = f"{destination}/.kuration-writer.lock"
    else:
        from pathlib import Path
        resolved = str(Path(_local(destination)).resolve())
        lock_root = Path(tempfile.gettempdir()) / "kuration-sdk-writer-locks"
        lock_root.mkdir(parents=True, exist_ok=True)
        lock_path = str(lock_root / f"{hashlib.sha256(resolved.encode()).hexdigest()}.lock")

    token = uuid.uuid4().hex.encode()
    handle = None
    fs = None
    memory_lease_key = None
    try:
        if remote:
            fs = _fs(lock_path)
            if destination.startswith("memory://"):
                memory_lease_key = fs._strip_protocol(lock_path)
                with _MEMORY_LEASE_GUARD:
                    if memory_lease_key in _MEMORY_LEASES:
                        raise FileExistsError(
                            f"output {destination!r} already has an active writer lease")
                    _MEMORY_LEASES.add(memory_lease_key)
                with fs.open(lock_path, "wb") as stream:
                    stream.write(token)
            else:
                try:
                    with fs.open(lock_path, "xb") as stream:
                        stream.write(token)
                except FileExistsError as error:
                    raise FileExistsError(
                        f"output {destination!r} already has an active writer lease; wait for "
                        f"that writer to finish (or remove {lock_path!r} only after confirming "
                        "it stopped)") from error
                except (NotImplementedError, ValueError) as error:
                    raise RuntimeError(
                        f"remote backend cannot provide an exclusive writer lease for "
                        f"{destination!r}; refusing race-prone publication") from error
        else:
            handle = open(lock_path, "a+b")
            if os.name == "nt":
                import msvcrt
                locking = msvcrt.locking  # type: ignore[attr-defined]
                if handle.seek(0, os.SEEK_END) == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                try:
                    locking(handle.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
                except OSError as error:
                    raise FileExistsError(
                        f"output {destination!r} already has an active writer lease") from error
            else:
                import fcntl
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise FileExistsError(
                        f"output {destination!r} already has an active writer lease") from error
        yield
    finally:
        if handle is not None:
            if os.name == "nt":
                import msvcrt
                locking = msvcrt.locking  # type: ignore[attr-defined]
                try:
                    handle.seek(0)
                    locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
                except OSError:
                    pass
            else:
                import fcntl
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            handle.close()
        if fs is not None:
            try:
                with fs.open(lock_path, "rb") as stream:
                    owned = stream.read() == token
                if owned:
                    fs.rm_file(fs._strip_protocol(lock_path))
            except (FileNotFoundError, OSError):
                pass
        if memory_lease_key is not None:
            with _MEMORY_LEASE_GUARD:
                _MEMORY_LEASES.discard(memory_lease_key)


def validate_source_destination_separation(
    out_dir: str,
    sources,
    *,
    adapter=None,
) -> None:
    """Reject source/output path overlap before a two-pass writer can mutate either side."""
    from pathlib import Path
    from .io import _fs, _local, is_remote

    if getattr(adapter, "in_memory", False):
        return

    from .io import _normalize_explicit_path
    destination = _normalize_explicit_path(out_dir, label="output path")
    destination_remote = is_remote(destination)

    def overlaps(child: str, parent: str) -> bool:
        normalized_child = child.rstrip("/") or "/"
        normalized_parent = parent.rstrip("/") or "/"
        return (
            normalized_child == normalized_parent
            or normalized_child.startswith(f"{normalized_parent}/")
        )

    def remote_path(fs, value: str, *, label: str) -> str:
        raw = str(fs._strip_protocol(value))
        if any(component in {".", ".."} for component in raw.split("/")):
            raise ValueError(
                f"{label} {value!r} contains an ambiguous dot path component; choose a "
                "normalized remote path"
            )
        return posixpath.normpath("/" + raw.lstrip("/"))

    if destination_remote:
        destination_fs = _fs(destination)
        destination_token = getattr(destination_fs, "_fs_token_", None)
        destination_path = remote_path(
            destination_fs, destination, label="output")
        for source in sources:
            source_text = str(source)
            if not is_remote(source_text):
                continue
            source_fs = _fs(source_text)
            if getattr(source_fs, "_fs_token_", None) != destination_token:
                continue
            source_path = remote_path(source_fs, source_text, label="source")
            if overlaps(source_path, destination_path) or overlaps(
                destination_path, source_path
            ):
                raise ValueError(
                    f"source {source_text!r} overlaps output {destination!r}; a publisher may "
                    "read sources more than once and must never replace or create files inside "
                    "its own input tree. Choose a separate output path."
                )
        return

    destination_lexical = Path(os.path.abspath(_local(destination)))
    destination_resolved = destination_lexical.resolve()
    destination_paths = {destination_lexical, destination_resolved}
    for source in sources:
        source_text = str(source)
        if is_remote(source_text):
            continue
        source_local = Path(_local(source_text))
        if not source_local.exists():
            continue
        source_lexical = Path(os.path.abspath(source_local))
        source_resolved = source_lexical.resolve()
        source_paths = {source_lexical, source_resolved}
        if any(
            source_path == output_path
            or source_path in output_path.parents
            or output_path in source_path.parents
            for source_path in source_paths
            for output_path in destination_paths
        ):
            raise ValueError(
                f"source {source_text!r} overlaps output {destination!r}; a publisher may read "
                "sources more than once and must never replace or create files inside its own "
                "input tree. Choose a separate output path."
            )


def segment_record(s: Segment) -> dict:
    return {
        "source_id": s.source_id,
        "time_range": list(s.time_range),
        "features": {k: float(v) for k, v in s.features.items()},
        "bin": list(s.bin),
        "tags": sorted(str(t) for t in s.tags),
        "group": s.group,
    }


def dump_segment_manifest(path: str, segments: list[Segment]) -> None:
    """Write scored segments and the artifact format version as JSON."""
    payload = {"artifact_format_version": ARTIFACT_FORMAT_VERSION,
               "count": len(segments), "segments": [segment_record(s) for s in segments]}
    write_text(path, json.dumps(payload, indent=2, default=float, allow_nan=False))


def portable_shard_entries(out_dir: str, entries: list[dict]) -> list[dict]:
    """Return manifest entries whose shard paths are relative to the dataset root."""
    root = str(out_dir).replace("\\", "/").rstrip("/")
    portable = []
    for entry in entries:
        path = str(entry["path"]).replace("\\", "/")
        prefix = f"{root}/"
        if path.startswith(prefix):
            relative = path[len(prefix):]
        else:
            # Writers are expected to place shards beneath out_dir. Keep a legacy-compatible
            # fallback, but never serialize a host-absolute path into a portable artifact.
            relative = f"{entry.get('split', 'all')}/{path.rsplit('/', 1)[-1]}"
        if relative.startswith("../") or relative.startswith("/") or "/../" in relative:
            raise ValueError(f"shard path {entry['path']!r} escapes dataset root {out_dir!r}")
        portable.append({**entry, "path": relative})
    return portable


def _publication_files(out_dir: str, *, require_config: bool) -> dict:
    from .io import exists, file_integrity
    files = {
        "dataset_manifest.json": file_integrity(f"{out_dir}/dataset_manifest.json"),
    }
    config_path = f"{out_dir}/config.json"
    if require_config or exists(config_path):
        files["config.json"] = file_integrity(config_path)
    resume_marker = f"{out_dir}/.kuration_output_spec.json"
    if exists(resume_marker):
        files[".kuration_output_spec.json"] = file_integrity(resume_marker)
    return files


def commit_dataset(out_dir: str) -> dict:
    """Publish the final small commit record after manifest and config are durable."""
    payload = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "protocol": "sidecar-v1",
        "state": "committed",
        "files": _publication_files(out_dir, require_config=True),
    }
    write_text(f"{out_dir}/dataset.commit.json",
               json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
    return payload


def pin_writing_dataset(out_dir: str) -> dict:
    """Refresh the writing sidecar after active-run metadata changes.

    In particular, an hours writer publishes its run fingerprint only after selection. Pinning
    that marker before any shard reuse makes an interrupted run resumable without trusting an
    uncommitted or tampered marker.
    """
    payload = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "protocol": "sidecar-v1",
        "state": "writing",
        "files": _publication_files(out_dir, require_config=False),
    }
    write_text(f"{out_dir}/dataset.commit.json",
               json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
    return payload


def begin_dataset(out_dir: str) -> dict:
    """Invalidate any prior commit before shard paths can be rewritten.

    The existing manifest entries are retained so a later ``resume=True`` run can checksum
    and reuse shards that the interrupted writer did not change. Readers reject the atomic
    ``state=writing`` sidecar, preventing them from observing a mixed old/new dataset.
    """
    from .io import exists, read_text
    manifest_path = f"{out_dir}/dataset_manifest.json"
    manifest = {"artifact_format_version": ARTIFACT_FORMAT_VERSION,
                "commit_protocol": "sidecar-v1", "count": 0, "shards": []}
    had_manifest = exists(manifest_path)
    if had_manifest:
        try:
            loaded = parse_json_object(
                read_text(manifest_path), label=f"dataset manifest {manifest_path!r}"
            )
            manifest = {**loaded, "commit_protocol": "sidecar-v1"}
        except (OSError, ValueError, TypeError):
            pass
        # Close the readable generation before changing its manifest bytes. For a legacy
        # manifest, readers continue to see the untouched legacy generation until the manifest
        # gains ``commit_protocol``; for a sidecar-v1 manifest, readers immediately reject the
        # explicit writing state.
        pin_writing_dataset(out_dir)
    write_text(
        manifest_path,
        json.dumps(manifest, indent=2, default=float, allow_nan=False),
    )
    return pin_writing_dataset(out_dir)


def _canonical_shard_paths(out_dir: str) -> list[str]:
    """List writer-shaped shard files exactly beneath ``out_dir`` without glob expansion."""
    from .io import _fs, _local, is_remote

    root = str(out_dir).rstrip("/")
    if is_remote(root):
        import posixpath
        from pathlib import PurePosixPath

        fs = _fs(root)
        raw_root = str(fs._strip_protocol(root)).rstrip("/")
        found = fs.find(raw_root, maxdepth=2, withdirs=False, detail=False)
        paths = []
        for candidate in found:
            raw_candidate = str(candidate)
            relative = posixpath.relpath(
                posixpath.normpath(raw_candidate), posixpath.normpath(raw_root)
            )
            parts = PurePosixPath(relative).parts
            if (
                len(parts) == 2
                and parts[0] not in {"", ".", ".."}
                and parts[1] not in {"", ".", ".."}
                and _SHARD_NAME.fullmatch(parts[1])
            ):
                paths.append(str(fs.unstrip_protocol(raw_candidate)))
        return sorted(paths)

    from pathlib import Path

    local_root = Path(_local(root))
    if not local_root.is_dir():
        return []
    paths = []
    for split_dir in local_root.iterdir():
        if not split_dir.is_dir() or split_dir.is_symlink():
            continue
        paths.extend(
            str(candidate)
            for candidate in split_dir.iterdir()
            if candidate.is_file() and _SHARD_NAME.fullmatch(candidate.name)
        )
    return sorted(paths)


def dataset_state(out_dir: str) -> str:
    """Classify a destination as ``absent``, ``writing``, ``committed``, or ``corrupt``.

    ``writing`` means a previous run published shards and the ``state=writing`` sidecar but
    never committed — readers reject that artifact, so nothing readable is lost by replacing
    it. A manifest without a sidecar predates the commit protocol and is treated as committed
    so a legacy artifact is never silently clobbered. Kuration metadata or canonical shard files
    without a manifest are ``corrupt`` rather than absent, so a damaged artifact also requires
    explicit replacement.
    """
    from .io import exists, read_text
    manifest_path = f"{out_dir}/dataset_manifest.json"
    if not exists(manifest_path):
        sentinels = (
            f"{out_dir}/dataset.commit.json",
            f"{out_dir}/config.json",
            f"{out_dir}/.kuration_output_spec.json",
        )
        if any(exists(path) for path in sentinels) or _canonical_shard_paths(out_dir):
            return "corrupt"
        return "absent"
    commit_path = f"{out_dir}/dataset.commit.json"
    if not exists(commit_path):
        try:
            manifest = parse_json_object(
                read_text(manifest_path), label=f"dataset manifest {manifest_path!r}"
            )
        except (OSError, ValueError, TypeError):
            return "corrupt"
        protocol = manifest.get("commit_protocol")
        # Only manifests that predate the sidecar protocol are readable without a commit.
        return "committed" if protocol is None else "corrupt"
    try:
        commit = parse_json_object(
            read_text(commit_path), label=f"dataset commit {commit_path!r}"
        )
    except (OSError, ValueError, TypeError):
        return "corrupt"
    state = commit.get("state")
    if state == "writing":
        return "writing"
    if state == "committed":
        return "committed"
    return "corrupt"


def validate_dataset_destination(out_dir: str, *, force: bool = False,
                                 resume: bool = False) -> str:
    """Validate a writer's intent against the destination's current publication state.

    This check is deliberately read-only. Writers call it before scoring/selection, then defer
    ``begin_dataset`` until they have a complete write plan. A committed artifact therefore stays
    readable if pre-publication work fails, while an explicit ``force=True`` remains the escape
    hatch for replacement. ``resume`` is distinct: it may continue either a matching committed
    run or an interrupted ``state=writing`` run without discarding its verified shards.
    """
    if not isinstance(force, bool):
        raise ValueError("force must be true or false")
    if not isinstance(resume, bool):
        raise ValueError("resume must be true or false")
    if force and resume:
        raise ValueError("force=True and resume=True are mutually exclusive: force replaces "
                         "an artifact, while resume reuses a matching run")
    state = dataset_state(out_dir)
    if resume and state == "corrupt":
        raise ValueError(
            f"resume=True cannot safely reuse {out_dir!r}: Kuration metadata or shards are "
            "present but the artifact publication state is damaged; use force=True to replace "
            "it or choose a fresh out_dir"
        )
    if resume and state != "absent":
        from .io import exists, file_integrity, read_text
        marker_path = f"{out_dir}/.kuration_output_spec.json"
        if not exists(marker_path):
            raise ValueError(
                f"resume=True cannot safely reuse {out_dir!r}: the existing artifact has no "
                "complete run fingerprint; use force=True to replace it or choose a fresh "
                "out_dir"
            )
        try:
            commit = parse_json_object(
                read_text(f"{out_dir}/dataset.commit.json"),
                label=f"dataset commit {out_dir!r}",
            )
            expected = (commit.get("files") or {}).get(".kuration_output_spec.json")
            marker_is_pinned = (
                commit.get("protocol") == "sidecar-v1"
                and commit.get("state") == state
                and isinstance(expected, dict)
                and file_integrity(marker_path) == expected
            )
        except (FileNotFoundError, OSError, TypeError, ValueError):
            marker_is_pinned = False
        if not marker_is_pinned:
            raise ValueError(
                f"resume=True cannot safely reuse {out_dir!r}: its run fingerprint is not "
                "integrity-pinned by the current commit; use force=True to replace it or "
                "choose a fresh out_dir"
            )
    if state == "committed" and not (force or resume):
        raise FileExistsError(
            f"output {out_dir!r} already contains a committed Kuration dataset; pass force=True "
            "to replace it, or resume=True to continue a matching hours run"
        )
    if state == "writing" and not (force or resume):
        raise FileExistsError(
            f"output {out_dir!r} holds a partially written Kuration dataset; pass resume=True "
            "to continue the matching hours run, or force=True to discard it"
        )
    if state == "corrupt" and not force:
        raise FileExistsError(
            f"output {out_dir!r} contains damaged or incomplete Kuration metadata/shards; "
            "pass force=True to replace it or choose a fresh out_dir"
        )
    return state


def _recorded_shard_paths(out_dir: str) -> list[str]:
    """Absolute shard paths the destination's current manifest points at (may be none)."""
    from .io import exists, read_text
    manifest_path = f"{out_dir}/dataset_manifest.json"
    if not exists(manifest_path):
        return []
    try:
        manifest = parse_json_object(
            read_text(manifest_path), label=f"dataset manifest {manifest_path!r}"
        )
    except (OSError, ValueError, TypeError):
        return []
    entries = manifest.get("shards")
    if not isinstance(entries, list):
        return []
    from .loaders import rebase_entries
    resolved = rebase_entries(
        [entry for entry in entries if isinstance(entry, dict) and entry.get("path")], out_dir)
    return [str(entry["path"]) for entry in resolved]


def clear_dataset_shards(out_dir: str) -> list[str]:
    """Delete a prior attempt's shard files and return the removed paths (sorted).

    Two sources, because an interrupted write leaves shards the manifest never learned about:
    the paths the current manifest records, plus every ``<split>/<name>_shard*`` file the
    writers can produce beneath the dataset root. Shard names are derived per run, so those
    leftovers are not guaranteed to be overwritten by the next attempt — they would linger
    unreferenced inside the artifact. Callers report what this removes; only ``force`` gets
    here.
    """
    from .io import _ensure_remote_mutation_supported, exists, is_remote
    if is_remote(out_dir):
        _ensure_remote_mutation_supported(out_dir)
    _validate_local_destination_symlinks(out_dir)
    candidates = set(_recorded_shard_paths(out_dir))
    candidates.update(_canonical_shard_paths(out_dir))
    removed: list[str] = []
    for path in sorted(candidates):
        if not exists(path):
            continue
        if is_remote(path):
            from .io import _ensure_remote_mutation_supported, _fs
            _ensure_remote_mutation_supported(path)
            fs = _fs(path)
            # AbstractFileSystem.rm expands glob metacharacters even for one explicit path.
            # Manifest values are data, never patterns; delete the exact backend key only.
            fs.rm_file(fs._strip_protocol(path))
        else:
            from pathlib import Path as _Path
            from .io import _local
            local = _Path(_local(path))
            if not local.is_file():
                continue
            local.unlink()
        removed.append(path)
    return removed


def clear_dataset_resume_marker(out_dir: str) -> bool:
    """Remove hours-only resume metadata before publishing a non-resumable artifact."""
    from .io import _ensure_remote_mutation_supported, exists, is_remote
    path = f"{str(out_dir).rstrip('/')}/.kuration_output_spec.json"
    if is_remote(path):
        _ensure_remote_mutation_supported(path)
    if not exists(path):
        return False
    if is_remote(path):
        from .io import _fs
        fs = _fs(path)
        fs.rm_file(fs._strip_protocol(path))
    else:
        from pathlib import Path as _Path
        from .io import _local
        _Path(_local(path)).unlink()
    return True


def build_config(schema: SchemaSpec, strategy: Strategy, *, results: dict, seed: int,
                 started: Optional[str] = None, ended: Optional[str] = None) -> dict:
    """Build the reproducibility configuration embedded in a curated dataset.

    The returned dictionary contains the artifact version, resolved schema and strategy, seed,
    optional timestamps, result accounting, and loaded-plugin provenance.
    """
    from .plugins import plugin_provenance
    return {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "schema": schema.to_dict(),
        "strategy": strategy.resolved,
        "seed": seed,
        "started": started,
        "ended": ended,
        "results": results,
        "plugins": plugin_provenance(),
    }


def write_config(path: str, config: dict) -> None:
    """Serialize a configuration dictionary as indented JSON to a local or remote path."""
    write_text(path, json.dumps(config, indent=2, allow_nan=False))
