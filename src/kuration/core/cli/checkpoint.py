"""kur-checkpoint — inspect, verify, and clear stage checkpoint directories.

    kur-checkpoint list runs/ckpt
    kur-checkpoint verify runs/ckpt
    kur-checkpoint clear runs/ckpt --stage feature-pass --yes

`checkpoint_dir` is an ordinary job key and `StageCheckpoint` commits are checksummed, but
nothing showed what a checkpoint tree already holds or removed a stale one — leaving
hand-deletion as the only repair. Verification here reads the committed metadata rather than
reconstructing the run's `CheckpointIdentity`, so it works from the artifact alone: a commit
whose data file is missing, resized, or re-hashed is reported as invalid.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import posixpath
import re

from ._common import cli_run, json_document


_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_FINGERPRINT = re.compile(r"[0-9a-f]{64}")
_LEGACY_IDENTITY = "legacy"


@dataclass(frozen=True)
class _PathEntry:
    path: str
    name: str
    is_dir: bool
    is_symlink: bool = False


@dataclass(frozen=True)
class _StageNamespace:
    stage: str
    stage_root: str
    identity: str | None = None

    @property
    def selector(self) -> str:
        return self.identity or _LEGACY_IDENTITY


def _normalize_root(root: str) -> str:
    from ..checkpoints import _normalize_checkpoint_root
    return _normalize_checkpoint_root(root)


def _join(root: str, *parts: str) -> str:
    base = root if root.endswith("://") else root.rstrip("/")
    return "/".join([base, *(part.strip("/") for part in parts)])


def _children(path: str) -> list[_PathEntry]:
    """List one directory literally; unlike globbing, metacharacters in roots stay data."""
    from ..io import _fs, _local, is_remote

    if is_remote(path):
        fs = _fs(path)
        native = fs._strip_protocol(str(path))
        try:
            raw_entries = fs.ls(native, detail=True)
        except FileNotFoundError:
            return []
        entries = []
        for raw in raw_entries:
            if isinstance(raw, dict):
                raw_name = str(raw.get("name", ""))
                kind = raw.get("type")
                is_symlink = kind in {"link", "symlink"}
                is_dir = not is_symlink and (kind in {"directory", "dir"} or (
                    kind is None and bool(getattr(fs, "isdir", lambda _value: False)(raw_name))
                ))
            else:
                raw_name = str(raw)
                is_symlink = False
                is_dir = bool(getattr(fs, "isdir", lambda _value: False)(raw_name))
            if not raw_name:
                continue
            usable = str(fs.unstrip_protocol(raw_name))
            entries.append(_PathEntry(
                path=usable,
                name=raw_name.rstrip("/").rsplit("/", 1)[-1],
                is_dir=is_dir,
                is_symlink=is_symlink,
            ))
        return sorted(entries, key=lambda entry: entry.name)

    directory = Path(_local(path))
    if not directory.is_dir():
        return []
    entries = []
    for child in directory.iterdir():
        is_symlink = child.is_symlink()
        entries.append(_PathEntry(
            path=str(child),
            name=child.name,
            # Never traverse a directory symlink while discovering deletion targets.
            is_dir=child.is_dir() and not is_symlink,
            is_symlink=is_symlink,
        ))
    return sorted(entries, key=lambda entry: entry.name)


def _is_stage_root(path: str) -> bool:
    entries = _children(path)
    return any(
        (entry.name in {"commits", "attempts"} and entry.is_dir and not entry.is_symlink)
        or (entry.name == "stage.commit.json"
            and not entry.is_dir and not entry.is_symlink)
        for entry in entries
    )


def _stage_namespaces(root: str) -> list[_StageNamespace]:
    """Discover fingerprint-namespaced and legacy stage roots without user-derived globs."""
    namespaces = []
    for child in _children(root):
        if not child.is_dir or _SAFE_COMPONENT.fullmatch(child.name) is None:
            continue
        if _is_stage_root(child.path):
            namespaces.append(_StageNamespace(stage=child.name, stage_root=child.path))
        if _FINGERPRINT.fullmatch(child.name) is None:
            continue
        for stage in _children(child.path):
            if (stage.is_dir and _SAFE_COMPONENT.fullmatch(stage.name) is not None
                    and _is_stage_root(stage.path)):
                namespaces.append(_StageNamespace(
                    stage=stage.name, stage_root=stage.path, identity=child.name))
    return sorted(namespaces, key=lambda item: (item.selector, item.stage, item.stage_root))


def _stage_dirs(root: str) -> list[str]:
    """Backward-compatible stage-name view over all checkpoint identity namespaces."""
    return sorted({namespace.stage for namespace in _stage_namespaces(root)})


def _present_namespaces(namespaces: list[_StageNamespace]) -> str:
    present = [f"{namespace.selector}/{namespace.stage}" for namespace in namespaces]
    return ", ".join(present) if present else "none"


def _select_namespaces(root: str, *, stage: str | None,
                       identity: str | None) -> list[_StageNamespace]:
    namespaces = _stage_namespaces(root)
    selected = namespaces
    if identity is not None:
        selected = [namespace for namespace in selected if namespace.selector == identity]
        if not selected:
            identities = sorted({namespace.selector for namespace in namespaces})
            raise ValueError(
                f"identity {identity!r} not found under {root}; "
                f"present: {identities or 'none'}")
    if stage is not None:
        selected = [namespace for namespace in selected if namespace.stage == stage]
        if not selected:
            raise ValueError(
                f"stage {stage!r} not found under {root}; "
                f"present: {_present_namespaces(namespaces)}")
    return selected


def scan_checkpoints(root: str, *, stage: str | None = None,
                     identity: str | None = None, verify: bool = False) -> dict:
    """Summarize each stage's commits, optionally re-checksumming every data file."""
    from ..io import exists, file_integrity, read_text
    root = _normalize_root(root)
    if not exists(root):
        raise FileNotFoundError(f"checkpoint root not found: {root}")
    namespaces = _select_namespaces(root, stage=stage, identity=identity)

    reports = []
    for namespace in namespaces:
        stage_root = namespace.stage_root
        partitions, invalid = [], 0
        total_rows = total_bytes = 0
        commit_paths = [
            entry.path
            for entry in _children(_join(stage_root, "commits"))
            if not entry.is_dir and not entry.is_symlink and entry.name.endswith(".json")
        ]
        for commit_path in commit_paths:
            record: dict = {"commit": commit_path}
            try:
                commit = json.loads(read_text(commit_path))
            except (OSError, ValueError) as error:
                record.update(partition_id=None, valid=False,
                              error=f"unreadable commit: {error}")
                partitions.append(record)
                invalid += 1
                continue
            if not isinstance(commit, dict):
                record.update(
                    partition_id=None,
                    valid=False,
                    error="invalid commit: expected a JSON object",
                )
                partitions.append(record)
                invalid += 1
                continue
            rows = commit.get("rows")
            size_bytes = commit.get("size_bytes")
            partition_id = commit.get("partition_id")
            commit_identity = commit.get("identity")
            digest = commit.get("sha256")
            if (not isinstance(partition_id, str) or not partition_id
                    or not isinstance(commit_identity, str) or not commit_identity
                    or not isinstance(rows, int) or isinstance(rows, bool) or rows < 0
                    or not isinstance(size_bytes, int) or isinstance(size_bytes, bool)
                    or size_bytes < 0
                    or not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
                record.update(
                    partition_id=partition_id,
                    identity=commit_identity,
                    rows=rows,
                    size_bytes=size_bytes,
                    data_file=commit.get("data_file"),
                    valid=False,
                    error="invalid commit schema",
                )
                partitions.append(record)
                invalid += 1
                continue
            data_file = commit.get("data_file")
            record.update(partition_id=partition_id,
                          identity=commit_identity,
                          rows=rows, size_bytes=size_bytes,
                          data_file=data_file)
            total_rows += rows
            total_bytes += size_bytes
            from ..checkpoints import _same_storage_path, _token
            expected_data_file = _join(
                stage_root, "data", f"{_token(partition_id)}.arrow")
            if namespace.identity is not None and commit_identity != namespace.identity:
                record.update(valid=False, error="namespace identity mismatch")
                invalid += 1
            elif (not isinstance(data_file, str)
                  or not _same_storage_path(data_file, expected_data_file)):
                record.update(
                    valid=False,
                    error="unsafe data file reference outside the canonical partition path",
                )
                invalid += 1
            elif not isinstance(data_file, str) or not data_file or not exists(data_file):
                record.update(valid=False, error="data file missing")
                invalid += 1
            elif verify:
                try:
                    integrity = file_integrity(data_file)
                except (OSError, ValueError) as error:
                    record.update(valid=False, error=f"data file unreadable: {error}")
                    invalid += 1
                else:
                    if integrity["sha256"] != digest:
                        record.update(valid=False, error="sha256 mismatch")
                        invalid += 1
                    elif integrity["size_bytes"] != size_bytes:
                        record.update(valid=False, error="size mismatch")
                        invalid += 1
                    else:
                        record["valid"] = True
            else:
                record["valid"] = None          # present, not checksummed
            partitions.append(record)

        attempts = [
            entry.path
            for entry in _children(_join(stage_root, "attempts"))
            if not entry.is_dir and not entry.is_symlink and entry.name.endswith(".arrow")
        ]
        marker = _join(stage_root, "stage.commit.json")
        finalized = False
        finalization_error = None
        if exists(marker):
            try:
                marker_payload = json.loads(read_text(marker))
                if not isinstance(marker_payload, dict):
                    raise ValueError("expected a JSON object")
                if marker_payload.get("commit") is not True:
                    raise ValueError("commit flag is not true")
                if marker_payload.get("stage_id") != namespace.stage:
                    raise ValueError("stage id mismatch")
                marker_identity = marker_payload.get("identity_fingerprint")
                if not isinstance(marker_identity, str) or not marker_identity:
                    raise ValueError("missing identity fingerprint")
                if namespace.identity is not None and marker_identity != namespace.identity:
                    raise ValueError("namespace identity mismatch")
                commit_identities = {
                    record.get("identity")
                    for record in partitions
                    if isinstance(record.get("identity"), str)
                }
                if commit_identities and commit_identities != {marker_identity}:
                    raise ValueError("marker identity does not match partition commits")
                marker_partitions = marker_payload.get("partitions")
                if not isinstance(marker_partitions, list):
                    raise ValueError("partitions must be a list")
                marker_ids: list[str] = []
                for entry in marker_partitions:
                    partition_id = entry.get("partition_id") if isinstance(entry, dict) else None
                    if not isinstance(partition_id, str):
                        raise ValueError("partition list contains an invalid id")
                    marker_ids.append(partition_id)
                commit_ids: list[str] = []
                for record in partitions:
                    committed_id = record.get("partition_id")
                    if isinstance(committed_id, str):
                        commit_ids.append(committed_id)
                if sorted(marker_ids) != sorted(commit_ids):
                    raise ValueError("partition list does not match commit files")
                finalized = True
            except (OSError, ValueError, json.JSONDecodeError) as error:
                finalization_error = f"invalid stage commit: {error}"
                invalid += 1
        identities = {str(p["identity"]) for p in partitions if p.get("identity")}
        reports.append({
            "stage": namespace.stage, "identity": namespace.identity,
            "layout": "fingerprint" if namespace.identity is not None else "legacy",
            "partitions": len(partitions), "invalid": invalid,
            "rows": total_rows, "size_bytes": total_bytes,
            "finalized": finalized, "finalization_error": finalization_error,
            "stale_attempts": len(attempts),
            # More than one identity means the tree mixes runs; only the current run's
            # commits would be reusable, so it is worth surfacing.
            "identities": sorted(identities), "details": partitions})
    return {"root": root, "verified": verify, "stages": reports,
            "invalid": sum(report["invalid"] for report in reports)}


def render_scan(scan: dict) -> str:
    lines = [f"checkpoint root: {scan['root']}"]
    if not scan["stages"]:
        lines.append("stages: none (no checkpoint stage namespace found)")
        return "\n".join(lines)
    for report in scan["stages"]:
        state = "finalized" if report["finalized"] else "in progress"
        namespace = report.get("identity") or _LEGACY_IDENTITY
        lines.append(
            f"stage {report['stage']}: {report['partitions']} partition(s), "
            f"{report['rows']} rows, {report['size_bytes']} bytes, {state} "
            f"(identity: {namespace})")
        if report["stale_attempts"]:
            lines.append(f"  stale attempts: {report['stale_attempts']} "
                         "(clear with --attempts-only)")
        if len(report["identities"]) > 1:
            lines.append(f"  warning: {len(report['identities'])} distinct identities; only "
                         "the current run's commits are reusable")
        for partition in report["details"]:
            if partition.get("valid") is False:
                lines.append(f"  invalid {partition.get('partition_id')}: "
                             f"{partition.get('error')}")
    verified = "checksums verified" if scan["verified"] else "metadata only (use verify)"
    lines.append(f"invalid partitions: {scan['invalid']} ({verified})")
    return "\n".join(lines)


def _is_descendant(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
    except ValueError:
        return False
    return target != root


def _remote_path(fs, path: str) -> str:
    raw = str(fs._strip_protocol(str(path))).replace("\\", "/")
    if any(component in {"", ".", ".."} for component in raw.strip("/").split("/")):
        raise ValueError(f"remote checkpoint path has an ambiguous component: {path!r}")
    return posixpath.normpath(f"/{raw.lstrip('/')}")


def _is_remote_descendant(root: str, target: str) -> bool:
    prefix = "/" if root == "/" else f"{root.rstrip('/')}/"
    return target != root and target.startswith(prefix)


def _remove_remote_literal(fs, path: str, *, root: str) -> None:
    """Recursively remove an exact backend path without invoking fsspec's glob-expanding rm."""
    normalized = _remote_path(fs, path)
    if not _is_remote_descendant(root, normalized):
        raise ValueError(
            f"refusing to delete checkpoint path outside root {root!r}: {path!r}")
    try:
        info = fs.info(path)
    except FileNotFoundError:
        return
    if info.get("type") not in {"directory", "dir"} and not fs.isdir(path):
        fs.rm_file(path)
        return
    for child in fs.ls(path, detail=True):
        child_name = str(child.get("name")) if isinstance(child, dict) else str(child)
        _remove_remote_literal(fs, child_name, root=root)
    try:
        fs.rmdir(path)
    except (FileNotFoundError, NotImplementedError):
        # Prefix-only object-store directories disappear when their last object is removed.
        pass


def _remove(root: str, path: str) -> None:
    """Remove one discovered node, refusing roots and paths outside the literal root."""
    from ..io import _fs, _local, is_remote

    if is_remote(root):
        if not is_remote(path):
            raise ValueError(f"refusing to delete non-remote path {path!r} from remote root {root!r}")
        root_scheme = str(root).split("://", 1)[0].lower()
        path_scheme = str(path).split("://", 1)[0].lower()
        if path_scheme != root_scheme:
            raise ValueError(f"refusing to delete path with a different protocol: {path!r}")
        fs = _fs(root)
        native_root = _remote_path(fs, root)
        native_path = _remote_path(fs, path)
        if not _is_remote_descendant(native_root, native_path):
            raise ValueError(
                f"refusing to delete checkpoint path outside root {root!r}: {path!r}")
        # ``AbstractFileSystem.rm`` expands glob metacharacters even when the caller supplies
        # one path. Walk through literal ``ls`` results and remove exact files/directories.
        _remove_remote_literal(
            fs, fs._strip_protocol(str(path)), root=native_root)
        return

    import shutil

    local_root = Path(_local(root)).expanduser().resolve(strict=True)
    local = Path(_local(path)).expanduser()
    if local.is_symlink():
        parent = local.parent.resolve(strict=True)
        if not _is_descendant(local_root, parent) and parent != local_root:
            raise ValueError(
                f"refusing to delete checkpoint link outside root {root!r}: {path!r}")
        local.unlink()
        return
    resolved = local.resolve(strict=False)
    if not _is_descendant(local_root, resolved):
        raise ValueError(f"refusing to delete checkpoint path outside root {root!r}: {path!r}")
    if resolved.is_dir():
        shutil.rmtree(resolved)
    else:
        resolved.unlink(missing_ok=True)


def clear_checkpoints(root: str, *, stage: str | None = None,
                      identity: str | None = None, attempts_only: bool = False) -> dict:
    """Delete stage checkpoint state and return what was removed."""
    from ..io import _ensure_remote_mutation_supported, exists, is_remote
    root = _normalize_root(root)
    if is_remote(root):
        _ensure_remote_mutation_supported(root)
    if not exists(root):
        raise FileNotFoundError(f"checkpoint root not found: {root}")
    namespaces = _select_namespaces(root, stage=stage, identity=identity)
    if stage is not None and identity is None:
        selectors = {namespace.selector for namespace in namespaces}
        if len(selectors) > 1:
            raise ValueError(
                f"stage {stage!r} exists in multiple identity namespaces "
                f"{sorted(selectors)}; pass --identity before deleting")

    removed = []
    for namespace in namespaces:
        if attempts_only:
            attempts_root = _join(namespace.stage_root, "attempts")
            for entry in _children(attempts_root):
                _remove(root, entry.path)
                removed.append(entry.path)
        else:
            _remove(root, namespace.stage_root)
            removed.append(namespace.stage_root)
    return {"root": root, "stages": [namespace.stage for namespace in namespaces],
            "identities": [namespace.selector for namespace in namespaces],
            "attempts_only": attempts_only,
            "removed": removed}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect, verify, and clear kuration stage checkpoints.")
    sub = ap.add_subparsers(dest="command", required=True)
    for name, description in (("list", "Summarize the stages a checkpoint root holds."),
                              ("verify", "Re-checksum every committed partition."),
                              ("clear", "Delete checkpoint state.")):
        parser = sub.add_parser(name, description=description, help=description)
        parser.add_argument("root", help="Checkpoint directory (the job's checkpoint_dir).")
        parser.add_argument("--stage", help="Restrict to one stage id.")
        parser.add_argument(
            "--identity",
            help="Restrict to one fingerprint namespace; use 'legacy' for root/stage trees.")
        parser.add_argument("--json", action="store_true", help="Emit structured JSON.")
        parser.add_argument("--debug", action="store_true",
                            help="Show full tracebacks on failure.")
        if name == "clear":
            parser.add_argument("--attempts-only", action="store_true",
                                help="Remove only stale attempt files, keeping commits.")
            parser.add_argument("--yes", action="store_true",
                                help="Confirm deletion; required.")
    args = ap.parse_args()

    def body():
        if args.command == "clear":
            if not args.yes:
                target = "stale attempts" if args.attempts_only else "committed checkpoints"
                raise SystemExit(
                    f"refusing to delete {target} under {args.root} without --yes "
                    f"(run 'kur-checkpoint list {args.root}' first)")
            result = clear_checkpoints(args.root, stage=args.stage,
                                       identity=args.identity,
                                       attempts_only=args.attempts_only)
            print(json_document(
                result, label="checkpoint clear result"
            ).rstrip() if args.json else
                  f"removed {len(result['removed'])} path(s) from "
                  f"{len(result['stages'])} stage(s) under {result['root']}")
            return
        scan = scan_checkpoints(args.root, stage=args.stage,
                                identity=args.identity,
                                verify=args.command == "verify")
        print(json_document(
            scan, label="checkpoint scan"
        ).rstrip() if args.json else render_scan(scan))
        if args.command == "verify" and scan["invalid"]:
            raise SystemExit(2)

    cli_run(body, debug=args.debug)


if __name__ == "__main__":
    main()
