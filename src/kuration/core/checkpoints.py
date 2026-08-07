"""Atomic physical-stage and partition checkpoints.

Partition commits are independent files, so concurrent workers never rewrite a shared manifest.
The stage commit is published last after reconstructing and validating all expected partition
commits.  A restart therefore trusts only checksum-valid partition commits with the same
execution identity.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
from typing import Any, Iterable, Iterator

from .columnar import SegmentBatch
from .io import (
    exists,
    file_integrity,
    is_remote,
    local_snapshot,
    read_text,
    staged_write,
)
from .jsonutil import parse_json_object


def canonical_hash(value: Any) -> str:
    """Return a deterministic SHA-256 hash of one JSON-compatible value."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CheckpointIdentity:
    logical_plan_hash: str
    physical_plan_hash: str
    input_fingerprints: tuple[dict, ...] = ()
    schema: dict = field(default_factory=dict)
    components: tuple[dict, ...] = ()
    parameters: dict = field(default_factory=dict)

    @classmethod
    def from_plans(cls, logical_plan: Any, physical_plan: Any, **metadata):
        logical = logical_plan.to_dict() if hasattr(logical_plan, "to_dict") else logical_plan
        physical = physical_plan.to_dict() if hasattr(physical_plan, "to_dict") else physical_plan
        return cls(logical_plan_hash=canonical_hash(logical),
                   physical_plan_hash=canonical_hash(physical), **metadata)

    @property
    def fingerprint(self) -> str:
        return canonical_hash(asdict(self))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PartitionCommit:
    partition_id: str
    identity: str
    data_file: str
    rows: int
    sha256: str
    columns: tuple[str, ...]
    size_bytes: int = 0
    key_range: tuple[Any, Any] | None = None
    reducer_state: dict | None = None
    errors: tuple[dict, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict) -> "PartitionCommit":
        if not isinstance(raw, dict):
            raise ValueError("partition commit must be an object")
        data = dict(raw)
        data.setdefault("size_bytes", 0)
        for name in ("partition_id", "identity", "data_file"):
            if not isinstance(data.get(name), str) or not data[name]:
                raise ValueError(f"partition commit {name} must be a non-empty string")
        rows = data.get("rows")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
            raise ValueError("partition commit rows must be a non-negative integer")
        digest = data.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("partition commit sha256 must be a lowercase SHA-256 digest")
        size = data["size_bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("partition commit size_bytes must be a non-negative integer")
        columns = data.get("columns")
        if (
            not isinstance(columns, (list, tuple))
            or any(not isinstance(column, str) or not column for column in columns)
            or len(set(columns)) != len(columns)
        ):
            raise ValueError(
                "partition commit columns must be unique non-empty strings"
            )
        data["columns"] = tuple(columns)
        if data.get("key_range") is not None:
            if (
                not isinstance(data["key_range"], (list, tuple))
                or len(data["key_range"]) != 2
            ):
                raise ValueError("partition commit key_range must contain two bounds")
            data["key_range"] = tuple(
                tuple(value) if isinstance(value, list) else value
                for value in data["key_range"])
        reducer_state = data.get("reducer_state")
        if reducer_state is not None and not isinstance(reducer_state, dict):
            raise ValueError("partition commit reducer_state must be an object or null")
        errors = data.get("errors", ())
        if (
            not isinstance(errors, (list, tuple))
            or any(not isinstance(error, dict) for error in errors)
        ):
            raise ValueError("partition commit errors must be a list of objects")
        data["errors"] = tuple(errors)
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


@contextmanager
def partition_snapshot(partition: PartitionCommit | str) -> Iterator[str]:
    """Yield one private, immutable local snapshot of a partition.

    ``PartitionCommit`` inputs are checksum-, size-, row-, and schema-validated against that
    same snapshot. Bare local paths require descriptor-relative POSIX traversal because they
    have no independent digest with which to detect a redirected fallback read.
    """
    commit = partition if isinstance(partition, PartitionCommit) else None
    path = commit.data_file if commit is not None else str(partition)
    if commit is None and not is_remote(path):
        from . import io as io_module
        if not io_module._HAS_POSIX_DESCRIPTOR_OPEN:
            raise ValueError(
                "bare local reducer partition paths require secure descriptor-relative "
                "traversal on this platform; provide a checksum-bound PartitionCommit"
            )
    with local_snapshot(path) as snapshot:
        if commit is not None:
            integrity = file_integrity(snapshot)
            if (
                integrity["sha256"] != commit.sha256
                or integrity["size_bytes"] != commit.size_bytes
            ):
                raise ValueError("checkpoint partition integrity does not match its commit")
            batch = SegmentBatch.read_ipc(snapshot)
            if len(batch) != commit.rows or batch.column_names != commit.columns:
                raise ValueError("checkpoint partition shape does not match its commit")
        yield snapshot


@contextmanager
def pinned_partition_files(
    partitions: Iterable[PartitionCommit | str],
) -> Iterator[tuple[str, ...]]:
    """Hold private snapshots for every partition until a reducer has finished."""
    with ExitStack() as stack:
        snapshots = tuple(
            stack.enter_context(partition_snapshot(partition))
            for partition in partitions
        )
        yield snapshots


def artifact_checkpoint_summary(marker: dict) -> dict:
    """Return portable checkpoint evidence suitable for a committed artifact config."""
    partitions = []
    for entry in marker.get("partitions", ()):
        if not isinstance(entry, dict):
            continue
        partition_id = str(entry.get("partition_id", ""))
        partitions.append({
            "partition_id_sha256": hashlib.sha256(partition_id.encode()).hexdigest(),
            "identity": entry.get("identity"),
            "rows": entry.get("rows"),
            "sha256": entry.get("sha256"),
            "columns": sorted(str(column) for column in (entry.get("columns") or ())),
            "size_bytes": entry.get("size_bytes", 0),
        })
    partitions.sort(key=lambda entry: (
        entry["partition_id_sha256"],
        str(entry.get("identity") or ""),
        str(entry.get("sha256") or ""),
    ))
    return {
        "stage_id": marker.get("stage_id"),
        "identity": marker.get("identity"),
        "identity_fingerprint": marker.get("identity_fingerprint"),
        "partitions": partitions,
        "partition_count": len(partitions),
        "rows": marker.get("rows", 0),
        "commit": marker.get("commit") is True,
    }


def _join(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/"), *(part.strip("/") for part in parts)])


def _normalize_checkpoint_root(root: str) -> str:
    """Keep checkpoint roots explicit, scheme-preserving, and narrower than a filesystem root."""
    from .io import _normalize_explicit_path
    return _normalize_explicit_path(root, label="checkpoint root")


def _same_storage_path(left: str, right: str) -> bool:
    """Compare local/fsspec paths without treating textual URL spelling as authority."""
    if is_remote(left) != is_remote(right):
        return False
    if is_remote(left):
        import posixpath
        from .io import _fs

        left_fs = _fs(left)
        right_fs = _fs(right)
        if getattr(left_fs, "_fs_token_", None) != getattr(right_fs, "_fs_token_", None):
            return False
        left_raw = str(left_fs._strip_protocol(left))
        right_raw = str(right_fs._strip_protocol(right))
        if any(
            component in {"", ".", ".."}
            for raw in (left_raw.strip("/"), right_raw.strip("/"))
            for component in raw.split("/")
        ):
            return False
        left_path = posixpath.normpath("/" + left_raw.lstrip("/"))
        right_path = posixpath.normpath("/" + right_raw.lstrip("/"))
        return left_path == right_path
    from .io import _local
    return Path(_local(left)).resolve(strict=False) == Path(_local(right)).resolve(strict=False)


def _token(partition_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "-", partition_id).strip("-")[:48] or "partition"
    return f"{readable}-{hashlib.sha256(partition_id.encode()).hexdigest()[:12]}"


def _atomic_json(path: str, value: dict) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with staged_write(path) as local_path:
        with open(local_path, "w", encoding="utf-8") as file:
            file.write(payload)


def _publish_attempt(attempt: str, final: str) -> None:
    if not is_remote(final):
        from .io import _local
        attempt, final = _local(attempt), _local(final)
        Path(final).parent.mkdir(parents=True, exist_ok=True)
        os.replace(attempt, final)
        return
    # fsspec object moves publish a completed attempt object under the stable name.
    from .io import _ensure_remote_mutation_supported, _fs
    _ensure_remote_mutation_supported(final)
    fs = _fs(final)
    fs.makedirs(final.rsplit("/", 1)[0], exist_ok=True)
    # Do not remove the last complete checkpoint before its replacement is publishable.
    fs.mv(attempt, final)


class StageCheckpoint:
    """Checkpoint namespace for one physical stage and one execution identity."""

    def __init__(self, root: str, stage_id: str, identity: CheckpointIdentity):
        if (not isinstance(stage_id, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", stage_id)
                or stage_id in {".", ".."}):
            raise ValueError("stage_id must be a safe non-empty name using letters, digits, "
                             "dot, underscore, or hyphen")
        self.root = _normalize_checkpoint_root(root)
        if is_remote(self.root) and any(
                character in self.root for character in "*?["):
            raise ValueError(
                "remote checkpoint root cannot contain '*', '?', or '[' because backend "
                "mutation APIs may interpret them as glob patterns"
            )
        self.stage_id = stage_id
        self.identity = identity
        self.stage_root = _join(self.root, stage_id)

    def _paths(self, partition_id: str) -> tuple[str, str]:
        token = _token(partition_id)
        return (_join(self.stage_root, "data", f"{token}.arrow"),
                _join(self.stage_root, "commits", f"{token}.json"))

    def _attempt_path(self, partition_id: str) -> str:
        token = _token(partition_id)
        return _join(self.stage_root, "attempts", f"{token}.{uuid.uuid4().hex}.arrow")

    @property
    def commit_marker(self) -> str:
        return _join(self.stage_root, "stage.commit.json")

    def load_commit(self, partition_id: str) -> PartitionCommit | None:
        expected_data_file, commit_path = self._paths(partition_id)
        if not exists(commit_path):
            return None
        try:
            commit = PartitionCommit.from_dict(parse_json_object(
                read_text(commit_path), label=f"partition commit {commit_path!r}"
            ))
        except (OSError, ValueError, TypeError):
            return None
        if (commit.partition_id != partition_id
                or commit.identity != self.identity.fingerprint
                or not _same_storage_path(commit.data_file, expected_data_file)):
            return None
        return commit

    def validate(self, partition_id: str) -> bool:
        commit = self.load_commit(partition_id)
        if commit is None or not exists(commit.data_file):
            return False
        try:
            with partition_snapshot(commit):
                return True
        except (OSError, ValueError):
            return False

    def reusable(self, partition_id: str) -> PartitionCommit | None:
        return self.load_commit(partition_id) if self.validate(partition_id) else None

    def write_partition(self, partition_id: str, batch: SegmentBatch, *,
                        key_range: tuple[Any, Any] | None = None,
                        reducer_state: dict | None = None,
                        errors=()) -> PartitionCommit:
        reusable = self.reusable(partition_id)
        if reusable is not None:
            return reusable
        final_path, commit_path = self._paths(partition_id)
        attempt_path = self._attempt_path(partition_id)
        try:
            metadata = batch.write_ipc(attempt_path)
            # Validate the attempt before it can become visible through a commit.
            attempt = SegmentBatch.read_ipc(attempt_path)
            if len(attempt) != len(batch) or attempt.column_names != batch.column_names:
                raise RuntimeError(f"checkpoint attempt validation failed for {partition_id!r}")
            _publish_attempt(attempt_path, final_path)
            commit = PartitionCommit(
                partition_id=partition_id, identity=self.identity.fingerprint,
                data_file=final_path, rows=len(batch), sha256=metadata["sha256"],
                columns=batch.column_names, size_bytes=metadata["size_bytes"], key_range=key_range,
                reducer_state=dict(reducer_state) if reducer_state is not None else None,
                errors=tuple(dict(error) for error in errors))
            _atomic_json(commit_path, commit.to_dict())
            if not self.validate(partition_id):
                raise RuntimeError(f"published checkpoint validation failed for {partition_id!r}")
            return commit
        finally:
            # A failed validation must not accumulate attempts indefinitely.
            if exists(attempt_path):
                if is_remote(attempt_path):
                    from .io import _ensure_remote_mutation_supported, _fs
                    _ensure_remote_mutation_supported(attempt_path)
                    fs = _fs(attempt_path)
                    fs.rm_file(fs._strip_protocol(attempt_path))
                else:
                    from .io import _local
                    Path(_local(attempt_path)).unlink(missing_ok=True)

    def committed(self, partition_ids) -> tuple[PartitionCommit, ...]:
        commits = []
        for partition_id in partition_ids:
            commit = self.reusable(partition_id)
            if commit is not None:
                commits.append(commit)
        return tuple(commits)

    def finalize(self, partition_ids) -> dict:
        expected = tuple(partition_ids)
        commits = self.committed(expected)
        present = {commit.partition_id for commit in commits}
        missing = [partition_id for partition_id in expected if partition_id not in present]
        if missing:
            raise RuntimeError(f"cannot commit stage {self.stage_id!r}; missing or invalid "
                               f"partitions {missing}")
        marker = {"stage_id": self.stage_id, "identity": self.identity.to_dict(),
                  "identity_fingerprint": self.identity.fingerprint,
                  "partitions": [commit.to_dict() for commit in commits],
                  "rows": sum(commit.rows for commit in commits),
                  "commit": True}
        _atomic_json(self.commit_marker, marker)
        return marker

    def load_stage(self) -> dict | None:
        if not exists(self.commit_marker):
            return None
        try:
            marker = parse_json_object(
                read_text(self.commit_marker),
                label=f"stage commit {self.commit_marker!r}",
            )
        except (OSError, ValueError):
            return None
        if (marker.get("commit") is not True
                or marker.get("identity_fingerprint") != self.identity.fingerprint):
            return None
        entries = marker.get("partitions")
        if (
            not isinstance(entries, list)
            or any(not isinstance(entry, dict) for entry in entries)
        ):
            return None
        ids = [entry.get("partition_id") for entry in entries]
        if (
            any(not isinstance(partition_id, str) or not partition_id
                for partition_id in ids)
            or len(set(ids)) != len(ids)
        ):
            return None
        return marker if len(self.committed(ids)) == len(ids) else None
