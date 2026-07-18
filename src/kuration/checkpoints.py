"""Atomic physical-stage and partition checkpoints.

Partition commits are independent files, so concurrent workers never rewrite a shared manifest.
The stage commit is published last after reconstructing and validating all expected partition
commits.  A restart therefore trusts only checksum-valid partition commits with the same
execution identity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
from typing import Any

from .columnar import SegmentBatch
from .io import exists, is_remote, local_copy, read_text, staged_write


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
        data = dict(raw)
        data.setdefault("size_bytes", 0)
        data["columns"] = tuple(data.get("columns", ()))
        if data.get("key_range") is not None:
            data["key_range"] = tuple(
                tuple(value) if isinstance(value, list) else value
                for value in data["key_range"])
        data["errors"] = tuple(data.get("errors", ()))
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


def _join(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/"), *(part.strip("/") for part in parts)])


def _token(partition_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "-", partition_id).strip("-")[:48] or "partition"
    return f"{readable}-{hashlib.sha256(partition_id.encode()).hexdigest()[:12]}"


def _atomic_json(path: str, value: dict) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with staged_write(path) as local_path:
        with open(local_path, "w", encoding="utf-8") as file:
            file.write(payload)


def _checksum(path: str) -> str:
    digest = hashlib.sha256()
    with local_copy(path) as local_path, open(local_path, "rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_attempt(attempt: str, final: str) -> None:
    if not is_remote(final):
        from .io import _local
        attempt, final = _local(attempt), _local(final)
        Path(final).parent.mkdir(parents=True, exist_ok=True)
        os.replace(attempt, final)
        return
    # fsspec object moves publish a completed attempt object under the stable name.
    from .io import _fs
    fs = _fs(final)
    fs.makedirs(final.rsplit("/", 1)[0], exist_ok=True)
    if fs.exists(final):
        fs.rm(final)
    fs.mv(attempt, final)


class StageCheckpoint:
    """Checkpoint namespace for one physical stage and one execution identity."""

    def __init__(self, root: str, stage_id: str, identity: CheckpointIdentity):
        if (not isinstance(stage_id, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", stage_id)
                or stage_id in {".", ".."}):
            raise ValueError("stage_id must be a safe non-empty name using letters, digits, "
                             "dot, underscore, or hyphen")
        self.root = str(root).rstrip("/")
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
        _, commit_path = self._paths(partition_id)
        if not exists(commit_path):
            return None
        try:
            commit = PartitionCommit.from_dict(json.loads(read_text(commit_path)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        if commit.partition_id != partition_id or commit.identity != self.identity.fingerprint:
            return None
        return commit

    def validate(self, partition_id: str) -> bool:
        commit = self.load_commit(partition_id)
        if commit is None or not exists(commit.data_file):
            return False
        try:
            if _checksum(commit.data_file) != commit.sha256:
                return False
            batch = SegmentBatch.read_ipc(commit.data_file)
            return len(batch) == commit.rows and batch.column_names == commit.columns
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
                    from .io import _fs
                    _fs(attempt_path).rm(attempt_path)
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
            marker = json.loads(read_text(self.commit_marker))
        except (OSError, json.JSONDecodeError):
            return None
        if (marker.get("commit") is not True
                or marker.get("identity_fingerprint") != self.identity.fingerprint):
            return None
        ids = [entry.get("partition_id") for entry in marker.get("partitions", ())]
        if any(not isinstance(partition_id, str) for partition_id in ids):
            return None
        return marker if len(self.committed(ids)) == len(ids) else None
