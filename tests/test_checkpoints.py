"""Partition/stage checkpoint commit, reuse, and corruption behavior."""
import json

import pytest

from kuration import (CheckpointIdentity, Segment, SegmentBatch, StageCheckpoint,
                       canonical_hash)

pytest.importorskip("pyarrow")


def _batch(source, offset=0):
    return SegmentBatch.from_segments([
        Segment(source, offset, offset + 4, (offset, offset + 3), {"score": 0.5})
    ])


def _identity(physical="physical-a"):
    return CheckpointIdentity(logical_plan_hash=canonical_hash({"pipeline": 1}),
                              physical_plan_hash=canonical_hash({"plan": physical}),
                              input_fingerprints=({"source": "a", "etag": "v1"},),
                              schema={"name": "demo"}, parameters={"seed": 7})


def test_partition_commit_is_validated_reused_and_stage_committed_last(tmp_path):
    checkpoint = StageCheckpoint(str(tmp_path), "features", _identity())
    first = checkpoint.write_partition("source:a", _batch("a"),
                                       key_range=(("a", 0, 4), ("a", 0, 4)),
                                       reducer_state={"count": 1})
    reused = checkpoint.write_partition("source:a", _batch("a", 10))
    assert reused == first
    assert checkpoint.validate("source:a")
    with pytest.raises(RuntimeError, match="missing or invalid"):
        checkpoint.finalize(["source:a", "source:b"])
    checkpoint.write_partition("source:b", _batch("b"))
    marker = checkpoint.finalize(["source:a", "source:b"])
    assert marker["rows"] == 2
    assert checkpoint.load_stage()["identity_fingerprint"] == _identity().fingerprint


def test_corrupt_partition_is_not_reused_and_is_recomputed(tmp_path):
    checkpoint = StageCheckpoint(str(tmp_path), "features", _identity())
    commit = checkpoint.write_partition("a", _batch("a"))
    with open(commit.data_file, "ab") as file:
        file.write(b"corrupt")
    assert checkpoint.reusable("a") is None
    repaired = checkpoint.write_partition("a", _batch("a", 4))
    assert checkpoint.validate("a")
    assert repaired.sha256 != commit.sha256


def test_changed_plan_identity_never_reuses_partition_or_stage(tmp_path):
    original = StageCheckpoint(str(tmp_path), "features", _identity())
    original.write_partition("a", _batch("a"))
    original.finalize(["a"])
    changed = StageCheckpoint(str(tmp_path), "features", _identity("physical-b"))
    assert changed.reusable("a") is None
    assert changed.load_stage() is None


def test_commit_files_are_canonical_and_attempts_are_not_left_behind(tmp_path):
    checkpoint = StageCheckpoint(str(tmp_path), "features", _identity())
    commit = checkpoint.write_partition("source/a", _batch("a"))
    commits = list((tmp_path / "features" / "commits").glob("*.json"))
    assert len(commits) == 1
    assert json.loads(commits[0].read_text())["data_file"] == commit.data_file
    assert not list((tmp_path / "features" / "attempts").glob("*.arrow"))


def test_checkpoint_rejects_stage_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="safe"):
        StageCheckpoint(str(tmp_path), "../outside", _identity())
