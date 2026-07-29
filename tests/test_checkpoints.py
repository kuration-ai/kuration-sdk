"""Partition/stage checkpoint commit, reuse, and corruption behavior."""
from contextlib import contextmanager
import json
import os
from pathlib import Path

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


@pytest.mark.parametrize(
    "root",
    ["", "   ", os.path.abspath(os.path.sep), f"file://{os.path.abspath(os.path.sep)}",
     "memory://"],
)
def test_checkpoint_root_must_be_explicit_and_bounded(root):
    with pytest.raises(ValueError, match="checkpoint root"):
        StageCheckpoint(root, "features", _identity())


def test_remote_checkpoint_root_keeps_its_scheme():
    checkpoint = StageCheckpoint("memory://checkpoints", "features", _identity())
    assert checkpoint.root == "memory://checkpoints"
    assert checkpoint.stage_root == "memory://checkpoints/features"


@pytest.mark.parametrize(
    "root", ["memory://checkpoints/../other", "memory://checkpoints//other"]
)
def test_remote_checkpoint_root_rejects_ambiguous_components(root):
    with pytest.raises(ValueError, match="ambiguous remote path component"):
        StageCheckpoint(root, "features", _identity())


def test_checkpoint_rejects_commit_data_path_outside_its_partition(tmp_path):
    checkpoint = StageCheckpoint(str(tmp_path / "checkpoint"), "features", _identity())
    commit = checkpoint.write_partition("source:a", _batch("a"))
    outside = tmp_path / "outside.arrow"
    outside.write_bytes(Path(commit.data_file).read_bytes())
    _data_path, commit_path = checkpoint._paths("source:a")
    payload = json.loads(Path(commit_path).read_text())
    payload["data_file"] = str(outside)
    Path(commit_path).write_text(json.dumps(payload))

    assert checkpoint.load_commit("source:a") is None
    assert checkpoint.reusable("source:a") is None


def test_checkpoint_validation_uses_the_opened_generation_during_snapshot(
        tmp_path, monkeypatch):
    from kuration.core import checkpoints as checkpoints_module

    checkpoint = StageCheckpoint(str(tmp_path), "features", _identity())
    checkpoint.write_partition("source:a", _batch("a"))
    replacement = tmp_path / "replacement.arrow"
    _batch("replacement", 8).write_ipc(str(replacement))
    real_snapshot = checkpoints_module.local_snapshot

    @contextmanager
    def replace_after_snapshot(path):
        with real_snapshot(path) as snapshot:
            os.replace(replacement, path)
            yield snapshot

    monkeypatch.setattr(checkpoints_module, "local_snapshot", replace_after_snapshot)

    assert checkpoint.validate("source:a") is True


def test_checkpoint_validation_rejects_symlinks_and_special_files(tmp_path):
    checkpoint = StageCheckpoint(str(tmp_path), "features", _identity())
    commit = checkpoint.write_partition("source:a", _batch("a"))
    data_path = Path(commit.data_file)
    target = tmp_path / "target.arrow"
    os.replace(data_path, target)
    data_path.symlink_to(target)

    assert checkpoint.validate("source:a") is False

    data_path.unlink()
    os.mkfifo(data_path)
    assert checkpoint.validate("source:a") is False


def test_checkpoint_metadata_rejects_duplicate_keys_and_malformed_stage_entries(
        tmp_path):
    checkpoint = StageCheckpoint(str(tmp_path), "features", _identity())
    checkpoint.write_partition("source:a", _batch("a"))
    _data_path, commit_path = checkpoint._paths("source:a")
    commit_document = Path(commit_path).read_text()
    Path(commit_path).write_text(
        commit_document.replace(
            '"partition_id":"source:a"',
            '"partition_id":"source:a","partition_id":"other"',
            1,
        )
    )
    assert checkpoint.load_commit("source:a") is None

    marker = {
        "stage_id": "features",
        "identity": _identity().to_dict(),
        "identity_fingerprint": _identity().fingerprint,
        "partitions": ["not-an-object"],
        "rows": 1,
        "commit": True,
    }
    Path(checkpoint.commit_marker).write_text(json.dumps(marker))
    assert checkpoint.load_stage() is None
