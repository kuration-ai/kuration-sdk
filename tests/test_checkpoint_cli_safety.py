"""Checkpoint CLI discovery and deletion stay literal, namespaced, and root-bounded."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

import pytest

from kuration.core.cli.checkpoint import clear_checkpoints, main, scan_checkpoints
from kuration.core.checkpoints import _token


def _commit_payload(data_file: str, identity: str, payload: bytes) -> dict:
    return {
        "partition_id": "source:a",
        "identity": identity,
        "data_file": data_file,
        "rows": 1,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "columns": ["source_id"],
        "size_bytes": len(payload),
    }


def _write_local_stage(root: Path, *, stage: str = "stage000",
                       identity: str | None = None, attempt: bool = False) -> Path:
    stage_root = root / identity / stage if identity is not None else root / stage
    data = stage_root / "data" / f"{_token('source:a')}.arrow"
    commit = stage_root / "commits" / "part.json"
    data.parent.mkdir(parents=True, exist_ok=True)
    commit.parent.mkdir(parents=True, exist_ok=True)
    payload = b"checkpoint"
    data.write_bytes(payload)
    commit_payload = _commit_payload(
        str(data), identity or "legacy-identity", payload)
    commit.write_text(json.dumps(commit_payload))
    (stage_root / "stage.commit.json").write_text(json.dumps({
        "stage_id": stage,
        "identity_fingerprint": identity or "legacy-identity",
        "partitions": [commit_payload],
        "rows": 1,
        "commit": True,
    }))
    if attempt:
        attempts = stage_root / "attempts"
        attempts.mkdir()
        (attempts / "stale.arrow").write_bytes(b"stale")
    return stage_root


def _write_memory_stage(fs, root: str, *, stage: str = "stage000",
                        identity: str | None = None, attempt: bool = False) -> str:
    stage_root = f"{root}/{identity}/{stage}" if identity is not None else f"{root}/{stage}"
    data = f"{stage_root}/data/{_token('source:a')}.arrow"
    commit = f"{stage_root}/commits/part.json"
    payload = b"checkpoint"
    fs.pipe(data, payload)
    commit_payload = _commit_payload(
        data, identity or "legacy-identity", payload)
    fs.pipe(commit, json.dumps(commit_payload).encode())
    fs.pipe(f"{stage_root}/stage.commit.json", json.dumps({
        "stage_id": stage,
        "identity_fingerprint": identity or "legacy-identity",
        "partitions": [commit_payload],
        "rows": 1,
        "commit": True,
    }).encode())
    if attempt:
        fs.pipe(f"{stage_root}/attempts/stale.arrow", b"stale")
    return stage_root


def test_scan_supports_fingerprint_namespaces_and_legacy_layout(tmp_path):
    root = tmp_path / "checkpoints"
    first = "a" * 64
    second = "b" * 64
    _write_local_stage(root, stage="legacy-features")
    _write_local_stage(root, identity=first)
    _write_local_stage(root, identity=second)

    scan = scan_checkpoints(str(root), verify=True)
    discovered = {(report["identity"], report["stage"]) for report in scan["stages"]}
    assert discovered == {
        (None, "legacy-features"),
        (first, "stage000"),
        (second, "stage000"),
    }
    assert scan["invalid"] == 0


@pytest.mark.parametrize(
    "root",
    ["", "   ", os.path.abspath(os.path.sep), f"file://{os.path.abspath(os.path.sep)}",
     "memory://"],
)
def test_checkpoint_root_must_be_explicit_and_bounded(root):
    with pytest.raises(ValueError, match="checkpoint root"):
        scan_checkpoints(root)
    with pytest.raises(ValueError, match="checkpoint root"):
        clear_checkpoints(root)


@pytest.mark.parametrize("root", ["sftp://host/checkpoints", "ftp://host/checkpoints"])
def test_checkpoint_clear_rejects_hierarchical_remote_backends(root):
    with pytest.raises(ValueError, match="remote mutation protocol"):
        clear_checkpoints(root)


def test_clear_requires_identity_for_an_ambiguous_stage_and_cli_selects_one(
        tmp_path, monkeypatch, capsys):
    root = tmp_path / "checkpoints"
    first = "a" * 64
    second = "b" * 64
    first_stage = _write_local_stage(root, identity=first)
    second_stage = _write_local_stage(root, identity=second)

    with pytest.raises(ValueError, match="--identity"):
        clear_checkpoints(str(root), stage="stage000")

    monkeypatch.setattr(sys, "argv", [
        "kur-checkpoint", "clear", str(root), "--stage", "stage000",
        "--identity", first, "--yes", "--json",
    ])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["identities"] == [first]
    assert not first_stage.exists()
    assert second_stage.exists()


def test_local_glob_metacharacters_are_literal_and_deletion_stays_under_root(tmp_path):
    root = tmp_path / "ckpt[1]"
    glob_match = tmp_path / "ckpt1"
    target = _write_local_stage(root, stage="features", attempt=True)
    outside = _write_local_stage(glob_match, stage="features", attempt=True)

    clear_checkpoints(str(root), stage="features", attempts_only=True)
    assert not (target / "attempts" / "stale.arrow").exists()
    assert (outside / "attempts" / "stale.arrow").exists()

    clear_checkpoints(str(root), stage="features")
    assert not target.exists()
    assert outside.exists()


def test_attempt_symlink_removal_never_follows_outside_the_root(tmp_path):
    root = tmp_path / "checkpoints"
    stage = _write_local_stage(root, stage="features")
    outside = tmp_path / "outside.arrow"
    outside.write_bytes(b"keep")
    attempts = stage / "attempts"
    attempts.mkdir()
    link = attempts / "stale.arrow"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    clear_checkpoints(str(root), stage="features", attempts_only=True)
    assert not link.exists()
    assert outside.read_bytes() == b"keep"


def test_memory_root_with_glob_metacharacters_is_literal_and_namespaced():
    fsspec = pytest.importorskip("fsspec")
    fs = fsspec.filesystem("memory")
    prefix = f"memory://checkpoint-cli-{uuid.uuid4().hex}"
    root = f"{prefix}/ckpt[1]"
    glob_match = f"{prefix}/ckpt1"
    identity = "c" * 64
    target = _write_memory_stage(fs, root, identity=identity, attempt=True)
    outside = _write_memory_stage(fs, glob_match, identity=identity, attempt=True)
    try:
        scan = scan_checkpoints(root, identity=identity, verify=True)
        assert [(report["identity"], report["stage"]) for report in scan["stages"]] == [
            (identity, "stage000")
        ]
        assert scan["invalid"] == 0

        clear_checkpoints(
            root, stage="stage000", identity=identity, attempts_only=True)
        assert not fs.exists(f"{target}/attempts/stale.arrow")
        assert fs.exists(f"{outside}/attempts/stale.arrow")

        clear_checkpoints(root, stage="stage000", identity=identity)
        assert not fs.exists(target)
        assert fs.exists(outside)
    finally:
        fs.rm(prefix, recursive=True)


@pytest.mark.parametrize("payload", [[], {"rows": "oops"}])
def test_scan_records_malformed_commit_instead_of_crashing(tmp_path, payload):
    root = tmp_path / "checkpoints"
    stage = _write_local_stage(root, stage="features")
    (stage / "commits" / "part.json").write_text(json.dumps(payload))

    scan = scan_checkpoints(str(root), verify=True)

    assert scan["invalid"] >= 1
    assert scan["stages"][0]["details"][0]["valid"] is False
    assert "commit" in scan["stages"][0]["details"][0]["error"]


def test_scan_does_not_treat_malformed_stage_marker_as_finalized(tmp_path):
    root = tmp_path / "checkpoints"
    stage = _write_local_stage(root, stage="features")
    (stage / "stage.commit.json").write_text("{}")

    scan = scan_checkpoints(str(root), verify=True)
    report = scan["stages"][0]

    assert report["finalized"] is False
    assert report["finalization_error"].startswith("invalid stage commit:")
    assert scan["invalid"] == 1
