"""Run registry: creation, reconciliation of dead workers, purge rules."""
from __future__ import annotations

import pytest

from kuration_sdk_lab.jobs import protocol
from kuration_sdk_lab.jobs.registry import RunRegistry, _resume_available


@pytest.fixture
def registry(tmp_path):
    return RunRegistry(tmp_path / "runs")


def test_create_and_info(registry):
    info = registry.create({"out": "/data/x", "train_hours": 1}, kind="curate",
                           submitter="eric")
    assert info["state"] == protocol.STATE_PENDING
    assert info["kind"] == "curate"
    assert info["out"] == "/data/x"
    assert info["submitter"] == "eric"
    assert info["has_result"] is False
    assert registry.job(info["run_id"])["train_hours"] == 1
    assert [r["run_id"] for r in registry.list()] == [info["run_id"]]


def test_running_with_dead_pid_reconciles_to_orphaned(registry):
    info = registry.create({}, kind="curate")
    registry.mark(info["run_id"], protocol.STATE_RUNNING, pid=2**22 + 12345)  # not a live pid
    reconciled = registry.info(info["run_id"])
    assert reconciled["state"] == protocol.STATE_ORPHANED
    assert "died" in reconciled["exit"]["message"]
    # and the reconciliation is durable, not a per-read illusion
    assert protocol.read_status(registry.run_dir(info["run_id"]))["state"] \
        == protocol.STATE_ORPHANED


def test_pending_ordering_is_fifo(registry):
    first = registry.create({}, kind="curate")
    second = registry.create({}, kind="curate")
    assert [r["run_id"] for r in registry.pending()] \
        == [first["run_id"], second["run_id"]]


def test_purge_refuses_live_runs(registry):
    info = registry.create({}, kind="curate")
    with pytest.raises(ValueError):
        registry.purge(info["run_id"])
    registry.mark(info["run_id"], protocol.STATE_FAILED, exit={"ok": False})
    registry.purge(info["run_id"])
    assert registry.info(info["run_id"]) is None


def test_invalid_run_ids_are_rejected(registry):
    with pytest.raises(ValueError):
        registry.run_dir("../escape")
    with pytest.raises(ValueError):
        registry.run_dir("")


def test_resume_not_available_for_absent_or_committed_out(tmp_path):
    assert _resume_available("curate", protocol.STATE_FAILED, str(tmp_path / "nope")) is False
    assert _resume_available("report", protocol.STATE_FAILED, str(tmp_path)) is False
    assert _resume_available("curate", protocol.STATE_SUCCEEDED, str(tmp_path)) is False
