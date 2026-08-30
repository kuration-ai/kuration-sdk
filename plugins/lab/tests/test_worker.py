"""The worker subprocess on real jobs: happy path, failure, and SIGTERM cancellation."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from conftest import make_csv, make_job

from kuration_sdk_lab.jobs import protocol
from kuration_sdk_lab.jobs.registry import RunRegistry


def _prepare_run(tmp_path, job) -> Path:
    registry = RunRegistry(tmp_path / "runs")
    info = registry.create(job, kind="curate")
    return registry.run_dir(info["run_id"])


def _run_worker(run_dir, *, wait=True):
    proc = subprocess.Popen(
        [sys.executable, "-m", "kuration_sdk_lab.jobs.worker", str(run_dir)],
        stdout=open(run_dir / protocol.LOG_FILE, "ab"),
        stderr=subprocess.STDOUT, start_new_session=True)
    if wait:
        assert proc.wait(timeout=180) is not None
    return proc


def test_worker_curates_and_persists_review(tmp_path):
    job = make_job(tmp_path)
    run_dir = _prepare_run(tmp_path, job)
    proc = _run_worker(run_dir)
    log = (run_dir / protocol.LOG_FILE).read_text()
    assert proc.returncode == 0, log

    status = protocol.read_status(run_dir)
    assert status["state"] == protocol.STATE_SUCCEEDED, log
    assert status["exit"] == {"ok": True}

    events, _ = protocol.read_events(run_dir)
    kinds = [(e["phase"], e["event"]) for e in events]
    assert kinds[0] == ("lifecycle", "start")
    assert kinds[-1] == ("lifecycle", "complete")
    assert any(phase == "scoring" for phase, _ in kinds)
    assert all(isinstance(e["seq"], int) for e in events)

    result = json.loads((run_dir / protocol.RESULT_FILE).read_text())
    assert result["kind"] == "curate"
    assert result["summary"]["shards"] >= 1
    assert "review" in result and "errors" in result

    from kuration.core.manifest import dataset_state

    assert dataset_state(job["out"]) == "committed"


def test_worker_marks_invalid_job_failed_with_hint_shape(tmp_path):
    run_dir = _prepare_run(tmp_path, {"job_schema_version": "1.0", "bogus_key": True})
    proc = _run_worker(run_dir)
    assert proc.returncode == 1

    status = protocol.read_status(run_dir)
    assert status["state"] == protocol.STATE_FAILED
    assert status["exit"]["ok"] is False
    assert status["exit"]["error_type"]
    assert status["exit"]["message"]
    assert "hint" in status["exit"]
    events, _ = protocol.read_events(run_dir)
    assert events[-1]["event"] == "error"
    assert not (run_dir / protocol.RESULT_FILE).exists()


def test_sigterm_cancels_via_progress_callback(tmp_path):
    # Many small sources -> one progress advance each, so a SIGTERM lands mid-run and the
    # callback raises within one source.
    sources = [str(make_csv(tmp_path, f"s{i:03d}.csv", steps=400)) for i in range(40)]
    job = make_job(tmp_path, sources=sources, train_hours=0.5)
    run_dir = _prepare_run(tmp_path, job)
    proc = _run_worker(run_dir, wait=False)

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        events, _ = protocol.read_events(run_dir)
        if any(e["phase"] == "scoring" and e["event"] == "advance" for e in events):
            break
        if proc.poll() is not None:
            break
        time.sleep(0.01)
    os.killpg(proc.pid, signal.SIGTERM)
    proc.wait(timeout=60)

    status = protocol.read_status(run_dir)
    log = (run_dir / protocol.LOG_FILE).read_text()
    assert status["state"] == protocol.STATE_CANCELLED, (status, log)
    assert status["exit"]["cancelled"] is True
    events, _ = protocol.read_events(run_dir)
    assert events[-1] == {**events[-1], "phase": "lifecycle", "event": "cancelled"}
