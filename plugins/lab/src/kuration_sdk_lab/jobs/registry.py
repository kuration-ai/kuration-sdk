"""Durable run registry: one directory per run under the configured runs dir.

The disk is the source of truth — the in-memory side (the JobManager) only holds live
subprocess handles. Everything here re-reads ``run.json``/``status.json`` per call, so runs
survive a Jupyter server restart and are discoverable by a fresh process.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from . import protocol

_RUN_ID = re.compile(r"[A-Za-z0-9_-]+\Z")


def _pid_alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _last_event(run_dir: Path) -> dict | None:
    """Read the last parseable event line without scanning the whole file."""
    path = run_dir / protocol.EVENTS_FILE
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 8192))
            tail = handle.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return None
    for line in reversed(tail.strip().splitlines()):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            return event
    return None


def _resume_available(kind: str, state: str, out) -> bool:
    """A curate run that died mid-write can be resumed (``write_dataset(resume=True)``)."""
    if kind != "curate" or not isinstance(out, str):
        return False
    if state not in {protocol.STATE_FAILED, protocol.STATE_CANCELLED, protocol.STATE_ORPHANED}:
        return False
    try:
        from kuration.core.manifest import dataset_state

        return dataset_state(out) == "writing"
    except Exception:
        return False


class RunRegistry:
    def __init__(self, runs_dir: str | Path):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        if not _RUN_ID.match(run_id or ""):
            raise ValueError(f"invalid run id: {run_id!r}")
        return self.runs_dir / run_id

    def create(self, job: dict, *, kind: str, submitter: str | None = None) -> dict:
        run_id = protocol.new_run_id()
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True)
        protocol.write_json_atomic(run_dir / protocol.JOB_FILE, job)
        protocol.write_json_atomic(run_dir / protocol.RUN_FILE, {
            "run_id": run_id,
            "kind": kind,
            "out": job.get("out"),
            "created": protocol.utcnow(),
            "submitter": submitter,
        })
        protocol.write_status(run_dir, protocol.STATE_PENDING)
        return self.info(run_id)

    def info(self, run_id: str) -> dict | None:
        """Merged, reconciled, JSON-ready view of one run."""
        run_dir = self.run_dir(run_id)
        run_meta = protocol.read_json(run_dir / protocol.RUN_FILE)
        if run_meta is None:
            return None
        status = self._reconciled_status(run_dir)
        kind = run_meta.get("kind") or "curate"
        out = run_meta.get("out")
        state = status.get("state", protocol.STATE_PENDING)
        return {
            "run_id": run_meta.get("run_id") or run_id,
            "kind": kind,
            "out": out,
            "created": run_meta.get("created"),
            "submitter": run_meta.get("submitter"),
            "state": state,
            "updated": status.get("updated"),
            "exit": status.get("exit"),
            "pid": status.get("pid"),
            "has_result": (run_dir / protocol.RESULT_FILE).is_file(),
            "resume_available": _resume_available(kind, state, out),
            "last_event": _last_event(run_dir),
        }

    def _reconciled_status(self, run_dir: Path) -> dict:
        """Fix up a status left behind by a dead worker (server crash, SIGKILL)."""
        status = protocol.read_status(run_dir)
        if status.get("state") == protocol.STATE_RUNNING and not _pid_alive(status.get("pid")):
            status = protocol.write_status(
                run_dir, protocol.STATE_ORPHANED,
                pid=status.get("pid"),
                exit={"ok": False, "message": "worker process died without a terminal status"},
            )
        return status

    def list(self) -> list[dict]:
        infos = []
        for entry in sorted(self.runs_dir.iterdir(), reverse=True):
            if not entry.is_dir() or not _RUN_ID.match(entry.name):
                continue
            info = self.info(entry.name)
            if info is not None:
                infos.append(info)
        return infos

    def pending(self) -> list[dict]:
        return sorted((i for i in self.list() if i["state"] == protocol.STATE_PENDING),
                      key=lambda i: (i["created"] or "", i["run_id"]))

    def job(self, run_id: str) -> dict | None:
        return protocol.read_json(self.run_dir(run_id) / protocol.JOB_FILE)

    def result(self, run_id: str) -> dict | None:
        return protocol.read_json(self.run_dir(run_id) / protocol.RESULT_FILE)

    def mark(self, run_id: str, state: str, **extra) -> dict:
        return protocol.write_status(self.run_dir(run_id), state, **extra)

    def purge(self, run_id: str) -> None:
        info = self.info(run_id)
        if info is None:
            raise FileNotFoundError(f"unknown run: {run_id}")
        if info["state"] not in protocol.TERMINAL_STATES:
            raise ValueError(f"run {run_id} is {info['state']}; cancel it before purging")
        shutil.rmtree(self.run_dir(run_id))
