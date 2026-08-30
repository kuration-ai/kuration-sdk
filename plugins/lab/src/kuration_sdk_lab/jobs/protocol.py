"""On-disk run protocol shared by the server (manager/registry) and the worker subprocess.

One directory per run:

    job.json      the submitted job document (server-written at submit, then immutable)
    run.json      submit metadata: run_id, kind, out, created, pid, submitter (server-owned)
    status.json   current state, atomically replaced (worker-owned while the worker lives;
                  the server writes it only before spawn and after the worker is dead)
    events.jsonl  progress events, one JSON object per line, monotonically increasing ``seq``
    worker.log    captured worker stdout/stderr (the SDK's skip diagnostics land here)
    result.json   JSON-safe result summary persisted by the worker before exit

Every event line is the SDK progress dict (``{"phase", "event", "current", "total",
"source", "status", "detail"}`` — ``source`` arrives already redacted) stamped with
``{"seq", "run_id", "ts"}``. Synthetic lifecycle events use ``phase: "lifecycle"`` with
``event`` in ``{start, complete, error, cancelled}``.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import secrets
from pathlib import Path

STATE_PENDING = "pending"
STATE_RUNNING = "running"
STATE_SUCCEEDED = "succeeded"
STATE_FAILED = "failed"
STATE_CANCELLED = "cancelled"
STATE_ORPHANED = "orphaned"

TERMINAL_STATES = frozenset({STATE_SUCCEEDED, STATE_FAILED, STATE_CANCELLED, STATE_ORPHANED})

JOB_FILE = "job.json"
RUN_FILE = "run.json"
STATUS_FILE = "status.json"
EVENTS_FILE = "events.jsonl"
LOG_FILE = "worker.log"
RESULT_FILE = "result.json"


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="microseconds")


def new_run_id() -> str:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def write_json_atomic(path: str | Path, payload: dict) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: str | Path) -> dict | None:
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def write_status(run_dir: str | Path, state: str, **extra) -> dict:
    payload = {"state": state, "updated": utcnow(), **extra}
    write_json_atomic(Path(run_dir) / STATUS_FILE, payload)
    return payload


def read_status(run_dir: str | Path) -> dict:
    return read_json(Path(run_dir) / STATUS_FILE) or {"state": STATE_PENDING, "updated": None}


def read_events(run_dir: str | Path, *, since: int = 0, limit: int = 1000) -> tuple[list[dict], int]:
    """Return ``(events with seq > since, cursor)``; tolerate a torn final line.

    The cursor is the seq of the last *returned* event (or ``since`` when none), so passing
    it back as the next ``since`` never skips events even when ``limit`` truncates a batch.
    """
    path = Path(run_dir) / EVENTS_FILE
    events: list[dict] = []
    cursor = since
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return events, cursor
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue  # a torn tail line — the writer flushes per line, so only the last can tear
        seq = event.get("seq")
        if not isinstance(seq, int) or seq <= since:
            continue
        events.append(event)
        cursor = seq
        if len(events) >= limit:
            break
    return events, cursor


class EventWriter:
    """Single writer of ``events.jsonl``; lives in the worker process only."""

    def __init__(self, run_dir: str | Path, run_id: str):
        self._run_id = run_id
        self._file = open(Path(run_dir) / EVENTS_FILE, "a", encoding="utf-8")
        self.seq = 0

    def emit(self, event: dict) -> dict:
        self.seq += 1
        line = {"seq": self.seq, "run_id": self._run_id, "ts": utcnow(), **event}
        self._file.write(json.dumps(line, allow_nan=False) + "\n")
        self._file.flush()
        return line

    def lifecycle(self, event: str, **fields) -> dict:
        return self.emit({"phase": "lifecycle", "event": event, **fields})

    def close(self) -> None:
        self._file.close()
