"""JobManager: spawn, watch, stream, and cancel worker subprocesses.

Design notes:
- The disk protocol (``protocol.py``) is the source of truth; the manager holds only live
  ``Process`` handles. A server restart loses nothing — pending runs are re-spawned by
  ``ensure_started()`` and event streams re-tail ``events.jsonl`` from any ``since``.
- Event streaming is per-subscriber file tailing (``stream()``), not a shared fan-out: each
  WebSocket polls the events file independently every ``poll`` seconds. Concurrent viewers
  are few; simplicity and restart-resilience win over an in-memory bus.
- Cancellation: SIGTERM to the worker's process group (the worker turns it into a raise from
  the SDK progress callback), SIGKILL after ``cancel_grace`` seconds.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
from pathlib import Path
from typing import AsyncIterator

from . import protocol
from .registry import RunRegistry


class JobManager:
    def __init__(self, registry: RunRegistry, *, max_concurrent: int = 2,
                 cancel_grace: float = 15.0, python: str = sys.executable):
        self.registry = registry
        self.max_concurrent = max(1, int(max_concurrent))
        self.cancel_grace = float(cancel_grace)
        self.python = python
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._cancelling: set[str] = set()
        self._started = False
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        # Created lazily so the manager can be constructed before the event loop exists.
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def ensure_started(self) -> None:
        """Idempotent: kick the pending queue (also re-adopts pending runs after a restart)."""
        if not self._started:
            self._started = True
        await self._maybe_start()

    async def submit(self, job: dict, *, kind: str, submitter: str | None = None) -> dict:
        info = self.registry.create(job, kind=kind, submitter=submitter)
        await self.ensure_started()
        return self.registry.info(info["run_id"]) or info

    async def _maybe_start(self) -> None:
        async with self._get_lock():
            for info in self.registry.pending():
                if len(self._procs) >= self.max_concurrent:
                    break
                await self._spawn(info["run_id"])

    async def _spawn(self, run_id: str) -> None:
        run_dir = self.registry.run_dir(run_id)
        log = open(run_dir / protocol.LOG_FILE, "ab")
        try:
            proc = await asyncio.create_subprocess_exec(
                self.python, "-m", "kuration_sdk_lab.jobs.worker", str(run_dir),
                stdout=log, stderr=log, stdin=asyncio.subprocess.DEVNULL,
                cwd=str(run_dir), start_new_session=True,
            )
        except Exception as error:
            self.registry.mark(run_id, protocol.STATE_FAILED,
                               exit={"ok": False, "error_type": type(error).__name__,
                                     "message": f"failed to spawn worker: {error}"})
            return
        finally:
            log.close()  # the child holds its own descriptor
        self._procs[run_id] = proc
        asyncio.ensure_future(self._watch(run_id, proc))

    async def _watch(self, run_id: str, proc: asyncio.subprocess.Process) -> None:
        returncode = await proc.wait()
        self._procs.pop(run_id, None)
        cancelled = run_id in self._cancelling
        self._cancelling.discard(run_id)
        status = protocol.read_status(self.registry.run_dir(run_id))
        if status.get("state") not in protocol.TERMINAL_STATES:
            # The worker died without writing a terminal status (SIGKILL, crash, OOM).
            if cancelled:
                self.registry.mark(run_id, protocol.STATE_CANCELLED,
                                   exit={"ok": False, "cancelled": True,
                                         "message": "worker killed after the cancel grace period"})
            else:
                self.registry.mark(run_id, protocol.STATE_FAILED,
                                   exit={"ok": False,
                                         "message": f"worker exited with code {returncode} "
                                                    "without writing a terminal status"})
        await self._maybe_start()

    async def cancel(self, run_id: str) -> dict:
        info = self.registry.info(run_id)
        if info is None:
            raise FileNotFoundError(f"unknown run: {run_id}")
        state = info["state"]
        if state == protocol.STATE_PENDING:
            self.registry.mark(run_id, protocol.STATE_CANCELLED,
                               exit={"ok": False, "cancelled": True,
                                     "message": "cancelled before start"})
            return self.registry.info(run_id) or info
        proc = self._procs.get(run_id)
        if state != protocol.STATE_RUNNING or proc is None:
            return info  # already terminal (or orphaned) — nothing to signal
        self._cancelling.add(run_id)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)  # start_new_session=True → pgid == pid
        asyncio.ensure_future(self._enforce_kill(run_id, proc))
        return self.registry.info(run_id) or info

    async def _enforce_kill(self, run_id: str, proc: asyncio.subprocess.Process) -> None:
        try:
            await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=self.cancel_grace)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)

    async def stream(self, run_id: str, *, since: int = 0,
                     poll: float = 0.5) -> AsyncIterator[dict]:
        """Yield events with ``seq > since``, then a final synthetic ``state`` message.

        Replay and live tailing are the same code path: keep reading the events file until
        the run is terminal and fully drained. Safe across server restarts and for runs that
        finished long ago.
        """
        run_dir = self.registry.run_dir(run_id)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"unknown run: {run_id}")
        while True:
            events, since = protocol.read_events(run_dir, since=since)
            for event in events:
                yield event
            status = protocol.read_status(run_dir)
            if status.get("state") in protocol.TERMINAL_STATES and not events:
                yield {"phase": "lifecycle", "event": "state", "state": status.get("state"),
                       "exit": status.get("exit"), "run_id": run_id, "seq": since}
                return
            await asyncio.sleep(poll)

    async def shutdown(self) -> None:
        """Terminate live workers (server is going down); runs become orphaned and resumable."""
        for proc in list(self._procs.values()):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)


def default_runs_dir() -> Path:
    from jupyter_core.paths import jupyter_data_dir

    return Path(jupyter_data_dir()) / "kuration-sdk-lab" / "runs"
