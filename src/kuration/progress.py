"""Structured progress events and a concise CLI renderer."""
from __future__ import annotations

import sys
from typing import Callable, Optional

ProgressCallback = Optional[Callable[[dict], None]]


def emit(progress: ProgressCallback, phase: str, event: str, **fields) -> None:
    if progress is not None:
        progress({"phase": phase, "event": event, **fields})


def cli_progress(*, file=None):
    """Return a throttled stderr renderer for structured progress events."""
    file = file or sys.stderr
    last: dict[str, int] = {}

    def report(e: dict) -> None:
        phase, event = e["phase"], e["event"]
        if event == "start":
            total = e.get("total")
            print(f"[{phase}] starting" + (f" ({total} total)" if total is not None else ""),
                  file=file)
            last[phase] = 0
        elif event == "advance":
            current, total = e.get("current", 0), e.get("total", 0)
            stride = max(1, total // 20) if total else 1
            if current == total or current - last.get(phase, 0) >= stride:
                detail = f" source={e['source']}" if e.get("source") is not None else ""
                print(f"[{phase}] {current}/{total}{detail}", file=file)
                last[phase] = current
        elif event == "complete":
            completion_detail = e.get("detail")
            print(f"[{phase}] complete" + (f" ({completion_detail})" if completion_detail else ""),
                  file=file)

    return report
