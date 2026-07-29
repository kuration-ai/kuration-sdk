"""Structured progress events and a concise CLI renderer."""
from __future__ import annotations

from html import escape
import sys
from typing import Any, Callable, Optional

ProgressCallback = Optional[Callable[[dict], None]]
ProgressSpec = ProgressCallback | str | bool


def emit(progress: ProgressCallback, phase: str, event: str, **fields) -> None:
    if progress is not None:
        # Progress callbacks commonly render into CI logs and persisted notebooks. Keep the
        # source field useful while applying the same credential/host-path redaction as reports.
        if fields.get("source") is not None:
            from .scoring import _redact_diagnostic

            fields["source"] = _redact_diagnostic(fields["source"])
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
                count = str(current) if total is None else f"{current}/{total}"
                print(f"[{phase}] {count}{detail}", file=file)
                last[phase] = current
        elif event == "complete":
            completion_detail = e.get("detail")
            print(f"[{phase}] complete" + (f" ({completion_detail})" if completion_detail else ""),
                  file=file)

    return report


def _notebook_progress_html(event: dict) -> str:
    phase = escape(str(event.get("phase", "work")))
    state = event.get("event", "advance")
    current = event.get("current")
    total = event.get("total")
    if state == "start":
        current = 0
    if state == "complete" and current is None:
        current = total
    if isinstance(total, int) and total > 0 and isinstance(current, int):
        fraction = min(1.0, max(0.0, current / total))
        count = f"{current}/{total}"
    else:
        fraction = 1.0 if state == "complete" else 0.0
        count = str(current) if current is not None else state
    detail = event.get("detail")
    if detail is None and event.get("source") is not None:
        detail = f"source={event['source']}"
    detail_html = (
        f"<span style=\"opacity:.7\">{escape(str(detail))}</span>"
        if detail is not None
        else ""
    )
    return (
        "<div class=\"kur-progress\" "
        "style=\"font:13px system-ui,sans-serif;margin:.25rem 0\">"
        f"<div style=\"display:flex;justify-content:space-between;gap:1rem\">"
        f"<strong>{phase}</strong><span>{escape(count)}</span></div>"
        "<div style=\"height:.45rem;background:#8883;border-radius:1rem;overflow:hidden\">"
        f"<div class=\"kur-progress-bar\" style=\"height:100%;width:{fraction * 100:.3f}%;"
        "background:#4f7cff;transition:width .15s\"></div></div>"
        f"{detail_html}</div>"
    )


def notebook_progress():
    """Return a compact, multi-phase progress renderer for Jupyter frontends.

    IPython is imported only when this renderer is selected. Each execution phase owns one
    display row, so repeated events update in place instead of flooding notebook output.
    """
    try:
        from IPython.display import HTML, display
    except ImportError as error:
        raise ImportError(
            "notebook progress needs IPython; install JupyterLab or IPython"
        ) from error

    handles: dict[str, Any] = {}

    def report(event: dict) -> None:
        phase = str(event.get("phase", "work"))
        value = HTML(_notebook_progress_html(event))
        handle = handles.get(phase)
        if handle is None:
            handles[phase] = display(value, display_id=True)
        else:
            handle.update(value)

    return report


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def progress_callback(progress: ProgressSpec = "auto") -> ProgressCallback:
    """Resolve a progress selection into the SDK's structured callback contract.

    ``"auto"`` chooses notebook rendering inside a Jupyter kernel and terminal rendering
    elsewhere. ``"notebook"``, ``"terminal"``, and ``"off"`` make the choice explicit;
    existing callbacks pass through unchanged.
    """
    if progress is None or progress is False or progress == "off":
        return None
    if callable(progress):
        return progress
    if progress is True:
        progress = "auto"
    if progress == "auto":
        return notebook_progress() if _in_notebook() else cli_progress()
    if progress == "notebook":
        return notebook_progress()
    if progress in {"terminal", "cli"}:
        return cli_progress()
    raise ValueError(
        "progress must be a callback, boolean, None, or "
        "'auto', 'notebook', 'terminal', 'cli', or 'off'"
    )
