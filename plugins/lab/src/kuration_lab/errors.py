"""API error envelope: ``{"error": {"type", "message", "hint"}}``.

The hint reuses the SDK's own failure→remedy mapping (`kuration.core.cli._common.error_hint`)
so the browser shows the same actionable guidance the CLIs print.
"""
from __future__ import annotations


def status_for(error: BaseException) -> int:
    if isinstance(error, FileNotFoundError):
        return 404
    if isinstance(error, PermissionError):
        return 403
    if isinstance(error, FileExistsError):
        return 409
    if isinstance(error, (ValueError, KeyError, TypeError, ImportError, NotImplementedError)):
        return 400
    return 500


def error_envelope(error: BaseException) -> dict:
    from kuration.core.cli._common import error_hint

    message = " ".join(str(error).split()) or "<no message>"
    return {
        "error": {
            "type": type(error).__name__,
            "message": message,
            "hint": error_hint(error),
        }
    }
