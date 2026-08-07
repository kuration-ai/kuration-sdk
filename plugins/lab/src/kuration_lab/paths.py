"""One path rule for every request field that names a server-side file or directory.

Relative paths resolve against the server's contents root (``ServerApp.root_dir``) so the UI
can pass paths straight from the file browser. Absolute paths are accepted by default —
datasets routinely live outside the notebook root — but are rejected when the deployment sets
``KurationLabApp.allow_absolute_paths = False``. Responses always carry the resolved absolute
form so the frontend has one canonical spelling per subject.
"""
from __future__ import annotations

import os


_JOB_PATH_KEYS = ("out", "checkpoint_dir", "spill_dir")
_JOB_DOC_KEYS = ("schema", "strategy")  # string form means "path to a JSON document"


def normalize_job_paths(job: dict, base_dir: str) -> dict:
    """Anchor a job document's relative paths to the directory it was authored in.

    A ``.kur.json`` edited in JupyterLab uses paths relative to its own location; the server
    and worker processes have unrelated working directories, so path-bearing keys are made
    absolute once, at the API boundary. Globs stay globs — joining preserves the pattern.
    """

    def _anchor(value):
        if isinstance(value, str) and value and "://" not in value \
                and not os.path.isabs(os.path.expanduser(value)):
            return os.path.normpath(os.path.join(base_dir, os.path.expanduser(value)))
        return value

    normalized = dict(job)
    for key in _JOB_PATH_KEYS + _JOB_DOC_KEYS:
        if key in normalized:
            normalized[key] = _anchor(normalized[key])
    if isinstance(normalized.get("sources"), list):
        normalized["sources"] = [_anchor(source) for source in normalized["sources"]]
    if isinstance(normalized.get("arms"), list):
        normalized["arms"] = [
            {**arm, "strategy": _anchor(arm["strategy"])}
            if isinstance(arm, dict) and isinstance(arm.get("strategy"), str) else arm
            for arm in normalized["arms"]
        ]
    return normalized


def resolve_path(value: str, root_dir: str, *, allow_absolute: bool = True) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("path must be a non-empty string")
    candidate = os.path.expanduser(value)
    if os.path.isabs(candidate):
        if not allow_absolute:
            raise PermissionError(
                f"absolute paths are disabled by the server configuration: {value!r}"
            )
        return os.path.realpath(candidate)
    resolved = os.path.realpath(os.path.join(root_dir, candidate))
    if not allow_absolute:
        root = os.path.realpath(root_dir)
        if os.path.commonpath([resolved, root]) != root:
            raise PermissionError(f"path escapes the server root: {value!r}")
    return resolved
