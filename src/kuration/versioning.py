"""Serialized job/artifact contract versions and deterministic migrations."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

ARTIFACT_FORMAT_VERSION = "1.0"
JOB_SCHEMA_VERSION = "1.0"
LEGACY_VERSION = "0.0"


def _major(value: str) -> int:
    try:
        return int(str(value).split(".", 1)[0])
    except (TypeError, ValueError) as e:
        raise ValueError(f"malformed format version {value!r}; expected '<major>.<minor>'") from e


def check_version(value: Any, *, kind: str) -> str:
    current = ARTIFACT_FORMAT_VERSION if kind == "artifact" else JOB_SCHEMA_VERSION
    value = LEGACY_VERSION if value is None else str(value)
    if _major(value) > _major(current):
        raise ValueError(f"unsupported future {kind} format {value}; this SDK supports {current}")
    return value


def migrate_job(job: dict) -> dict:
    """Return a deep-copied job updated to the current job-schema version.

    A malformed or unsupported future major version raises ``ValueError``.
    """
    check_version(job.get("job_schema_version"), kind="job")
    out = copy.deepcopy(job)
    out["job_schema_version"] = JOB_SCHEMA_VERSION
    return out


def migrate_config(config: dict) -> dict:
    """Return a deep-copied config updated to the current artifact version.

    A malformed or unsupported future major version raises ``ValueError``.
    """
    check_version(config.get("artifact_format_version"), kind="artifact")
    out = copy.deepcopy(config)
    out["artifact_format_version"] = ARTIFACT_FORMAT_VERSION
    return out


def migrate(value, out=None):
    """Migrate a job/config dict or JSON path, optionally writing the result."""
    source = json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value
    if not isinstance(source, dict):
        raise ValueError("migration input must be a JSON object")
    result = migrate_job(source) if ("command" in source or "job_schema_version" in source) \
        else migrate_config(source)
    if out is not None:
        Path(out).write_text(json.dumps(result, indent=2) + "\n")
    return result
