"""Serialized job/artifact contract versions and deterministic migrations."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

ARTIFACT_FORMAT_VERSION = "1.1"
JOB_SCHEMA_VERSION = "1.0"
LEGACY_VERSION = "0.0"

# Changes that keep an older document structurally valid but make it *select differently* on the
# same data, keyed by the version that introduced them. Migration is structural, so these notes
# are the only honest way to tell a user that re-running their old config here is not the run they
# recorded. Add an entry with every semantic bump; ``migration_notes`` reports the ones a document
# has not seen yet.
ARTIFACT_SEMANTIC_CHANGES: dict[str, tuple[str, ...]] = {
    "1.1": (
        "metric semantics: 'dispersion' and 'min_pair_dist' now use every declared position "
        "dimension; 1.0 used only the first two, so a 3D-position dataset scores (and selects) "
        "differently",
        "metric semantics: scope 'per_entity_mean' is now honored for categorical fields; 1.0 "
        "silently pooled them, so a per-entity categorical metric scores (and selects) "
        "differently",
    ),
}
JOB_SEMANTIC_CHANGES: dict[str, tuple[str, ...]] = {}

# Keys that exist only on executable job documents. Resolved artifact configs deliberately share
# ``schema``/``strategy``/``output`` with jobs, so those common provenance keys cannot classify an
# unversioned document by themselves.
_JOB_MARKERS = {
    "command", "adapter", "sources", "segmenter", "arms", "baseline", "target_size",
    "output_preset", "encoder", "train_hours", "test_hours", "out", "resume", "streaming",
    "executor", "mode", "workers", "on_error", "span_steps", "span_seconds",
    "clip_aggregate", "memory_budget", "batch_size", "checkpoint_dir", "spill_dir",
    "approximation", "max_in_flight", "include_plugin_entry_points",
}

# Resolved dataset configs can retain job-like presentation fields such as ``mode`` and
# ``output_preset``. Prefer fields written only after execution when classifying an unversioned
# document so those artifacts are never migrated as executable jobs.
_ARTIFACT_MARKERS = {
    "results", "pipeline", "output_request", "provenance", "started", "ended",
}


def _major(value: str) -> int:
    try:
        return int(str(value).split(".", 1)[0])
    except (TypeError, ValueError) as e:
        raise ValueError(f"malformed format version {value!r}; expected '<major>.<minor>'") from e


def _version_key(value: str) -> tuple[int, int]:
    parts = str(value).split(".")
    try:
        major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (TypeError, ValueError) as e:
        raise ValueError(f"malformed format version {value!r}; expected '<major>.<minor>'") from e
    return major, minor


def migration_notes(value: Any, *, kind: str = "artifact") -> list[str]:
    """Return the selection-affecting changes between a document's version and the current one.

    ``value`` is the declared version string (``None``/absent means the pre-versioning legacy
    document). Notes are ordered oldest change first; an up-to-date document yields ``[]``.
    """
    if kind not in {"artifact", "job"}:
        raise ValueError(f"unknown document kind {kind!r}; expected 'artifact' or 'job'")
    table = ARTIFACT_SEMANTIC_CHANGES if kind == "artifact" else JOB_SEMANTIC_CHANGES
    current = ARTIFACT_FORMAT_VERSION if kind == "artifact" else JOB_SCHEMA_VERSION
    declared = _version_key(LEGACY_VERSION if value is None else str(value))
    notes = []
    for version in sorted(table, key=_version_key):
        if declared < _version_key(version) <= _version_key(current):
            notes.extend(f"{version}: {note}" for note in table[version])
    return notes


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


def document_kind(document: dict) -> str:
    """Classify a versioned or unversioned document as a ``'job'`` or an ``'artifact'`` config."""
    if not isinstance(document, dict):
        raise ValueError("migration input must be a JSON object")
    if "job_schema_version" in document and "artifact_format_version" in document:
        raise ValueError("migration input declares both job_schema_version and "
                         "artifact_format_version")
    is_job = "job_schema_version" in document or (
        "artifact_format_version" not in document
        and not bool(set(document) & _ARTIFACT_MARKERS)
        and bool(set(document) & _JOB_MARKERS))
    return "job" if is_job else "artifact"


def document_migration_notes(document: dict) -> list[str]:
    """Return the selection-affecting notes a job/config document has not seen yet.

    Reported by ``kur-migrate`` next to the structural migration: the migrated document is valid
    here, but a run of it can legitimately differ from the recorded one.
    """
    kind = document_kind(document)
    key = "job_schema_version" if kind == "job" else "artifact_format_version"
    return migration_notes(document.get(key), kind=kind)


def migrate(value, out=None):
    """Migrate a job/config dict or JSON path, optionally writing the result.

    Structural only — see ``document_migration_notes`` for semantic changes the bump implies.
    """
    source = json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value
    kind = document_kind(source)
    result = migrate_job(source) if kind == "job" else migrate_config(source)
    if out is not None:
        Path(out).write_text(json.dumps(result, indent=2) + "\n")
    return result
