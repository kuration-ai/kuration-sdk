"""Serialized job/artifact contract versions and deterministic migrations."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
from typing import Any

ARTIFACT_FORMAT_VERSION = "1.3"
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
    "1.2": (
        "selection semantics: bounded uniform-bin selection now uses canonical bin order rather "
        "than partition encounter order, so bounded and materialized execution select the same "
        "segments regardless of source or partition ordering",
        "experiment semantics: each arm now enforces tag coverage only for its own declarative "
        "tag sources, so adding an unrelated arm cannot change another arm's selection",
        "tag-coverage semantics: min_tag_frac=0 now means no tag floor, positive fractional "
        "count floors round up, and coverage cannot restore predicate-failing candidates; "
        "1.1 count selection still forced one item per tag",
        "group/tag identity semantics: disk-backed hours fills now use Python hash/equality "
        "like the in-memory path, so values such as 1, true, and '1' no longer diverge by "
        "execution mode",
    ),
    "1.3": (
        "shard naming: output shard filenames are now derived from each span's own "
        "(start_idx, end_idx) instead of its rank among selected/admitted spans for that "
        "source; two curations that each pick a different span at the same rank no longer "
        "alias each other's filenames. Resuming an in-progress 1.2 write under 1.3 will not "
        "recognize already-written shards by name and will write new ones alongside them "
        "rather than corrupting the dataset -- start a fresh output directory instead of "
        "resuming across this version boundary.",
    ),
}
JOB_SEMANTIC_CHANGES: dict[str, tuple[str, ...]] = {}

# Keys that exist only on executable job documents. Resolved artifact configs deliberately share
# ``schema``/``strategy``/``output`` with jobs, so those common provenance keys cannot classify an
# unversioned document by themselves.
_JOB_MARKERS = {
    "command", "adapter", "sources", "segmenter", "arms", "baseline", "target_size",
    "output_preset", "encoder", "train_hours", "test_hours", "out", "resume", "force",
    "streaming",
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


def _parse_version(value: Any) -> tuple[int, int]:
    text = str(value)
    match = re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", text)
    if match is None:
        raise ValueError(
            f"malformed format version {value!r}; expected canonical '<major>.<minor>' "
            "with non-negative integers"
        )
    return int(match.group(1)), int(match.group(2))


def _major(value: Any) -> int:
    """Return a canonical format version's major component."""
    return _parse_version(value)[0]


def _version_key(value: str) -> tuple[int, int]:
    return _parse_version(value)


def migration_notes(value: Any, *, kind: str = "artifact") -> list[str]:
    """Return the selection-affecting changes between a document's version and the current one.

    ``value`` is the declared version string (``None``/absent means the pre-versioning legacy
    document). Notes are ordered oldest change first; an up-to-date document yields ``[]``.
    """
    if kind not in {"artifact", "job"}:
        raise ValueError(f"unknown document kind {kind!r}; expected 'artifact' or 'job'")
    table = ARTIFACT_SEMANTIC_CHANGES if kind == "artifact" else JOB_SEMANTIC_CHANGES
    current = ARTIFACT_FORMAT_VERSION if kind == "artifact" else JOB_SCHEMA_VERSION
    check_version(value, kind=kind)
    declared = _version_key(LEGACY_VERSION if value is None else str(value))
    notes: list[str] = []
    for version in sorted(table, key=_version_key):
        if declared < _version_key(version) <= _version_key(current):
            notes.extend(f"{version}: {note}" for note in table[version])
    return notes


def check_version(value: Any, *, kind: str) -> str:
    if kind not in {"artifact", "job"}:
        raise ValueError(f"unknown document kind {kind!r}; expected 'artifact' or 'job'")
    current = ARTIFACT_FORMAT_VERSION if kind == "artifact" else JOB_SCHEMA_VERSION
    value = LEGACY_VERSION if value is None else str(value)
    if _version_key(value) > _version_key(current):
        raise ValueError(f"unsupported future {kind} format {value}; this SDK supports {current}")
    return value


def migrate_job(job: dict) -> dict:
    """Return a deep-copied job updated to the current job-schema version.

    A malformed or unsupported future version raises ``ValueError``.
    """
    check_version(job.get("job_schema_version"), kind="job")
    out = copy.deepcopy(job)
    out["job_schema_version"] = JOB_SCHEMA_VERSION
    return out


def migrate_config(config: dict) -> dict:
    """Return a deep-copied config updated to the current artifact version.

    A malformed or unsupported future version raises ``ValueError``.
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
    if isinstance(value, (str, Path)):
        path = Path(value)

        def reject_constant(token: str) -> None:
            raise ValueError(
                f"migration input {str(path)!r} is not valid JSON: "
                f"non-finite numeric token {token!r} is unsupported"
            )

        def unique_object(pairs):
            parsed = {}
            for key, item in pairs:
                if key in parsed:
                    raise ValueError(
                        f"migration input {str(path)!r} is ambiguous JSON: "
                        f"duplicate object key {key!r}"
                    )
                parsed[key] = item
            return parsed

        source = json.loads(
            path.read_text(),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    else:
        source = value
    kind = document_kind(source)
    result = migrate_job(source) if kind == "job" else migrate_config(source)
    if out is not None:
        try:
            encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"migrated document cannot be represented as strict JSON: {error}"
            ) from error
        from .io import write_text
        write_text(out, encoded)
    return result
