"""Post-run artifact inspection without requiring the original job or input sources."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .io import exists
from .loaders import (
    _ResolvedDatasetSnapshot,
    _assert_snapshot_current,
    _resolve_operation_snapshot,
    _snapshot_entries,
    _validate_snapshot_shard,
    read_shard_arrays,
)
from .results import DatasetComparison, DatasetInspection


def inspect_dataset(dataset, *, verify: bool = False,
                    container: str | None = None, registry=None) -> DatasetInspection:
    """Summarize a curated artifact and optionally read every shard for integrity.

    ``dataset`` accepts the same directory, manifest path, or loaded manifest dictionary as
    :func:`shard_entries`. Metadata inspection does not read shard payloads. ``verify=True``
    checks existence and decodes every shard, returning failures in ``verification.errors``.
    """
    if not isinstance(verify, bool):
        raise ValueError("verify must be true or false")
    snapshot = _resolve_operation_snapshot(dataset, container)
    return _inspect_resolved_dataset(snapshot, verify=verify, registry=registry)


def _inspect_resolved_dataset(
    snapshot: _ResolvedDatasetSnapshot,
    *,
    verify: bool,
    registry=None,
) -> DatasetInspection:
    """Inspect one already-resolved metadata generation without reopening its config."""
    entries = _snapshot_entries(snapshot)
    config: dict = snapshot.config or {}

    splits: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"shards": 0, "steps": 0, "padded_shards": 0, "padding_steps": 0})
    modality_variants: dict[str, set[tuple[tuple[int, ...], str]]] = defaultdict(set)
    shard_length = int((config.get("output") or {}).get("shard_length") or 0)
    for entry in entries:
        split = str(entry.get("split", "all"))
        stats = splits[split]
        stats["shards"] += 1
        stats["steps"] += int(entry.get("n_steps", 0))
        if entry.get("padded"):
            stats["padded_shards"] += 1
            stats["padding_steps"] += max(0, shard_length - int(entry.get("n_steps", 0)))
        for name, shape in (entry.get("shape") or {}).items():
            modality_variants[name].add((tuple(shape), str((entry.get("dtype") or {}).get(name))))

    sample_hz = ((config.get("schema") or {}).get("sample_hz") if config else None)
    for stats in splits.values():
        stats["seconds"] = stats["steps"] / sample_hz if sample_hz else None
    modalities = {}
    for name, variants in sorted(modality_variants.items()):
        ordered = sorted(variants)
        modalities[name] = ({"shape": list(ordered[0][0]), "dtype": ordered[0][1]}
                            if len(ordered) == 1 else
                            {"variants": [{"shape": list(shape), "dtype": dtype}
                                          for shape, dtype in ordered]})

    verification_errors = []
    checked = 0
    output = config.get("output") or {}
    expected_modalities = dict(output.get("modalities") or {})
    if output.get("presence_mask", True):
        expected_modalities["_present"] = {"shape": [], "dtype": "bool"}
    for entry in entries:
        _assert_snapshot_current(snapshot, operation="dataset inspection")
        _validate_snapshot_shard(snapshot, entry)
        path = entry.get("path")
        if not exists(path):
            verification_errors.append({"path": path, "error": "missing shard"})
            _assert_snapshot_current(snapshot, operation="dataset inspection")
            continue
        if verify:
            try:
                if expected_modalities:
                    expected_names = set(expected_modalities)
                    manifest_names = set(entry.get("shape") or {})
                    if manifest_names != expected_names:
                        raise ValueError(f"manifest modalities mismatch: expected "
                                         f"{sorted(expected_names)}, got {sorted(manifest_names)}")
                    for name, expected in expected_modalities.items():
                        shape = list((entry.get("shape") or {}).get(name, ()))
                        dtype = str((entry.get("dtype") or {}).get(name))
                        if shape != list(expected.get("shape") or ()) or dtype != expected["dtype"]:
                            raise ValueError(f"manifest contract mismatch for {name!r}: "
                                             f"shape={shape}, dtype={dtype}")
                if entry.get("sha256") is None or entry.get("size_bytes") is None:
                    raise ValueError(
                        "integrity unavailable: shard entry needs both SHA-256 and byte size"
                    )
                read_shard_arrays(entry, snapshot.container,
                                  modalities=list(expected_modalities) or None,
                                  registry=registry, verify=True)
                checked += 1
            except Exception as error:
                verification_errors.append({"path": path, "error": repr(error)})
        _validate_snapshot_shard(snapshot, entry)
        _assert_snapshot_current(snapshot, operation="dataset inspection")

    execution = config.get("execution") or {}
    physical = execution.get("physical_plan") or {}
    provenance = config.get("provenance") or {}
    inspection = DatasetInspection(
        dataset=snapshot.base, container=snapshot.container, shards=len(entries),
        sources=len({str(entry.get("source_id")) for entry in entries}),
        splits=dict(sorted(splits.items())), modalities=modalities,
        budget=(config.get("results") or {}).get("budget"),
        errors=list(config.get("errors") or ()),
        sources_skipped=(config.get("results") or {}).get("sources_skipped", 0),
        provenance_fingerprint=provenance.get("fingerprint"),
        execution={"bounded": execution.get("bounded", False),
                   "backend": physical.get("backend", physical.get("mode"))},
        verification={"requested": verify, "checked": checked,
                      "valid": not verification_errors, "errors": verification_errors})
    _assert_snapshot_current(snapshot, operation="dataset inspection")
    return inspection


def _changes(left: Any, right: Any, path: str = "", *, limit: int = 500) -> list[dict]:
    if left == right:
        return []
    if isinstance(left, dict) and isinstance(right, dict):
        changes = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left:
                changes.append({"path": child, "left": None, "right": right[key]})
            elif key not in right:
                changes.append({"path": child, "left": left[key], "right": None})
            else:
                changes.extend(_changes(left[key], right[key], child,
                                        limit=limit - len(changes)))
            if len(changes) >= limit:
                break
        return changes[:limit]
    if isinstance(left, list) and isinstance(right, list):
        changes = []
        for index in range(max(len(left), len(right))):
            child = f"{path}[{index}]"
            if index >= len(left):
                changes.append({"path": child, "left": None, "right": right[index]})
            elif index >= len(right):
                changes.append({"path": child, "left": left[index], "right": None})
            else:
                changes.extend(_changes(left[index], right[index], child,
                                        limit=limit - len(changes)))
            if len(changes) >= limit:
                break
        return changes[:limit]
    return [{"path": path or "$", "left": left, "right": right}]


def compare_datasets(left, right, *, verify: bool = False) -> DatasetComparison:
    """Compare two written datasets' schemas, output contracts, provenance, and sizes."""
    if not isinstance(verify, bool):
        raise ValueError("verify must be true or false")
    left_snapshot = _resolve_operation_snapshot(left, None)
    right_snapshot = _resolve_operation_snapshot(right, None)
    left_base, right_base = left_snapshot.base, right_snapshot.base
    if left_base is None or right_base is None:
        raise ValueError("compare_datasets needs dataset directories or manifest paths, not "
                         "loaded manifest dictionaries")
    left_inspection = _inspect_resolved_dataset(
        left_snapshot, verify=verify
    )
    right_inspection = _inspect_resolved_dataset(
        right_snapshot, verify=verify
    )
    left_config = left_snapshot.config or {}
    right_config = right_snapshot.config or {}
    comparable_left = {key: left_config.get(key) for key in (
        "artifact_format_version", "schema", "strategy", "output", "pipeline",
        "output_request", "output_preset", "mode", "results", "warnings")}
    comparable_right = {key: right_config.get(key) for key in comparable_left}
    changes = _changes(comparable_left, comparable_right)
    left_summary = {"dataset": left_base, "shards": left_inspection.shards,
                    "splits": left_inspection.splits,
                    "modalities": left_inspection.modalities,
                    "provenance_fingerprint": left_inspection.provenance_fingerprint,
                    "verification": left_inspection.verification}
    right_summary = {"dataset": right_base, "shards": right_inspection.shards,
                     "splits": right_inspection.splits,
                     "modalities": right_inspection.modalities,
                     "provenance_fingerprint": right_inspection.provenance_fingerprint,
                     "verification": right_inspection.verification}
    summary_changes = _changes(
        {key: value for key, value in left_summary.items() if key != "dataset"},
        {key: value for key, value in right_summary.items() if key != "dataset"},
        "artifact")
    all_changes = summary_changes + changes
    comparison = DatasetComparison(
        left=left_summary,
        right=right_summary,
        identical=not all_changes,
        changes=all_changes,
        verified=verify,
    )
    _assert_snapshot_current(left_snapshot, operation="dataset comparison")
    _assert_snapshot_current(right_snapshot, operation="dataset comparison")
    return comparison
