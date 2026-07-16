"""Post-run artifact inspection without requiring the original job or input sources."""
from __future__ import annotations

from collections import defaultdict
import json
from typing import Any

from .io import exists, read_text
from .loaders import read_shard_arrays, shard_entries
from .results import DatasetInspection
from .versioning import check_version


def _base_path(dataset) -> str | None:
    if isinstance(dataset, dict):
        return None
    path = str(dataset).rstrip("/")
    return path.rsplit("/", 1)[0] if path.endswith(".json") and "/" in path else \
        "." if path.endswith(".json") else path


def inspect_dataset(dataset, *, verify: bool = False,
                    container: str | None = None) -> DatasetInspection:
    """Summarize a curated artifact and optionally read every shard for integrity.

    ``dataset`` accepts the same directory, manifest path, or loaded manifest dictionary as
    :func:`shard_entries`. Metadata inspection does not read shard payloads. ``verify=True``
    checks existence and decodes every shard, returning failures in ``verification.errors``.
    """
    if not isinstance(verify, bool):
        raise ValueError("verify must be true or false")
    entries, resolved_container = shard_entries(dataset, container=container)
    base = _base_path(dataset)
    config: dict = {}
    if base is not None:
        config = json.loads(read_text(f"{base}/config.json"))
        check_version(config.get("artifact_format_version"), kind="artifact")

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
    for entry in entries:
        path = entry.get("path")
        if not exists(path):
            verification_errors.append({"path": path, "error": "missing shard"})
            continue
        if verify:
            try:
                read_shard_arrays(entry, resolved_container)
                checked += 1
            except Exception as error:
                verification_errors.append({"path": path, "error": repr(error)})

    execution = config.get("execution") or {}
    physical = execution.get("physical_plan") or {}
    provenance = config.get("provenance") or {}
    return DatasetInspection(
        dataset=base, container=resolved_container, shards=len(entries),
        splits=dict(sorted(splits.items())), modalities=modalities,
        budget=(config.get("results") or {}).get("budget"),
        errors=list(config.get("errors") or ()),
        sources_skipped=(config.get("results") or {}).get("sources_skipped", 0),
        provenance_fingerprint=provenance.get("fingerprint"),
        execution={"bounded": execution.get("bounded", False),
                   "backend": physical.get("backend", physical.get("mode"))},
        verification={"requested": verify, "checked": checked,
                      "valid": not verification_errors, "errors": verification_errors})
