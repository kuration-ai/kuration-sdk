"""Manifests + reproducible config.

`config.json` embeds the resolved schema AND strategy, so a curated dataset reproduces from
its own artifact alone — the same reproducibility guarantee the reference pipeline gives,
now spanning the (data-defined) schema too. Local-path writers for now; gs:// I/O arrives
with the output layer (Phase D).
"""
from __future__ import annotations

import json
from typing import Optional

from .io import write_text
from .schema import SchemaSpec
from .segment import Segment
from .strategy import Strategy
from .versioning import ARTIFACT_FORMAT_VERSION


def segment_record(s: Segment) -> dict:
    return {
        "source_id": s.source_id,
        "time_range": list(s.time_range),
        "features": {k: float(v) for k, v in s.features.items()},
        "bin": list(s.bin),
        "tags": sorted(str(t) for t in s.tags),
        "group": s.group,
    }


def dump_segment_manifest(path: str, segments: list[Segment]) -> None:
    """Write scored segments and the artifact format version as JSON."""
    payload = {"artifact_format_version": ARTIFACT_FORMAT_VERSION,
               "count": len(segments), "segments": [segment_record(s) for s in segments]}
    write_text(path, json.dumps(payload, indent=2, default=float))


def portable_shard_entries(out_dir: str, entries: list[dict]) -> list[dict]:
    """Return manifest entries whose shard paths are relative to the dataset root."""
    root = str(out_dir).replace("\\", "/").rstrip("/")
    portable = []
    for entry in entries:
        path = str(entry["path"]).replace("\\", "/")
        prefix = f"{root}/"
        if path.startswith(prefix):
            relative = path[len(prefix):]
        else:
            # Writers are expected to place shards beneath out_dir. Keep a legacy-compatible
            # fallback, but never serialize a host-absolute path into a portable artifact.
            relative = f"{entry.get('split', 'all')}/{path.rsplit('/', 1)[-1]}"
        if relative.startswith("../") or relative.startswith("/") or "/../" in relative:
            raise ValueError(f"shard path {entry['path']!r} escapes dataset root {out_dir!r}")
        portable.append({**entry, "path": relative})
    return portable


def commit_dataset(out_dir: str) -> dict:
    """Publish the final small commit record after manifest and config are durable."""
    from .io import file_integrity
    payload = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "protocol": "sidecar-v1",
        "state": "committed",
        "files": {
            "dataset_manifest.json": file_integrity(f"{out_dir}/dataset_manifest.json"),
            "config.json": file_integrity(f"{out_dir}/config.json"),
        },
    }
    write_text(f"{out_dir}/dataset.commit.json",
               json.dumps(payload, sort_keys=True, indent=2))
    return payload


def begin_dataset(out_dir: str) -> dict:
    """Invalidate any prior commit before shard paths can be rewritten.

    The existing manifest entries are retained so a later ``resume=True`` run can checksum
    and reuse shards that the interrupted writer did not change. Readers reject the atomic
    ``state=writing`` sidecar, preventing them from observing a mixed old/new dataset.
    """
    from .io import exists, file_integrity, read_text
    manifest_path = f"{out_dir}/dataset_manifest.json"
    manifest = {"artifact_format_version": ARTIFACT_FORMAT_VERSION,
                "commit_protocol": "sidecar-v1", "count": 0, "shards": []}
    if exists(manifest_path):
        try:
            loaded = json.loads(read_text(manifest_path))
            if isinstance(loaded, dict):
                manifest = {**loaded, "commit_protocol": "sidecar-v1"}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    write_text(manifest_path, json.dumps(manifest, indent=2, default=float))
    files = {"dataset_manifest.json": file_integrity(manifest_path)}
    config_path = f"{out_dir}/config.json"
    if exists(config_path):
        files["config.json"] = file_integrity(config_path)
    payload = {"artifact_format_version": ARTIFACT_FORMAT_VERSION,
               "protocol": "sidecar-v1", "state": "writing", "files": files}
    write_text(f"{out_dir}/dataset.commit.json",
               json.dumps(payload, sort_keys=True, indent=2))
    return payload


def build_config(schema: SchemaSpec, strategy: Strategy, *, results: dict, seed: int,
                 started: Optional[str] = None, ended: Optional[str] = None) -> dict:
    """Build the reproducibility configuration embedded in a curated dataset.

    The returned dictionary contains the artifact version, resolved schema and strategy, seed,
    optional timestamps, result accounting, and loaded-plugin provenance.
    """
    return {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "schema": schema.to_dict(),
        "strategy": strategy.resolved,
        "seed": seed,
        "started": started,
        "ended": ended,
        "results": results,
        "plugins": __import__("kuration.core.plugins", fromlist=["plugin_provenance"]).plugin_provenance(),
    }


def write_config(path: str, config: dict) -> None:
    """Serialize a configuration dictionary as indented JSON to a local or remote path."""
    write_text(path, json.dumps(config, indent=2, default=float))
