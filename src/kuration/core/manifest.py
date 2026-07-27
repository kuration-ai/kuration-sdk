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


def dataset_state(out_dir: str) -> str:
    """Classify what a destination already holds: ``absent``, ``writing``, or ``committed``.

    ``writing`` means a previous run published shards and the ``state=writing`` sidecar but
    never committed — readers reject that artifact, so nothing readable is lost by replacing
    it. A manifest without a sidecar predates the commit protocol and is treated as committed
    so a legacy artifact is never silently clobbered.
    """
    from .io import exists, read_text
    if not exists(f"{out_dir}/dataset_manifest.json"):
        return "absent"
    commit_path = f"{out_dir}/dataset.commit.json"
    if not exists(commit_path):
        return "committed"
    try:
        commit = json.loads(read_text(commit_path))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return "committed"
    state = commit.get("state") if isinstance(commit, dict) else None
    return "writing" if state == "writing" else "committed"


def _recorded_shard_paths(out_dir: str) -> list[str]:
    """Absolute shard paths the destination's current manifest points at (may be none)."""
    from .io import exists, read_text
    manifest_path = f"{out_dir}/dataset_manifest.json"
    if not exists(manifest_path):
        return []
    try:
        manifest = json.loads(read_text(manifest_path))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return []
    entries = manifest.get("shards") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        return []
    from .loaders import rebase_entries
    resolved = rebase_entries(
        [entry for entry in entries if isinstance(entry, dict) and entry.get("path")], out_dir)
    return [str(entry["path"]) for entry in resolved]


def clear_dataset_shards(out_dir: str) -> list[str]:
    """Delete a prior attempt's shard files and return the removed paths (sorted).

    Two sources, because an interrupted write leaves shards the manifest never learned about:
    the paths the current manifest records, plus every ``<split>/<name>_shard*`` file the
    writers can produce beneath the dataset root. Shard names are derived per run, so those
    leftovers are not guaranteed to be overwritten by the next attempt — they would linger
    unreferenced inside the artifact. Callers report what this removes; only ``force`` gets
    here.
    """
    from .io import exists, glob, is_remote
    candidates = set(_recorded_shard_paths(out_dir))
    candidates.update(glob(f"{str(out_dir).rstrip('/')}/*/*_shard*"))
    removed: list[str] = []
    for path in sorted(candidates):
        if not exists(path):
            continue
        if is_remote(path):
            from .io import _fs
            _fs(path).rm(path)
        else:
            from pathlib import Path as _Path
            from .io import _local
            local = _Path(_local(path))
            if not local.is_file():
                continue
            local.unlink()
        removed.append(path)
    return removed


def build_config(schema: SchemaSpec, strategy: Strategy, *, results: dict, seed: int,
                 started: Optional[str] = None, ended: Optional[str] = None) -> dict:
    """Build the reproducibility configuration embedded in a curated dataset.

    The returned dictionary contains the artifact version, resolved schema and strategy, seed,
    optional timestamps, result accounting, and loaded-plugin provenance.
    """
    from .plugins import plugin_provenance
    return {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "schema": schema.to_dict(),
        "strategy": strategy.resolved,
        "seed": seed,
        "started": started,
        "ended": ended,
        "results": results,
        "plugins": plugin_provenance(),
    }


def write_config(path: str, config: dict) -> None:
    """Serialize a configuration dictionary as indented JSON to a local or remote path."""
    write_text(path, json.dumps(config, indent=2, default=float))
