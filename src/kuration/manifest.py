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
        "plugins": __import__("kuration.plugins", fromlist=["plugin_provenance"]).plugin_provenance(),
    }


def write_config(path: str, config: dict) -> None:
    """Serialize a configuration dictionary as indented JSON to a local or remote path."""
    write_text(path, json.dumps(config, indent=2, default=float))
