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
    payload = {"count": len(segments), "segments": [segment_record(s) for s in segments]}
    write_text(path, json.dumps(payload, indent=2, default=float))


def build_config(schema: SchemaSpec, strategy: Strategy, *, results: dict, seed: int,
                 started: Optional[str] = None, ended: Optional[str] = None) -> dict:
    return {
        "schema": schema.to_dict(),
        "strategy": strategy.resolved,
        "seed": seed,
        "started": started,
        "ended": ended,
        "results": results,
    }


def write_config(path: str, config: dict) -> None:
    write_text(path, json.dumps(config, indent=2, default=float))
