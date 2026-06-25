"""Shared job-spec builders for the kur-* CLIs.

A *job* is one JSON document wiring the pipeline from data: schema, adapter, sources,
segmenter, strategy, and (for curate) output. Each piece may be inline or a path/name, so a
whole curation run is a single reproducible file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..adapters import get_adapter
from ..schema import SchemaSpec
from ..segment import Segmenter


def _maybe_path(value):
    """A dict is inline; a str is a path to a JSON document."""
    if isinstance(value, str):
        return json.loads(Path(value).read_text())
    return value


def build_schema(job: dict) -> Optional[SchemaSpec]:
    """The schema, or None (e.g. the curated adapter derives its own from the dataset)."""
    if "schema" not in job:
        return None
    return SchemaSpec.from_dict(_maybe_path(job["schema"]))


def build_adapter(job: dict, schema: Optional[SchemaSpec]):
    a = dict(job["adapter"])
    kind = a.pop("type")
    return get_adapter(kind, schema=schema, **a) if schema is not None else get_adapter(kind, **a)


def build_segmenter(job: dict, schema: Optional[SchemaSpec]) -> Segmenter:
    sg = job["segmenter"]
    hz = sg.get("sample_hz") or (schema.sample_hz if schema else None)
    return Segmenter(sg["window"], sg["stride"], unit=sg.get("unit", "steps"), sample_hz=hz)


def resolve_sources(job: dict, adapter) -> list:
    """Explicit `sources`, else the adapter's own (the curated adapter lists its shards)."""
    if job.get("sources"):
        return list(job["sources"])
    if hasattr(adapter, "sources"):
        return list(adapter.sources())
    raise ValueError("job has no 'sources' and the adapter exposes no sources()")
