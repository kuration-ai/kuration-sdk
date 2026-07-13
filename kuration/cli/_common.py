"""Shared job-spec builders for the kur-* CLIs.

A *job* is one JSON document wiring the pipeline from data: schema, adapter, sources,
segmenter, strategy, and (for curate) output. Each piece may be inline or a path/name, so a
whole curation run is a single reproducible file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from ..adapters import get_adapter
from ..schema import SchemaSpec
from ..segment import Segmenter


def read_job(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"job spec not found: {path}")
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"job spec {path} is not valid JSON: {e}") from e


def load_job_plugins(job: dict) -> list[dict]:
    """Load explicit registration modules before constructing any registry-backed objects."""
    from ..plugins import load_plugins
    return load_plugins(job.get("plugins") or [])


def cli_progress_for(*, quiet: bool = False):
    """Render progress only on an interactive stderr; redirected CLIs stay machine-clean."""
    if quiet or not sys.stderr.isatty():
        return None
    from ..progress import cli_progress
    return cli_progress(file=sys.stderr)


def cli_run(body, *, debug: bool = False) -> None:
    """Error surface shared by the kur-* mains: a failure prints one `error:` line to stderr
    and exits 1; `--debug` re-raises so the full traceback is available on demand."""
    try:
        body()
    except Exception as e:
        if debug:
            raise
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(1)


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
    hz = sg.get("sample_hz") if "sample_hz" in sg else (schema.sample_hz if schema else None)
    return Segmenter(sg["window"], sg["stride"], unit=sg.get("unit", "steps"), sample_hz=hz)


def resolve_sources(job: dict, adapter) -> list:
    """Explicit `sources`, else the adapter's own (the curated adapter lists its shards)."""
    if job.get("sources"):
        return list(job["sources"])
    if hasattr(adapter, "sources"):
        return list(adapter.sources())
    raise ValueError("job has no 'sources' and the adapter exposes no sources()")
