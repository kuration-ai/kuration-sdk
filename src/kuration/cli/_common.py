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
        job = json.loads(p.read_text())
        from ..versioning import check_version
        check_version(job.get("job_schema_version"), kind="job")
        return job
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


def error_hint(error: BaseException) -> str | None:
    """Return one stable, actionable remedy for common user-facing failures."""
    message = str(error).lower()
    if "segment pool is empty" in message:
        return ("Run kur-validate --check-sources, use on_error='raise', and confirm the "
                "window is not longer than each source.")
    if "undefined metric" in message or "unknown registered metric" in message:
        return "Define the metric with measure()/strategy.define or register its plugin first."
    if "missing value for output modality" in message:
        return ("Inspect the field mapping and _present mask, or omit/encode fields that are "
                "not defined for every present entity.")
    if "categorical" in message and ("output" in message or "policy" in message):
        return "Choose categorical='omit', 'integer-vocab', or 'encoder' explicitly."
    if "memory_budget" in message or "bounded exact" in message:
        return ("Use top-k/uniform-bins without global filters or scorers, or remove the "
                "memory budget after sizing the pool.")
    if isinstance(error, ImportError) and "pip install" not in message:
        return "Install the optional extra named by the adapter, method, or output container."
    if "resume=true" in message or "resume" in message and "fingerprint" in message:
        return "Use a fresh output directory or restore the exact original inputs and options."
    return None


def format_error(error: BaseException, *, prefix: str = "error") -> str:
    """Format a one-line error with a remedy when the failure class is recognized."""
    rendered = f"{prefix}: {type(error).__name__}: {error}"
    hint = error_hint(error)
    return f"{rendered} | hint: {hint}" if hint else rendered


def cli_run(body, *, debug: bool = False) -> None:
    """Error surface shared by the kur-* mains: a failure prints one `error:` line to stderr
    and exits 1; `--debug` re-raises so the full traceback is available on demand."""
    try:
        body()
    except Exception as e:
        if debug:
            raise
        print(format_error(e), file=sys.stderr)
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
