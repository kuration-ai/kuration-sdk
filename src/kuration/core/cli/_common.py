"""Shared job-spec builders for the kur-* CLIs.

A *job* is one JSON document wiring the pipeline from data: schema, adapter, sources,
segmenter, strategy, and (for curate) output. Each piece may be inline or a path/name, so a
whole curation run is a single reproducible file.
"""
from __future__ import annotations

import json
import posixpath
import sys
from pathlib import Path, PurePosixPath
from typing import Optional

from ..adapters import get_adapter
from ..jsonutil import parse_json
from ..schema import SchemaSpec
from ..segment import Segmenter


def json_document(value, *, label: str) -> str:
    """Encode one indented, newline-terminated strict JSON document."""
    try:
        return json.dumps(value, indent=2, allow_nan=False) + "\n"
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} cannot be represented as strict JSON: {error}") from error


def _storage_path_relation(left, right) -> str | None:
    """Return the component-aware relationship of two local or fsspec paths.

    The result describes ``left`` relative to ``right``: ``"same"``, ``"ancestor"``,
    ``"descendant"``, or ``None``. Prefix strings are deliberately insufficient here:
    ``data-report.html`` is not inside ``data``, while a symlinked local spelling or a
    normalized remote spelling may still identify the same tree.
    """
    from ..io import _fs, _local, is_remote

    left, right = str(left), str(right)
    if is_remote(left) != is_remote(right):
        return None
    if is_remote(left):
        left_fs, right_fs = _fs(left), _fs(right)
        if getattr(left_fs, "_fs_token_", None) != getattr(right_fs, "_fs_token_", None):
            return None

        def remote_path(fs, value: str) -> PurePosixPath:
            raw = str(fs._strip_protocol(value))
            if any(component in {".", ".."} for component in raw.split("/")):
                raise ValueError(
                    f"path {value!r} contains an ambiguous dot path component; choose a "
                    "normalized remote path"
                )
            return PurePosixPath(posixpath.normpath("/" + raw.lstrip("/")))

        left_parts = remote_path(left_fs, left).parts
        right_parts = remote_path(right_fs, right).parts
    else:
        left_parts = Path(_local(left)).resolve(strict=False).parts
        right_parts = Path(_local(right)).resolve(strict=False).parts

    if left_parts == right_parts:
        return "same"
    if (len(left_parts) < len(right_parts)
            and right_parts[:len(left_parts)] == left_parts):
        return "ancestor"
    if (len(right_parts) < len(left_parts)
            and left_parts[:len(right_parts)] == right_parts):
        return "descendant"
    return None


def _same_storage_path(left, right) -> bool:
    """Compare local or fsspec paths without trusting superficial path spelling."""
    return _storage_path_relation(left, right) == "same"


def artifact_root_path(path) -> str:
    """Normalize a dataset directory or accepted manifest spelling to its artifact root."""
    from ..io import _local, is_remote

    raw = str(path).rstrip("/\\")
    if is_remote(raw):
        scheme, remote_path = raw.split("://", 1)
        if posixpath.basename(remote_path) == "dataset_manifest.json":
            return f"{scheme}://{posixpath.dirname(remote_path)}"
        return raw
    local = Path(_local(raw))
    return str(local.parent) if local.name == "dataset_manifest.json" else str(local)


def guard_file_output(
    path,
    *,
    force: bool | None,
    protected_inputs=(),
    label: str = "output",
) -> None:
    """Fail before work when a file output overlaps input or an existing result.

    ``force=None`` means the calling command exposes no overwrite flag, so the diagnostic
    directs the user to another path instead of advertising an option that does not exist.
    """
    from ..io import _fs, _local, exists, is_remote

    output = str(path)
    if not output:
        raise ValueError(f"{label} must be a non-empty path")
    if is_remote(output) and any(character in output for character in "*?["):
        raise ValueError(
            f"{label} {output!r} contains a glob metacharacter; remote output paths must "
            "not contain '*', '?', or '['"
        )
    for source in protected_inputs:
        relation = _storage_path_relation(output, source)
        if relation == "same":
            raise ValueError(
                f"{label} {output!r} is also an input path; choose a separate output path"
            )
        if relation is not None:
            raise ValueError(
                f"{label} {output!r} overlaps input path {str(source)!r} "
                f"({relation}); choose a separate output path outside the input tree"
            )
    if exists(output):
        is_directory = (
            _fs(output).isdir(output)
            if is_remote(output)
            else Path(_local(output)).is_dir()
        )
        if is_directory:
            raise IsADirectoryError(
                f"{label} {output!r} is a directory; choose a file path"
            )
        if force is not True:
            remedy = (
                "choose a new output path"
                if force is None
                else "pass --force to overwrite"
            )
            raise FileExistsError(f"{label} {output!r} already exists; {remedy}")


def read_job(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"job spec not found: {path}")
    job = parse_json(p.read_text(), label=f"job spec {path}")
    if not isinstance(job, dict):
        raise ValueError(f"job spec {path} must contain a JSON object")
    from ..versioning import check_version
    check_version(job.get("job_schema_version"), kind="job")
    return job


def load_job_plugins(job: dict):
    """Load job plugins and return an isolated registry snapshot for all later resolution."""
    from ..plugins import load_plugins, plugin_registry_snapshot
    modules = job.get("plugins") or []
    include_entry_points = job.get("include_plugin_entry_points", True)
    load_plugins(modules, include_entry_points=include_entry_points)
    return plugin_registry_snapshot(modules, include_entry_points=include_entry_points)


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
    # Parser and plugin exceptions may contain newlines or tabs. Normal CLI mode stays
    # log- and shell-friendly; ``--debug`` is the explicit multiline traceback surface.
    message = " ".join(str(error).split()) or "<no message>"
    rendered = f"{prefix}: {type(error).__name__}: {message}"
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
        return parse_json(Path(value).read_text(), label=f"referenced JSON document {value}")
    return value


def parse_mapping_option(value: str | None, *, label: str) -> dict | None:
    """Parse a mapping-style CLI option that accepts either inline JSON or a path to a JSON
    file (`--kinds`/`--fields`/`--adapter-options` across kur-infer/init/preview/import).

    A value naming an existing file is read from disk; anything else is parsed as inline JSON
    text. Either form must decode to a JSON object — loud, one-line failure names both
    accepted forms so the fix is obvious from the error alone.
    """
    if value is None:
        return None
    path = Path(value)
    if path.is_file():
        text, source = path.read_text(), f"file {value!r}"
    else:
        text, source = value, "inline JSON"
    try:
        parsed = parse_json(text, label=source)
    except ValueError as e:
        raise ValueError(
            f"{label} must be inline JSON or a path to a JSON file; could not parse it as "
            f"{source}: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object (inline JSON or path to JSON file)")
    return parsed


def build_schema(job: dict) -> Optional[SchemaSpec]:
    """The schema, or None (e.g. the curated adapter derives its own from the dataset)."""
    if "schema" not in job:
        return None
    return SchemaSpec.from_dict(_maybe_path(job["schema"]))


def build_adapter(job: dict, schema: Optional[SchemaSpec], *, registry=None):
    a = dict(job["adapter"])
    kind = a.pop("type")
    return (get_adapter(kind, registry=registry, schema=schema, **a) if schema is not None
            else get_adapter(kind, registry=registry, **a))


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
