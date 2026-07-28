"""kur-validate — preflight a job spec without reading any data.

    kur-validate job.json [--check-sources] [--json-schema] [--print-schema]

Runs every load-time validator the real run would — SchemaSpec parsing, adapter field-map
checks, the strategy's DSL/tags/scorers/referenced-metric resolution, the segmenter, and (for
a curate job) the OutputSpec — and reports the first failure with a non-zero exit. A typo fails
in milliseconds instead of after a parallel feature pass. No source data is read unless
`--check-sources` is given, which stats existence and calls the adapter's `check_source`
preflight — every built-in adapter implements it (CSV/JSONL headers, Parquet/HDF5/NPZ/Zarr
schema metadata, HF builder features, curated manifests) — verifying each source can serve
the adapter's mapping, still without reading data rows.

`--print-schema` dumps the shipped JSON Schema (point your editor's `$schema` at it for
autocomplete); `--json-schema` additionally validates the job against it (needs `jsonschema`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._common import format_error, json_document, parse_json, read_job
from .job import job_kind as job_kind, resolve_job

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "job.schema.json"


def validate_job(job: dict, *, check_sources: bool = False) -> dict:
    """Run the SDK's load-time validators. Raises ValueError/etc on the first problem;
    returns a summary dict on success."""
    resolved = resolve_job(job, enumerate_sources=check_sources)
    summary = {"kind": resolved.kind, "schema": resolved.schema.name}
    if resolved.kind == "experiment":
        summary["arms"] = [name for name, _strategy in resolved.arms]
    if resolved.sources is None:
        # Enumerating adapter-resolved sources can be arbitrarily expensive (the HF adapter
        # materializes the whole dataset) — the no-data-read default must not touch them.
        summary["sources"] = "implicit (adapter-resolved; --check-sources enumerates)"
        from ..results import ValidationResult
        return ValidationResult(summary)
    sources = list(resolved.sources)
    summary["n_sources"] = len(sources)
    if check_sources:                                 # existence failure IS fatal (propagates)
        if getattr(resolved.adapter, "sources_are_paths", True):   # HF names/dataset globs
            from ..io import exists                       # aren't statable paths
            missing = [s for s in sources if not exists(s)]
            if missing:
                raise ValueError(f"{len(missing)} source(s) do not exist, e.g. {missing[0]!r}")
            summary["sources_exist"] = True
        check = getattr(resolved.adapter, "check_source", None)
        if check is not None:                         # header/metadata sniff, no data rows read
            for s in sources:
                check(s)
            summary["sources_checked"] = True
    from ..results import ValidationResult
    return ValidationResult(summary)


def _json_schema_check(job: dict) -> None:
    try:
        import jsonschema
    except ImportError as e:
        raise ImportError("--json-schema needs the optional extra: "
                          "pip install 'kuration-sdk[cli]'  (or: pip install jsonschema)") from e
    jsonschema.validate(
        job,
        parse_json(SCHEMA_PATH.read_text(), label="shipped job JSON Schema"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Preflight a kuration job spec (no data read).")
    ap.add_argument("job", nargs="?", help="Path to the job JSON.")
    ap.add_argument("--check-sources", action="store_true",
                    help="Also verify each resolved source exists, and (adapters with "
                         "check_source) that its header/schema serves the column map — "
                         "no data rows read.")
    ap.add_argument("--json-schema", action="store_true",
                    help="Also validate the job against the shipped JSON Schema (needs jsonschema).")
    ap.add_argument("--print-schema", action="store_true",
                    help="Print the shipped JSON Schema and exit.")
    ap.add_argument("--debug", action="store_true", help="Show full tracebacks on failure.")
    args = ap.parse_args()

    if args.print_schema:
        schema = parse_json(SCHEMA_PATH.read_text(), label="shipped job JSON Schema")
        print(json_document(schema, label="shipped job JSON Schema").rstrip())
        return
    if not args.job:
        ap.error("a job path is required (or use --print-schema)")

    try:
        job = read_job(args.job)
        if args.json_schema:
            _json_schema_check(job)
        summary = validate_job(job, check_sources=args.check_sources)
    except ImportError as e:
        if args.debug:
            raise
        # a missing optional dependency is an environment problem, not an invalid spec
        print(format_error(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if args.debug:
            raise
        print(format_error(e, prefix="INVALID"), file=sys.stderr)
        sys.exit(1)

    bits = [f"kind={summary['kind']}", f"schema={summary['schema']}"]
    if "arms" in summary:
        bits.append(f"arms={summary['arms']}")
    if "n_sources" in summary:
        bits.append(f"sources={summary['n_sources']}"
                    + ("✓ (exist + column maps)" if summary.get("sources_checked")
                       else "✓" if summary.get("sources_exist") else ""))
    elif "sources" in summary:
        bits.append(summary["sources"])
    print("OK: " + "  ".join(bits))


if __name__ == "__main__":
    main()
