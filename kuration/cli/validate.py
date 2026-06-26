"""kur-validate — preflight a job spec without reading any data.

    kur-validate job.json [--check-sources] [--json-schema] [--print-schema]

Runs every load-time validator the real run would — SchemaSpec parsing, adapter field-map
checks, the strategy's DSL/tags/scorers/referenced-metric resolution, the segmenter, and (for
a curate job) the OutputSpec — and reports the first failure with a non-zero exit. A typo fails
in milliseconds instead of after a parallel feature pass. No source data is read unless
`--check-sources` is given (which only stats existence, never parses).

`--print-schema` dumps the shipped JSON Schema (point your editor's `$schema` at it for
autocomplete); `--json-schema` additionally validates the job against it (needs `jsonschema`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..output import OutputSpec
from ..strategy import load_strategy
from ._common import build_adapter, build_schema, build_segmenter, resolve_sources

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "job.schema.json"


def job_kind(job: dict) -> str:
    if "arms" in job:
        return "experiment"
    if "output" in job or "train_hours" in job:
        return "curate"
    return "report"


def validate_job(job: dict, *, check_sources: bool = False) -> dict:
    """Run the SDK's load-time validators. Raises ValueError/etc on the first problem;
    returns a summary dict on success."""
    kind = job_kind(job)
    schema = build_schema(job)
    adapter = build_adapter(job, schema)
    schema = schema if schema is not None else getattr(adapter, "schema", None)
    if schema is None:
        raise ValueError("no 'schema' in job and the adapter exposes none")
    build_segmenter(job, schema)                      # validates window/stride/unit/sample_hz

    if kind == "experiment":
        if not job.get("arms"):
            raise ValueError("experiment job needs a non-empty 'arms'")
        for a in job["arms"]:
            if "name" not in a or "strategy" not in a:
                raise ValueError(f"each arm needs 'name' and 'strategy' (offending: {a!r})")
            load_strategy(a["strategy"], schema)      # compiles DSL, validates tags/scorers/refs
        if "target_size" not in job:
            raise ValueError("experiment job needs 'target_size'")
    else:
        if "strategy" not in job:
            raise ValueError("job needs a 'strategy'")
        load_strategy(job["strategy"], schema)

    if kind == "curate":
        if "output" not in job:
            raise ValueError("curate job needs an 'output' spec")
        OutputSpec.from_dict(job["output"])           # validates modalities/dtype/container
        if "train_hours" not in job:
            raise ValueError("curate job needs 'train_hours'")
        if "out" not in job:
            raise ValueError("curate job needs an 'out' directory")

    summary = {"kind": kind, "schema": schema.name}
    if kind == "experiment":
        summary["arms"] = [a["name"] for a in job["arms"]]
    try:                                              # resolution is cheap (no shard reads)
        sources = resolve_sources(job, adapter)
    except ValueError as e:                           # not fatal: sources may be supplied at run time
        summary["sources"] = f"unresolved ({e})"
        return summary
    summary["n_sources"] = len(sources)
    if check_sources:                                 # existence failure IS fatal (propagates)
        from ..io import exists
        missing = [s for s in sources if not exists(s)]
        if missing:
            raise ValueError(f"{len(missing)} source(s) do not exist, e.g. {missing[0]!r}")
        summary["sources_exist"] = True
    return summary


def _json_schema_check(job: dict) -> None:
    try:
        import jsonschema
    except ImportError as e:
        raise ImportError("--json-schema needs jsonschema: pip install jsonschema") from e
    jsonschema.validate(job, json.loads(SCHEMA_PATH.read_text()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Preflight a kuration job spec (no data read).")
    ap.add_argument("job", nargs="?", help="Path to the job JSON.")
    ap.add_argument("--check-sources", action="store_true",
                    help="Also verify each resolved source exists (stat only, no read).")
    ap.add_argument("--json-schema", action="store_true",
                    help="Also validate the job against the shipped JSON Schema (needs jsonschema).")
    ap.add_argument("--print-schema", action="store_true",
                    help="Print the shipped JSON Schema and exit.")
    args = ap.parse_args()

    if args.print_schema:
        print(SCHEMA_PATH.read_text())
        return
    if not args.job:
        ap.error("a job path is required (or use --print-schema)")

    job = json.loads(Path(args.job).read_text())
    try:
        if args.json_schema:
            _json_schema_check(job)
        summary = validate_job(job, check_sources=args.check_sources)
    except Exception as e:
        print(f"INVALID: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    bits = [f"kind={summary['kind']}", f"schema={summary['schema']}"]
    if "arms" in summary:
        bits.append(f"arms={summary['arms']}")
    if "n_sources" in summary:
        bits.append(f"sources={summary['n_sources']}"
                    + ("✓" if summary.get("sources_exist") else ""))
    elif "sources" in summary:
        bits.append(summary["sources"])
    print("OK: " + "  ".join(bits))


if __name__ == "__main__":
    main()
