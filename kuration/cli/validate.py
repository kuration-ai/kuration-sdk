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
import json
import math
import sys
from pathlib import Path

from ..output import OutputSpec
from ..strategy import load_strategy
from ._common import build_adapter, build_schema, build_segmenter, load_job_plugins, read_job

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
    workers = job.get("workers", 1)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    seed = job.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if job.get("on_error", "skip") not in {"skip", "raise"}:
        raise ValueError("on_error must be 'skip' or 'raise'")
    for flag in ("resume", "streaming"):
        if flag in job and not isinstance(job[flag], bool):
            raise ValueError(f"{flag} must be true or false")
    load_job_plugins(job)
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
        names = [a["name"] for a in job["arms"]]
        if len(set(names)) != len(names):
            raise ValueError("experiment arm names must be unique")
        if "target_size" not in job:
            raise ValueError("experiment job needs 'target_size'")
        from ..strategy import validate_target_size
        validate_target_size(job["target_size"])
    else:
        if "strategy" not in job:
            raise ValueError("job needs a 'strategy'")
        load_strategy(job["strategy"], schema)

    if kind == "curate":
        if "output" not in job:
            raise ValueError("curate job needs an 'output' spec")
        output = OutputSpec.from_dict(job["output"])
        output.validate_schema(schema)
        if "train_hours" not in job:
            raise ValueError("curate job needs 'train_hours'")
        for name in ("train_hours", "test_hours"):
            value = job.get(name, 0.0)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite non-negative number")
        if "out" not in job:
            raise ValueError("curate job needs an 'out' directory")

    summary = {"kind": kind, "schema": schema.name}
    if kind == "experiment":
        summary["arms"] = [a["name"] for a in job["arms"]]
    if job.get("sources"):
        sources = list(job["sources"])
    elif check_sources and hasattr(adapter, "sources"):
        sources = list(adapter.sources())             # user opted into source I/O
    else:
        # Enumerating adapter-resolved sources can be arbitrarily expensive (the HF adapter
        # materializes the whole dataset) — the no-data-read default must not touch them.
        summary["sources"] = "implicit (adapter-resolved; --check-sources enumerates)"
        return summary
    summary["n_sources"] = len(sources)
    if check_sources:                                 # existence failure IS fatal (propagates)
        if getattr(adapter, "sources_are_paths", True):   # HF names / dataset-mode globs
            from ..io import exists                       # aren't statable paths
            missing = [s for s in sources if not exists(s)]
            if missing:
                raise ValueError(f"{len(missing)} source(s) do not exist, e.g. {missing[0]!r}")
            summary["sources_exist"] = True
        check = getattr(adapter, "check_source", None)
        if check is not None:                         # header/metadata sniff, no data rows read
            for s in sources:
                check(s)
            summary["sources_checked"] = True
    return summary


def _json_schema_check(job: dict) -> None:
    try:
        import jsonschema
    except ImportError as e:
        raise ImportError("--json-schema needs the optional extra: "
                          "pip install -e '.[cli]'  (or: pip install jsonschema)") from e
    jsonschema.validate(job, json.loads(SCHEMA_PATH.read_text()))


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
        print(SCHEMA_PATH.read_text())
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
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if args.debug:
            raise
        print(f"INVALID: {type(e).__name__}: {e}", file=sys.stderr)
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
