"""kur-infer — bootstrap a `SchemaSpec` (+ adapter hints) from a data sample.

    kur-infer data.csv --out schema.json

The on-ramp before a job exists: sample one source, classify its columns, and emit the
`schema` block a job needs plus the `fields`/`time_column`/`entity_column` hints its adapter
takes. `--out` writes the bare schema document, which a job may reference by path
(`"schema": "schema.json"`). Inference stays conservative and loud: an unclassifiable column
raises with a pointer at `--kinds` or an explicit hand-written schema.
"""
from __future__ import annotations

import argparse

from ._common import (cli_run, guard_file_output, json_document,
                      parse_mapping_option)

_ADAPTER_FOR_KIND = {
    "timeseries": "timeseries", "jsonl": "jsonl", "json-rows": "timeseries",
    "npz": "npz", "hdf5": "hdf5", "zarr": "zarr", "parquet-dataset": "parquet-dataset",
}


def infer_source(source: str, *, name: str = "inferred", time: str | None = None,
                 entity: str | None = None, hz: float | None = None,
                 kinds: dict | None = None, sample: int = 200) -> dict:
    """Sample one source and return its adapter kind, schema, hints, and a job fragment."""
    from ..infer import infer_schema, sample_source
    kind, rows = sample_source(source, n=sample)
    schema, hints = infer_schema(rows, name=name, time=time, entity=entity, hz=hz,
                                 kinds=kinds)
    adapter: dict = {"type": _ADAPTER_FOR_KIND.get(kind, kind)}
    # Only the tabular adapters take a column mapping; array adapters read their own layout.
    if kind in {"timeseries", "jsonl", "json-rows"}:
        adapter["fields"] = hints["fields"]
        if hints.get("time_column"):
            adapter["time_column"] = hints["time_column"]
        if hints.get("entity_column"):
            adapter["entity_column"] = hints["entity_column"]
    warnings = []
    if schema.sample_hz is None:
        warnings.append({
            "code": "sample-rate-unknown",
            "message": "no sample rate could be derived from the time column",
            "hint": "Pass --hz so hours budgets and segment durations are meaningful."})
    if hints.get("list_columns"):
        warnings.append({
            "code": "list-columns-expanded",
            "message": f"list cells expanded per dimension: {sorted(hints['list_columns'])}",
            "hint": "Confirm the expanded dimension names match the field's meaning."})
    from ..provenance import publishable_source_id
    return {"source": publishable_source_id(source), "adapter_kind": kind,
            "schema": schema.to_dict(),
            "hints": hints, "adapter": adapter, "warnings": warnings}


def render_inference(result: dict) -> str:
    schema = result["schema"]
    fields = schema.get("fields") or {}
    lines = [f"source: {result['source']}",
             f"adapter: {result['adapter_kind']}",
             f"schema: {schema.get('name', 'inferred')} ({len(fields)} field(s))",
             f"entity axis: {schema.get('entity_axis') or 'none'}",
             f"sample rate: {schema.get('sample_hz') or 'unknown'}"]
    for field_name, spec in fields.items():
        detail = spec.get("kind", "unknown")
        if spec.get("dims"):
            detail += f"[{', '.join(spec['dims'])}]"
        lines.append(f"field {field_name}: {detail}")
    for warning in result["warnings"]:
        lines.append(f"warning [{warning['code']}]: {warning['message']}")
        lines.append(f"  fix: {warning['hint']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Infer a kuration schema and adapter hints from a data sample.")
    ap.add_argument("source", help="One data file, or a directory of .parquet parts.")
    ap.add_argument("--out", metavar="PATH",
                    help="Write the schema document here; refuses an existing file.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing --out file (never the sampled source).")
    ap.add_argument("--name", default="inferred", help="Schema name (default: inferred).")
    ap.add_argument("--time", help="Time-column override.")
    ap.add_argument("--entity", help="Entity-column override.")
    ap.add_argument("--hz", type=float, help="Sample-rate override.")
    ap.add_argument("--kinds", metavar="JSON|PATH",
                    help="Per-column kind overrides for ambiguous columns, as inline JSON or "
                         "path to JSON file.")
    ap.add_argument("--sample", type=int, default=200,
                    help="Rows/timesteps to sample (default: 200).")
    ap.add_argument("--json", action="store_true",
                    help="Emit schema, hints, and the adapter fragment as JSON.")
    ap.add_argument("--debug", action="store_true", help="Show full tracebacks on failure.")
    args = ap.parse_args()

    def body():
        if args.force and not args.out:
            raise ValueError("--force requires --out")
        if args.out:
            guard_file_output(args.out, force=args.force, protected_inputs=(args.source,),
                              label="schema output")
        if args.sample < 1:
            raise ValueError("--sample must be a positive integer")
        kinds = parse_mapping_option(args.kinds, label="--kinds")
        result = infer_source(args.source, name=args.name, time=args.time,
                              entity=args.entity, hz=args.hz, kinds=kinds,
                              sample=args.sample)
        if args.out:
            from ..io import write_text
            write_text(args.out, json_document(result["schema"], label="inferred schema"))
        print(json_document(result, label="schema inference").rstrip()
              if args.json else render_inference(result))
        if args.out and not args.json:
            print(f"wrote schema: {args.out}")

    cli_run(body, debug=args.debug)


if __name__ == "__main__":
    main()
