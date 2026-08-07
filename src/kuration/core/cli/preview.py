"""kur-preview — read a bounded sample of the corpus a job points at.

    kur-preview job.json
    kur-preview --infer data.csv

`kur-explain` reads no rows and `kur-validate --check-sources` only proves sources exist, so
neither answers "what is actually in this data". This decodes a bounded prefix of a few
sources and reports field ranges, missingness, and representative ticks — the cheap check
between a job that parses and a run that takes an hour.
"""
from __future__ import annotations

import argparse

from ._common import (cli_run, guard_file_output, json_document, parse_mapping_option,
                      read_job)


def _job_subject(path: str, *, html_output: str | None = None):
    """Resolve a job spec into the (schema, adapter, sources) preview needs."""
    from .job import resolve_job
    job = read_job(path)
    if html_output:
        raw_sources = job.get("sources")
        raw_sources = raw_sources if isinstance(raw_sources, (list, tuple)) else ()
        referenced = [
            value for value in (job.get("schema"), job.get("strategy"))
            if isinstance(value, str)
        ]
        guard_file_output(
            html_output,
            force=None,
            protected_inputs=(
                path,
                *(source for source in raw_sources
                  if isinstance(source, str)),
                *referenced,
            ),
            label="preview HTML output",
        )
    resolved = resolve_job(job)
    if resolved.sources is None:
        raise ValueError(
            f"job {path!r} resolves no explicit sources and its adapter lists none; "
            "preview needs a concrete source list")
    return resolved.schema, resolved.adapter, resolved.sources


def _inferred_subject(sources, **infer_kwargs):
    """Infer a schema from the first source and build its adapter for an ad-hoc preview."""
    from ..schema import SchemaSpec
    from ._common import build_adapter
    from .infer_schema_cli import infer_source
    inferred = infer_source(sources[0], **infer_kwargs)
    schema = SchemaSpec.from_dict(inferred["schema"])
    adapter = build_adapter({"adapter": inferred["adapter"]}, schema)
    return schema, adapter, list(sources)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Preview a bounded sample of the sources a job reads.")
    ap.add_argument("target", nargs="+",
                    help="A job spec, or (with --infer) one or more data files.")
    ap.add_argument("--infer", action="store_true",
                    help="Treat the arguments as data files and infer their schema.")
    ap.add_argument("--max-sources", type=int, default=3,
                    help="Sources to sample (default: 3).")
    ap.add_argument("--max-ticks", type=int, default=10_000,
                    help="Ticks to read per source (default: 10000).")
    ap.add_argument("--examples", type=int, default=5,
                    help="Representative ticks to retain (default: 5).")
    ap.add_argument("--time", help="Time-column override (--infer only).")
    ap.add_argument("--entity", help="Entity-column override (--infer only).")
    ap.add_argument("--hz", type=float, help="Sample-rate override (--infer only).")
    ap.add_argument("--kinds", metavar="JSON|PATH",
                    help="Per-column kind overrides (--infer only), as inline JSON or path "
                         "to JSON file.")
    ap.add_argument("--json", action="store_true", help="Emit structured JSON.")
    ap.add_argument("--html", metavar="PATH", help="Write a self-contained HTML preview.")
    ap.add_argument("--debug", action="store_true", help="Show full tracebacks on failure.")
    args = ap.parse_args()

    def body():
        infer_only = ("--time", "--entity", "--hz", "--kinds")
        if args.html:
            guard_file_output(
                args.html,
                force=None,
                protected_inputs=tuple(args.target),
                label="preview HTML output",
            )
        if not args.infer:
            if len(args.target) != 1:
                raise ValueError("job mode takes exactly one job spec; pass --infer to "
                                 "preview several data files directly")
            supplied = [name for name, value in
                        zip(infer_only, (args.time, args.entity, args.hz, args.kinds))
                        if value is not None]
            if supplied:
                raise ValueError(f"{', '.join(supplied)} apply to --infer mode only; a job "
                                 "spec carries its own schema and adapter")
            schema, adapter, sources = _job_subject(
                args.target[0], html_output=args.html)
        else:
            kinds = parse_mapping_option(args.kinds, label="--kinds")
            schema, adapter, sources = _inferred_subject(
                args.target, time=args.time, entity=args.entity, hz=args.hz, kinds=kinds)

        if args.html:
            guard_file_output(
                args.html,
                force=None,
                protected_inputs=tuple(sources),
                label="preview HTML output",
            )
        from ..review import preview_dataset
        preview = preview_dataset(schema=schema, adapter=adapter, sources=sources,
                                  max_sources=args.max_sources,
                                  max_ticks_per_source=args.max_ticks,
                                  examples=args.examples)
        if args.html:
            from ..html_report import write_html_report
            write_html_report(args.html, preview, title="kuration dataset preview")
        print(json_document(
            preview.to_dict(), label="dataset preview"
        ).rstrip() if args.json else preview.render())

    cli_run(body, debug=args.debug)


if __name__ == "__main__":
    main()
