"""kur-report — profile a dataset's metric distributions from a JSON job spec.

    kur-report job.json [--out report.json]

Uses the same job keys as kur-curate (schema, adapter, sources, segmenter, strategy) but runs
no selection: it scores every segment and reports per-metric distributions + group/tag coverage.
"""
from __future__ import annotations

import argparse

from ..report import dataset_report
from ..scoring import score_sources
from ._common import (cli_progress_for, cli_run, guard_file_output, json_document,
                      read_job)
from .job import resolve_job


def main() -> None:
    ap = argparse.ArgumentParser(description="Distribution/quality report for a dataset.")
    ap.add_argument("job", help="Path to the job JSON.")
    ap.add_argument("--out", default=None,
                    help="Write the full report JSON here (refuses an existing file).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing --out file (never the input job).")
    ap.add_argument("--debug", action="store_true", help="Show full tracebacks on failure.")
    ap.add_argument("--quiet", action="store_true", help="Suppress interactive progress output.")
    args = ap.parse_args()
    cli_run(lambda: _run(args), debug=args.debug)


def _run(args) -> None:
    if args.force and not args.out:
        raise ValueError("--force requires --out")
    if args.out:
        guard_file_output(args.out, force=args.force, protected_inputs=(args.job,),
                          label="report output")
    job = read_job(args.job)
    referenced = [
        value for value in (job.get("schema"), job.get("strategy"), job.get("out"))
        if isinstance(value, str)
    ]
    raw_sources = job.get("sources")
    raw_sources = raw_sources if isinstance(raw_sources, (list, tuple)) else ()
    if args.out:
        guard_file_output(
            args.out,
            force=args.force,
            protected_inputs=(
                args.job,
                *(source for source in raw_sources if isinstance(source, str)),
                *referenced,
            ),
            label="report output",
        )
    resolved = resolve_job(job)
    # Reporting a fully specified curate job is an intentional convenience: all curate-only
    # fields are validated/resolved above, then the report consumes its common scoring plan.
    if resolved.kind not in {"report", "curate"}:
        raise ValueError(f"kur-report requires a report or curate job, got {resolved.kind!r}")
    schema, adapter, strategy = resolved.schema, resolved.adapter, resolved.strategy
    if strategy is None or resolved.sources is None:  # guaranteed by the resolver for these kinds
        raise RuntimeError("resolved report job is missing its strategy or sources")
    if args.out:
        guard_file_output(
            args.out,
            force=args.force,
            protected_inputs=(args.job, *resolved.sources, *referenced),
            label="report output",
        )
    progress = cli_progress_for(quiet=args.quiet)

    # Scorer outputs are produced by the scorer stage, not the feature pass — exclude them
    # here, then apply the configured scorers over the complete pool below.
    registered = sorted(
        r for r in strategy.referenced_metrics()
        if r not in strategy.define and r not in strategy.scorer_outputs()
    )
    segs, errors, _ = score_sources(
        schema=schema, adapter=adapter, sources=resolved.sources, define=strategy.define,
        registered=registered, segmenter=resolved.segmenter, tags_spec=strategy.tags,
        workers=job.get("workers", 1), on_error=resolved.on_error,
        streaming=job.get("streaming", False), progress=progress,
        executor=job.get("executor", "local"), registry=resolved.registry)

    if strategy.scorers:
        from ..scorers import apply_scorers
        apply_scorers(strategy.scorers, segs, schema=schema, adapter=adapter,
                      seed=job.get("seed", 0), progress=progress, registry=resolved.registry)

    rep = dataset_report(segs)
    print(f"{rep['n_segments']} segments from {rep['n_sources']} sources "
          f"({len(errors)} skipped)\n")
    def _fmt(value):
        # An all-non-finite metric reports None statistics (JSON-safe) — render as a dash.
        return "—" if value is None else f"{value:.3f}"

    for m, d in rep["metrics"].items():
        print(f"  {m:<24} mean={_fmt(d['mean'])}  p50={_fmt(d['p50'])}  p95={_fmt(d['p95'])}  "
              f"[{_fmt(d['min'])}, {_fmt(d['max'])}]")
    if rep["tags"]:
        top = sorted(rep["tags"].items(), key=lambda x: -x[1])[:8]
        print("  tags:", "  ".join(f"{k}={v}" for k, v in top))
    if args.out:
        from ..io import write_text
        write_text(args.out, json_document(rep, label="dataset report"))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
