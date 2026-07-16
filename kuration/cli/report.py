"""kur-report — profile a dataset's metric distributions from a JSON job spec.

    kur-report job.json [--out report.json]

Uses the same job keys as kur-curate (schema, adapter, sources, segmenter, strategy) but runs
no selection: it scores every segment and reports per-metric distributions + group/tag coverage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..report import dataset_report
from ..scoring import score_sources
from ..strategy import load_strategy
from ._common import (build_adapter, build_schema, build_segmenter, cli_progress_for, cli_run,
                      load_job_plugins, read_job, resolve_sources)


def main() -> None:
    ap = argparse.ArgumentParser(description="Distribution/quality report for a dataset.")
    ap.add_argument("job", help="Path to the job JSON.")
    ap.add_argument("--out", default=None, help="Write the full report JSON here.")
    ap.add_argument("--debug", action="store_true", help="Show full tracebacks on failure.")
    ap.add_argument("--quiet", action="store_true", help="Suppress interactive progress output.")
    args = ap.parse_args()
    cli_run(lambda: _run(args), debug=args.debug)


def _run(args) -> None:
    job = read_job(args.job)
    load_job_plugins(job)
    schema = build_schema(job)
    adapter = build_adapter(job, schema)
    schema = schema or adapter.schema
    strategy = load_strategy(job["strategy"], schema)
    segmenter = build_segmenter(job, schema)
    sources = resolve_sources(job, adapter)
    progress = cli_progress_for(quiet=args.quiet)

    # Scorer outputs are produced by the scorer stage, not the feature pass — exclude them
    # here, then apply the configured scorers over the complete pool below.
    registered = [r for r in strategy.referenced_metrics()
                  if r not in strategy.define and r not in strategy.scorer_outputs()]
    segs, errors, _ = score_sources(
        schema=schema, adapter=adapter, sources=sources, define=strategy.define,
        registered=registered, segmenter=segmenter, tags_spec=strategy.tags,
        workers=job.get("workers", 1), on_error=job.get("on_error", "skip"),
        streaming=job.get("streaming", False), progress=progress,
        executor=job.get("executor", "local"))

    if strategy.scorers:
        from ..scorers import apply_scorers
        apply_scorers(strategy.scorers, segs, schema=schema, adapter=adapter,
                      seed=job.get("seed", 0), progress=progress)

    rep = dataset_report(segs)
    print(f"{rep['n_segments']} segments from {rep['n_sources']} sources "
          f"({len(errors)} skipped)\n")
    for m, d in rep["metrics"].items():
        print(f"  {m:<24} mean={d['mean']:.3f}  p50={d['p50']:.3f}  p95={d['p95']:.3f}  "
              f"[{d['min']:.3f}, {d['max']:.3f}]")
    if rep["tags"]:
        top = sorted(rep["tags"].items(), key=lambda x: -x[1])[:8]
        print("  tags:", "  ".join(f"{k}={v}" for k, v in top))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
