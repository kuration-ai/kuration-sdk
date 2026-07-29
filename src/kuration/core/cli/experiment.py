"""kur-experiment — compare several strategies over the same sources, from a JSON spec.

    kur-experiment job.json [--out comparison.json]

Adds `arms: [{name, strategy}]` and `baseline` to the shared job keys; `target_size` instead
of an hour budget. Prints per-arm JS-vs-baseline on each metric; optionally writes the full
comparison JSON.
"""
from __future__ import annotations

import argparse

from ..experiment import Arm, run_experiment
from ._common import (cli_progress_for, cli_run, guard_file_output, json_document,
                      read_job)
from .job import resolve_job


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-arm curation comparison from a JSON job spec.")
    ap.add_argument("job", help="Path to the job JSON.")
    ap.add_argument("--out", default=None,
                    help="Write the full comparison JSON here (refuses an existing file).")
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
                          label="experiment output")
    job = read_job(args.job)
    raw_arms = job.get("arms")
    raw_arms = raw_arms if isinstance(raw_arms, (list, tuple)) else ()
    referenced = [
        arm.get("strategy")
        for arm in raw_arms
        if isinstance(arm, dict) and isinstance(arm.get("strategy"), str)
    ]
    if isinstance(job.get("schema"), str):
        referenced.append(job["schema"])
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
            label="experiment output",
        )
    resolved = resolve_job(job)
    if resolved.kind != "experiment":
        raise ValueError(f"kur-experiment requires an experiment job, got {resolved.kind!r}")
    if resolved.sources is None:
        raise RuntimeError("resolved experiment job is missing its sources")
    if args.out:
        guard_file_output(
            args.out,
            force=args.force,
            protected_inputs=(args.job, *resolved.sources, *referenced),
            label="experiment output",
        )
    arms = [Arm(name, strategy) for name, strategy in resolved.arms]

    out = run_experiment(schema=resolved.schema, adapter=resolved.adapter,
                         sources=resolved.sources, arms=arms,
                         segmenter=resolved.segmenter, target_size=job["target_size"],
                         baseline=job.get("baseline"), seed=job.get("seed", 0),
                         workers=job.get("workers", 1), on_error=resolved.on_error,
                         streaming=job.get("streaming", False),
                         progress=cli_progress_for(quiet=args.quiet),
                         executor=job.get("executor", "local"))

    base = out["baseline"]
    print(f"baseline={base}  ({out['n_total_segments']} segments)\n")
    for name, a in out["arms"].items():
        if name == base:
            continue
        top = sorted(((m, d["js_vs_baseline"]) for m, d in a["metrics"].items()
                      if "js_vs_baseline" in d), key=lambda x: -x[1])[:3]
        shown = "  ".join(f"{m}={js:.3f}" for m, js in top)
        print(f"  {name:<16} n={a['n_selected']:5d}  top JS-vs-{base}: {shown}")
    if args.out:
        from ..io import write_text
        write_text(args.out, json_document(out, label="experiment comparison"))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
