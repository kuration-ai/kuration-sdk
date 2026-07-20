"""kur-curate — run a curation job (sources → hours-budget shards) from a JSON spec.

    kur-curate job.json

job keys: schema, adapter{type,...}, sources?(else adapter.sources()), segmenter{window,stride,unit?},
strategy(inline/path/name), output(OutputSpec), train_hours, test_hours?, out, seed?.
"""
from __future__ import annotations

import argparse

from ..output import write_dataset
from ._common import cli_progress_for, cli_run, read_job
from .job import resolve_job


def main() -> None:
    ap = argparse.ArgumentParser(description="Curate a dataset from a JSON job spec.")
    ap.add_argument("job", help="Path to the job JSON.")
    ap.add_argument("--debug", action="store_true", help="Show full tracebacks on failure.")
    ap.add_argument("--quiet", action="store_true", help="Suppress interactive progress output.")
    args = ap.parse_args()
    cli_run(lambda: _run(args), debug=args.debug)


def _run(args) -> None:
    job = read_job(args.job)
    resolved = resolve_job(job)
    if resolved.kind != "curate":
        raise ValueError(f"kur-curate requires a curate job, got {resolved.kind!r}")
    if resolved.sources is None or resolved.strategy is None:
        raise RuntimeError("resolved curate job is missing its sources or strategy")

    res = write_dataset(job["out"], schema=resolved.schema, adapter=resolved.adapter,
                        sources=resolved.sources, strategy=resolved.strategy,
                        segmenter=resolved.segmenter, spec=resolved.output,
                        encoder=resolved.encoder,
                        train_hours=job["train_hours"], test_hours=job.get("test_hours", 0.0),
                        seed=job.get("seed", 0), workers=job.get("workers", 1),
                        on_error=resolved.on_error, resume=job.get("resume", False),
                        streaming=job.get("streaming", False),
                        span_steps=job.get("span_steps"), span_seconds=job.get("span_seconds"),
                        clip_aggregate=job.get("clip_aggregate", "max"),
                        progress=cli_progress_for(quiet=args.quiet),
                        executor=job.get("executor", "local"),
                        memory_budget=job.get("memory_budget"),
                        batch_size=job.get("batch_size", 1024),
                        checkpoint_dir=job.get("checkpoint_dir"),
                        spill_dir=job.get("spill_dir"),
                        approximation=job.get("approximation", "exact"),
                        max_in_flight=job.get("max_in_flight", 2), registry=resolved.registry)
    r = res["results"]
    b = r["budget"]
    emitted = b["emitted_seconds"]
    requested = b["requested_seconds"]
    print(f"wrote {r['shards']} shards from {r['tracks_admitted']}/{r['tracks_total']} tracks "
          f"({r['n_train']} train / {r['n_test']} test sources, "
          f"{r['sources_skipped']} skipped; emitted "
          f"{emitted['train'] / 3600:.3f}/{requested['train'] / 3600:.3f} train h, "
          f"{emitted['test'] / 3600:.3f}/{requested['test'] / 3600:.3f} test h) → {job['out']}")


if __name__ == "__main__":
    main()
