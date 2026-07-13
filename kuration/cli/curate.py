"""kur-curate — run a curation job (sources → hours-budget shards) from a JSON spec.

    kur-curate job.json

job keys: schema, adapter{type,...}, sources?(else adapter.sources()), segmenter{window,stride,unit?},
strategy(inline/path/name), output(OutputSpec), train_hours, test_hours?, out, seed?.
"""
from __future__ import annotations

import argparse

from ..output import OutputSpec, write_dataset
from ..strategy import load_strategy
from ._common import (build_adapter, build_schema, build_segmenter, cli_progress_for, cli_run,
                      load_job_plugins, read_job, resolve_sources)


def main() -> None:
    ap = argparse.ArgumentParser(description="Curate a dataset from a JSON job spec.")
    ap.add_argument("job", help="Path to the job JSON.")
    ap.add_argument("--debug", action="store_true", help="Show full tracebacks on failure.")
    ap.add_argument("--quiet", action="store_true", help="Suppress interactive progress output.")
    args = ap.parse_args()
    cli_run(lambda: _run(args), debug=args.debug)


def _run(args) -> None:
    job = read_job(args.job)
    load_job_plugins(job)
    schema = build_schema(job)
    adapter = build_adapter(job, schema)
    schema = schema or adapter.schema           # curated adapter derives its own
    strategy = load_strategy(job["strategy"], schema)
    segmenter = build_segmenter(job, schema)
    sources = resolve_sources(job, adapter)
    spec = OutputSpec.from_dict(job["output"])

    res = write_dataset(job["out"], schema=schema, adapter=adapter, sources=sources,
                        strategy=strategy, segmenter=segmenter, spec=spec,
                        train_hours=job["train_hours"], test_hours=job.get("test_hours", 0.0),
                        seed=job.get("seed", 0), workers=job.get("workers", 1),
                        on_error=job.get("on_error", "skip"), resume=job.get("resume", False),
                        streaming=job.get("streaming", False),
                        progress=cli_progress_for(quiet=args.quiet))
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
