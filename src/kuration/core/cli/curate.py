"""kur-curate — run a curation job (sources → hours-budget shards) from a JSON spec.

    kur-curate job.json

job keys: schema, adapter{type,...}, sources?(else adapter.sources()), segmenter{window,stride,unit?},
strategy(inline/path/name), output(OutputSpec), train_hours, test_hours?, out, seed?.
"""
from __future__ import annotations

import argparse

from ..output import write_dataset
from ._common import (cli_progress_for, cli_run, guard_file_output, json_document,
                      read_job)
from .job import resolve_job


def main() -> None:
    ap = argparse.ArgumentParser(description="Curate a dataset from a JSON job spec.")
    ap.add_argument("job", help="Path to the job JSON.")
    ap.add_argument("--debug", action="store_true", help="Show full tracebacks on failure.")
    ap.add_argument("--quiet", action="store_true", help="Suppress interactive progress output.")
    ap.add_argument("--force", action="store_true",
                    help="Replace an existing committed or interrupted output artifact.")
    ap.add_argument("--review", action="store_true",
                    help="Explain the hours-budget admission after writing (no rescoring).")
    ap.add_argument("--review-html", metavar="PATH",
                    help="Write the admission review as a self-contained HTML report.")
    ap.add_argument("--review-json", metavar="PATH",
                    help="Write the admission review as JSON.")
    args = ap.parse_args()
    cli_run(lambda: _run(args), debug=args.debug)


def _guard_review_outputs(args, *, protected_inputs=()) -> None:
    """Protect every optional review file before the dataset writer can mutate anything."""
    outputs = [
        (args.review_html, "review HTML output"),
        (args.review_json, "review JSON output"),
    ]
    for index, (path, label) in enumerate(outputs):
        if path:
            guard_file_output(
                path,
                force=None,
                protected_inputs=(
                    *protected_inputs,
                    *(other for other_index, (other, _other_label) in enumerate(outputs)
                      if other and other_index != index),
                ),
                label=label,
            )


def _run(args) -> None:
    _guard_review_outputs(args, protected_inputs=(args.job,))
    job = read_job(args.job)
    job_sources = job.get("sources")
    job_sources = job_sources if isinstance(job_sources, (list, tuple)) else ()
    raw_sources = tuple(
        source for source in job_sources if isinstance(source, str)
    )
    referenced = tuple(
        value for value in (job.get("schema"), job.get("strategy"))
        if isinstance(value, str)
    )
    mutable_trees = tuple(
        value for value in (
            job.get("out"), job.get("checkpoint_dir"), job.get("spill_dir")
        )
        if isinstance(value, str)
    )
    _guard_review_outputs(
        args,
        protected_inputs=(args.job, *raw_sources, *referenced, *mutable_trees),
    )
    if args.force:
        job = {**job, "force": True}
    resolved = resolve_job(job)
    if resolved.kind != "curate":
        raise ValueError(f"kur-curate requires a curate job, got {resolved.kind!r}")
    if resolved.sources is None or resolved.strategy is None:
        raise RuntimeError("resolved curate job is missing its sources or strategy")
    _guard_review_outputs(
        args,
        protected_inputs=(
            args.job, *resolved.sources, *referenced, *mutable_trees,
        ),
    )

    res = write_dataset(job["out"], schema=resolved.schema, adapter=resolved.adapter,
                        sources=resolved.sources, strategy=resolved.strategy,
                        segmenter=resolved.segmenter, spec=resolved.output,
                        encoder=resolved.encoder,
                        train_hours=job["train_hours"], test_hours=job.get("test_hours", 0.0),
                        seed=job.get("seed", 0), workers=job.get("workers", 1),
                        on_error=resolved.on_error, resume=job.get("resume", False),
                        force=job.get("force", False),
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
    # `tracks_admitted` counts admitted (source, entity, span) items, not whole tracks: with
    # span_steps/span_seconds (kur-init's default) one track can contribute many items, so
    # "admitted/total tracks" is only truthful when every item is a whole, unspanned track.
    n_items = r["tracks_admitted"]
    n_admitted_tracks = len({(src, eid) for (src, eid, _k) in res["admitted"]})
    tracks_desc = (f"{n_items}/{r['tracks_total']} tracks" if n_items == n_admitted_tracks
                   else f"{n_items} span items from {n_admitted_tracks}/{r['tracks_total']} tracks")
    print(f"wrote {r['shards']} shards from {tracks_desc} "
          f"({r['n_train_sources']} train / {r['n_test_sources']} test sources, "
          f"{r['sources_skipped']} skipped; emitted "
          f"{emitted['train'] / 3600:.3f}/{requested['train'] / 3600:.3f} train h, "
          f"{emitted['test'] / 3600:.3f}/{requested['test'] / 3600:.3f} test h) → {job['out']}")

    if args.review or args.review_html or args.review_json:
        # The hours path already recorded its admission decisions, so this reuses them
        # rather than scoring the pool a second time.
        from ..review import hours_selection_review
        review = hours_selection_review(res)
        if args.review_html:
            from ..html_report import write_html_report
            write_html_report(args.review_html, review,
                              title="kuration hours selection review")
        if args.review_json:
            from ..io import write_text
            write_text(
                args.review_json,
                json_document(review.to_dict(), label="hours selection review"),
            )
        if args.review:
            print(review.render())


if __name__ == "__main__":
    main()
