"""kur-stats — unified lifecycle statistics for a written dataset artifact.

    kur-stats curated/ --html stats.html

`stats()` accepts a pipeline, a selection result, or an artifact; a shell can only name the
artifact, so that is what this exposes. It is the distribution-level counterpart to
`kur-inspect`: inspect answers "is this artifact intact", stats answers "what is in it".
Domain packages that register a stats enricher contribute sections here for free.
"""
from __future__ import annotations

import argparse
import json

from ._common import cli_progress_for, cli_run


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Report distribution statistics for a written dataset artifact.")
    ap.add_argument("dataset", help="Dataset directory or dataset_manifest.json path.")
    ap.add_argument("--verify", action="store_true",
                    help="Verify shard integrity while scanning.")
    ap.add_argument("--no-content", action="store_true",
                    help="Summarize the manifest only; do not decode shard contents.")
    ap.add_argument("--hist-bins", type=int, default=10,
                    help="Histogram resolution (default: 10).")
    ap.add_argument("--max-values", type=int, default=100_000,
                    help="Values sampled per modality (default: 100000).")
    ap.add_argument("--examples", type=int, default=5,
                    help="Representative examples to retain (default: 5).")
    ap.add_argument("--capabilities", action="store_true",
                    help="Print the plots and evidence this report supports, then exit.")
    ap.add_argument("--json", action="store_true", help="Emit the structured report as JSON.")
    ap.add_argument("--html", metavar="PATH", help="Write a self-contained HTML report.")
    ap.add_argument("--quiet", action="store_true", help="Suppress interactive progress.")
    ap.add_argument("--debug", action="store_true", help="Show full tracebacks on failure.")
    args = ap.parse_args()

    def body():
        from ..stats import stats
        report = stats(args.dataset, hist_bins=args.hist_bins, max_values=args.max_values,
                       examples=args.examples, content=not args.no_content,
                       verify=args.verify, progress=cli_progress_for(quiet=args.quiet))
        if args.capabilities:
            print(json.dumps(report.available(), indent=2))
            return
        if args.html:
            from ..stats import write_stats_html
            write_stats_html(args.html, report, title="kuration stats")
        print(json.dumps(report.to_dict(), indent=2) if args.json else report.render())

    cli_run(body, debug=args.debug)


if __name__ == "__main__":
    main()
