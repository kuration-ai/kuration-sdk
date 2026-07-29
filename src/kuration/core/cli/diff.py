"""Compare two curated dataset artifacts."""
from __future__ import annotations

import argparse

from ._common import artifact_root_path, cli_run, guard_file_output, json_document


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two kuration dataset artifacts and their provenance.")
    parser.add_argument("left", help="First dataset directory or manifest.")
    parser.add_argument("right", help="Second dataset directory or manifest.")
    parser.add_argument("--verify", action="store_true",
                        help="Decode every shard in both datasets before comparing.")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON.")
    parser.add_argument("--html", metavar="PATH",
                        help="Write a self-contained HTML comparison report.")
    parser.add_argument("--debug", action="store_true", help="Show tracebacks on failure.")
    args = parser.parse_args()

    def body():
        if args.html:
            guard_file_output(
                args.html,
                force=None,
                protected_inputs=(
                    artifact_root_path(args.left),
                    artifact_root_path(args.right),
                ),
                label="comparison HTML output",
            )
        from ..inspection import compare_datasets
        comparison = compare_datasets(args.left, args.right, verify=args.verify)
        if args.html:
            from ..html_report import write_html_report
            write_html_report(args.html, comparison, title="kuration dataset comparison")
        print(json_document(
            comparison.to_dict(), label="dataset comparison"
        ).rstrip() if args.json
              else comparison.render())

    cli_run(body, debug=args.debug)


if __name__ == "__main__":
    main()
