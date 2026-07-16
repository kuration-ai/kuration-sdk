"""Compare two curated dataset artifacts."""
from __future__ import annotations

import argparse
import json

from ._common import cli_run


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
        from ..inspection import compare_datasets
        comparison = compare_datasets(args.left, args.right, verify=args.verify)
        if args.html:
            from ..html_report import write_html_report
            write_html_report(args.html, comparison, title="kuration dataset comparison")
        print(json.dumps(comparison.to_dict(), indent=2) if args.json
              else comparison.render())

    cli_run(body, debug=args.debug)


if __name__ == "__main__":
    main()
