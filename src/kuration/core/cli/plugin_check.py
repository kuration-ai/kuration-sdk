"""Validate installed Kuration plugin discovery and descriptor contracts."""
from __future__ import annotations

import argparse

from ._common import cli_run, json_document


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate installed Kuration plugin entry points and descriptors.")
    parser.add_argument(
        "target", nargs="?",
        help="Installed entry-point name or explicit registration module; omit for all plugins.")
    parser.add_argument("--json", action="store_true", help="Emit structured results.")
    parser.add_argument("--debug", action="store_true", help="Show tracebacks on failure.")
    args = parser.parse_args()

    def body() -> None:
        from ..plugin_validation import check_plugins

        results = check_plugins(args.target)
        if args.json:
            print(json_document(
                [result.to_dict() for result in results],
                label="plugin validation results",
            ).rstrip())
        elif results:
            print("\n\n".join(result.render() for result in results))
        else:
            print("PASS no installed kuration.plugins entry points")
        if any(not result.passed for result in results):
            raise SystemExit(2)

    cli_run(body, debug=args.debug)


if __name__ == "__main__":
    main()
