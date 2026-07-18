from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(description="Migrate a kuration job or config JSON.")
    ap.add_argument("path")
    ap.add_argument("--out")
    ap.add_argument("--debug", action="store_true", help="Show full tracebacks on failure.")
    args = ap.parse_args()
    from ._common import cli_run
    cli_run(lambda: _run(args), debug=args.debug)


def _run(args) -> None:
    from ..versioning import migrate
    result = migrate(args.path, args.out)
    if not args.out:
        print(json.dumps(result, indent=2))
