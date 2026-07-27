from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Migrate a kuration job or config JSON.")
    ap.add_argument("path")
    ap.add_argument("--out")
    ap.add_argument("--debug", action="store_true", help="Show full tracebacks on failure.")
    args = ap.parse_args()
    from ._common import cli_run
    cli_run(lambda: _run(args), debug=args.debug)


def _run(args) -> None:
    from ..versioning import document_migration_notes, migrate
    source = json.loads(Path(args.path).read_text())
    # Report semantic changes before the structural result: the migrated document is valid here,
    # but on some data it selects differently than it did under the version that wrote it.
    for note in document_migration_notes(source):
        print(f"note: {note}", file=sys.stderr)
    result = migrate(source, args.out)
    if not args.out:
        print(json.dumps(result, indent=2))
