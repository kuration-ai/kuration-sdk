from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._common import (cli_run, guard_file_output, json_document, parse_json)


def main() -> None:
    ap = argparse.ArgumentParser(description="Migrate a kuration job or config JSON.")
    ap.add_argument("path", help="Job or artifact config JSON to migrate.")
    ap.add_argument("--out", help="Write migrated JSON here (refuses an existing file).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing --out file; permits explicit in-place migration.")
    ap.add_argument("--debug", action="store_true", help="Show full tracebacks on failure.")
    args = ap.parse_args()
    cli_run(lambda: _run(args), debug=args.debug)


def _run(args) -> None:
    from ..versioning import document_migration_notes, migrate
    if args.force and not args.out:
        raise ValueError("--force requires --out")
    if args.out:
        guard_file_output(args.out, force=args.force, label="migration output")
    source = parse_json(Path(args.path).read_text(), label=f"migration input {args.path}")
    # Report semantic changes before the structural result: the migrated document is valid here,
    # but on some data it selects differently than it did under the version that wrote it.
    for note in document_migration_notes(source):
        print(f"note: {note}", file=sys.stderr)
    result = migrate(source, args.out)
    if not args.out:
        print(json_document(result, label="migrated document").rstrip())


if __name__ == "__main__":
    main()
