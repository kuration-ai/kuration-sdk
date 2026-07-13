from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(description="Migrate a kuration job or config JSON.")
    ap.add_argument("path")
    ap.add_argument("--out")
    args = ap.parse_args()
    from ..versioning import migrate
    result = migrate(args.path, args.out)
    if not args.out:
        print(json.dumps(result, indent=2))
