"""Component discovery CLIs."""
from __future__ import annotations

import argparse
import json

_PLURAL = {"adapters": "adapter", "metrics": "metric", "scorers": "scorer",
           "methods": "method", "filters": "filter", "encoders": "encoder",
           "containers": "container", "presets": "preset"}


def list_main() -> None:
    ap = argparse.ArgumentParser(description="List installed kuration components.")
    ap.add_argument("kind", choices=sorted(_PLURAL))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    from ..registry import list_components
    rows = list_components(_PLURAL[args.kind])
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(f"{row['name']:<28} {row.get('description', '')}")


def describe_main() -> None:
    ap = argparse.ArgumentParser(description="Describe an installed kuration component.")
    ap.add_argument("kind", choices=sorted(set(_PLURAL) | set(_PLURAL.values())))
    ap.add_argument("name")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    from ..registry import describe_component
    row = describe_component(_PLURAL.get(args.kind, args.kind), args.name)
    if args.json:
        print(json.dumps(row, indent=2))
    else:
        print(f"{row['kind']} {row['name']}: {row.get('description', '')}")
        print(json.dumps(row.get("parameters", {}), indent=2))
