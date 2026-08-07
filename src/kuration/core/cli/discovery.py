"""Component discovery CLIs."""
from __future__ import annotations

import argparse
import sys

from ._common import cli_run, json_document

_PLURAL = {"adapters": "adapter", "metrics": "metric", "scorers": "scorer",
           "methods": "method", "filters": "filter", "encoders": "encoder",
           "containers": "container", "presets": "preset", "operators": "operator",
           "transforms": "transform", "analyzers": "analyzer"}


def _report_plugin_failures() -> None:
    from ..plugins import plugin_failures

    for failure in plugin_failures():
        print(f"ERROR plugin {failure['name']} ({failure['value']}): {failure['error']}",
              file=sys.stderr)


def list_main() -> None:
    ap = argparse.ArgumentParser(description="List installed kuration components.")
    ap.add_argument("kind", choices=sorted(_PLURAL))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--debug", action="store_true", help="Show full tracebacks on failure.")
    args = ap.parse_args()
    cli_run(lambda: _list(args), debug=args.debug)


def _list(args) -> None:
    from ..registry import list_components
    rows = list_components(_PLURAL[args.kind])
    _report_plugin_failures()
    if args.json:
        print(json_document(rows, label=f"{args.kind} component listing").rstrip())
    else:
        for row in rows:
            print(f"{row['name']:<28} {row.get('description', '')}")


def describe_main() -> None:
    ap = argparse.ArgumentParser(description="Describe an installed kuration component.")
    ap.add_argument("kind", choices=sorted(set(_PLURAL) | set(_PLURAL.values())))
    ap.add_argument("name")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--debug", action="store_true", help="Show full tracebacks on failure.")
    args = ap.parse_args()
    cli_run(lambda: _describe(args), debug=args.debug)


def _describe(args) -> None:
    from ..plugins import load_plugins
    from ..registry import describe_component
    load_plugins()
    _report_plugin_failures()
    row = describe_component(_PLURAL.get(args.kind, args.kind), args.name)
    if args.json:
        print(json_document(row, label="component description").rstrip())
    else:
        print(f"{row['kind']} {row['name']}: {row.get('description', '')}")
        print(json_document(
            row.get("parameters", {}), label="component parameters"
        ).rstrip())
