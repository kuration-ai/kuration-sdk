"""Evaluate a JSON quality policy against a written dataset artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._common import cli_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check a curated dataset against explicit quality expectations.")
    parser.add_argument("dataset", help="Dataset directory or dataset_manifest.json path.")
    parser.add_argument("--policy", required=True, help="Quality-policy JSON file.")
    parser.add_argument("--verify", action="store_true",
                        help="Decode every shard before evaluating the policy.")
    parser.add_argument("--json", action="store_true", help="Emit structured gate results.")
    parser.add_argument("--html", metavar="PATH",
                        help="Write inspection and gate results as static HTML.")
    parser.add_argument("--debug", action="store_true", help="Show tracebacks on failure.")
    args = parser.parse_args()

    def body():
        policy = json.loads(Path(args.policy).read_text())
        if not isinstance(policy, dict):
            raise ValueError("quality policy must be a JSON object")
        policy = policy.get("expectations", policy)
        from ..inspection import inspect_dataset
        report = inspect_dataset(
            args.dataset, verify=args.verify or bool(policy.get("require_verified")))
        from ..quality import evaluate_quality
        gates = evaluate_quality(report, policy)
        if args.html:
            from ..html_report import write_html_report
            write_html_report(args.html, report, gates, title="kuration quality check")
        print(json.dumps(gates.to_dict(), indent=2) if args.json else gates.render())
        if not gates.passed:
            raise SystemExit(2)

    cli_run(body, debug=args.debug)


if __name__ == "__main__":
    main()
