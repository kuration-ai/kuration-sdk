"""Evaluate quality gates and/or a registered analyzer against a dataset artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._common import artifact_root_path, cli_run, guard_file_output, json_document


def _read_object(path: str | None, *, label: str) -> dict:
    if path is None:
        return {}
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check a curated dataset with quality expectations and/or an analyzer.")
    parser.add_argument("dataset", help="Dataset directory or dataset_manifest.json path.")
    parser.add_argument("--policy", help="Quality-policy JSON file.")
    parser.add_argument("--analyzer", help="Registered analyzer name.")
    parser.add_argument(
        "--analyzer-config",
        metavar="PATH",
        help="JSON object of analyzer construction parameters.",
    )
    parser.add_argument(
        "--analyzer-options",
        metavar="PATH",
        help="JSON object passed to the analyzer's dataset hook.",
    )
    parser.add_argument("--verify", action="store_true",
                        help="Decode every shard before evaluating the policy.")
    parser.add_argument("--json", action="store_true", help="Emit structured gate results.")
    parser.add_argument("--html", metavar="PATH",
                        help="Write inspection and gate results as static HTML.")
    parser.add_argument("--debug", action="store_true", help="Show tracebacks on failure.")
    args = parser.parse_args()

    def body():
        if not args.policy and not args.analyzer:
            raise ValueError("provide --policy and/or --analyzer")
        if (args.analyzer_config or args.analyzer_options) and not args.analyzer:
            raise ValueError("--analyzer-config/--analyzer-options require --analyzer")
        if args.html and not args.policy:
            raise ValueError("--html currently requires --policy")
        if args.verify and not args.policy:
            # --verify decodes every shard before the *policy* is evaluated; an analyzer
            # reads the manifest only (see analyzers.check_dataset), so a bare --verify in
            # analyzer-only mode would silently do nothing. Say so instead of ignoring it.
            raise ValueError("--verify requires --policy (analyzer-only checks read the "
                              "manifest, not decoded shard contents)")
        if args.html:
            guard_file_output(
                args.html,
                force=None,
                protected_inputs=(
                    artifact_root_path(args.dataset),
                    *(path for path in (
                        args.policy, args.analyzer_config, args.analyzer_options
                    ) if path),
                ),
                label="quality-check HTML output",
            )
        gates = None
        analyzer_result = None
        report = None
        if args.policy:
            policy = _read_object(args.policy, label="quality policy")
            policy = policy.get("expectations", policy)
            if not isinstance(policy, dict):
                raise ValueError("quality policy expectations must be a JSON object")
            from ..inspection import inspect_dataset
            report = inspect_dataset(
                args.dataset, verify=args.verify or bool(policy.get("require_verified")))
            from ..quality import evaluate_quality
            gates = evaluate_quality(report, policy)
        if args.analyzer:
            from ..analyzers import run_dataset_analyzer
            analyzer_result = run_dataset_analyzer(
                args.analyzer,
                args.dataset,
                parameters=_read_object(args.analyzer_config, label="analyzer config"),
                options=_read_object(args.analyzer_options, label="analyzer options"),
            )
        if args.html:
            if report is None or gates is None:  # guarded by the option check above
                raise RuntimeError("quality policy result is unavailable")
            from ..html_report import write_html_report
            write_html_report(args.html, report, gates, title="kuration quality check")
        if args.json:
            rendered: object
            if gates is not None and analyzer_result is not None:
                rendered = {"quality": gates.to_dict(), "analyzer": analyzer_result}
            elif gates is not None:
                rendered = gates.to_dict()
            else:
                rendered = analyzer_result
            print(json_document(rendered, label="dataset check result").rstrip())
        else:
            if gates is not None:
                print(gates.render())
            if analyzer_result is not None:
                status = "PASS" if analyzer_result["passed"] else "FAIL"
                if gates is not None:
                    print()
                print(f"analyzer {args.analyzer}: {status}")
                print(json_document(
                    analyzer_result["summary"], label="analyzer summary"
                ).rstrip())
        if (gates is not None and not gates.passed) or (
            analyzer_result is not None and not analyzer_result["passed"]
        ):
            raise SystemExit(2)

    cli_run(body, debug=args.debug)


if __name__ == "__main__":
    main()
