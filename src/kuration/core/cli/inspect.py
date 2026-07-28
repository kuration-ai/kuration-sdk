"""Inspect a written dataset artifact without the original job or source corpus."""
from __future__ import annotations

import argparse

from ._common import artifact_root_path, cli_run, guard_file_output, json_document


def render_inspection(report: dict) -> str:
    lines = [f"dataset: {report['dataset']}",
             f"container: {report['container']}",
             f"shards: {report['shards']}"]
    for split, stats in report["splits"].items():
        duration = f", {stats['seconds']:.3f}s" if stats["seconds"] is not None else ""
        lines.append(f"split {split}: {stats['shards']} shards, {stats['steps']} steps"
                     f"{duration}, padding={stats['padding_steps']} steps")
    modalities = ", ".join(
        f"{name}{tuple(info.get('shape', ()))}/{info.get('dtype', 'variants')}"
        for name, info in report["modalities"].items())
    lines.append(f"modalities: {modalities or 'none'}")
    lines.append(f"skipped sources: {report['sources_skipped']} "
                 f"({len(report['errors'])} recorded errors)")
    lines.append(f"provenance: {report['provenance_fingerprint'] or 'unavailable'}")
    execution = report["execution"]
    lines.append(f"execution: {'bounded' if execution['bounded'] else 'reference'}; "
                 f"backend={execution['backend'] or 'unrecorded'}")
    verification = report["verification"]
    if verification["requested"]:
        lines.append(f"verification: {'PASS' if verification['valid'] else 'FAIL'}; "
                     f"{verification['checked']} shards decoded")
        for error in verification["errors"][:5]:
            lines.append(f"  error: {error['path']}: {error['error']}")
    elif not verification["valid"]:
        lines.append(f"verification: {len(verification['errors'])} missing shard(s); "
                     "run with --verify for payload checks")
    else:
        lines.append("verification: metadata only; run with --verify to decode every shard")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a kuration dataset manifest, configuration, and shards.")
    parser.add_argument("dataset", help="Dataset directory or dataset_manifest.json path.")
    parser.add_argument("--verify", action="store_true",
                        help="Decode every shard and report missing/corrupt payloads.")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON.")
    parser.add_argument("--html", metavar="PATH",
                        help="Write a self-contained HTML inspection report.")
    parser.add_argument("--debug", action="store_true", help="Show full tracebacks on failure.")
    args = parser.parse_args()

    def body():
        if args.html:
            guard_file_output(
                args.html,
                force=None,
                protected_inputs=(artifact_root_path(args.dataset),),
                label="inspection HTML output",
            )
        from ..inspection import inspect_dataset
        report = inspect_dataset(args.dataset, verify=args.verify)
        if args.html:
            from ..html_report import write_html_report
            write_html_report(args.html, report, title="kuration dataset inspection")
        print(json_document(
            report.to_dict(), label="dataset inspection"
        ).rstrip() if args.json
              else render_inspection(report))
        if args.verify and not report["verification"]["valid"]:
            raise SystemExit(2)

    cli_run(body, debug=args.debug)


if __name__ == "__main__":
    main()
