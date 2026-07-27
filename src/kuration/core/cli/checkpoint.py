"""kur-checkpoint — inspect, verify, and clear stage checkpoint directories.

    kur-checkpoint list runs/ckpt
    kur-checkpoint verify runs/ckpt
    kur-checkpoint clear runs/ckpt --stage feature-pass --yes

`checkpoint_dir` is an ordinary job key and `StageCheckpoint` commits are checksummed, but
nothing showed what a checkpoint tree already holds or removed a stale one — leaving
hand-deletion as the only repair. Verification here reads the committed metadata rather than
reconstructing the run's `CheckpointIdentity`, so it works from the artifact alone: a commit
whose data file is missing, resized, or re-hashed is reported as invalid.
"""
from __future__ import annotations

import argparse
import json

from ._common import cli_run


def _stage_dirs(root: str) -> list[str]:
    """Stage names under a checkpoint root, identified by their commits/ namespace."""
    from ..io import glob
    root = str(root).rstrip("/")
    names = set()
    for path in glob(f"{root}/*/commits"):
        parts = str(path).rstrip("/").split("/")
        if len(parts) >= 2:
            names.add(parts[-2])
    return sorted(names)


def scan_checkpoints(root: str, *, stage: str | None = None, verify: bool = False) -> dict:
    """Summarize each stage's commits, optionally re-checksumming every data file."""
    from ..io import exists, file_integrity, glob, read_text
    root = str(root).rstrip("/")
    if not exists(root):
        raise FileNotFoundError(f"checkpoint root not found: {root}")
    stages = _stage_dirs(root)
    if stage is not None:
        if stage not in stages:
            raise ValueError(f"stage {stage!r} not found under {root}; "
                             f"present: {stages or 'none'}")
        stages = [stage]

    reports = []
    for name in stages:
        stage_root = f"{root}/{name}"
        partitions, invalid = [], 0
        total_rows = total_bytes = 0
        for commit_path in glob(f"{stage_root}/commits/*.json"):
            record: dict = {"commit": commit_path}
            try:
                commit = json.loads(read_text(commit_path))
            except (OSError, ValueError) as error:
                record.update(partition_id=None, valid=False,
                              error=f"unreadable commit: {error}")
                partitions.append(record)
                invalid += 1
                continue
            data_file = commit.get("data_file")
            record.update(partition_id=commit.get("partition_id"),
                          identity=commit.get("identity"),
                          rows=commit.get("rows"), size_bytes=commit.get("size_bytes", 0),
                          data_file=data_file)
            total_rows += int(commit.get("rows") or 0)
            total_bytes += int(commit.get("size_bytes") or 0)
            if not data_file or not exists(data_file):
                record.update(valid=False, error="data file missing")
                invalid += 1
            elif verify:
                integrity = file_integrity(data_file)
                if integrity["sha256"] != commit.get("sha256"):
                    record.update(valid=False, error="sha256 mismatch")
                    invalid += 1
                elif (commit.get("size_bytes")
                      and integrity["size_bytes"] != commit["size_bytes"]):
                    record.update(valid=False, error="size mismatch")
                    invalid += 1
                else:
                    record["valid"] = True
            else:
                record["valid"] = None          # present, not checksummed
            partitions.append(record)

        attempts = glob(f"{stage_root}/attempts/*.arrow")
        marker = f"{stage_root}/stage.commit.json"
        identities = {str(p["identity"]) for p in partitions if p.get("identity")}
        reports.append({
            "stage": name, "partitions": len(partitions), "invalid": invalid,
            "rows": total_rows, "size_bytes": total_bytes,
            "finalized": exists(marker), "stale_attempts": len(attempts),
            # More than one identity means the tree mixes runs; only the current run's
            # commits would be reusable, so it is worth surfacing.
            "identities": sorted(identities), "details": partitions})
    return {"root": root, "verified": verify, "stages": reports,
            "invalid": sum(report["invalid"] for report in reports)}


def render_scan(scan: dict) -> str:
    lines = [f"checkpoint root: {scan['root']}"]
    if not scan["stages"]:
        lines.append("stages: none (no commits/ namespace found)")
        return "\n".join(lines)
    for report in scan["stages"]:
        state = "finalized" if report["finalized"] else "in progress"
        lines.append(
            f"stage {report['stage']}: {report['partitions']} partition(s), "
            f"{report['rows']} rows, {report['size_bytes']} bytes, {state}")
        if report["stale_attempts"]:
            lines.append(f"  stale attempts: {report['stale_attempts']} "
                         "(clear with --attempts-only)")
        if len(report["identities"]) > 1:
            lines.append(f"  warning: {len(report['identities'])} distinct identities; only "
                         "the current run's commits are reusable")
        for partition in report["details"]:
            if partition.get("valid") is False:
                lines.append(f"  invalid {partition.get('partition_id')}: "
                             f"{partition.get('error')}")
    verified = "checksums verified" if scan["verified"] else "metadata only (use verify)"
    lines.append(f"invalid partitions: {scan['invalid']} ({verified})")
    return "\n".join(lines)


def _remove(path: str) -> None:
    from ..io import is_remote
    if is_remote(path):
        from ..io import _fs
        _fs(path).rm(path, recursive=True)
        return
    import shutil
    from pathlib import Path

    from ..io import _local
    local = Path(_local(path))
    if local.is_dir():
        shutil.rmtree(local)
    else:
        local.unlink(missing_ok=True)


def clear_checkpoints(root: str, *, stage: str | None = None,
                      attempts_only: bool = False) -> dict:
    """Delete stage checkpoint state and return what was removed."""
    from ..io import exists, glob
    root = str(root).rstrip("/")
    if not exists(root):
        raise FileNotFoundError(f"checkpoint root not found: {root}")
    stages = _stage_dirs(root)
    if stage is not None:
        if stage not in stages:
            raise ValueError(f"stage {stage!r} not found under {root}; "
                             f"present: {stages or 'none'}")
        stages = [stage]
    removed = []
    for name in stages:
        stage_root = f"{root}/{name}"
        if attempts_only:
            for path in glob(f"{stage_root}/attempts/*"):
                _remove(path)
                removed.append(path)
        else:
            _remove(stage_root)
            removed.append(stage_root)
    return {"root": root, "stages": stages, "attempts_only": attempts_only,
            "removed": removed}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect, verify, and clear kuration stage checkpoints.")
    sub = ap.add_subparsers(dest="command", required=True)
    for name, description in (("list", "Summarize the stages a checkpoint root holds."),
                              ("verify", "Re-checksum every committed partition."),
                              ("clear", "Delete checkpoint state.")):
        parser = sub.add_parser(name, description=description, help=description)
        parser.add_argument("root", help="Checkpoint directory (the job's checkpoint_dir).")
        parser.add_argument("--stage", help="Restrict to one stage id.")
        parser.add_argument("--json", action="store_true", help="Emit structured JSON.")
        parser.add_argument("--debug", action="store_true",
                            help="Show full tracebacks on failure.")
        if name == "clear":
            parser.add_argument("--attempts-only", action="store_true",
                                help="Remove only stale attempt files, keeping commits.")
            parser.add_argument("--yes", action="store_true",
                                help="Confirm deletion; required.")
    args = ap.parse_args()

    def body():
        if args.command == "clear":
            if not args.yes:
                target = "stale attempts" if args.attempts_only else "committed checkpoints"
                raise SystemExit(
                    f"refusing to delete {target} under {args.root} without --yes "
                    f"(run 'kur-checkpoint list {args.root}' first)")
            result = clear_checkpoints(args.root, stage=args.stage,
                                       attempts_only=args.attempts_only)
            print(json.dumps(result, indent=2) if args.json else
                  f"removed {len(result['removed'])} path(s) from "
                  f"{len(result['stages'])} stage(s) under {result['root']}")
            return
        scan = scan_checkpoints(args.root, stage=args.stage,
                                verify=args.command == "verify")
        print(json.dumps(scan, indent=2) if args.json else render_scan(scan))
        if args.command == "verify" and scan["invalid"]:
            raise SystemExit(2)

    cli_run(body, debug=args.debug)


if __name__ == "__main__":
    main()
