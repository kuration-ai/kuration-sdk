"""kur-export — turn a written dataset back into ordinary array files.

    kur-export curated/ --out arrays/ --format npz --split train

The read side (`iter_shards`) is public but Python-only, so handing curated shards to a
non-Python trainer, or transcoding a container, meant writing a script or re-running
curation. This walks the manifest once and writes one file per shard (or per shard and
modality for `npy`), plus an `export_index.json` carrying the manifest metadata that would
otherwise be lost.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import uuid

from ._common import cli_run, json_document

_FORMATS = ("npz", "npy", "parquet")
_GENERATIONS_DIR = ".kuration-export-generations"
_WRITER_LEASE = ".kuration-writer.lock"


def _portable_token(value: object, *, fallback: str) -> str:
    """Return a portable filename token without merging distinct original values."""
    raw = str(value)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip(". _")[:96] or fallback
    safe = safe.rstrip(". ")
    if safe == raw and safe not in {".", ".."}:
        return safe
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"{safe}-{digest}"


def _stem(entry: dict) -> str:
    """Derive a portable, split-aware export name from the artifact shard identity."""
    base = str(entry.get("path", "shard")).rstrip("/").rsplit("/", 1)[-1]
    shard = _portable_token(os.path.splitext(base)[0] or "shard", fallback="shard")
    split = _portable_token(entry.get("split") or "unsplit", fallback="unsplit")
    return f"{split}__{shard}"


def _dataset_root(dataset: str) -> str:
    """Normalize the accepted directory/manifest spelling to the artifact root."""
    from ..io import _local, is_remote

    raw = str(dataset).rstrip("/\\")
    if is_remote(raw):
        return raw.rsplit("/", 1)[0] if raw.rsplit("/", 1)[-1] == "dataset_manifest.json" else raw
    local = Path(_local(raw))
    return str(local.parent) if local.name == "dataset_manifest.json" else str(local)


def _prepare_export_destination(dataset: str, out: str, *, force: bool) -> str:
    """Reject source overlap and accidental replacement before decoding any shard."""
    from ..io import _normalize_explicit_path, is_remote
    from ..manifest import validate_source_destination_separation

    destination = _normalize_explicit_path(out, label="export destination")
    if is_remote(destination) and any(character in destination for character in "*?["):
        raise ValueError(
            f"export destination {destination!r} contains a glob metacharacter; remote output "
            "paths must not contain '*', '?', or '['"
        )
    validate_source_destination_separation(destination, [_dataset_root(dataset)])
    _validate_destination_state(destination, force=force)
    return destination


def _validate_destination_state(destination: str, *, force: bool) -> None:
    """Apply no-clobber semantics, ignoring only the export's active lease object."""
    from ..io import _fs, _local, exists, is_remote

    if exists(destination):
        if is_remote(destination):
            fs = _fs(destination)
            is_directory = fs.isdir(destination)
            entries = fs.ls(destination, detail=False) if is_directory else []
            has_entries = any(
                str(entry).rstrip("/").rsplit("/", 1)[-1] != _WRITER_LEASE
                for entry in entries
            )
        else:
            local = Path(_local(destination))
            if local.is_symlink():
                raise ValueError(
                    f"export destination {destination!r} is a symlink; choose a real directory"
                )
            is_directory = local.is_dir()
            has_entries = any(local.iterdir()) if is_directory else False
        if not is_directory:
            raise NotADirectoryError(
                f"export destination {destination!r} exists and is not a directory"
            )
        if has_entries and not force:
            raise FileExistsError(
                f"export destination {destination!r} is not empty; pass --force to overwrite "
                "files named by this export"
            )


def _generation_path(destination: str, generation: str) -> tuple[str, str]:
    relative = f"{_GENERATIONS_DIR}/{generation}"
    return relative, f"{destination}/{relative}"


def _cleanup_generation(destination: str, generation_root: str) -> None:
    """Best-effort removal of one uncommitted namespace owned by this invocation."""
    from ..io import _fs, _local, is_remote

    try:
        if is_remote(generation_root):
            fs = _fs(generation_root)
            if fs.exists(generation_root):
                fs.rm(generation_root, recursive=True)
            return
        local_generation = Path(_local(generation_root))
        if local_generation.is_dir() and not local_generation.is_symlink():
            shutil.rmtree(local_generation)
        elif local_generation.exists() and not local_generation.is_symlink():
            local_generation.unlink()
        namespace = Path(_local(destination)) / _GENERATIONS_DIR
        if namespace.is_dir() and not namespace.is_symlink():
            namespace.rmdir()
    except (FileNotFoundError, OSError):
        # A crash can leave the same kind of unreferenced generation. It is never authoritative:
        # readers follow only the generation named by the commit-last export index.
        pass


def _claim_path(claims: dict[str, str], path: str, owner: str) -> None:
    prior = claims.get(path)
    if prior is not None:
        raise ValueError(
            f"export filename collision at {path!r}: both {prior} and {owner} claim it"
        )
    claims[path] = owner


def _write_parquet(path: str, arrays: dict) -> dict[str, int | str]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError("parquet export needs pyarrow: "
                          "pip install 'kuration-sdk[parquet]'") from error
    columns: dict[str, object] = {}
    owners: dict[str, str] = {}
    for name, array in arrays.items():
        if array.ndim == 1:
            produced = [(name, array)]
        else:
            # Flatten trailing dimensions into stable, order-preserving column names.
            flat = array.reshape(array.shape[0], -1)
            produced = [(f"{name}_{index}", flat[:, index]) for index in range(flat.shape[1])]
        for column, values in produced:
            if column in owners:
                # Flattening is name-mangling: silently overwriting a sibling modality would
                # export the wrong data under a plausible column name.
                raise ValueError(
                    f"parquet export column {column!r} is claimed by both modality "
                    f"{owners[column]!r} and modality {name!r}; flattened tensor columns are "
                    f"named '<modality>_<index>'. Export those modalities separately with "
                    "--modalities, or use --format npz/npy which keeps tensor shapes."
                )
            owners[column] = name
            columns[column] = values
    from ..io import file_integrity, staged_write
    with staged_write(path) as local_path:
        pq.write_table(pa.table(columns), local_path)
        return file_integrity(local_path)


def _write_numpy(path: str, arrays, *, archive: bool) -> dict[str, int | str]:
    import numpy as np

    from ..io import file_integrity, staged_write

    with staged_write(path) as local_path:
        if archive:
            np.savez(local_path, **arrays)
        else:
            np.save(local_path, arrays)
        return file_integrity(local_path)


def export_dataset(dataset: str, out: str, *, fmt: str = "npz", split: str | None = None,
                   modalities=None, trim: bool = True, verify: bool = True,
                   force: bool = False) -> dict:
    """Publish one complete exported generation and return its commit index."""
    if fmt not in _FORMATS:
        raise ValueError(f"--format must be one of {list(_FORMATS)}, got {fmt!r}")
    out = _prepare_export_destination(dataset, out, force=force)

    from ..io import write_text
    from ..loaders import iter_shards
    from ..manifest import dataset_writer_lease
    from ..provenance import publishable_source_id

    with dataset_writer_lease(out):
        # Re-check under the lease: another publisher may have committed after the first
        # no-clobber preflight but before this invocation acquired exclusivity.
        _validate_destination_state(out, force=force)
        generation = uuid.uuid4().hex
        generation_relative, generation_root = _generation_path(out, generation)
        committed = False
        try:
            files: list[dict] = []
            integrity: dict[str, dict[str, int | str]] = {}
            claimed_paths: dict[str, str] = {}
            shards = 0
            for arrays, entry in iter_shards(
                dataset,
                split=split,
                modalities=modalities,
                trim=trim,
                with_meta=True,
                verify=verify,
            ):
                stem = _stem(entry)
                source_id = entry.get("source_id")
                record = {
                    "source_id": (
                        publishable_source_id(source_id) if source_id is not None else None
                    ),
                    "entity_id": entry.get("entity_id"),
                    "split": entry.get("split"),
                    "shard_index": entry.get("shard_index"),
                    "n_steps": entry.get("n_steps"),
                    "modalities": sorted(arrays),
                }
                if fmt == "npy":
                    written = []
                    for name, array in sorted(arrays.items()):
                        modality = _portable_token(name, fallback="modality")
                        filename = f"{stem}__{modality}.npy"
                        relative = f"{generation_relative}/{filename}"
                        path = f"{out}/{relative}"
                        _claim_path(
                            claimed_paths,
                            relative,
                            f"shard {entry.get('path')!r} modality {name!r}",
                        )
                        integrity[relative] = _write_numpy(path, array, archive=False)
                        written.append(relative)
                    record["files"] = written
                else:
                    filename = f"{stem}.{fmt}"
                    relative = f"{generation_relative}/{filename}"
                    path = f"{out}/{relative}"
                    _claim_path(
                        claimed_paths, relative, f"shard {entry.get('path')!r}"
                    )
                    if fmt == "npz":
                        integrity[relative] = _write_numpy(path, arrays, archive=True)
                    else:
                        integrity[relative] = _write_parquet(path, arrays)
                    record["files"] = [relative]
                files.append(record)
                shards += 1

            if not shards:
                raise RuntimeError(
                    f"no shards matched in {dataset!r}"
                    + (f" for split {split!r}" if split else "")
                    + "; check the split name with kur-inspect"
                )
            index = {
                "index_version": 1,
                "state": "committed",
                "generation": generation,
                "dataset": publishable_source_id(_dataset_root(dataset)),
                "format": fmt,
                "split": split,
                "shards": shards,
                "trimmed": trim,
                "entries": files,
                "integrity": integrity,
            }
            encoded = json_document(index, label="export index")
            # The index is both the commit marker and the only pointer to this generation.
            # Atomic replacement therefore switches readers from the old complete generation to
            # the new complete generation without exposing a mixture.
            write_text(f"{out}/export_index.json", encoded)
            committed = True
            return index
        finally:
            if not committed:
                _cleanup_generation(out, generation_root)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export a written dataset's shards as plain array files.")
    ap.add_argument("dataset", help="Dataset directory or dataset_manifest.json path.")
    ap.add_argument("--out", required=True, metavar="DIR", help="Destination directory.")
    ap.add_argument("--format", default="npz", choices=list(_FORMATS),
                    help="Output file format (default: npz).")
    ap.add_argument("--split", help="Export one split only (e.g. train).")
    ap.add_argument("--modalities", help="Comma-separated subset of modalities to export.")
    ap.add_argument("--no-trim", action="store_true",
                    help="Keep trailing pad rows instead of trimming to n_steps.")
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip manifest checksum verification while reading.")
    ap.add_argument("--force", action="store_true",
                    help="Publish a new generation in a non-empty destination.")
    ap.add_argument("--json", action="store_true", help="Emit the export index as JSON.")
    ap.add_argument("--debug", action="store_true", help="Show full tracebacks on failure.")
    args = ap.parse_args()

    def body():
        modalities = ([name.strip() for name in args.modalities.split(",") if name.strip()]
                      if args.modalities else None)
        if modalities is not None and not modalities:
            raise ValueError("--modalities listed no names")
        index = export_dataset(args.dataset, args.out, fmt=args.format, split=args.split,
                               modalities=modalities, trim=not args.no_trim,
                               verify=not args.no_verify, force=args.force)
        if args.json:
            print(json_document(index, label="export index").rstrip())
        else:
            files = sum(len(entry["files"]) for entry in index["entries"])
            print(f"exported {index['shards']} shards ({files} file(s), "
                  f"format={index['format']}, split={index['split'] or 'all'}) "
                  f"→ {args.out}")

    cli_run(body, debug=args.debug)


if __name__ == "__main__":
    main()
