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
import json
import os

from ._common import cli_run

_FORMATS = ("npz", "npy", "parquet")


def _stem(entry: dict) -> str:
    """Reuse the artifact's own collision-proof shard name as the export name."""
    base = str(entry.get("path", "shard")).rstrip("/").rsplit("/", 1)[-1]
    return os.path.splitext(base)[0] or "shard"


def _write_parquet(path: str, arrays: dict) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError("parquet export needs pyarrow: "
                          "pip install 'kuration-sdk[parquet]'") from error
    columns: dict[str, object] = {}
    for name, array in arrays.items():
        if array.ndim == 1:
            columns[name] = array
            continue
        # Flatten trailing dimensions into stable, order-preserving column names.
        flat = array.reshape(array.shape[0], -1)
        for index in range(flat.shape[1]):
            columns[f"{name}_{index}"] = flat[:, index]
    from ..io import staged_write
    with staged_write(path) as local_path:
        pq.write_table(pa.table(columns), local_path)


def export_dataset(dataset: str, out: str, *, fmt: str = "npz", split: str | None = None,
                   modalities=None, trim: bool = True, verify: bool = True) -> dict:
    """Write every shard of `dataset` into `out` as `fmt` and return an index document."""
    if fmt not in _FORMATS:
        raise ValueError(f"--format must be one of {list(_FORMATS)}, got {fmt!r}")
    import numpy as np

    from ..io import staged_write, write_text
    from ..loaders import iter_shards

    out = str(out).rstrip("/")
    files: list[dict] = []
    shards = 0
    for arrays, entry in iter_shards(dataset, split=split, modalities=modalities,
                                     trim=trim, with_meta=True, verify=verify):
        stem = _stem(entry)
        record = {"source_id": entry.get("source_id"), "entity_id": entry.get("entity_id"),
                  "split": entry.get("split"), "shard_index": entry.get("shard_index"),
                  "n_steps": entry.get("n_steps"), "modalities": sorted(arrays)}
        if fmt == "npy":
            written = []
            for name, array in sorted(arrays.items()):
                path = f"{out}/{stem}__{name}.npy"
                with staged_write(path) as local_path:
                    np.save(local_path, array)
                written.append(path)
            record["files"] = written
        else:
            path = f"{out}/{stem}.{fmt}"
            if fmt == "npz":
                with staged_write(path) as local_path:
                    np.savez(local_path, **arrays)
            else:
                _write_parquet(path, arrays)
            record["files"] = [path]
        files.append(record)
        shards += 1

    if not shards:
        raise RuntimeError(
            f"no shards matched in {dataset!r}"
            + (f" for split {split!r}" if split else "")
            + "; check the split name with kur-inspect")
    index = {"dataset": str(dataset), "format": fmt, "split": split, "shards": shards,
             "trimmed": trim, "entries": files}
    write_text(f"{out}/export_index.json", json.dumps(index, indent=2) + "\n")
    return index


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
                               verify=not args.no_verify)
        if args.json:
            print(json.dumps(index, indent=2))
        else:
            files = sum(len(entry["files"]) for entry in index["entries"])
            print(f"exported {index['shards']} shards ({files} file(s), "
                  f"format={index['format']}, split={index['split'] or 'all'}) "
                  f"→ {args.out}")

    cli_run(body, debug=args.debug)


if __name__ == "__main__":
    main()
