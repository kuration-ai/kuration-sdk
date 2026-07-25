"""Import NPZ tensor datasets into a lossless Kuration artifact."""
from __future__ import annotations

import argparse
import json

from ..importing import import_dataset
from ._common import cli_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import .npz files into a Kuration dataset without selecting rows."
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help="One or more .npz files or directories (directories are searched recursively).",
    )
    parser.add_argument("--out", required=True, help="Destination Kuration dataset directory.")
    parser.add_argument("--source-id", help="Source ID override; only valid for one input file.")
    parser.add_argument("--split", default="train", help="Manifest split (default: train).")
    parser.add_argument(
        "--shard-length",
        type=int,
        help="Rows per output shard; by default each input file remains one shard.",
    )
    parser.add_argument(
        "--container",
        choices=("npz", "json", "hdf5", "parquet"),
        default="npz",
        help="Output container (default: npz).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing Kuration manifest at the destination.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a structured import result.")
    parser.add_argument("--debug", action="store_true", help="Show tracebacks on failure.")
    args = parser.parse_args()

    def body() -> None:
        result = import_dataset(
            args.sources,
            args.out,
            source_id=args.source_id,
            split=args.split,
            shard_length=args.shard_length,
            container=args.container,
            force=args.force,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(
                f"imported {result['sources']} source(s), {result['steps']} rows, "
                f"{result['shards']} shard(s) with modalities "
                f"{', '.join(result['modalities'])} → {result['dataset']}"
            )

    cli_run(body, debug=args.debug)


if __name__ == "__main__":
    main()
