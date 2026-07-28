"""Detect an input adapter and import its complete dataset into Kuration."""
from __future__ import annotations

import argparse

from ..importing import import_dataset
from ._common import cli_run, json_document, parse_mapping_option


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-detect an input adapter and import without selecting rows."
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help="One or more input files, or a supported dataset directory.",
    )
    parser.add_argument("--out", required=True, help="Destination Kuration dataset directory.")
    parser.add_argument(
        "--adapter",
        help="Registered adapter override (default: detect from every source).",
    )
    parser.add_argument(
        "--schema",
        metavar="PATH",
        help="Schema JSON override; required for custom adapters without array import hooks.",
    )
    parser.add_argument(
        "--fields",
        metavar="JSON|PATH",
        help="Field mapping override, as inline JSON or path to JSON file.",
    )
    parser.add_argument("--time", help="Time-column override for tabular adapters.")
    parser.add_argument("--entity", help="Entity-column override for tabular adapters.")
    parser.add_argument("--hz", type=float, help="Sample-rate override.")
    parser.add_argument(
        "--kinds",
        metavar="JSON|PATH",
        help="Scalar/categorical inference overrides for tabular columns, as inline JSON or "
             "path to JSON file.",
    )
    parser.add_argument(
        "--adapter-options",
        metavar="JSON|PATH",
        help="Adapter-specific construction options, as inline JSON or path to JSON file.",
    )
    parser.add_argument("--source-id", help="Source ID override; only valid for one input file.")
    parser.add_argument("--split", default="train", help="Manifest split (default: train).")
    parser.add_argument(
        "--shard-length",
        type=int,
        help="Rows per output shard; by default each logical source/track remains one shard.",
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
        help="Replace an existing Kuration dataset at the destination, "
             "removing its manifest and any stale shard files.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a structured import result.")
    parser.add_argument("--debug", action="store_true", help="Show tracebacks on failure.")
    args = parser.parse_args()

    def body() -> None:
        result = import_dataset(
            args.sources,
            args.out,
            adapter=args.adapter,
            schema=args.schema,
            fields=parse_mapping_option(args.fields, label="--fields"),
            time=args.time,
            entity=args.entity,
            hz=args.hz,
            kinds=parse_mapping_option(args.kinds, label="--kinds"),
            adapter_options=parse_mapping_option(args.adapter_options, label="--adapter-options"),
            source_id=args.source_id,
            split=args.split,
            shard_length=args.shard_length,
            container=args.container,
            force=args.force,
        )
        if args.json:
            print(json_document(result, label="dataset import result").rstrip())
        else:
            print(
                f"imported with {result['adapter']}: {result['sources']} source(s), "
                f"{result['steps']} rows, "
                f"{result['shards']} shard(s) with modalities "
                f"{', '.join(result['modalities'])} → {result['dataset']}"
            )

    cli_run(body, debug=args.debug)


if __name__ == "__main__":
    main()
