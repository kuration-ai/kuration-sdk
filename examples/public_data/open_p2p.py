"""Curate Open P2P annotations for cross-game action and instruction coverage.

    python examples/public_data/open_p2p.py --demo --out open-p2p-out
    python examples/public_data/open_p2p.py dataset/*/annotation.proto --out open-p2p-out
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from kuration import OutputSpec, input, iter_shards

try:
    from .adapters.open_p2p import (
        OpenP2PAnnotationAdapter,
        OpenP2PTextEncoder,
        open_p2p_schema,
        write_demo_annotation,
    )
except ImportError:
    from adapters.open_p2p import (
        OpenP2PAnnotationAdapter,
        OpenP2PTextEncoder,
        open_p2p_schema,
        write_demo_annotation,
    )


def output_spec() -> OutputSpec:
    return OutputSpec.from_dict({
        "container": "npz",
        "shard_length": 40,
        "short_policy": "keep",
        "modalities": {
            "frame_index": {"dtype": "int64", "shape": [], "source": "field:frame_index"},
            "mouse_dx": {"dtype": "float32", "shape": [], "source": "field:mouse_dx"},
            "mouse_dy": {"dtype": "float32", "shape": [], "source": "field:mouse_dy"},
            "known_action": {"dtype": "uint8", "shape": [], "source": "field:known_action"},
            "keys": {"dtype": "U256", "shape": [], "source": "encoder:keys"},
            "buttons": {"dtype": "U128", "shape": [], "source": "encoder:buttons"},
            "instruction": {
                "dtype": "U1024", "shape": [], "source": "encoder:instruction"},
        },
    })


def run(sources: list[str], out_dir: str, *, n: int = 12) -> dict:
    if not sources:
        raise ValueError("provide annotation.proto paths or use --demo")
    schema = open_p2p_schema()
    adapter = OpenP2PAnnotationAdapter(schema)
    for source in sources:
        adapter.check_source(source)

    pipeline = (
        input(adapter=adapter, sources=sources, schema=schema)
        .window(40, stride=20)
        .measure("key_density", "mean", of="keys_down")
        .measure("mouse_activity", "mean", of="mouse_motion")
        .measure("instruction_rate", "rate", of="instruction_present")
        .bin("key_density", bins=4)
        .bin("mouse_activity", bins=4)
        .tag("environment", "action_source", "instruction_state", min_frac=0.05)
        .sample("equal", n=n)
        .options(mode="strict", seed=33)
    )
    if len(sources) > 1:
        pipeline = pipeline.split(test=0.2)

    result = pipeline.output(out_dir, spec=output_spec(), encoder=OpenP2PTextEncoder())
    arrays, metadata = next(iter_shards(out_dir, with_meta=True))
    instruction_steps = sum(
        int(np.count_nonzero(shard["instruction"] != "")) for shard in iter_shards(out_dir))
    system_like_steps = sum(
        int(np.count_nonzero(shard["known_action"])) for shard in iter_shards(out_dir))
    print(f"input: {len(sources)} recording(s)")
    print(f"curation: {result['results']['segments_total']} candidate windows -> "
          f"{result['selected']} selected")
    print(f"output: {len(result['shards'])} shards in {out_dir}")
    print(f"first shard: split={metadata['split']}, modalities={sorted(arrays)}")
    print(f"selected annotation coverage: instruction_steps={instruction_steps}, "
          f"known_action_steps={system_like_steps}")
    return result


def _demo_sources(directory: str) -> list[str]:
    sources = []
    for recording in range(4):
        path = Path(directory) / f"recording_{recording}" / "annotation.proto"
        write_demo_annotation(path, recording=recording)
        sources.append(str(path))
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", help="Open P2P annotation.proto paths")
    parser.add_argument("--out", default="open-p2p-out", help="output dataset directory")
    parser.add_argument("--n", type=int, default=12, help="number of windows to select")
    parser.add_argument("--demo", action="store_true",
                        help="use generated protobuf fixtures instead of downloaded data")
    args = parser.parse_args()
    if args.demo and args.sources:
        parser.error("--demo cannot be combined with source paths")
    if not args.demo and not args.sources:
        parser.error("provide source paths or use --demo")

    if args.demo:
        with tempfile.TemporaryDirectory(prefix="kuration_open_p2p_demo_") as directory:
            run(_demo_sources(directory), args.out, n=args.n)
    else:
        run(args.sources, args.out, n=args.n)


if __name__ == "__main__":
    main()
