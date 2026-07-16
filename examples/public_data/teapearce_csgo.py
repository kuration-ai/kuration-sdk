"""Curate TeaPearce Counter-Strike HDF5 recordings for action and combat coverage.

Try the offline structural demo:

    python examples/public_data/teapearce_csgo.py --demo --out teapearce-out

Or pass one or more extracted upstream recordings:

    python examples/public_data/teapearce_csgo.py data/*.hdf5 --out teapearce-out
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from kuration import OutputSpec, input, iter_shards

try:  # package import in tests / notebooks
    from .adapters.teapearce import (
        ACTION_WIDTH,
        FrameEncoder,
        TeaPearceHDF5Adapter,
        teapearce_schema,
        write_demo_recording,
    )
except ImportError:  # direct: python examples/public_data/teapearce_csgo.py
    from adapters.teapearce import (
        ACTION_WIDTH,
        FrameEncoder,
        TeaPearceHDF5Adapter,
        teapearce_schema,
        write_demo_recording,
    )


def output_spec(*, include_frames: bool, frame_shape: tuple[int, ...] | None) -> OutputSpec:
    modalities = {
        "action": {"dtype": "float32", "shape": [ACTION_WIDTH], "source": "field:action"},
        "kill": {"dtype": "uint8", "shape": [], "source": "field:kill"},
        "death": {"dtype": "uint8", "shape": [], "source": "field:death"},
        "mouse_dx": {"dtype": "float32", "shape": [], "source": "field:mouse_dx"},
        "mouse_dy": {"dtype": "float32", "shape": [], "source": "field:mouse_dy"},
    }
    if include_frames:
        assert frame_shape is not None
        modalities["frame"] = {
            "dtype": "uint8", "shape": list(frame_shape), "source": "encoder:frame"}
    return OutputSpec.from_dict({
        "container": "npz",
        "shard_length": 96,
        "short_policy": "keep",
        "modalities": modalities,
    })


def run(sources: list[str], out_dir: str, *, n: int = 12,
        include_frames: bool = False) -> dict:
    if not sources:
        raise ValueError("provide at least one extracted .hdf5 recording or use --demo")
    schema = teapearce_schema()
    adapter = TeaPearceHDF5Adapter(schema, include_frames=include_frames)
    for source in sources:
        adapter.check_source(source)

    frame_shape = None
    if include_frames:
        shapes = {adapter.frame_shape(source) for source in sources}
        if len(shapes) != 1:
            raise ValueError(f"all frame shapes must agree, got {sorted(shapes)}")
        frame_shape = shapes.pop()

    pipeline = (
        input(adapter=adapter, sources=sources, schema=schema)
        .window(96, stride=48)
        .measure("action_density", "mean", of="keys_down")
        .measure("mouse_activity", "mean", of="mouse_motion")
        .measure("shoot_rate", "rate", of="left_click")
        .bin("action_density", bins=4)
        .bin("mouse_activity", bins=4)
        .tag("combat.kind", min_frac=0.08)
        .sample("equal", n=n)
        .options(mode="strict", seed=21)
    )
    if len(sources) > 1:
        pipeline = pipeline.split(test=0.2)

    result = pipeline.output(
        out_dir,
        spec=output_spec(include_frames=include_frames, frame_shape=frame_shape),
        encoder=FrameEncoder() if include_frames else None,
    )
    arrays, metadata = next(iter_shards(out_dir, with_meta=True))
    print(f"input: {len(sources)} recording(s)")
    print(f"curation: {result['results']['segments_total']} candidate windows -> "
          f"{result['selected']} selected")
    print(f"output: {len(result['shards'])} shards in {out_dir}")
    print(f"first shard: split={metadata['split']}, modalities={sorted(arrays)}")
    print(f"combat totals in output: kills={sum(int(s['kill'].sum()) for s in iter_shards(out_dir))}, "
          f"deaths={sum(int(s['death'].sum()) for s in iter_shards(out_dir))}")
    return result


def _demo_sources(directory: str) -> list[str]:
    sources = []
    for recording in range(4):
        path = Path(directory) / f"synthetic_recording_{recording}.hdf5"
        write_demo_recording(path, recording=recording)
        sources.append(str(path))
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", help="extracted TeaPearce .hdf5 recordings")
    parser.add_argument("--out", default="teapearce-out", help="output dataset directory")
    parser.add_argument("--n", type=int, default=12, help="number of windows to select")
    parser.add_argument("--include-frames", action="store_true",
                        help="copy image tensors into output shards (substantially larger)")
    parser.add_argument("--demo", action="store_true",
                        help="use generated structural fixtures instead of upstream data")
    args = parser.parse_args()
    if args.demo and args.sources:
        parser.error("--demo cannot be combined with source paths")
    if not args.demo and not args.sources:
        parser.error("provide source paths or use --demo")

    if args.demo:
        with tempfile.TemporaryDirectory(prefix="kuration_teapearce_demo_") as directory:
            run(_demo_sources(directory), args.out, n=args.n,
                include_frames=args.include_frames)
    else:
        run(args.sources, args.out, n=args.n, include_frames=args.include_frames)


if __name__ == "__main__":
    main()
