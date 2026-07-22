"""Command-line golden path for SLAM preparation and trajectory curation."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import importlib
import inspect
import json
from pathlib import Path
import sys

from .pointcloud import curate_point_cloud
from .preflight import preflight_slam
from .runner import prepare_slam_input
from .workflow import curate_slam_gameplay


def _symbol(spec: str):
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("runtime factory must use 'module:attribute' syntax")
    module = importlib.import_module(module_name)
    value = getattr(module, attribute)
    if not callable(value):
        raise TypeError(f"runtime factory {spec!r} is not callable")
    return value


def _runtime_options(spec: str | None, *, preflight: bool = False) -> dict:
    if spec is None:
        return {}
    factory = _symbol(spec)
    try:
        inspect.signature(factory).bind(preflight=preflight)
    except (TypeError, ValueError):
        options = factory()
    else:
        options = factory(preflight=preflight)
    if not isinstance(options, Mapping):
        raise TypeError("runtime factory must return a mapping of run_slam_pipeline options")
    return dict(options)


def _videos(values: list[str] | None) -> dict[str, str] | None:
    if not values:
        return None
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError("--video values must use NAME=PATH syntax")
        if name in result:
            raise ValueError(f"duplicate video clip name {name!r}")
        result[name] = path
    return result


def _source(args):
    videos = _videos(getattr(args, "video", None))
    if args.manifest is not None and videos is not None:
        raise ValueError("pass --manifest or --video, not both")
    if args.manifest is not None:
        return Path(args.manifest)
    if videos is not None:
        return videos
    raise ValueError("pass --manifest or at least one --video NAME=PATH")


def _progress(quiet: bool):
    if quiet or not sys.stderr.isatty():
        return None
    from kuration.progress import cli_progress

    return cli_progress(file=sys.stderr)


def _add_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", help="Existing SLAM manifest or export directory.")
    parser.add_argument(
        "--video",
        action="append",
        metavar="NAME=PATH",
        help="Raw video clip; repeat for multiple clips.",
    )
    parser.add_argument(
        "--runtime-factory",
        help="Callable module:attribute returning runner options and initialized models.",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kur-slam",
        description="Prepare SLAM reconstruction data and curate diverse gameplay windows.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="Check runtime and input readiness.")
    _add_source(preflight)
    preflight.add_argument("--out", help="Preparation output directory for raw input.")
    preflight.add_argument("--require-gpu", action="store_true")
    preflight.add_argument("--json", action="store_true", dest="as_json")

    run = commands.add_parser("run", help="Prepare input and write a curated selection manifest.")
    _add_source(run)
    run.add_argument("--out", required=True, help="Workflow output directory.")
    run.add_argument("--window", type=int, default=32)
    run.add_argument("--stride", type=int, default=16)
    run.add_argument("--count", type=int, default=2_000)
    run.add_argument("--min-quality", type=float, default=0.5)
    run.add_argument(
        "--method", choices=("uniform-bins", "k-center"), default="uniform-bins"
    )
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--cache-key")
    run.add_argument("--require-gpu", action="store_true")
    run.add_argument("--quiet", action="store_true")

    demo = commands.add_parser("demo", help="Run the model-free offline golden path.")
    demo.add_argument("--out", required=True)
    demo.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        source = _source(args)
        report = preflight_slam(
            source,
            args.out,
            runner_options=_runtime_options(args.runtime_factory, preflight=True),
            require_gpu=args.require_gpu,
        )
        print(json.dumps(report.to_dict(), indent=2) if args.as_json else report.render())
        return 0 if report.ready else 2

    if args.command == "demo":
        from .demo import write_demo_reconstruction

        root = Path(args.out)
        manifest = write_demo_reconstruction(root / "slam-export")
        result = curate_slam_gameplay(
            manifest,
            root / "selection.json",
            window=16,
            stride=8,
            n=12,
            min_quality=0.5,
            progress=_progress(args.quiet),
        )
        curate_point_cloud(
            manifest,
            method="voxel",
            voxel_size=0.05,
            max_points=500,
            output=root / "scene_pointcloud.npz",
        )
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    source = _source(args)
    root = Path(args.out)
    progress = _progress(args.quiet)
    runner_options = _runtime_options(args.runtime_factory)
    manifest = prepare_slam_input(
        source,
        root / "slam-export",
        runner_options=runner_options,
        resume=args.resume,
        cache_key=args.cache_key,
        progress=progress,
        require_gpu=args.require_gpu,
    )
    result = curate_slam_gameplay(
        manifest,
        root / "selection.json",
        window=args.window,
        stride=args.stride,
        n=args.count,
        min_quality=args.min_quality,
        selection_method=args.method,
        seed=args.seed,
        progress=progress,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
