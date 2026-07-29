"""Command-line golden path for SLAM preparation and trajectory curation."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import importlib
import inspect
from pathlib import Path
import sys

from ...core.cli._common import _storage_path_relation, cli_run, json_document
from ._paths import local_path
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


def _runtime_options(
    spec: str | None,
    *,
    depth_model: str | None = None,
    depth_model_id: str | None = None,
    preflight: bool = False,
) -> dict:
    if spec is not None and depth_model is not None:
        raise ValueError("pass --runtime-factory or --depth-model, not both")
    if spec is None:
        if depth_model is None:
            return {}
        return {
            "depth_model": depth_model,
            "depth_model_id": depth_model_id or depth_model,
        }
    factory = _symbol(spec)
    try:
        inspect.signature(factory).bind(preflight=preflight)
    except (TypeError, ValueError):
        options = factory()
    else:
        options = factory(preflight=preflight)
    if not isinstance(options, Mapping):
        raise TypeError("runtime factory must return a mapping of reconstruction options")
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
    from kuration.core.progress import cli_progress

    return cli_progress(file=sys.stderr)


def _decoder_options(args) -> dict:
    return {
        "frame_skip": args.frame_skip,
        "batch_size": args.batch_size,
        "color_space": args.color_space,
    }


def _source_paths(source) -> tuple[str | Path, ...]:
    if isinstance(source, Mapping):
        return tuple(
            value for value in source.values() if isinstance(value, (str, Path))
        )
    return (source,) if isinstance(source, (str, Path)) else ()


def _output_root(value: str) -> Path:
    if not value.strip():
        raise ValueError("--out must be a non-empty local path")
    root = local_path(value, name="kur-slam --out")
    resolved = root.resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise ValueError("kur-slam --out must not be the filesystem root")
    return root


def _guard_output_root(
    root: Path,
    *,
    source,
    force: bool,
    resume: bool,
) -> None:
    """Protect source trees and existing results before reconstruction begins."""
    if force and resume:
        raise ValueError("--force and --resume are mutually exclusive")
    if root.is_symlink():
        raise ValueError(f"output root {str(root)!r} must not be a symlink")
    for source_path in _source_paths(source):
        relation = _storage_path_relation(root, source_path)
        if relation is not None:
            raise ValueError(
                f"output root {str(root)!r} overlaps input path {str(source_path)!r} "
                f"({relation}); choose an output outside the input tree"
            )
    if not root.exists():
        return
    if not root.is_dir():
        raise FileExistsError(
            f"output root {str(root)!r} exists and is not a directory"
        )
    if any(root.iterdir()) and not force and not resume:
        raise FileExistsError(
            f"output root {str(root)!r} is not empty; pass --force to update the "
            "command-owned files or --resume to reuse a matching reconstruction"
        )


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
    parser.add_argument(
        "--depth-model",
        help="Depth Anything 3 preset or Hub model id; an alternative to --runtime-factory.",
    )
    parser.add_argument(
        "--depth-model-id",
        help="Stable model revision recorded in provenance (defaults to --depth-model).",
    )
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--color-space", choices=("RGB", "BGR"), default="RGB")


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
    preflight.add_argument("--debug", action="store_true",
                           help="Show full tracebacks on failure.")

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
    run.add_argument(
        "--force",
        action="store_true",
        help="Update command-owned files in a non-empty output; unrelated files are retained.",
    )
    run.add_argument("--cache-key")
    run.add_argument("--require-gpu", action="store_true")
    run.add_argument("--quiet", action="store_true")
    run.add_argument("--debug", action="store_true", help="Show full tracebacks on failure.")

    demo = commands.add_parser("demo", help="Run the model-free offline golden path.")
    demo.add_argument("--out", required=True)
    demo.add_argument(
        "--force",
        action="store_true",
        help="Update command-owned files in a non-empty output; unrelated files are retained.",
    )
    demo.add_argument("--quiet", action="store_true")
    demo.add_argument("--debug", action="store_true", help="Show full tracebacks on failure.")
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.command == "preflight":
        source = _source(args)
        report = preflight_slam(
            source,
            args.out,
            runner_options=_runtime_options(
                args.runtime_factory,
                depth_model=args.depth_model,
                depth_model_id=args.depth_model_id,
                preflight=True,
            ),
            decoder_options=_decoder_options(args),
            require_gpu=args.require_gpu,
        )
        print(
            json_document(report.to_dict(), label="SLAM preflight report").rstrip()
            if args.as_json
            else report.render()
        )
        return 0 if report.ready else 2

    if args.command == "demo":
        from .demo import write_demo_reconstruction

        root = _output_root(args.out)
        _guard_output_root(root, source=None, force=args.force, resume=False)
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
        print(json_document(result.to_dict(), label="SLAM curation result").rstrip())
        return 0

    source = _source(args)
    root = _output_root(args.out)
    _guard_output_root(root, source=source, force=args.force, resume=args.resume)
    progress = _progress(args.quiet)
    runner_options = _runtime_options(
        args.runtime_factory,
        depth_model=args.depth_model,
        depth_model_id=args.depth_model_id,
    )
    manifest = prepare_slam_input(
        source,
        root / "slam-export",
        runner_options=runner_options,
        decoder_options=_decoder_options(args),
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
    print(json_document(result.to_dict(), label="SLAM curation result").rstrip())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    status = 0

    def body() -> None:
        nonlocal status
        status = _run(args)

    cli_run(body, debug=args.debug)
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
