"""Optional global SLAM execution before bounded Kuration curation begins."""
from __future__ import annotations

from collections.abc import Callable, Mapping
import importlib
import json
from pathlib import Path

import numpy as np

from .export import EXPORT_FORMAT, EXPORT_VERSION, export_slam_output


_CACHE_FORMAT = "kuration-slam-prepare-cache"
_CACHE_VERSION = 1


def _emit(progress, phase: str, event: str, **fields) -> None:
    if progress is not None:
        progress({"phase": phase, "event": event, **fields})


def _pipeline_factory(factory: Callable | str | None) -> Callable:
    if factory is None:
        factory = "kuration.slam.slam_pipeline:SLAMOperator"
    if isinstance(factory, str):
        module_name, separator, attribute = factory.partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError("pipeline_factory strings must use 'module:attribute' syntax")
        try:
            module = importlib.import_module(module_name)
            factory = getattr(module, attribute)
        except (ImportError, AttributeError) as error:
            raise ImportError(
                f"cannot load SLAM pipeline factory {module_name}:{attribute}; install or "
                "provide the SLAMOperator implementation, or pass pipeline_factory= explicitly"
            ) from error
    if not callable(factory):
        raise TypeError("pipeline_factory must be callable or an import string")
    return factory


def _validated_frames(data: Mapping[str, object]) -> dict[str, np.ndarray]:
    if not isinstance(data, Mapping) or not data:
        raise ValueError("raw SLAM input must be a non-empty {clip_name: frames} mapping")
    validated = {}
    for name, frames in data.items():
        if not isinstance(name, str) or not name:
            raise ValueError("raw SLAM clip names must be non-empty strings")
        value = np.asarray(frames)
        if value.ndim != 4 or value.shape[0] < 1 or value.shape[-1] != 3:
            raise ValueError(f"raw SLAM frames for {name!r} must have shape (T, H, W, 3)")
        if value.dtype.kind not in "uif" or not np.isfinite(value).all():
            raise ValueError(f"raw SLAM frames for {name!r} must be a finite numeric array")
        validated[name] = value
    return validated


def decode_video_clips(
    videos: Mapping[str, str | Path],
    *,
    frame_skip: int = 1,
    batch_size: int = 64,
    color_space: str = "RGB",
    provider=None,
    progress=None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Decode video paths into the frame mapping and timestamps expected by SLAMOperator."""
    if not isinstance(videos, Mapping) or not videos:
        raise ValueError("video input must be a non-empty {clip_name: path} mapping")
    if not isinstance(frame_skip, int) or isinstance(frame_skip, bool) or frame_skip < 1:
        raise ValueError("frame_skip must be a positive integer")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("video decode batch_size must be a positive integer")
    if color_space not in {"RGB", "BGR"}:
        raise ValueError("SLAM video color_space must be 'RGB' or 'BGR'")
    if provider is None:
        try:
            from kuration import PyAVMediaProvider

            provider = PyAVMediaProvider()
        except ImportError as error:
            raise ImportError(
                "decoding video for SLAM needs the slam extra: "
                "pip install 'kuration-sdk[slam]'"
            ) from error
    frames_by_clip = {}
    timestamps = {}
    _emit(progress, "slam-decode", "start", total=len(videos))
    for clip_index, (name, uri) in enumerate(videos.items(), 1):
        if not isinstance(name, str) or not name:
            raise ValueError("video clip names must be non-empty strings")
        if not isinstance(uri, (str, Path)) or not str(uri):
            raise ValueError(f"video path for {name!r} must be a non-empty path or URI")
        refs = list(provider.iter_refs(str(uri), source_id=name))[::frame_skip]
        if not refs:
            raise ValueError(f"video for {name!r} has no frames")
        batches = []
        for start in range(0, len(refs), batch_size):
            batches.append(np.asarray(provider.decode_batch(refs[start:start + batch_size])))
        frames = np.concatenate(batches, axis=0)
        if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) != len(refs):
            raise ValueError(
                f"video provider for {name!r} must decode shape (T, H, W, 3), got {frames.shape}"
            )
        # Kuration's video provider declares RGB, matching what the vendored Depth Anything 3
        # input processor assumes (it never swaps channels). Pass color_space="BGR" only for a
        # SLAM implementation ported from a cv2-based (BGR) loader.
        if color_space == "BGR":
            frames = frames[..., ::-1].copy()
        frames_by_clip[name] = frames
        timestamps[name] = np.asarray([ref.timestamp for ref in refs], dtype=np.float64)
        _emit(
            progress,
            "slam-decode",
            "advance",
            current=clip_index,
            total=len(videos),
            source=name,
            frames=len(refs),
        )
    _emit(
        progress,
        "slam-decode",
        "complete",
        current=len(videos),
        total=len(videos),
        detail=f"{sum(len(value) for value in frames_by_clip.values())} frames",
    )
    return frames_by_clip, timestamps


def run_slam_pipeline(
    data: Mapping[str, object],
    *,
    depth_model=None,
    embedding_model=None,
    embeddings=None,
    pipeline_factory: Callable | str | None = None,
    progress=None,
    **pipeline_options,
):
    """Run the notebook-style ``SLAMOperator`` over raw in-memory clip frames.

    Models are injected by the caller so this module has no mandatory Torch, Depth Anything,
    SALAD, or FAISS dependency. With no explicit factory, the function lazily imports the
    vendored ``kuration.slam.slam_pipeline.SLAMOperator``.
    """
    frames = _validated_frames(data)
    if depth_model is None:
        raise ValueError("running SLAM needs depth_model; precomputed SLAM input does not")
    if embedding_model is not None and embeddings is not None:
        raise ValueError("pass embedding_model or embeddings, not both")
    options = dict(pipeline_options)
    options["depth_model"] = depth_model
    if embedding_model is not None:
        options["embedding_model"] = embedding_model
    if embeddings is not None:
        options["embeddings"] = embeddings
    pipeline = _pipeline_factory(pipeline_factory)(**options)
    if not callable(pipeline):
        raise TypeError("pipeline_factory must construct a callable SLAM pipeline")
    _emit(progress, "slam-reconstruction", "start", total=len(frames))
    try:
        output = pipeline(frames)
    except Exception:
        _emit(progress, "slam-reconstruction", "error", total=len(frames))
        raise
    if not isinstance(getattr(output, "trajectories", None), Mapping) \
            or not isinstance(getattr(output, "poses", None), Mapping):
        raise ValueError("SLAM pipeline output must expose trajectory and pose mappings")
    _emit(
        progress,
        "slam-reconstruction",
        "complete",
        current=len(frames),
        total=len(frames),
        detail=f"{len(output.trajectories)} reconstructed clips",
    )
    return output


def _existing_manifest(source) -> Path | None:
    if not isinstance(source, (str, Path)):
        return None
    path = Path(source)
    if path.is_dir():
        path = path / "manifest.json"
    if not path.is_file():
        raise ValueError(f"SLAM input path does not contain a manifest: {path}")
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read SLAM manifest {str(path)!r}") from error
    if document.get("format") != EXPORT_FORMAT or document.get("version") != EXPORT_VERSION:
        raise ValueError(
            f"SLAM manifest must declare format={EXPORT_FORMAT!r}, version={EXPORT_VERSION}"
        )
    return path


def _cache_manifest(out_dir: str | Path, cache_key: str) -> Path | None:
    root = Path(out_dir)
    metadata_path = root / ".kuration-slam-prepare.json"
    manifest_path = root / "manifest.json"
    if not metadata_path.is_file() or not manifest_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if metadata != {
        "format": _CACHE_FORMAT,
        "version": _CACHE_VERSION,
        "cache_key": cache_key,
    }:
        return None
    return _existing_manifest(manifest_path)


def _write_cache(out_dir: str | Path, cache_key: str) -> None:
    root = Path(out_dir)
    target = root / ".kuration-slam-prepare.json"
    temporary = root / ".kuration-slam-prepare.json.tmp"
    document = {"format": _CACHE_FORMAT, "version": _CACHE_VERSION, "cache_key": cache_key}
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def prepare_slam_input(
    source,
    out_dir: str | Path | None = None,
    *,
    run_if_needed: bool = True,
    runner_options: Mapping[str, object] | None = None,
    decoder_options: Mapping[str, object] | None = None,
    export_options: Mapping[str, object] | None = None,
    resume: bool = False,
    cache_key: str | None = None,
    progress=None,
    preflight: bool = True,
    require_gpu: bool = False,
) -> Path:
    """Normalize existing manifests, completed outputs, or raw frames into one manifest.

    - A manifest path is returned unchanged.
    - A completed SLAM output object is exported without constructing a pipeline.
    - A raw ``{clip_name: frames}`` mapping runs SLAM only when ``run_if_needed`` is true.
    """
    if not isinstance(resume, bool) or not isinstance(preflight, bool):
        raise ValueError("resume and preflight must be true or false")
    if cache_key is not None and (not isinstance(cache_key, str) or not cache_key.strip()):
        raise ValueError("cache_key must be a non-empty string or None")
    _emit(progress, "slam-prepare", "start", total=None)
    manifest = _existing_manifest(source)
    if manifest is not None:
        _emit(progress, "slam-cache", "complete", detail="existing manifest reused")
        _emit(progress, "slam-prepare", "complete", detail=str(manifest))
        return manifest
    if out_dir is None:
        raise ValueError("out_dir is required when the input is not an existing SLAM manifest")
    if resume:
        if cache_key is None:
            raise ValueError("resume=True requires cache_key for non-manifest input")
        cached = _cache_manifest(out_dir, cache_key)
        if cached is not None:
            _emit(progress, "slam-cache", "complete", detail="matching preparation reused")
            _emit(progress, "slam-prepare", "complete", detail=str(cached))
            return cached
        _emit(progress, "slam-cache", "complete", detail="cache miss")
    if preflight:
        from .preflight import preflight_slam

        report = preflight_slam(
            source,
            out_dir,
            run_if_needed=run_if_needed,
            runner_options=runner_options,
            decoder_options=decoder_options,
            require_gpu=require_gpu,
        )
        _emit(
            progress,
            "slam-preflight",
            "complete",
            detail="ready" if report.ready else "failed",
            report=report.to_dict(),
        )
        report.raise_for_errors()
    export_kwargs = dict(export_options or {})
    if isinstance(getattr(source, "trajectories", None), Mapping) \
            and isinstance(getattr(source, "poses", None), Mapping):
        _emit(progress, "slam-export", "start", total=1)
        prepared = export_slam_output(source, out_dir, **export_kwargs)
        _emit(progress, "slam-export", "complete", current=1, total=1)
        if cache_key is not None:
            _write_cache(out_dir, cache_key)
        _emit(progress, "slam-prepare", "complete", detail=str(prepared))
        return prepared
    if isinstance(source, Mapping):
        if not run_if_needed:
            raise ValueError("raw frame input requires SLAM, but run_if_needed=False")
        if source and all(isinstance(value, (str, Path)) for value in source.values()):
            decode_kwargs = dict(decoder_options or {})
            decode_kwargs.setdefault("progress", progress)
            frames, timestamps = decode_video_clips(source, **decode_kwargs)
            export_kwargs.setdefault("timestamps", timestamps)
            export_kwargs.setdefault("videos", {name: str(path) for name, path in source.items()})
        else:
            frames = source
        run_kwargs = dict(runner_options or {})
        run_kwargs.setdefault("progress", progress)
        output = run_slam_pipeline(frames, **run_kwargs)
        _emit(progress, "slam-export", "start", total=1)
        prepared = export_slam_output(output, out_dir, **export_kwargs)
        _emit(progress, "slam-export", "complete", current=1, total=1)
        if cache_key is not None:
            _write_cache(out_dir, cache_key)
        _emit(progress, "slam-prepare", "complete", detail=str(prepared))
        return prepared
    raise TypeError(
        "SLAM input must be a manifest path, a completed output object, or raw frame mapping"
    )
