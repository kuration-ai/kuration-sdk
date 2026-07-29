"""Optional global SLAM execution before bounded Kuration curation begins."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ._paths import local_path
from .export import (
    EXPORT_FORMAT,
    EXPORT_VERSION,
    _atomic_write_text,
    export_slam_output,
)


_CACHE_FORMAT = "kuration-slam-prepare-cache"
_CACHE_VERSION = 2


def _callable_identity(value: object) -> str:
    if isinstance(value, str):
        return value
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}:{name}"


def _model_identity(value: object, explicit: str | None, *, name: str) -> dict | None:
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise ValueError(f"{name}_id must be a non-empty string or None")
        return {"id": explicit.strip(), "source": "explicit"}
    if value is None:
        return None
    if isinstance(value, str):
        return {"id": value, "source": "model-name"}
    return {"id": _callable_identity(type(value)), "source": "python-type"}


def _stable_options(options: Mapping[str, object]) -> dict[str, object]:
    """Record plain runtime options verbatim, rejecting anything unrecordable.

    These options are both provenance and part of the automatic-resume cache key, so silently
    dropping an option that cannot be JSON-encoded would let two runs that differ *only* in that
    option share one cache entry (and one recorded identity). Opaque values are refused the same
    way opaque model identities are (`_validate_automatic_resume_identity`).
    """
    stable: dict[str, object] = {}
    for key, value in sorted(options.items()):
        if value is None or isinstance(value, (str, int, float, bool)):
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(
                    f"SLAM runtime option {key!r} must be JSON-safe and finite"
                )
            stable[key] = value
            continue
        if isinstance(value, Path):
            stable[key] = str(value)
            continue
        if isinstance(value, tuple):
            value = list(value)
        try:
            stable[key] = json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"SLAM runtime option {key!r} of type {type(value).__name__} cannot be recorded "
                "in reconstruction provenance or the automatic-resume cache key; pass a "
                "JSON-safe primitive form instead (e.g. devices=['cuda:0'] rather than "
                "torch.device objects), or an explicit cache_key"
            ) from error
    return stable


def slam_provenance(options: Mapping[str, object]) -> dict[str, object]:
    """Return stable, JSON-safe reconstruction identity without object reprs."""
    factory = (
        options.get("runtime_factory")
        or "kuration.gameplay.slam._runtime:SLAMOperator"
    )
    depth_id = options.get("depth_model_id")
    embedding_id = options.get("embedding_model_id")
    if depth_id is not None and not isinstance(depth_id, str):
        raise ValueError("depth_model_id must be a non-empty string or None")
    if embedding_id is not None and not isinstance(embedding_id, str):
        raise ValueError("embedding_model_id must be a non-empty string or None")
    depth = _model_identity(
        options.get("depth_model"), depth_id, name="depth_model"
    )
    embedding = _model_identity(
        options.get("embedding_model"),
        embedding_id,
        name="embedding_model",
    )
    ignored = {
        "depth_model",
        "depth_model_id",
        "embedding_model",
        "embedding_model_id",
        "embeddings",
        "runtime_factory",
        "progress",
    }
    document: dict[str, object] = {
        "engine": _callable_identity(factory),
        "models": {key: value for key, value in {"depth": depth, "embedding": embedding}.items()
                   if value is not None},
        "options": _stable_options({key: value for key, value in options.items() if key not in ignored}),
    }
    if options.get("embeddings") is not None:
        document["embeddings"] = {
            "id": options.get("embedding_model_id") or "precomputed",
            "source": "precomputed",
        }
    return document


def _provenance_key(provenance: Mapping[str, object]) -> str:
    payload = json.dumps(dict(provenance), allow_nan=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _array_identity(value: object) -> dict[str, object]:
    array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    digest = sha256()
    digest.update(memoryview(contiguous).cast("B"))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": digest.hexdigest(),
    }


def _identity_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _array_identity(value)
    if isinstance(value, Mapping):
        return {str(key): _identity_value(item) for key, item in sorted(
            value.items(), key=lambda pair: str(pair[0])
        )}
    if isinstance(value, (list, tuple)):
        return [_identity_value(item) for item in value]
    return {"python_type": _callable_identity(type(value))}


def _source_identity(source: object) -> dict[str, object]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        return {"kind": "manifest", "path": str(path.resolve())}
    if isinstance(source, Mapping):
        if source and all(isinstance(value, (str, Path)) for value in source.values()):
            clips = {}
            for name, value in sorted(source.items(), key=lambda pair: str(pair[0])):
                if not isinstance(name, str) or not name:
                    raise ValueError("video clip names must be non-empty strings")
                uri = str(value)
                entry: dict[str, object] = {"uri": uri}
                path = Path(uri)
                if path.is_file():
                    stat = path.stat()
                    entry.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
                clips[str(name)] = entry
            return {"kind": "videos", "clips": clips}
        frames = _validated_frames(source)
        return {
            "kind": "frames",
            "clips": {name: _array_identity(value) for name, value in sorted(frames.items())},
        }
    trajectories = getattr(source, "trajectories", None)
    if isinstance(trajectories, Mapping):
        return {
            "kind": "completed-output",
            "trajectories": {
                str(name): _array_identity(value)
                for name, value in sorted(trajectories.items(), key=lambda pair: str(pair[0]))
            },
        }
    return {"kind": "python-object", "type": _callable_identity(type(source))}


def _validate_automatic_resume_identity(options: Mapping[str, object]) -> None:
    """Require stable caller identities for opaque runtime state used by automatic resume."""
    depth_model = options.get("depth_model")
    if depth_model is not None and not isinstance(depth_model, str) \
            and not options.get("depth_model_id"):
        raise ValueError(
            "automatic SLAM resume with a depth-model object requires depth_model_id; "
            "pass a stable model revision or an explicit cache_key"
        )
    embedding_model = options.get("embedding_model")
    embeddings = options.get("embeddings")
    if (embedding_model is not None or embeddings is not None) \
            and not options.get("embedding_model_id"):
        raise ValueError(
            "automatic SLAM resume with an embedding model or precomputed embeddings requires "
            "embedding_model_id; pass a stable model/data revision or an explicit cache_key"
        )


@dataclass(frozen=True)
class SLAMReconstruction:
    """A portable reconstruction artifact and its Kuration pipeline entry point."""

    manifest: Path

    def to_pipeline(
        self,
        *,
        join_video: bool = False,
        tolerance: float | None = None,
        mismatch: str = "error",
        provider=None,
    ):
        from .adapter import input_slam

        return input_slam(
            self.manifest,
            join_video=join_video,
            tolerance=tolerance,
            mismatch=mismatch,
            provider=provider,
        )

    def __fspath__(self) -> str:
        return str(self.manifest)


# Keyword arguments the default runtime (``_runtime.SLAMOperator``) accepts beyond the model
# inputs named above. Kept static so validation never imports torch; pinned against the real
# signature in tests/test_slam_regressions.py.
DEFAULT_RUNTIME_OPTIONS = frozenset({
    "output_pointcloud", "devices", "window_size", "overlap", "bridge_size",
    "process_res", "conf_thresh", "num_max_points", "top_k",
    "similarity_threshold", "seed", "verbose",
})


def _validate_runtime_options(runtime_factory, runtime_options: Mapping[str, Any]) -> None:
    """Reject options the default runtime cannot accept, before any cached path can hide them.

    A custom ``runtime_factory`` owns its own signature, so only the default factory is
    checked. Without this, a typo of a caller-level keyword (``tolerence=``) lands here and is
    silently ignored whenever an existing manifest or cache hit skips runtime construction.
    """
    if runtime_factory is not None or not runtime_options:
        return
    unknown = sorted(set(runtime_options) - DEFAULT_RUNTIME_OPTIONS)
    if unknown:
        raise TypeError(
            f"unknown SLAM runtime option(s) {unknown}; the default runtime accepts "
            f"{sorted(DEFAULT_RUNTIME_OPTIONS)}"
        )


class SLAMReconstructor:
    """Video/frame-to-reconstruction workflow that feeds Kuration's curation pipeline.

    Model construction remains lazy when ``depth_model`` is a string. Use :meth:`run` for the
    notebook-style in-memory output, or :meth:`reconstruct` for a portable manifest that opens
    directly as a normal Kuration pipeline.
    """

    def __init__(
        self,
        depth_model=None,
        *,
        embedding_model=None,
        embeddings=None,
        depth_model_id: str | None = None,
        embedding_model_id: str | None = None,
        runtime_factory: Callable | str | None = None,
        **runtime_options,
    ) -> None:
        if embedding_model is not None and embeddings is not None:
            raise ValueError("pass embedding_model or embeddings, not both")
        _validate_runtime_options(runtime_factory, runtime_options)
        self.runner_options: dict[str, Any] = {
            "depth_model": depth_model,
            "embedding_model": embedding_model,
            "embeddings": embeddings,
            "depth_model_id": depth_model_id,
            "embedding_model_id": embedding_model_id,
            "runtime_factory": runtime_factory,
            **runtime_options,
        }
        self.provenance = slam_provenance(self.runner_options)

    def run(self, data: Mapping[str, object], *, progress=None):
        options = dict(self.runner_options)
        options["progress"] = progress
        output = run_slam_reconstruction(data, **options)
        try:
            output.provenance = self.provenance
        except (AttributeError, TypeError):
            pass
        return output

    __call__ = run

    def reconstruct(
        self,
        source,
        out_dir: str | Path | None = None,
        *,
        run_if_needed: bool = True,
        decoder_options: Mapping[str, object] | None = None,
        export_options: Mapping[str, object] | None = None,
        resume: bool = False,
        cache_key: str | None = None,
        progress=None,
        preflight: bool = True,
        require_gpu: bool = False,
    ) -> SLAMReconstruction:
        exports = dict(export_options or {})
        exports.setdefault("provenance", self.provenance)
        resolved_cache_key = cache_key
        if resume and resolved_cache_key is None:
            _validate_automatic_resume_identity(self.runner_options)
            resolved_cache_key = _provenance_key({
                "runtime": self.provenance,
                "source": _source_identity(source),
                "decoder": _identity_value(dict(decoder_options or {})),
                "export": _identity_value(exports),
            })
        manifest = prepare_slam_input(
            source,
            out_dir,
            run_if_needed=run_if_needed,
            runner_options=self.runner_options,
            decoder_options=decoder_options,
            export_options=exports,
            resume=resume,
            cache_key=resolved_cache_key,
            progress=progress,
            preflight=preflight,
            require_gpu=require_gpu,
        )
        return SLAMReconstruction(manifest)


def _emit(progress, phase: str, event: str, **fields) -> None:
    if progress is not None:
        progress({"phase": phase, "event": event, **fields})


def _runtime_factory(factory: Callable | str | None) -> Callable:
    if factory is None:
        factory = "kuration.gameplay.slam._runtime:SLAMOperator"
    if isinstance(factory, str):
        module_name, separator, attribute = factory.partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError("runtime_factory strings must use 'module:attribute' syntax")
        try:
            module = importlib.import_module(module_name)
            factory = getattr(module, attribute)
        except (ImportError, AttributeError) as error:
            raise ImportError(
                f"cannot load SLAM runtime factory {module_name}:{attribute}; install or "
                "provide the SLAMOperator implementation, or pass runtime_factory= explicitly"
            ) from error
    if not callable(factory):
        raise TypeError("runtime_factory must be callable or an import string")
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


def run_slam_reconstruction(
    data: Mapping[str, object],
    *,
    depth_model=None,
    embedding_model=None,
    embeddings=None,
    depth_model_id: str | None = None,
    embedding_model_id: str | None = None,
    runtime_factory: Callable | str | None = None,
    progress=None,
    **runtime_options,
):
    """Run the notebook-style ``SLAMOperator`` over raw in-memory clip frames.

    Models are injected by the caller so this module has no mandatory Torch, Depth Anything,
    SALAD, or FAISS dependency. With no explicit factory, the function lazily imports the
    private ``kuration.gameplay.slam._runtime.SLAMOperator``.
    """
    _validate_runtime_options(runtime_factory, runtime_options)
    frames = _validated_frames(data)
    if depth_model is None:
        raise ValueError("running SLAM needs depth_model; precomputed SLAM input does not")
    if embedding_model is not None and embeddings is not None:
        raise ValueError("pass embedding_model or embeddings, not both")
    _model_identity(depth_model, depth_model_id, name="depth_model")
    _model_identity(embedding_model, embedding_model_id, name="embedding_model")
    options = dict(runtime_options)
    options["depth_model"] = depth_model
    if embedding_model is not None:
        options["embedding_model"] = embedding_model
    if embeddings is not None:
        options["embeddings"] = embeddings
    runtime = _runtime_factory(runtime_factory)(**options)
    if not callable(runtime):
        raise TypeError("runtime_factory must construct a callable SLAM runtime")
    _emit(progress, "slam-reconstruction", "start", total=len(frames))
    try:
        output = runtime(frames)
    except Exception:
        _emit(progress, "slam-reconstruction", "error", total=len(frames))
        raise
    if not isinstance(getattr(output, "trajectories", None), Mapping) \
            or not isinstance(getattr(output, "poses", None), Mapping):
        raise ValueError("SLAM runtime output must expose trajectory and pose mappings")
    _emit(
        progress,
        "slam-reconstruction",
        "complete",
        current=len(frames),
        total=len(frames),
        detail=f"{len(output.trajectories)} reconstructed clips",
    )
    provenance_options = {
        **runtime_options,
        "depth_model": depth_model,
        "embedding_model": embedding_model,
        "embeddings": embeddings,
        "depth_model_id": depth_model_id,
        "embedding_model_id": embedding_model_id,
        "runtime_factory": runtime_factory,
    }
    try:
        output.provenance = slam_provenance(provenance_options)
    except (AttributeError, TypeError):
        pass
    return output


def _existing_manifest(source) -> Path | None:
    if not isinstance(source, (str, Path)):
        return None
    path = local_path(source, name="SLAM input path")
    if path.is_dir():
        path = path / "manifest.json"
    if not path.is_file():
        raise ValueError(f"SLAM input path does not contain a manifest: {path}")
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read SLAM manifest {str(path)!r}") from error
    if not isinstance(document, dict):
        raise ValueError("SLAM manifest must be a JSON object")
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
    if metadata.get("format") != _CACHE_FORMAT \
            or metadata.get("version") != _CACHE_VERSION \
            or metadata.get("cache_key") != cache_key:
        return None
    try:
        manifest_digest = sha256(manifest_path.read_bytes()).hexdigest()
    except OSError:
        return None
    if metadata.get("manifest_sha256") != manifest_digest:
        return None
    return _existing_manifest(manifest_path)


def _write_cache(out_dir: str | Path, cache_key: str) -> None:
    root = Path(out_dir)
    target = root / ".kuration-slam-prepare.json"
    manifest_path = root / "manifest.json"
    document = {
        "format": _CACHE_FORMAT,
        "version": _CACHE_VERSION,
        "cache_key": cache_key,
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
    }
    _atomic_write_text(target, json.dumps(document, indent=2, sort_keys=True) + "\n")


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
    - A completed SLAM output object is exported without constructing a runtime.
    - A raw ``{clip_name: frames}`` mapping runs SLAM only when ``run_if_needed`` is true.
    """
    if not isinstance(resume, bool) or not isinstance(preflight, bool):
        raise ValueError("resume and preflight must be true or false")
    if cache_key is not None and (not isinstance(cache_key, str) or not cache_key.strip()):
        raise ValueError("cache_key must be a non-empty string or None")
    # Validate before the manifest/cache short-circuits below, which never build a runtime and
    # would otherwise ignore an unusable option without a word.
    runner = dict(runner_options or {})
    named = {"depth_model", "embedding_model", "embeddings", "depth_model_id",
             "embedding_model_id", "runtime_factory", "progress"}
    _validate_runtime_options(
        runner.get("runtime_factory"),
        {key: value for key, value in runner.items() if key not in named},
    )
    _emit(progress, "slam-prepare", "start", total=None)
    manifest = _existing_manifest(source)
    if manifest is not None:
        _emit(progress, "slam-cache", "complete", detail="existing manifest reused")
        _emit(progress, "slam-prepare", "complete", detail=str(manifest))
        return manifest
    if out_dir is None:
        raise ValueError("out_dir is required when the input is not an existing SLAM manifest")
    out_dir = local_path(out_dir, name="SLAM out_dir")
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
    export_kwargs: dict[str, Any] = dict(export_options or {})
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
        frames: Mapping[str, object]
        if source and all(isinstance(value, (str, Path)) for value in source.values()):
            decode_kwargs: dict[str, Any] = dict(decoder_options or {})
            decode_kwargs.setdefault("progress", progress)
            frames, timestamps = decode_video_clips(source, **decode_kwargs)
            export_kwargs.setdefault("timestamps", timestamps)
            export_kwargs.setdefault("videos", {name: str(path) for name, path in source.items()})
        else:
            frames = dict(source)
        run_kwargs: dict[str, Any] = dict(runner_options or {})
        run_kwargs.setdefault("progress", progress)
        output = run_slam_reconstruction(frames, **run_kwargs)
        export_kwargs.setdefault("provenance", getattr(output, "provenance", None))
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
