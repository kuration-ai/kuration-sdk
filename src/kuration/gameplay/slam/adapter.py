"""Input adapter for an exported SLAM reconstruction manifest."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from ...core.adapters.base import iter_arrays_to_ticks
from ...core.buffer import Tick

from ._paths import local_path
from .export import EXPORT_FORMAT, EXPORT_VERSION
from .schema import FIELD_TO_ARRAY, TRACKING_CONFIDENCE, slam_schema


def _resolve_reference(
    root: Path,
    value: str,
    *,
    allow_remote: bool = False,
) -> str:
    parsed = urlparse(value)
    if parsed.scheme and len(parsed.scheme) > 1:
        if allow_remote:
            return value
        raise ValueError(
            "SLAM trajectory archives must use local paths; "
            f"remote reference {value!r} is unsupported"
        )
    path = Path(value)
    return str(path if path.is_absolute() else root / path)


class SLAMTrajectoryAdapter:
    """Read portable per-clip trajectory archives referenced by a SLAM manifest."""

    supports_streaming = False
    sources_are_paths = False

    def __init__(self, schema=None, *, manifest: str | Path):
        self.manifest_path = local_path(manifest, name="SLAM manifest")
        try:
            document = json.loads(self.manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read SLAM manifest {str(self.manifest_path)!r}") from error
        if not isinstance(document, dict):
            raise ValueError("SLAM manifest must be a JSON object")
        if document.get("format") != EXPORT_FORMAT or document.get("version") != EXPORT_VERSION:
            raise ValueError(
                f"SLAM manifest must declare format={EXPORT_FORMAT!r}, version={EXPORT_VERSION}"
            )
        clips = document.get("clips")
        if not isinstance(clips, dict) or not clips:
            raise ValueError("SLAM manifest 'clips' must be a non-empty object")
        root = self.manifest_path.parent
        self._sources = {}
        self.video_mapping = {}
        for name, entry in clips.items():
            if not isinstance(name, str) or not name or not isinstance(entry, dict):
                raise ValueError("SLAM manifest clips must map non-empty names to objects")
            trajectory = entry.get("trajectory")
            if not isinstance(trajectory, str) or not trajectory:
                raise ValueError(f"SLAM clip {name!r} needs a non-empty trajectory path")
            self._sources[name] = _resolve_reference(root, trajectory)
            video = entry.get("video")
            if video is not None:
                if not isinstance(video, str) or not video:
                    raise ValueError(f"SLAM clip {name!r} video must be a non-empty path or URI")
                self.video_mapping[name] = _resolve_reference(
                    root,
                    video,
                    allow_remote=True,
                )
        generated = slam_schema(sample_hz=document.get("sample_hz"))
        self.schema = generated if schema is None else schema
        missing = sorted(set(FIELD_TO_ARRAY) - set(self.schema.fields))
        if missing:
            raise ValueError(f"SLAM adapter schema is missing required field(s) {missing}")
        self.timebase = document.get("timebase", "frame_index")
        if self.timebase not in {"timestamp", "frame_index"}:
            raise ValueError("SLAM manifest timebase must be 'timestamp' or 'frame_index'")
        self.available_fields = frozenset(FIELD_TO_ARRAY)
        self.available_globals: frozenset[str] = frozenset()
        scene = document.get("scene")
        if scene is not None and not isinstance(scene, dict):
            raise ValueError("SLAM manifest 'scene' must be an object or null")
        self.scene = scene or {}

    def sources(self) -> list[str]:
        return sorted(self._sources)

    def _path(self, source: str) -> str:
        try:
            return self._sources[source]
        except KeyError as error:
            raise ValueError(f"unknown SLAM clip {source!r}; known: {self.sources()}") from error

    def ticks(self, source: str) -> list[Tick]:
        path = self._path(source)
        try:
            with np.load(path, allow_pickle=False) as archive:
                times = np.asarray(archive["time"], dtype=np.float64)
                arrays = {}
                for field, key in FIELD_TO_ARRAY.items():
                    if field == TRACKING_CONFIDENCE and key not in archive.files:
                        arrays[field] = np.ones(len(times), dtype=np.float64)
                    else:
                        arrays[field] = np.asarray(archive[key])
        except (OSError, KeyError, ValueError) as error:
            raise ValueError(f"cannot read SLAM trajectory for clip {source!r}: {path}") from error
        try:
            lengths = {field: len(value) for field, value in arrays.items()}
        except TypeError as error:
            raise ValueError(
                f"SLAM clip {source!r} arrays must have a timestep dimension"
            ) from error
        if len(set(lengths.values())) != 1:
            raise ValueError(f"SLAM clip {source!r} arrays disagree on step count: {lengths}")
        length = next(iter(lengths.values()))
        if length < 1:
            raise ValueError(f"SLAM clip {source!r} must contain at least one trajectory row")
        if times.shape != (length,):
            raise ValueError(f"SLAM clip {source!r} time must have shape ({length},)")
        if not np.isfinite(times).all() or (length > 1 and np.any(np.diff(times) <= 0)):
            raise ValueError(f"SLAM clip {source!r} time must be finite and strictly increasing")
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError(f"SLAM clip {source!r} arrays must contain only finite values")
        rows = list(iter_arrays_to_ticks(self.schema, arrays))
        return [Tick(float(time), row.entities, row.events, row.globals, row.media)
                for time, row in zip(times, rows, strict=True)]

    def check_source(self, source: str) -> None:
        # NPZ is not a streaming format; materializing one trajectory here also validates the
        # timestamp and finite-value contract that cannot be expressed by shape metadata alone.
        self.ticks(source)


def input_slam(
    manifest: str | Path,
    *,
    join_video: bool = False,
    tolerance: float | None = None,
    mismatch: str = "error",
    provider=None,
):
    """Open an exported SLAM manifest as a Kuration pipeline."""
    # Registration is idempotent, which also keeps this direct API safe if an isolated test
    # registry or unusual import order invokes the module explicitly.
    from . import register

    register()
    adapter = SLAMTrajectoryAdapter(manifest=manifest)
    from ...core.pipeline import input

    pipeline = input(adapter=adapter, schema=adapter.schema)
    if not join_video:
        return pipeline
    missing = sorted(set(adapter.sources()) - set(adapter.video_mapping))
    if missing:
        raise ValueError(f"SLAM manifest has no video path for clip(s) {missing}")
    if adapter.timebase == "timestamp":
        if tolerance is None:
            hz = adapter.schema.sample_hz
            if hz is None:
                raise ValueError("timestamp video join needs tolerance when sample_hz is absent")
            tolerance = 0.51 / hz
        return pipeline.join_video(
            adapter.video_mapping,
            on="timestamp",
            tolerance=tolerance,
            mismatch=mismatch,
            provider=provider,
        )
    return pipeline.join_video(
        adapter.video_mapping,
        on="frame_index",
        mismatch=mismatch,
        provider=provider,
    )
