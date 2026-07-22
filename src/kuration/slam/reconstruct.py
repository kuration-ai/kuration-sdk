"""Run a live ``SLAMOperator`` as a compositional operator over a video Pipeline."""
from __future__ import annotations

from pathlib import Path

from ..media import VideoAdapter
from .adapter import SLAMTrajectoryAdapter
from .export import export_slam_output
from .runner import decode_video_clips


RECONSTRUCT_NAME = "aintropy/slam-reconstruct"

RECONSTRUCT_PARAMETERS = {
    "type": "object",
    "properties": {
        # Deliberately untyped: carries a live SLAMOperator instance, not a JSON value.
        "pipeline": {},
        "out_dir": {"type": "string", "minLength": 1},
        "frame_skip": {"type": "integer", "minimum": 1},
        "batch_size": {"type": "integer", "minimum": 1},
        "color_space": {"type": "string", "enum": ["RGB", "BGR"]},
        "provider": {},
        "join_video": {"type": "boolean"},
        "tolerance": {"type": "number", "exclusiveMinimum": 0},
        "mismatch": {"type": "string", "enum": ["error", "drop"]},
        "progress": {},
    },
    "required": ["pipeline", "out_dir"],
    "additionalProperties": False,
}


def slam_reconstruct(
    dataset,
    *,
    pipeline,
    out_dir,
    frame_skip: int = 1,
    batch_size: int = 64,
    color_space: str = "RGB",
    provider=None,
    join_video: bool = False,
    tolerance: float | None = None,
    mismatch: str = "error",
    progress=None,
):
    """Reconstruct a video-backed Pipeline in place with an already-built ``SLAMOperator``.

    ``dataset`` must come from ``input(<video paths>)`` (a ``VideoAdapter``-backed Pipeline).
    ``pipeline`` is constructed exactly as it is standalone (``SLAMOperator(depth_model=...)``);
    this operator only decodes frames, calls it, and exports the result. The returned Pipeline
    is swapped onto the resulting SLAM trajectory manifest, so ``.window()``/``.transform()``/
    ``.curate()`` keep working unmodified, the same as after ``input_slam()``.
    """
    if not isinstance(dataset.adapter, VideoAdapter):
        raise ValueError(
            f"{RECONSTRUCT_NAME} needs a VideoAdapter-backed Pipeline; build it with "
            f"input(<video paths>), got {type(dataset.adapter).__name__}"
        )
    if not callable(pipeline):
        raise ValueError(f"{RECONSTRUCT_NAME}: pipeline= must be a callable SLAMOperator instance")

    names = [Path(source).stem for source in dataset.sources]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"{RECONSTRUCT_NAME} needs unique clip names from video filenames; "
            f"duplicate stem(s) {duplicates}"
        )
    videos = dict(zip(names, dataset.sources))

    frames, timestamps = decode_video_clips(
        videos, frame_skip=frame_skip, batch_size=batch_size, color_space=color_space,
        provider=provider, progress=progress,
    )
    output = pipeline(frames)
    manifest = export_slam_output(output, out_dir, timestamps=timestamps, videos=videos)

    adapter = SLAMTrajectoryAdapter(manifest=manifest)
    result = dataset._next(
        "slam_reconstruct", out_dir=str(out_dir), frame_skip=frame_skip, batch_size=batch_size,
        color_space=color_space, join_video=join_video,
    )
    result.adapter = adapter
    result.schema = adapter.schema
    result.sources = adapter.sources()
    result._in_memory = False
    if not join_video:
        return result

    missing = sorted(set(adapter.sources()) - set(adapter.video_mapping))
    if missing:
        raise ValueError(f"{RECONSTRUCT_NAME}: SLAM manifest has no video path for clip(s) {missing}")
    if adapter.timebase == "timestamp":
        if tolerance is None:
            hz = adapter.schema.sample_hz
            if hz is None:
                raise ValueError(
                    f"{RECONSTRUCT_NAME}: timestamp video join needs tolerance= when sample_hz "
                    "is absent"
                )
            tolerance = 0.51 / hz
        return result.join_video(
            adapter.video_mapping, on="timestamp", tolerance=tolerance, mismatch=mismatch,
            provider=provider,
        )
    return result.join_video(
        adapter.video_mapping, on="frame_index", mismatch=mismatch, provider=provider,
    )
