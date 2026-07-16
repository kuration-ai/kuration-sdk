"""Lazy media contracts and the optional CPU video adapter.

The module itself depends only on NumPy. PyAV is imported only when the default provider is
actually asked to probe or decode a video.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Iterator, Mapping, Optional, Protocol, runtime_checkable

import numpy as np

from .buffer import Tick
from .schema import MediaSpec, SchemaSpec


def _iter_provider_refs(provider, uri: str, source_id: str):
    """Call both the minimal ``iter_refs(uri)`` contract and providers accepting source_id."""
    import inspect
    parameters = inspect.signature(provider.iter_refs).parameters
    refs = (provider.iter_refs(uri, source_id=source_id) if "source_id" in parameters
            else provider.iter_refs(uri))
    for ref in refs:
        if ref.source_id == source_id:
            yield ref
        else:
            yield FrameRef(ref.uri, source_id, ref.frame_index, ref.timestamp,
                           ref.stream_index, ref.pts, ref.time_base)


@dataclass(frozen=True)
class FrameRef:
    uri: str
    source_id: str
    frame_index: int
    timestamp: float
    stream_index: int = 0
    pts: Optional[int] = None
    time_base: Optional[tuple[int, int]] = None


@dataclass(frozen=True)
class VideoMetadata:
    uri: str
    width: int
    height: int
    frame_count: Optional[int]
    duration: Optional[float]
    average_rate: Optional[float]
    stream_index: int = 0
    variable_rate: bool = False
    codec: Optional[str] = None


@runtime_checkable
class MediaProvider(Protocol):
    capabilities: Mapping[str, bool]
    def probe(self, uri: str) -> VideoMetadata: ...
    def iter_refs(self, uri: str, *, source_id: Optional[str] = None) -> Iterator[FrameRef]: ...
    def decode_batch(self, refs: Iterable[FrameRef]) -> np.ndarray: ...


def _av():
    try:
        import av
    except ImportError as e:
        raise ImportError("MP4 video support needs the optional extra: "
                          "pip install 'kuration-sdk[video]'") from e
    return av


class PyAVMediaProvider:
    """CPU RGB provider backed by the optional PyAV package."""
    capabilities = {"seeking": True, "variable_frame_rate": True, "streaming": True}

    def probe(self, uri: str) -> VideoMetadata:
        av = _av()
        from .io import local_copy
        with local_copy(uri) as path, av.open(path) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                raise ValueError(f"video {uri!r} contains no video stream")
            rate = float(stream.average_rate) if stream.average_rate else None
            duration = (float(stream.duration * stream.time_base)
                        if stream.duration is not None and stream.time_base else None)
            return VideoMetadata(uri, stream.width, stream.height,
                                 int(stream.frames) if stream.frames else None, duration, rate,
                                 stream.index, False,
                                 getattr(getattr(stream, "codec_context", None), "name", None))

    def iter_refs(self, uri: str, *, source_id: Optional[str] = None) -> Iterator[FrameRef]:
        av = _av()
        from .io import local_copy
        with local_copy(uri) as path, av.open(path) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                raise ValueError(f"video {uri!r} contains no video stream")
            previous = float("-inf")
            for i, frame in enumerate(container.decode(stream)):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError(f"video {uri!r}: decoded frame {i} has no presentation time")
                timestamp = float(frame.pts * frame.time_base)
                if timestamp < previous:
                    raise ValueError(f"video {uri!r}: presentation timestamps are not ordered")
                previous = timestamp
                tb = Fraction(frame.time_base)
                yield FrameRef(uri, source_id or uri, i, timestamp, stream.index, frame.pts,
                               (tb.numerator, tb.denominator))

    def decode_batch(self, refs: Iterable[FrameRef]) -> np.ndarray:
        refs = list(refs)
        if not refs:
            return np.empty((0,), dtype=np.uint8)
        uris = {r.uri for r in refs}
        if len(uris) != 1:
            chunks = [self.decode_batch([r]) for r in refs]
            return np.concatenate(chunks, axis=0)
        wanted: dict[int, list[int]] = {}
        for i, ref in enumerate(refs):
            wanted.setdefault(ref.frame_index, []).append(i)
        out: list[Optional[np.ndarray]] = [None] * len(refs)
        av = _av()
        from .io import local_copy
        uri = refs[0].uri
        with local_copy(uri) as path, av.open(path) as container:
            stream = next((s for s in container.streams
                           if s.type == "video" and s.index == refs[0].stream_index), None)
            if stream is None:
                raise ValueError(f"video {uri!r}: stream {refs[0].stream_index} is unavailable")
            for i, frame in enumerate(container.decode(stream)):
                positions = wanted.get(i)
                if positions:
                    value = frame.to_ndarray(format="rgb24")
                    for pos in positions:
                        out[pos] = value
                if all(x is not None for x in out):
                    break
        missing = [refs[i].frame_index for i, x in enumerate(out) if x is None]
        if missing:
            raise ValueError(f"video {uri!r}: could not decode frame(s) {missing}")
        shapes = {x.shape for x in out if x is not None}
        if len(shapes) != 1:
            raise ValueError(f"video {uri!r}: frame shape changed within a decode batch: {shapes}")
        return np.stack(out)  # type: ignore[arg-type]


class VideoAdapter:
    """One video source becomes ticks containing a lazy ``media:frame`` reference."""
    supports_streaming = True

    def __init__(self, sources: Iterable[str], *, provider: Optional[MediaProvider] = None,
                 column: str = "frame"):
        self._sources = list(sources)
        self.provider = provider or PyAVMediaProvider()
        self.column = column
        if not self._sources:
            raise ValueError("VideoAdapter needs at least one source")
        first = self.provider.probe(self._sources[0])
        self.schema = SchemaSpec.from_dict({
            "name": "video", "fields": {}, "sample_hz": first.average_rate,
            "media": {column: {"kind": "video-frame", "dtype": "uint8",
                                "shape": [first.height, first.width, 3], "layout": "HWC",
                                "color_space": "RGB", "timing": "pts",
                                "capabilities": dict(self.provider.capabilities)}}})

    def sources(self) -> list[str]:
        return list(self._sources)

    def iter_ticks(self, source_uri: str):
        for ref in _iter_provider_refs(self.provider, source_uri, source_uri):
            yield Tick(ref.timestamp, {0: {}}, media={self.column: ref})

    def ticks(self, source_uri: str):
        return self.iter_ticks(source_uri)

    def check_source(self, source_uri: str) -> None:
        meta = self.provider.probe(source_uri)
        spec = self.schema.media[self.column]
        if (meta.height, meta.width, 3) != spec.shape:
            raise ValueError(f"video {source_uri!r}: frame shape {(meta.height, meta.width, 3)} "
                             f"does not match declared {spec.shape}")


class JoinedMediaAdapter:
    """Align one video with every source from an existing adapter."""
    supports_streaming = True

    def __init__(self, adapter, videos: Mapping[str, str], *, provider=None, column="frame",
                 on="timestamp", tolerance=None, mismatch="error"):
        if on not in {"timestamp", "frame_index"}:
            raise ValueError("join_video on= must be 'timestamp' or 'frame_index'")
        if mismatch not in {"error", "drop"}:
            raise ValueError("join_video mismatch= must be 'error' or 'drop'")
        if on == "timestamp" and (tolerance is None or tolerance < 0):
            raise ValueError("timestamp video alignment needs a non-negative tolerance")
        self.adapter, self.videos = adapter, dict(videos)
        self.provider = provider or PyAVMediaProvider()
        self.column, self.on, self.tolerance, self.mismatch = column, on, tolerance, mismatch
        if not self.videos:
            raise ValueError("join_video resolved to an empty source mapping")
        first = self.provider.probe(next(iter(self.videos.values())))
        raw = adapter.schema.to_dict()
        media = raw.setdefault("media", {})
        if column in media:
            raise ValueError(f"media column {column!r} already exists")
        media[column] = {"kind": "video-frame", "dtype": "uint8",
                         "shape": [first.height, first.width, 3], "layout": "HWC",
                         "color_space": "RGB", "timing": "pts",
                         "capabilities": dict(self.provider.capabilities)}
        self.schema = SchemaSpec.from_dict(raw)

    def _uri(self, source):
        if source not in self.videos:
            raise ValueError(f"source {source!r} has no resolved video mapping")
        return self.videos[source]

    def iter_ticks(self, source):
        # ``ticks`` is the adapter contract and may itself be lazy. Some adapters inherit a
        # file-oriented ``iter_ticks`` while overriding ``ticks`` for in-memory sources.
        ticks = iter(self.adapter.ticks(source))
        refs = iter(_iter_provider_refs(self.provider, self._uri(source), source))
        if self.on == "frame_index":
            count = 0
            for i, tick in enumerate(ticks):
                try:
                    ref = next(refs)
                except StopIteration:
                    if self.mismatch == "drop": break
                    raise ValueError(f"source {source!r}: data has more rows than video frames")
                count = i + 1
                yield Tick(tick.time, tick.entities, tick.events, tick.globals,
                           {**tick.media, self.column: ref})
            if self.mismatch == "error" and next(refs, None) is not None:
                raise ValueError(f"source {source!r}: data has {count} rows but video has more frames")
            return
        previous = None
        current = next(refs, None)
        if current is None:
            raise ValueError(f"video for source {source!r} has no decoded frames")
        last_target = float("-inf")
        for tick in ticks:
            try:
                target = float(tick.time)
            except (TypeError, ValueError) as e:
                raise ValueError(f"source {source!r}: timestamp alignment needs numeric tick times") from e
            if target < last_target:
                raise ValueError(f"source {source!r}: data timestamps are not ordered")
            last_target = target
            while current is not None and current.timestamp < target:
                previous, current = current, next(refs, None)
            choices = [ref for ref in (previous, current) if ref is not None]
            best = min(choices, key=lambda ref: (abs(ref.timestamp - target), ref.frame_index))
            drift = abs(best.timestamp - target)
            if drift > self.tolerance:
                if self.mismatch == "drop": continue
                raise ValueError(f"source {source!r}: nearest video frame is {drift:.6g}s away, "
                                 f"beyond tolerance {self.tolerance}")
            yield Tick(tick.time, tick.entities, tick.events, tick.globals,
                       {**tick.media, self.column: best})

    def ticks(self, source):
        return self.iter_ticks(source)

    def check_source(self, source):
        self.provider.probe(self._uri(source))
        sample = list(self.iter_ticks(source))
        if not sample:
            raise ValueError(f"source {source!r}: alignment produced no ticks")


def media_provider_for(adapter) -> Optional[MediaProvider]:
    return getattr(adapter, "provider", None)
