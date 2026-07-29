"""Lazy media contracts and the optional CPU video adapter.

The module itself depends only on NumPy. PyAV is imported only when the default provider is
actually asked to probe or decode a video.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any, Iterable, Iterator, Mapping, Optional, Protocol, runtime_checkable

import numpy as np

from .buffer import Tick
from .schema import SchemaSpec


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
    """CPU RGB provider backed by PyAV, with bounded reuse of staged remote videos."""
    capabilities = {"seeking": True, "variable_frame_rate": True, "streaming": True}

    def __init__(self, *, remote_cache_size: int = 4):
        if (not isinstance(remote_cache_size, int) or isinstance(remote_cache_size, bool)
                or remote_cache_size < 0):
            raise ValueError("remote_cache_size must be a non-negative integer")
        from collections import OrderedDict
        from threading import RLock
        self.remote_cache_size = remote_cache_size
        self._remote_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._remote_cache_lock = RLock()

    def __getstate__(self):
        # Worker processes receive their own secure cache; host temp paths and locks are not
        # portable or pickleable.
        return {"remote_cache_size": self.remote_cache_size}

    def __setstate__(self, state):
        PyAVMediaProvider.__init__(self, remote_cache_size=state["remote_cache_size"])

    def _download_remote(self, uri: str) -> str:
        import os
        import shutil
        import tempfile
        from pathlib import Path
        from urllib.parse import urlsplit
        from .io import open_file
        suffix = Path(urlsplit(uri).path).suffix
        handle = tempfile.NamedTemporaryFile(prefix="kuration-media-", suffix=suffix,
                                             delete=False)
        try:
            with open_file(uri, "rb") as source:
                shutil.copyfileobj(source, handle, length=1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
            return handle.name
        except Exception:
            handle.close()
            try:
                os.unlink(handle.name)
            except FileNotFoundError:
                pass
            raise
        finally:
            if not handle.closed:
                handle.close()

    def _evict_remote_locked(self) -> None:
        import os
        while len(self._remote_cache) > self.remote_cache_size:
            victim = next((uri for uri, entry in self._remote_cache.items()
                           if entry["uses"] == 0), None)
            if victim is None:
                return
            entry = self._remote_cache.pop(victim)
            try:
                os.unlink(entry["path"])
            except FileNotFoundError:
                pass

    def _local_path(self, uri: str):
        from contextlib import contextmanager

        @contextmanager
        def resolved():
            from .io import is_remote, local_copy
            if not is_remote(uri) or self.remote_cache_size == 0:
                with local_copy(uri) as path:
                    yield path
                return
            with self._remote_cache_lock:
                entry = self._remote_cache.get(uri)
                if entry is None:
                    entry = {"path": self._download_remote(uri), "uses": 0}
                    self._remote_cache[uri] = entry
                else:
                    self._remote_cache.move_to_end(uri)
                entry["uses"] += 1
            try:
                yield entry["path"]
            finally:
                with self._remote_cache_lock:
                    entry["uses"] -= 1
                    self._evict_remote_locked()

        return resolved()

    def close(self) -> None:
        """Release every staged remote video owned by this provider."""
        import os
        cache = getattr(self, "_remote_cache", None)
        lock = getattr(self, "_remote_cache_lock", None)
        if cache is None or lock is None:
            return
        with lock:
            for entry in cache.values():
                try:
                    os.unlink(entry["path"])
                except FileNotFoundError:
                    pass
            cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def probe(self, uri: str) -> VideoMetadata:
        av = _av()
        with self._local_path(uri) as path, av.open(path) as container:
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
        with self._local_path(uri) as path, av.open(path) as container:
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
            grouped: dict[str, list[tuple[int, FrameRef]]] = {}
            for position, ref in enumerate(refs):
                grouped.setdefault(ref.uri, []).append((position, ref))
            values: list[Optional[np.ndarray]] = [None] * len(refs)
            for group in grouped.values():
                decoded = self.decode_batch([ref for _, ref in group])
                for (position, _), value in zip(group, decoded):
                    values[position] = value
            return np.stack(values)  # type: ignore[arg-type]
        wanted: dict[int, list[int]] = {}
        for i, ref in enumerate(refs):
            wanted.setdefault(ref.frame_index, []).append(i)
        out: list[Optional[np.ndarray]] = [None] * len(refs)
        av = _av()
        uri = refs[0].uri
        with self._local_path(uri) as path, av.open(path) as container:
            stream = next((s for s in container.streams
                           if s.type == "video" and s.index == refs[0].stream_index), None)
            if stream is None:
                raise ValueError(f"video {uri!r}: stream {refs[0].stream_index} is unavailable")
            wanted_pts = ({int(ref.pts): positions for ref, positions in
                           ((refs[indexes[0]], indexes) for indexes in wanted.values())
                           if ref.pts is not None})
            use_pts = len(wanted_pts) == len(wanted)
            if use_pts:
                container.seek(min(wanted_pts), stream=stream, backward=True, any_frame=False)
            for i, frame in enumerate(container.decode(stream)):
                positions = wanted_pts.get(int(frame.pts)) if use_pts and frame.pts is not None \
                    else wanted.get(i)
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
                 column: str = "frame", rate_tolerance: float = 1e-9):
        self._sources = list(sources)
        self.provider = provider or PyAVMediaProvider()
        self.column = column
        if (isinstance(rate_tolerance, bool) or not isinstance(rate_tolerance, (int, float))
                or not math.isfinite(rate_tolerance) or rate_tolerance < 0):
            raise ValueError("VideoAdapter rate_tolerance must be a non-negative finite number")
        self._rate_tolerance = float(rate_tolerance)
        if not self._sources:
            raise ValueError("VideoAdapter needs at least one source")
        first = self.provider.probe(self._sources[0])
        self.schema = SchemaSpec.from_dict({
            "name": "video", "fields": {}, "sample_hz": first.average_rate,
            "media": {column: {"kind": "video-frame", "dtype": "uint8",
                                "shape": [first.height, first.width, 3], "layout": "HWC",
                                "color_space": "RGB", "timing": "pts",
                                "capabilities": dict(self.provider.capabilities)}}})
        for source in self._sources[1:]:
            self._validate_metadata(self.provider.probe(source))

    def _validate_metadata(self, meta: VideoMetadata) -> None:
        spec = self.schema.media[self.column]
        if (meta.height, meta.width, 3) != spec.shape:
            raise ValueError(f"video {meta.uri!r}: frame shape "
                             f"{(meta.height, meta.width, 3)} does not match declared "
                             f"{spec.shape}")
        expected_rate = self.schema.sample_hz
        actual_rate = meta.average_rate
        rates_match = (expected_rate is None and actual_rate is None)
        if expected_rate is not None and actual_rate is not None:
            rates_match = math.isclose(expected_rate, actual_rate,
                                       rel_tol=self._rate_tolerance, abs_tol=0.0)
        if not rates_match:
            raise ValueError(f"video {meta.uri!r}: average frame rate {actual_rate!r} does not "
                             f"match the dataset rate {expected_rate!r} "
                             f"(rate_tolerance={self._rate_tolerance!r})")

    def sources(self) -> list[str]:
        return list(self._sources)

    def iter_ticks(self, source_uri: str):
        for ref in _iter_provider_refs(self.provider, source_uri, source_uri):
            yield Tick(ref.timestamp, {0: {}}, media={self.column: ref})

    def ticks(self, source_uri: str):
        return self.iter_ticks(source_uri)

    def check_source(self, source_uri: str) -> None:
        self._validate_metadata(self.provider.probe(source_uri))


class JoinedMediaAdapter:
    """Align one video with every source from an existing adapter."""
    supports_streaming = True

    def __init__(self, adapter, videos: Mapping[str, str], *, provider=None, column="frame",
                 on="timestamp", tolerance=None, mismatch="error"):
        if on not in {"timestamp", "frame_index"}:
            raise ValueError("join_video on= must be 'timestamp' or 'frame_index'")
        if mismatch not in {"error", "drop"}:
            raise ValueError("join_video mismatch= must be 'error' or 'drop'")
        if on == "timestamp" and (
                not isinstance(tolerance, (int, float, np.integer, np.floating))
                or isinstance(tolerance, (bool, np.bool_))
                or not math.isfinite(float(tolerance)) or tolerance < 0):
            raise ValueError("timestamp video alignment needs a finite non-negative tolerance")
        self.adapter, self.videos = adapter, dict(videos)
        cls = type(adapter)
        overrides_ticks_only = "ticks" in cls.__dict__ and "iter_ticks" not in cls.__dict__
        self.supports_streaming = (getattr(adapter, "supports_streaming", True)
                                   and hasattr(adapter, "iter_ticks")
                                   and not overrides_ticks_only)
        self.provider = provider or PyAVMediaProvider()
        self.column, self.on, self.tolerance, self.mismatch = column, on, tolerance, mismatch
        if not self.videos:
            raise ValueError("join_video resolved to an empty source mapping")
        first_uri = next(iter(self.videos.values()))
        first = self.provider.probe(first_uri)
        raw = adapter.schema.to_dict()
        media = raw.setdefault("media", {})
        if column in media:
            raise ValueError(f"media column {column!r} already exists")
        media[column] = {"kind": "video-frame", "dtype": "uint8",
                         "shape": [first.height, first.width, 3], "layout": "HWC",
                         "color_space": "RGB", "timing": "pts",
                         "capabilities": dict(self.provider.capabilities)}
        self.schema = SchemaSpec.from_dict(raw)
        self._validate_metadata(first)
        for uri in dict.fromkeys(self.videos.values()):
            if uri != first_uri:
                self._validate_metadata(self.provider.probe(uri))

    def _validate_metadata(self, meta: VideoMetadata) -> None:
        spec = self.schema.media[self.column]
        if (meta.height, meta.width, 3) != spec.shape:
            raise ValueError(f"video {meta.uri!r}: frame shape "
                             f"{(meta.height, meta.width, 3)} does not match declared "
                             f"{spec.shape}")

    def _uri(self, source):
        if source not in self.videos:
            raise ValueError(f"source {source!r} has no resolved video mapping")
        return self.videos[source]

    def source_fingerprint(self, source):
        """Compose the wrapped data identity with the auxiliary video's content identity."""
        from .provenance import source_descriptor
        return {"data": source_descriptor(source, self.adapter),
                "video": source_descriptor(self._uri(source), None)}

    def iter_ticks(self, source):
        ticks = iter(self.adapter.iter_ticks(source) if self.supports_streaming
                     else self.adapter.ticks(source))
        refs = iter(_iter_provider_refs(self.provider, self._uri(source), source))
        if self.on == "frame_index":
            count = 0
            for i, tick in enumerate(ticks):
                try:
                    ref = next(refs)
                except StopIteration:
                    if self.mismatch == "drop":
                        break
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
            if not math.isfinite(target):
                raise ValueError(f"source {source!r}: timestamp alignment needs finite tick times")
            if target < last_target:
                raise ValueError(f"source {source!r}: data timestamps are not ordered")
            last_target = target
            while current is not None and current.timestamp < target:
                previous, current = current, next(refs, None)
            choices = [ref for ref in (previous, current) if ref is not None]
            best = min(choices, key=lambda ref: (abs(ref.timestamp - target), ref.frame_index))
            drift = abs(best.timestamp - target)
            if drift > self.tolerance:
                if self.mismatch == "drop":
                    continue
                raise ValueError(f"source {source!r}: nearest video frame is {drift:.6g}s away, "
                                 f"beyond tolerance {self.tolerance}")
            yield Tick(tick.time, tick.entities, tick.events, tick.globals,
                       {**tick.media, self.column: best})

    def ticks(self, source):
        return self.iter_ticks(source)

    def check_source(self, source):
        if hasattr(self.adapter, "check_source"):
            self.adapter.check_source(source)
        self._validate_metadata(self.provider.probe(self._uri(source)))
        sample = list(self.iter_ticks(source))
        if not sample:
            raise ValueError(f"source {source!r}: alignment produced no ticks")


def media_provider_for(adapter) -> Optional[MediaProvider]:
    return getattr(adapter, "provider", None)
