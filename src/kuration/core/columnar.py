"""Bounded Arrow-backed segment feature batches.

``Segment`` remains the convenient Python boundary.  ``SegmentBatch`` is the physical
execution representation: fixed identity columns plus projected feature/bin columns backed by
one Arrow ``RecordBatch``.  PyArrow stays lazy and optional until a columnar execution path is
used.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
import hashlib
import json
from typing import Any

import numpy as np

from .io import local_copy, staged_write
from .segment import Segment


IDENTITY_COLUMNS = ("source_id", "start_idx", "end_idx", "start_time", "end_time",
                    "group", "tags")


def _arrow():
    try:
        import pyarrow as pa
    except ImportError as e:
        raise ImportError("columnar execution needs PyArrow: "
                          "pip install 'kuration-sdk[scale]'") from e
    return pa


def _pack(value: Any) -> Any:
    if isinstance(value, tuple):
        return {"__tuple__": [_pack(item) for item in value]}
    if isinstance(value, frozenset):
        return {"__frozenset__": sorted((_pack(item) for item in value), key=repr)}
    if isinstance(value, list):
        return [_pack(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _pack(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"columnar segment metadata must be JSON-compatible, got "
                     f"{type(value).__name__}")


def _unpack(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"__tuple__"}:
        return tuple(_unpack(item) for item in value["__tuple__"])
    if isinstance(value, dict) and set(value) == {"__frozenset__"}:
        return frozenset(_unpack(item) for item in value["__frozenset__"])
    if isinstance(value, list):
        return [_unpack(item) for item in value]
    if isinstance(value, dict):
        return {key: _unpack(item) for key, item in value.items()}
    return value


def _metadata(value: Any) -> str:
    return json.dumps(_pack(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


class SegmentBatch:
    """A bounded, immutable Arrow ``RecordBatch`` of segment metadata and features."""

    def __init__(self, record_batch):
        pa = _arrow()
        if not isinstance(record_batch, pa.RecordBatch):
            raise TypeError("SegmentBatch needs a pyarrow.RecordBatch")
        missing = set(IDENTITY_COLUMNS) - set(record_batch.schema.names)
        if missing:
            raise ValueError(f"SegmentBatch is missing identity columns {sorted(missing)}")
        if len(set(record_batch.schema.names)) != len(record_batch.schema.names):
            raise ValueError("SegmentBatch column names must be unique")
        self._batch = record_batch

    @classmethod
    def from_segments(cls, segments: Sequence[Segment], *,
                      feature_names: Sequence[str] | None = None,
                      bin_names: Sequence[str] | None = None) -> "SegmentBatch":
        pa = _arrow()
        segments = list(segments)
        if feature_names is None:
            feature_names = sorted({name for segment in segments for name in segment.features})
        feature_names = tuple(feature_names)
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("feature_names must be unique")
        width = max((len(segment.bin) for segment in segments), default=0)
        if bin_names is None:
            bin_names = tuple(str(index) for index in range(width))
        bin_names = tuple(bin_names)
        if len(bin_names) != width:
            raise ValueError(f"bin_names has length {len(bin_names)}, expected {width}")
        if len(set(bin_names)) != len(bin_names):
            raise ValueError("bin_names must be unique")

        columns: dict[str, Any] = {
            "source_id": pa.array([str(segment.source_id) for segment in segments],
                                  type=pa.string()),
            "start_idx": pa.array([segment.start_idx for segment in segments], type=pa.int64()),
            "end_idx": pa.array([segment.end_idx for segment in segments], type=pa.int64()),
            "start_time": pa.array([_metadata(segment.time_range[0]) for segment in segments],
                                   type=pa.string()),
            "end_time": pa.array([_metadata(segment.time_range[1]) for segment in segments],
                                 type=pa.string()),
            "group": pa.array([_metadata(segment.group) for segment in segments],
                              type=pa.string()),
            "tags": pa.array([sorted(map(str, segment.tags)) for segment in segments],
                             type=pa.list_(pa.string())),
        }
        for name in feature_names:
            if not isinstance(name, str) or not name:
                raise ValueError("feature names must be non-empty strings")
            columns[f"feature:{name}"] = pa.array(
                [segment.features.get(name) for segment in segments], type=pa.float64())
        for index, name in enumerate(bin_names):
            if not isinstance(name, str) or not name:
                raise ValueError("bin names must be non-empty strings")
            columns[f"bin:{name}"] = pa.array(
                [segment.bin[index] if index < len(segment.bin) else None
                 for segment in segments])
        # ``pa.record_batch(mapping)`` was not accepted by every supported PyArrow release
        # (notably our declared 15.x minimum). The explicit arrays/names constructor is stable
        # across the full compatibility range.
        return cls(pa.RecordBatch.from_arrays(list(columns.values()), names=list(columns)))

    @classmethod
    def from_record_batch(cls, record_batch) -> "SegmentBatch":
        return cls(record_batch)

    @property
    def record_batch(self):
        return self._batch

    @property
    def schema(self):
        return self._batch.schema

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(self.schema.names)

    def __len__(self) -> int:
        return self._batch.num_rows

    def project(self, columns: Sequence[str]) -> "SegmentBatch":
        requested = tuple(dict.fromkeys((*IDENTITY_COLUMNS, *columns)))
        unknown = set(requested) - set(self.column_names)
        if unknown:
            raise KeyError(f"unknown SegmentBatch column(s) {sorted(unknown)}")
        return SegmentBatch(self._batch.select(requested))

    def slice(self, offset: int, length: int | None = None) -> "SegmentBatch":
        if offset < 0 or (length is not None and length < 0):
            raise ValueError("SegmentBatch slice offset/length must be non-negative")
        return SegmentBatch(self._batch.slice(offset, length))

    def numpy(self, column: str, *, zero_copy_only: bool = True) -> np.ndarray:
        """Return an Arrow-backed NumPy view when the column representation permits it."""
        index = self.schema.get_field_index(column)
        if index < 0:
            raise KeyError(column)
        return self._batch.column(index).to_numpy(zero_copy_only=zero_copy_only)

    def to_segments(self) -> list[Segment]:
        data = self._batch.to_pydict()
        feature_columns = [name for name in self.column_names if name.startswith("feature:")]
        bin_columns = [name for name in self.column_names if name.startswith("bin:")]
        out = []
        for index in range(len(self)):
            features = {name.split(":", 1)[1]: float(data[name][index])
                        for name in feature_columns if data[name][index] is not None}
            bins = tuple(data[name][index] for name in bin_columns
                         if data[name][index] is not None)
            out.append(Segment(
                source_id=data["source_id"][index], start_idx=data["start_idx"][index],
                end_idx=data["end_idx"][index],
                time_range=(_unpack(json.loads(data["start_time"][index])),
                            _unpack(json.loads(data["end_time"][index]))),
                features=features, bin=bins,
                tags=frozenset(data["tags"][index] or ()),
                group=_unpack(json.loads(data["group"][index]))))
        return out

    def write_ipc(self, path: str) -> dict[str, Any]:
        """Atomically write one Arrow IPC file and return reusable partition metadata."""
        pa = _arrow()
        with staged_write(path) as local_path:
            with pa.OSFile(local_path, "wb") as sink:
                with pa.ipc.new_file(sink, self.schema) as writer:
                    writer.write_batch(self._batch)
            import os
            size_bytes = os.path.getsize(local_path)
            with open(local_path, "rb") as file:
                hasher = hashlib.sha256()
                while chunk := file.read(1024 * 1024):
                    hasher.update(chunk)
                digest = hasher.hexdigest()
        return {"path": str(path), "rows": len(self), "sha256": digest,
                "columns": list(self.column_names), "size_bytes": size_bytes}

    @classmethod
    def read_ipc(cls, path: str, *, columns: Sequence[str] | None = None) -> "SegmentBatch":
        pa = _arrow()
        with local_copy(path) as local_path:
            with pa.memory_map(local_path, "r") as source:
                reader = pa.ipc.open_file(source)
                table = reader.read_all().combine_chunks()
                if columns is not None:
                    wanted = tuple(dict.fromkeys((*IDENTITY_COLUMNS, *columns)))
                    table = table.select(wanted)
                batches = table.to_batches(max_chunksize=max(1, table.num_rows))
                batch = batches[0] if batches else pa.record_batch(
                    [pa.array([], type=field.type) for field in table.schema],
                    schema=table.schema)
        return cls(batch)


def iter_segment_batches(segments: Iterable[Segment], batch_size: int, **kwargs) -> Iterator[SegmentBatch]:
    """Convert an iterable without retaining more than ``batch_size`` Python segments."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    pending = []
    for segment in segments:
        pending.append(segment)
        if len(pending) == batch_size:
            yield SegmentBatch.from_segments(pending, **kwargs)
            pending = []
    if pending:
        yield SegmentBatch.from_segments(pending, **kwargs)
