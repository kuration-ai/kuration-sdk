"""Bounded Arrow segment representation and IPC round trips."""
import hashlib

import numpy as np
import pytest

from kuration import Segment, SegmentBatch, iter_segment_batches

pytest.importorskip("pyarrow")


def _segments():
    return [
        Segment("source-a", 0, 4, (0.0, 0.3), {"speed": 1.5, "quality": 0.2},
                bin=(1, "low"), tags=frozenset({"indoor"}), group=("site", 1)),
        Segment("source-a", 4, 8, (0.4, 0.7), {"speed": 2.5, "quality": 0.8},
                bin=(2, "high"), tags=frozenset({"indoor", "north"}), group=("site", 1)),
    ]


def test_segment_batch_round_trip_and_zero_copy_numeric_view():
    original = _segments()
    batch = SegmentBatch.from_segments(original, bin_names=("speed", "band"))
    assert len(batch) == 2
    assert batch.numpy("feature:speed").tolist() == [1.5, 2.5]
    restored = batch.to_segments()
    assert restored == original


def test_segment_batch_projection_keeps_identity_and_requested_columns():
    batch = SegmentBatch.from_segments(_segments(), bin_names=("speed", "band"))
    projected = batch.project(["feature:quality"])
    assert projected.column_names[-1] == "feature:quality"
    assert "feature:speed" not in projected.column_names
    assert projected.to_segments()[0].features == {"quality": 0.2}
    with pytest.raises(KeyError, match="unknown SegmentBatch"):
        batch.project(["feature:missing"])


def test_segment_batch_ipc_is_atomic_projectable_and_checksummed(tmp_path):
    batch = SegmentBatch.from_segments(_segments(), bin_names=("speed", "band"))
    path = tmp_path / "part-000.arrow"
    metadata = batch.write_ipc(str(path))
    assert metadata["rows"] == 2
    assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    restored = SegmentBatch.read_ipc(str(path), columns=["feature:speed"])
    assert restored.column_names[-1] == "feature:speed"
    assert np.array_equal(restored.numpy("feature:speed"), [1.5, 2.5])
    assert not list(tmp_path.glob("*.__tmp__*"))


def test_iter_segment_batches_never_exceeds_bound():
    segments = [_segments()[0] for _ in range(5)]
    batches = list(iter_segment_batches(iter(segments), 2, bin_names=("speed", "band")))
    assert [len(batch) for batch in batches] == [2, 2, 1]
    with pytest.raises(ValueError, match="batch_size"):
        list(iter_segment_batches(segments, 0))
