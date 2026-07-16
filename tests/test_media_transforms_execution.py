from __future__ import annotations

import numpy as np
import pytest

from kuration import (FrameRef, MediaSpec, SchemaSpec, VideoMetadata, input, iter_shards,
                      register_batch_transform)


class FakeVideoProvider:
    capabilities = {"streaming": True, "variable_frame_rate": True}

    def __init__(self, timestamps=(0.0, 0.1, 0.25, 0.4)):
        self.timestamps = timestamps
        self.decode_calls = []

    def probe(self, uri):
        return VideoMetadata(uri, 2, 2, len(self.timestamps), self.timestamps[-1], 10.0)

    def iter_refs(self, uri, *, source_id=None):
        for i, timestamp in enumerate(self.timestamps):
            yield FrameRef(uri, source_id or uri, i, timestamp)

    def decode_batch(self, refs):
        refs = list(refs)
        self.decode_calls.append([r.frame_index for r in refs])
        return np.stack([np.full((2, 2, 3), r.frame_index, np.uint8) for r in refs])


@register_batch_transform("tests/frame-mean", outputs=("brightness",),
                          inputs=("media:frame",),
                          media_requirements={"kind": "video-frame", "layout": "HWC"})
class FrameMean:
    loads = 0

    def __init__(self, device="cpu"):
        type(self).loads += 1

    def transform_batch(self, batch, ctx):
        return {"brightness": batch["media:frame"].mean(axis=(1, 2, 3, 4))}


def test_media_only_schema_round_trip():
    schema = SchemaSpec.from_dict({
        "name": "video", "fields": {},
        "media": {"frame": {"kind": "video-frame", "dtype": "uint8",
                              "shape": [2, 2, 3], "layout": "HWC",
                              "color_space": "RGB", "timing": "pts"}},
    })
    assert isinstance(schema.media["frame"], MediaSpec)
    assert SchemaSpec.from_dict(schema.to_dict()) == schema


def test_video_transform_and_batched_media_output(tmp_path):
    provider = FakeVideoProvider()
    FrameMean.loads = 0
    pipeline = (input(["clip.mp4"], provider=provider)
                .window(2, stride=2)
                .transform("tests/frame-mean", batch_size=2)
                .curate("top-k", n=1, weights={"brightness": 1}))
    result = pipeline.collect()
    assert FrameMean.loads == 1
    assert result.selected[0].features["brightness"] == pytest.approx(2.5)

    pipeline.output(tmp_path, shard_length=2, modalities={
        "frame": {"dtype": "uint8", "shape": [2, 2, 3], "source": "media:frame"}})
    assert next(iter_shards(tmp_path))["frame"][:, 0, 0, 0].tolist() == [2, 3]
    assert provider.decode_calls[-1] == [2, 3]


def test_timestamp_join_uses_nearest_variable_rate_frame():
    provider = FakeVideoProvider()
    rows = [{"timestamp": t, "value": i} for i, t in enumerate((0.01, 0.12, 0.24, 0.39))]
    pipeline = input(rows, time="timestamp", hz=10).join_video(
        {"mem": "clip.mp4"}, provider=provider, tolerance=0.03)
    assert [tick.media["frame"].frame_index for tick in pipeline.adapter.ticks("mem")] == [0, 1, 2, 3]


def test_timestamp_join_fails_on_drift():
    provider = FakeVideoProvider()
    pipeline = input([{"timestamp": 2.0, "value": 1}], time="timestamp", hz=10).join_video(
        {"mem": "clip.mp4"}, provider=provider, tolerance=0.01)
    with pytest.raises(ValueError, match="beyond tolerance"):
        list(pipeline.adapter.ticks("mem"))
