from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from kuration import (FrameRef, MediaSpec, PyAVMediaProvider, SchemaSpec, VideoMetadata,
                      input, iter_shards, register_batch_transform)


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


@register_batch_transform("tests/internal-type-error", outputs=("broken",),
                          inputs=("field:value",))
class InternalTypeError:
    calls = 0

    def transform_batch(self, batch, ctx):
        type(self).calls += 1
        raise TypeError("plugin implementation failed")


@register_batch_transform("tests/constructor-type-error", outputs=("broken",),
                          inputs=("field:value",))
class ConstructorTypeError:
    calls = 0

    def __init__(self, device="cpu"):
        type(self).calls += 1
        raise TypeError("constructor implementation failed")


@register_batch_transform(
    "tests/context",
    outputs=("context_code",),
    inputs=("field:value",),
    lookbehind=1,
    lookahead=1,
    ownership="rightmost-start",
)
class ContextTransform:
    def transform_batch(self, batch, ctx):
        assert ctx["schema"] is ctx.schema
        assert ctx.ownership == "rightmost-start"
        assert not ctx.owned_lengths.flags.writeable
        values = []
        for before, after, owned in zip(
                ctx.before("field:value"), ctx.after("field:value"),
                ctx.owned_lengths, strict=True):
            # Legacy batch keys remain available throughout the rc3 compatibility window.
            assert "lookbehind:field:value" in batch
            assert "lookahead:field:value" in batch
            assert "owned_lengths" in batch
            values.append((0 if before is None else len(before))
                          + 10 * (0 if after is None else len(after)) + 100 * int(owned))
        return {"context_code": np.asarray(values, dtype=np.float64)}


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


def test_transform_receives_bounded_halo_and_rightmost_start_ownership():
    rows = [{"value": float(index)} for index in range(6)]
    result = (input(rows)
              .window(4, stride=2)
              .transform("tests/context", batch_size=2)
              .curate("top-k", n=2, weights={"context_code": 1})
              .collect())

    by_start = {segment.start_idx: segment.features["context_code"]
                for segment in result.all_segments}
    assert by_start == {0: 210.0, 2: 401.0}


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


def test_pyav_real_roundtrip(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "tiny.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for value in (10, 80, 160):
            pixels = np.full((16, 16, 3), value, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    provider = PyAVMediaProvider()
    metadata = provider.probe(str(path))
    refs = list(provider.iter_refs(str(path), source_id="tiny"))
    frames = provider.decode_batch(refs)
    assert (metadata.width, metadata.height) == (16, 16)
    assert [ref.source_id for ref in refs] == ["tiny"] * 3
    assert frames.shape == (3, 16, 16, 3)
    assert frames.dtype == np.uint8
    assert np.diff(frames.mean(axis=(1, 2, 3))).min() > 40


def test_pyav_provider_reuses_and_cleans_remote_staging(monkeypatch):
    opens = []

    def open_file(uri, mode):
        opens.append((uri, mode))
        return io.BytesIO(b"video-bytes")

    monkeypatch.setattr("kuration.io.is_remote", lambda uri: str(uri).startswith("s3://"))
    monkeypatch.setattr("kuration.io.open_file", open_file)
    provider = PyAVMediaProvider(remote_cache_size=1)
    with provider._local_path("s3://bucket/clip.mp4") as first:
        assert Path(first).read_bytes() == b"video-bytes"
    with provider._local_path("s3://bucket/clip.mp4") as second:
        assert second == first
    assert len(opens) == 1
    provider.close()
    assert not Path(first).exists()


def test_internal_transform_type_error_is_not_retried():
    InternalTypeError.calls = 0
    pipeline = (input([{"value": float(i)} for i in range(4)])
                .window(2, stride=2)
                .transform("tests/internal-type-error")
                .sample("top-k", n=2, weights={"broken": 1.0}))
    with pytest.raises(TypeError, match="plugin implementation failed"):
        pipeline.collect()
    assert InternalTypeError.calls == 1


def test_internal_constructor_type_error_is_not_retried():
    ConstructorTypeError.calls = 0
    pipeline = (input([{"value": float(i)} for i in range(4)])
                .window(2, stride=2)
                .transform("tests/constructor-type-error")
                .sample("top-k", n=2, weights={"broken": 1.0}))
    with pytest.raises(TypeError, match="constructor implementation failed"):
        pipeline.collect()
    assert ConstructorTypeError.calls == 1
