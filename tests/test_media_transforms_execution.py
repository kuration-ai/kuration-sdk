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
            # Legacy batch keys remain available throughout the release-candidate compatibility
            # window.
            assert "lookbehind:field:value" in batch
            assert "lookahead:field:value" in batch
            assert "owned_lengths" in batch
            values.append((0 if before is None else len(before))
                          + 10 * (0 if after is None else len(after)) + 100 * int(owned))
        return {"context_code": np.asarray(values, dtype=np.float64)}


@register_batch_transform(
    "tests/roles",
    outputs=("role_code",),
    inputs=("media:frame", "field:value"),
    media_requirements={"kind": "video-frame", "layout": "HWC"},
    # Only the field input needs boundary context; the media input must not be decoded for it.
    lookbehind={"field:value": 1},
)
class RoleTransform:
    """Reads every input through its declared role, as a remappable transform must."""

    seen: dict = {}

    def transform_batch(self, batch, ctx):
        frames_token = ctx.input_token("media:frame")
        values_token = ctx.input_token("field:value")
        type(self).seen = {"frames": frames_token, "values": values_token,
                           "inputs": dict(ctx.inputs), "batch_keys": sorted(batch)}
        with pytest.raises(KeyError, match="no lookbehind context"):
            ctx.before(frames_token)
        counts = [0 if before is None else len(before) for before in ctx.before(values_token)]
        brightness = batch[frames_token].mean(axis=(1, 2, 3, 4))
        return {"role_code": brightness + np.asarray(counts, dtype=np.float64)}


def test_transform_reads_remapped_input_tokens_through_declared_roles():
    """Pins the review finding that a transform hardcoding its registration tokens passes
    pipeline construction and then fails mid-run on a per-use ``inputs=`` remapping: the
    resolved token for every declared role is now published on the context."""
    provider = FakeVideoProvider()
    rows = [{"signal": float(index)} for index in range(4)]
    result = (input(rows)
              .join_video({"mem": "clip.mp4"}, provider=provider, column="shot",
                          on="frame_index")
              .window(2, stride=2)
              .transform("tests/roles", inputs=("media:shot", "field:signal"), batch_size=2)
              .curate("top-k", n=2, weights={"role_code": 1})
              .collect())

    assert not result.errors
    assert RoleTransform.seen["frames"] == "media:shot"
    assert RoleTransform.seen["values"] == "field:signal"
    assert RoleTransform.seen["inputs"] == {"media:frame": "media:shot",
                                            "field:value": "field:signal"}
    # The remapped tokens are the batch keys; the registration tokens are not present at all.
    assert "media:shot" in RoleTransform.seen["batch_keys"]
    assert "media:frame" not in RoleTransform.seen["batch_keys"]
    assert {segment.start_idx: segment.features["role_code"]
            for segment in result.all_segments} == {0: 0.5, 2: 3.5}


def test_transform_input_remapping_must_match_the_declared_roles():
    """A remapping that drops a role or changes its kind is rejected while the pipeline is
    being built, not when the transform reaches for a token that is not in its batch."""
    rows = [{"value": float(index)} for index in range(4)]
    base = input(rows).window(2, stride=2)

    with pytest.raises(ValueError, match="one token per role"):
        base.transform("tests/roles", inputs=("field:value",))
    with pytest.raises(ValueError, match="different kind"):
        base.transform("tests/context", inputs=("media:frame",))


def test_per_input_lookbehind_only_extracts_the_inputs_that_declared_it():
    """Pins the review finding that a single ``lookbehind`` count made the runner decode one
    unused video frame per segment: a per-input declaration decodes only the owned window."""
    provider = FakeVideoProvider()
    rows = [{"value": float(index)} for index in range(4)]
    (input(rows)
     .join_video({"mem": "clip.mp4"}, provider=provider, on="frame_index")
     .window(2, stride=2)
     .transform("tests/roles", batch_size=1)
     .curate("top-k", n=2, weights={"role_code": 1})
     .collect())

    # Exactly the two owned windows are decoded -- no single-frame lookbehind decode.
    assert provider.decode_calls == [[0, 1], [2, 3]]


@register_batch_transform("tests/valid-steps", outputs=("valid_mean",),
                          inputs=("field:x",))
class ValidStepsMean:
    """Averages only the real timesteps of a ragged, NaN-padded batch."""

    def transform_batch(self, batch, ctx):
        values = np.asarray(batch["field:x"], dtype=np.float64)
        assert np.array_equal(batch["valid_steps"], ctx.valid_steps)
        assert not ctx.valid_steps.flags.writeable
        return {"valid_mean": np.asarray(
            [values[row, :steps].mean() for row, steps in enumerate(ctx.valid_steps)],
            dtype=np.float64)}


def test_ragged_field_batches_expose_their_valid_step_counts():
    """Pins the review finding that ``_stack_field_windows`` NaN-pads a ragged batch where
    ``np.stack`` used to fail loudly: every transform (not only ownership-aware ones) is told
    how many timesteps each row really has, so the padding is observable instead of silent."""
    from kuration import Registry, Segment, Tick
    from kuration.core.transforms import apply_transforms

    schema = SchemaSpec.from_dict({"name": "ragged", "fields": {"x": {"kind": "scalar"}}})

    class Adapter:
        def ticks(self, source):
            return [Tick(index, {0: {"x": float(index)}}) for index in range(6)]

    Adapter.schema = schema
    registry = Registry()
    registry.register_transform("tests/valid-steps", ValidStepsMean)
    calls = [{"name": "tests/valid-steps", "implementation": "tests/valid-steps",
              "outputs": ["valid_mean"], "inputs": ["field:x"], "batch_size": 2,
              "device": "cpu", "params": {}}]
    segments = [Segment("a", 0, 4, (0, 3)), Segment("a", 4, 6, (4, 5))]

    apply_transforms(calls, segments, adapter=Adapter(), registry=registry)

    # (0+1+2+3)/4 and (4+5)/2 -- the shorter row's two pad cells are excluded, not averaged.
    assert [segment.features["valid_mean"] for segment in segments] == [1.5, 4.5]


@register_batch_transform("tests/field-echo-a", outputs=("echo_a",), inputs=("field:x",))
class FieldEchoA:
    def transform_batch(self, batch, ctx):
        values = np.asarray(batch["field:x"], dtype=np.float64)
        return {"echo_a": np.asarray(
            [values[row, :steps].mean() for row, steps in enumerate(ctx.valid_steps)])}


@register_batch_transform("tests/field-echo-b", outputs=("echo_b",), inputs=("field:x",))
class FieldEchoB:
    def transform_batch(self, batch, ctx):
        values = np.asarray(batch["field:x"], dtype=np.float64)
        return {"echo_b": np.asarray(
            [values[row, :steps].mean() for row, steps in enumerate(ctx.valid_steps)])}


def test_multiple_transforms_sharing_a_field_build_the_window_buffer_once(monkeypatch):
    """F12: ``build_window`` is documented as "the amortization point: one buffer per segment,
    shared by all of a profile's metrics" -- but the transform runner rebuilt the whole-schema
    buffer once per transform per window even when two transforms read the same field, so wall
    time scaled with the number of transforms rather than the work they did. One
    `build_window` per window, shared across every transform in the same batch pass, fixes
    that."""
    from kuration import Registry, Segment, Tick
    from kuration.core import buffer as buffer_module
    from kuration.core.transforms import apply_transforms

    schema = SchemaSpec.from_dict({"name": "shared", "fields": {"x": {"kind": "scalar"}}})

    class Adapter:
        def ticks(self, source):
            return [Tick(index, {0: {"x": float(index)}}) for index in range(6)]

    Adapter.schema = schema
    registry = Registry()
    registry.register_transform("tests/field-echo-a", FieldEchoA)
    registry.register_transform("tests/field-echo-b", FieldEchoB)
    calls = [
        {"name": "tests/field-echo-a", "implementation": "tests/field-echo-a",
         "outputs": ["echo_a"], "inputs": ["field:x"], "batch_size": 2, "device": "cpu",
         "params": {}},
        {"name": "tests/field-echo-b", "implementation": "tests/field-echo-b",
         "outputs": ["echo_b"], "inputs": ["field:x"], "batch_size": 2, "device": "cpu",
         "params": {}},
    ]
    segments = [Segment("a", 0, 2, (0, 1)), Segment("a", 2, 4, (2, 3)),
               Segment("a", 4, 6, (4, 5))]

    build_calls = []
    real_build_window = buffer_module.build_window

    def counting_build_window(schema_arg, ticks_arg):
        build_calls.append(ticks_arg)
        return real_build_window(schema_arg, ticks_arg)

    monkeypatch.setattr(buffer_module, "build_window", counting_build_window)

    apply_transforms(calls, segments, adapter=Adapter(), registry=registry)

    # 3 windows, 2 transforms both reading field:x -- one build_window call per window, not
    # one per (window, transform) pair.
    assert len(build_calls) == 3
    assert [segment.features["echo_a"] for segment in segments] == [0.5, 2.5, 4.5]
    assert [segment.features["echo_b"] for segment in segments] == [0.5, 2.5, 4.5]


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

    monkeypatch.setattr("kuration.core.io.is_remote", lambda uri: str(uri).startswith("s3://"))
    monkeypatch.setattr("kuration.core.io.open_file", open_file)
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
                .sample("top-k", n=2, weights={"broken": 1.0})
                .options(on_error="raise"))
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
