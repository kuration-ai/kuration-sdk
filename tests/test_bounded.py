"""Bounded fused feature-stage execution and partition reuse."""
import numpy as np
import pytest

from kuration import (CheckpointIdentity, Registry, SegmentBatch, StageCheckpoint,
                       canonical_hash, checkpoint_feature_stage, default_registry, input)

pytest.importorskip("pyarrow")


class _DoubleTransform:
    output_features = ("doubled",)
    inputs = ("feature:mean_x",)
    calls = 0
    instances = 0

    def __init__(self):
        type(self).instances += 1

    def transform_batch(self, batch, ctx):
        type(self).calls += 1
        return {"doubled": batch["feature:mean_x"] * 2}


class _ContextTransform:
    output_features = ("context_code",)
    inputs = ("field:x",)
    lookbehind = 1
    lookahead = 1
    ownership = "rightmost-start"

    def transform_batch(self, batch, ctx):
        values = []
        for before, after, owned in zip(
                batch["lookbehind:field:x"], batch["lookahead:field:x"],
                batch["owned_lengths"], strict=True):
            values.append((0 if before is None else len(before))
                          + 10 * (0 if after is None else len(after)) + 100 * int(owned))
        return {"context_code": np.asarray(values, dtype=np.float64)}


def _setup(tmp_path):
    registry = Registry(default_registry)
    registry.register_transform("test/double", _DoubleTransform)
    pipeline = (input([{"x": 1.0}, {"x": 2.0}, {"x": 3.0}, {"x": 4.0}, {"x": 5.0}])
                .using(registry).window(2, stride=1)
                .measure("mean_x", "mean", of="x")
                .transform("test/double", inputs=["feature:mean_x"], batch_size=2)
                .filter("doubled", ">=", 5))
    identity = CheckpointIdentity(canonical_hash(pipeline.describe()),
                                  canonical_hash(pipeline.explain().physical_plan))
    checkpoint = StageCheckpoint(str(tmp_path), "features", identity)
    return pipeline, registry, checkpoint


def _run(pipeline, registry, checkpoint):
    return checkpoint_feature_stage(
        schema=pipeline.schema, adapter=pipeline.adapter, sources=pipeline.sources,
        define=pipeline._define, registered=(), segmenter=pipeline._segmenter_or_default(),
        checkpoint=checkpoint, batch_size=2, predicates=pipeline._predicates,
        transforms=pipeline._transforms, registry=registry, on_error="raise")


def test_bounded_feature_stage_fuses_scoring_predicates_and_transforms(tmp_path):
    pipeline, registry, checkpoint = _setup(tmp_path)
    result = _run(pipeline, registry, checkpoint)
    assert [commit.rows for commit in result.commits] == [1, 2]
    assert result.rows == 3
    assert result.telemetry["checkpoint_misses"] == 2
    assert result.telemetry["checkpoint_bytes"] > 0
    restored = [segment for commit in result.commits
                for segment in SegmentBatch.read_ipc(commit.data_file).to_segments()]
    assert [(segment.start_idx, segment.features["doubled"]) for segment in restored] == [
        (1, 5.0), (2, 7.0), (3, 9.0)]
    assert list(result.source_tracks.values()) == [((0, 5),)]


def test_bounded_feature_stage_reuses_valid_partitions_without_transform_work(tmp_path):
    pipeline, registry, checkpoint = _setup(tmp_path)
    _DoubleTransform.calls = 0
    _DoubleTransform.instances = 0
    first = _run(pipeline, registry, checkpoint)
    assert _DoubleTransform.calls == 2
    assert _DoubleTransform.instances == 1
    _DoubleTransform.calls = 0
    second = _run(pipeline, registry, checkpoint)
    assert _DoubleTransform.calls == 0
    assert _DoubleTransform.instances == 1
    assert second.commits == first.commits
    assert second.telemetry["checkpoint_hits"] == 2
    assert second.telemetry["checkpoint_misses"] == 0
    marker = checkpoint.finalize([commit.partition_id for commit in second.commits])
    assert marker["rows"] == 3


def test_bounded_skip_discards_all_partitions_from_failed_source(tmp_path):
    pipeline, _, _ = _setup(tmp_path)

    class Adapter:
        schema = pipeline.schema

        def ticks(self, source):
            return pipeline.adapter.ticks(pipeline.sources[0])

    identity = CheckpointIdentity(canonical_hash({"logical": 1}),
                                  canonical_hash({"physical": 1}))
    checkpoint = StageCheckpoint(str(tmp_path / "rollback"), "features", identity)

    def fail_late(segment):
        if segment.source_id == "bad" and segment.start_idx >= 2:
            raise RuntimeError("late source failure")
        return ()

    result = checkpoint_feature_stage(
        schema=pipeline.schema, adapter=Adapter(), sources=["bad", "good"],
        define=pipeline._define, registered=(), segmenter=pipeline._segmenter_or_default(),
        checkpoint=checkpoint, batch_size=2, tags_of=fail_late, on_error="skip")

    assert [error["source"] for error in result.errors] == ["bad"]
    assert set(result.source_tracks) == {"good"}
    assert result.commits
    assert all(commit.partition_id.startswith("good:") for commit in result.commits)


def test_bounded_feature_stage_supplies_halo_and_ownership(tmp_path):
    registry = Registry(default_registry)
    registry.register_transform("test/context", _ContextTransform)
    pipeline = (input([{"x": float(value)} for value in range(6)])
                .using(registry).window(4, stride=2)
                .transform("test/context", batch_size=2)
                .sample("top-k", n=2, weights={"context_code": 1}))
    identity = CheckpointIdentity(canonical_hash({"logical": "context"}),
                                  canonical_hash({"physical": "context"}))
    checkpoint = StageCheckpoint(str(tmp_path / "context"), "features", identity)

    result = checkpoint_feature_stage(
        schema=pipeline.schema, adapter=pipeline.adapter, sources=pipeline.sources,
        define={}, registered=(), segmenter=pipeline._segmenter_or_default(),
        checkpoint=checkpoint, batch_size=2, transforms=pipeline._transforms,
        registry=registry, on_error="raise")
    restored = [segment for commit in result.commits
                for segment in SegmentBatch.read_ipc(commit.data_file).to_segments()]

    assert {segment.start_idx: segment.features["context_code"] for segment in restored} == {
        0: 210.0,
        2: 401.0,
    }
