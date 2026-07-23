"""Regression tests for core-runtime correctness findings from the comprehensive review."""
from __future__ import annotations

import numpy as np
import pytest

from kuration import (Arm, CheckpointIdentity, FrameRef, Registry, SchemaSpec,
                       Segmenter, StageCheckpoint, Tick, VideoMetadata, build_window,
                       checkpoint_feature_stage, curate, default_registry, input,
                       get_scorer, load_strategy, run_experiment)


SCHEMA = SchemaSpec.from_dict({
    "name": "core-review",
    "entity_axis": "entity",
    "fields": {"x": {"kind": "scalar"}, "y": {"kind": "scalar"}},
})


class _Adapter:
    def __init__(self, schema=SCHEMA):
        self.schema = schema

    def ticks(self, source):
        return [Tick(index, {0: {"x": float(index), "y": float(index + 1)}})
                for index in range(4)]


class _FakeVideoProvider:
    capabilities = {"streaming": True}

    def probe(self, uri):
        return VideoMetadata(uri, 2, 2, 4, 3.0, 1.0)

    def iter_refs(self, uri, *, source_id=None):
        for index in range(4):
            yield FrameRef(uri, source_id or uri, index, float(index))

    def decode_batch(self, refs):  # pragma: no cover - this test only aligns lazy references
        raise AssertionError("frames should stay lazy")


class _RegisteredFeatureTransform:
    output_features = ("doubled",)
    inputs = ("feature:tests/window-length",)

    def transform_batch(self, batch, ctx):
        return {"doubled": batch["feature:tests/window-length"] * 2}


class _CollisionTransform:
    output_features = ("shared",)
    inputs = ("field:x",)

    def transform_batch(self, batch, ctx):
        return {"shared": np.full(len(batch["field:x"]), 99.0)}


def test_vector_values_must_match_the_declared_width_exactly():
    schema = SchemaSpec.from_dict({
        "name": "vectors", "fields": {
            "pos": {"kind": "vector", "dims": ["x", "y", "z"]},
        },
    })
    with pytest.raises(ValueError, match=r"3-dim vector.*shape \(1,\)"):
        build_window(schema, [Tick(0, {"entity": {"pos": [7.0]}})])

    window = build_window(
        schema, [Tick(0, {"entity": {"pos": [1.0, 2.0, 3.0]}})])
    assert window.numeric["pos"][0, 0].tolist() == [1.0, 2.0, 3.0]


def test_duplicate_sources_are_rejected_for_explicit_and_file_inputs(tmp_path):
    with pytest.raises(ValueError, match="duplicate source ID"):
        input(adapter=_Adapter(), schema=SCHEMA, sources=["same", "same"])

    path = tmp_path / "rows.csv"
    path.write_text("x,y\n1,2\n")
    with pytest.raises(ValueError, match="duplicate source ID"):
        input([str(path), str(path)])


@pytest.mark.parametrize(
    ("sources", "message"),
    [(["same", "same"], "duplicate source ID"), ([""], "non-empty strings")],
)
def test_checkpoint_feature_stage_rejects_invalid_source_ids(tmp_path, sources, message):
    checkpoint = StageCheckpoint(
        str(tmp_path), "features", CheckpointIdentity("logical", "physical"))
    with pytest.raises(ValueError, match=message):
        checkpoint_feature_stage(
            schema=SCHEMA, adapter=_Adapter(), sources=sources, define={}, registered=(),
            segmenter=Segmenter(1, 1), checkpoint=checkpoint, on_error="raise")


def test_measure_rejects_transform_and_scorer_output_collisions():
    registry = Registry(default_registry)
    registry.register_transform("tests/collision", _CollisionTransform)
    registry.register_scorer("tests/constant", lambda: object())
    base = input([{"x": 1.0}, {"x": 2.0}]).using(registry)

    with pytest.raises(ValueError, match="already declared"):
        base.transform("tests/collision").measure("shared", "mean", of="x")
    with pytest.raises(ValueError, match="already declared"):
        base.score("tests/constant", out="shared").measure("shared", "mean", of="x")


def test_experiment_rejects_cross_kind_feature_producers_before_reading():
    metric_arm = Arm("metric", load_strategy({
        "define": {"shared": {"source": "entity.x", "reduce": "mean"}},
        "selection": {"method": "top-k", "weights": {"shared": 1}},
    }, SCHEMA))
    transform_strategy = load_strategy({
        "selection": {"method": "top-k", "weights": {"shared": 1}},
    }, SCHEMA, available_metrics=("shared",))
    transform_call = {
        "name": "tests/collision", "implementation": "tests/collision",
        "outputs": ("shared",), "inputs": ("field:x",), "batch_size": 1,
        "device": "cpu", "params": {}, "resources": {}, "lookbehind": 0,
        "lookahead": 0, "ownership": None,
    }
    transform_arm = Arm("transform", transform_strategy, (transform_call,))

    class NeverRead:
        def ticks(self, source):
            raise AssertionError("producer collisions must fail before source reads")

    with pytest.raises(ValueError, match="multiple producer kinds"):
        run_experiment(schema=SCHEMA, adapter=NeverRead(), sources=["source"],
                       arms=[metric_arm, transform_arm], segmenter=Segmenter(1, 1),
                       target_size=1)


def test_join_video_then_position_refinement_updates_the_wrapped_adapter():
    rows = [{"time": index, "x": float(index), "y": float(index + 10)}
            for index in range(4)]
    joined = input(rows, time="time", hz=1).join_video(
        {"mem": "video.mp4"}, provider=_FakeVideoProvider(), on="frame_index")
    refined = joined.as_position("pos", dims=["x", "y"])
    axis = refined.schema.entity_axis
    pipeline = (refined.window(2, stride=2)
                .measure("mean_x", {"source": f"{axis}.pos.x", "reduce": "mean"})
                .sample("top-k", n=2, weights={"mean_x": 1})
                .options(on_error="raise"))

    assert set(joined.adapter.adapter.schema.fields) == {"x", "y"}
    assert set(refined.adapter.adapter.schema.fields) == {"pos"}
    assert refined.adapter.adapter.fields == {"pos": ["x", "y"]}
    assert refined.adapter.adapter.available_fields == frozenset({"pos"})
    result = pipeline.collect()
    assert [segment.features["mean_x"] for segment in result.all_segments] == [0.5, 2.5]


def test_registered_metric_can_feed_a_transform():
    registry = Registry(default_registry)
    registry.register_metric("tests/window-length", lambda window: float(window.T))
    registry.register_transform("tests/registered-input", _RegisteredFeatureTransform)
    result = (input([{"x": 1.0}, {"x": 2.0}]).using(registry).window(2)
              .measure("tests/window-length").transform("tests/registered-input")
              .sample("top-k", n=1, weights={"doubled": 1})
              .options(on_error="raise").collect())
    assert result.selected[0].features["doubled"] == 4.0


def test_registered_transform_input_is_scheduled_for_experiments():
    registry = Registry(default_registry)
    registry.register_metric("tests/window-length", lambda window: float(window.T))
    registry.register_transform("tests/registered-input", _RegisteredFeatureTransform)
    base = (input([{"x": float(index)} for index in range(4)])
            .using(registry).window(2))
    arm = (base.measure("tests/window-length").transform("tests/registered-input")
           .sample("top-k", n=1, weights={"doubled": 1}))
    result = base.experiment({"arm": arm}, n=1)
    assert result["arms"]["arm"]["metrics"]["doubled"]["mean"] == 4.0


def test_registered_transform_input_is_scheduled_for_bounded_outputs(tmp_path):
    pytest.importorskip("pyarrow")
    registry = Registry(default_registry)
    registry.register_metric("tests/window-length", lambda window: float(window.T))
    registry.register_transform("tests/registered-input", _RegisteredFeatureTransform)
    base = (input([{"x": float(index)} for index in range(4)], hz=1)
            .using(registry).window(2, stride=2)
            .measure("tests/window-length").transform("tests/registered-input"))

    count = base.sample("top-k", n=1, weights={"doubled": 1}).output(
        tmp_path / "count", categorical="omit", shard_length=2)
    assert count["selected"] == 1

    hours = base.sample("top-k", hours=2 / 3600, weights={"doubled": 1}).output(
        tmp_path / "hours", categorical="omit", shard_length=2, span_steps=2)
    assert hours["admitted"]


def test_measure_lazily_loads_ambient_plugin_metrics(monkeypatch):
    name = "tests/lazy-plugin-metric"
    default_registry._entries["metric"].pop(name, None)

    def load_plugin():
        if name not in default_registry.names("metric"):
            default_registry.register_metric(name, lambda window: float(window.T))

    monkeypatch.setattr("kuration.core.plugins.load_plugins", load_plugin)
    try:
        pipeline = input([{"x": 1.0}, {"x": 2.0}]).measure(name)
        assert pipeline._registered_metrics == [name]
    finally:
        default_registry._entries["metric"].pop(name, None)


def test_bounded_count_checkpoint_identity_includes_partition_batch_policy(tmp_path):
    pytest.importorskip("pyarrow")
    base = (input([{"x": float(index)} for index in range(8)]).window(2, stride=1)
            .measure("level", "mean", of="x")
            .sample("top-k", n=3, weights={"level": 1.0}))
    first_pipeline = base.options(batch_size=2, on_error="raise")
    first = first_pipeline._bounded_count_result(
        str(tmp_path), first_pipeline._compile())
    second_pipeline = base.options(batch_size=4, on_error="raise")
    second = second_pipeline._bounded_count_result(
        str(tmp_path), second_pipeline._compile())

    keys = lambda result: [(segment.source_id, segment.start_idx) for segment in result.selected]
    assert len(first.all_segments) == len(second.all_segments) == 7
    assert keys(second) == keys(first)
    assert second.execution["telemetry"]["checkpoint_hits"] == 0
    policy = second.execution["checkpoint"]["identity"]["parameters"]["partition_policy"]
    assert policy["batch_size"] == 4 and policy["version"] == 1


@pytest.mark.parametrize("test_frac", [-0.1, float("nan"), True])
def test_curate_rejects_invalid_test_fraction_before_reading(test_frac):
    strategy = load_strategy({
        "selection": {"method": "top-k", "weights": {"m": 1}},
    }, SCHEMA, available_metrics=("m",))

    class NeverRead:
        def ticks(self, source):
            raise AssertionError("test_frac validation must precede source reads")

    with pytest.raises(ValueError, match=r"test_frac must be a finite number in \[0, 1\]"):
        curate(schema=SCHEMA, adapter=NeverRead(), sources=["source"], strategy=strategy,
               segmenter=Segmenter(1, 1), target_size=1, test_frac=test_frac)


def test_pipeline_score_validates_parameters_eagerly():
    pipeline = input([{"x": 1.0}, {"x": 2.0}])
    with pytest.raises(ValueError, match="unknown parameter"):
        pipeline.score("action_residual", obs="x", action="x", typo=True)
    with pytest.raises(ValueError, match="missing parameter"):
        pipeline.score("action_residual", obs="x")
    assert pipeline.score("action_residual", obs="x", action="x", k=2)._scorers


def test_get_scorer_resolves_default_registry_only_registrations():
    name = "tests/default-registry-only-scorer"
    scorer = object()
    default_registry.register_scorer(name, lambda: scorer)
    try:
        assert get_scorer(name) is scorer
        from kuration.core.scorers import has_scorer
        assert has_scorer(name)
    finally:
        default_registry._entries["scorer"].pop(name, None)
