from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kuration import (ComponentDescriptor, Registry, SchemaSpec, Segment, Segmenter, Tick,
                       default_registry, input)
from kuration.core.execution import ResourceSpec
from kuration.core.features import FeatureEngine
from kuration.core.output.containers import Container, read_shard
from kuration.core.output.spec import Modality, OutputSpec
from kuration.core.output.stream import stream_source_shards
from kuration.core.plugins import plugin_provenance
from kuration.core.scoring import score_sources


SCHEMA = SchemaSpec.from_dict({
    "name": "architecture",
    "entity_axis": "e",
    "fields": {"x": {"kind": "scalar"}, "y": {"kind": "scalar"}},
})


def _window_length(window):
    return float(window.T)


class _DoubleTransform:
    output_features = ("doubled",)
    inputs = ("feature:mean_x",)

    def transform_batch(self, batch, ctx):
        return {"doubled": batch["feature:mean_x"] * 2.0}


class _BatchOnlyEncoder:
    def encode_batch(self, records, ctx):
        return {"twice": np.asarray(
            [0.0 if record is None else 2.0 * record["x"] for record in records],
            dtype=np.float32)}


def _rows():
    return {"s0": [{"x": float(i), "y": float(10 - i)} for i in range(8)]}


def test_pipeline_overlay_metric_is_used_by_collect_and_report():
    registry = Registry(default_registry)
    registry.register_metric("acme/window-length", _window_length)
    base = (input(_rows()).using(registry).window(2)
            .measure("acme/window-length").options(on_error="raise"))
    report = base.report()
    assert report["metrics"]["acme/window-length"]["mean"] == 2.0
    result = base.sample("top-k", n=2, weights={"acme/window-length": 1.0}).collect()
    assert all(segment.features["acme/window-length"] == 2.0
               for segment in result.selected)


def test_callable_metrics_are_pipeline_scoped_and_clones_are_isolated():
    base = input(_rows()).window(2)
    one = base.measure("acme/local", lambda window: 1.0)
    two = base.measure("acme/local", lambda window: 2.0)
    a = one.sample("top-k", n=1, weights={"acme/local": 1}).collect()
    b = two.sample("top-k", n=1, weights={"acme/local": 1}).collect()
    assert a.selected[0].features["acme/local"] == 1.0
    assert b.selected[0].features["acme/local"] == 2.0
    assert "acme/local" not in default_registry.names("metric")

    derived = base.measure("x")
    derived.sources.append("new-source")
    derived.adapter.fields["x"] = "other"
    assert base.sources == ["s0"]
    assert base.adapter.fields["x"] == "x"
    assert set(base.schema.fields) == {"x", "y"}
    assert derived.schema is not base.schema


def test_component_descriptors_validate_and_defensively_freeze_metadata():
    parameters = {"type": "object", "properties": {"n": {"type": "integer"}}}
    descriptor = ComponentDescriptor("acme/item", "operator", parameters=parameters,
                                     capabilities={"streaming": True})
    parameters["properties"]["n"]["type"] = "string"
    assert descriptor.to_dict()["parameters"]["properties"]["n"]["type"] == "integer"
    with pytest.raises(TypeError, match="immutable"):
        descriptor.parameters["properties"]["n"]["type"] = "number"
    with pytest.raises(ValueError, match="booleans"):
        ComponentDescriptor("acme/bad", "operator", capabilities={"streaming": "yes"})


def test_metric_extension_output_must_be_one_finite_scalar():
    registry = Registry(default_registry)
    registry.register_metric("acme/vector", lambda window: np.ones(2))
    engine = FeatureEngine(SCHEMA, registered=["acme/vector"], registry=registry)
    ticks = [Tick(i, {0: {"x": float(i), "y": 0.0}}) for i in range(2)]
    with pytest.raises(ValueError, match="one finite scalar"):
        engine.score_window(Segment("s", 0, 2, (0, 2)), ticks)


def test_partial_adapter_mapping_fails_before_reading_sources():
    class PartialAdapter:
        schema = SCHEMA
        available_fields = frozenset({"x"})
        available_globals = frozenset()

        def ticks(self, source):
            raise AssertionError("preflight should fail before a source is read")

    with pytest.raises(ValueError, match=r"required fields \['y'\]"):
        score_sources(
            schema=SCHEMA, adapter=PartialAdapter(), sources=["s"],
            define={"mean_y": {"source": "e.y", "reduce": "mean"}}, registered=[],
            segmenter=Segmenter(2, 2), on_error="raise")


def test_overlay_transform_runs_in_experiment_and_unknown_dependency_is_eager():
    registry = Registry(default_registry)
    registry.register_transform("acme/double", _DoubleTransform)
    base = (input(_rows()).using(registry).window(2)
            .measure("mean_x", "mean", of="x").options(on_error="raise"))
    with pytest.raises(ValueError, match="no earlier producer"):
        base.transform("acme/double", inputs=["feature:missing"])
    transformed = (base.transform("acme/double")
                   .sample("top-k", n=2, weights={"doubled": 1.0}))
    low = base.sample("top-k", n=2, weights={"mean_x": -1.0})
    result = base.experiment({"high": transformed, "low": low}, n=2, baseline="high")
    assert "doubled" in result["metrics"]
    assert result["arms"]["high"]["metrics"]["doubled"]["mean"] > 0


def test_overlay_encoder_preset_and_container_are_resolved_end_to_end(tmp_path):
    registry = Registry(default_registry)
    registry.register_encoder("acme/batch", lambda: _BatchOnlyEncoder())

    def preset(schema):
        return {"shard_length": 2, "modalities": {
            "twice": {"dtype": "float32", "shape": [], "source": "encoder:twice"}}}

    registry.register_preset("acme/preset", preset)

    def writer(path, arrays, n_steps):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            np.savez(handle, _n_steps=n_steps, **arrays)

    def reader(path):
        with np.load(path, allow_pickle=False) as archive:
            return ({key: np.asarray(archive[key]) for key in archive.files
                     if key != "_n_steps"}, int(archive["_n_steps"]))

    registry.register_container("acme/container", Container(".acme", writer, reader))
    pipeline = (input(_rows()).using(registry).window(2).measure("mean_x", "mean", of="x")
                .sample("top-k", n=1, weights={"mean_x": 1.0})
                .options(on_error="raise"))
    result = pipeline.output(
        tmp_path, spec={"shard_length": 2, "container": "acme/container", "modalities": {
            "twice": {"dtype": "float32", "shape": [], "source": "encoder:twice"}}},
        encoder="acme/batch")
    arrays, n_steps = read_shard(result["shards"][0]["path"], "acme/container",
                                 registry=registry)
    assert n_steps == 2
    assert arrays["twice"].tolist() == [12.0, 14.0]
    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text())
    assert all(not Path(entry["path"]).is_absolute() for entry in manifest["shards"])
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["provenance"]["payload"]["adapter"]["type"].endswith("RowsAdapter")

    plan = pipeline.output(tmp_path / "dry", preset="acme/preset",
                           encoder="acme/batch", dry_run=True)
    assert plan.output["preset"]["name"] == "acme/preset"


def test_output_internal_names_and_unsafe_dtypes_are_rejected():
    base = {"shard_length": 2, "modalities": {
        "_present": {"dtype": "bool", "shape": [], "source": "field:x"}}}
    with pytest.raises(ValueError, match="reserved"):
        OutputSpec.from_dict(base)
    base["modalities"] = {
        "x": {"dtype": "object", "shape": [], "source": "field:x"}}
    with pytest.raises(ValueError, match="unsupported"):
        OutputSpec.from_dict(base)

    for container, dtype in (("npz", "U1024"), ("npz", "S32"), ("json", "U32")):
        spec = OutputSpec.from_dict({"shard_length": 2, "container": container,
                                     "modalities": {"text": {
                                         "dtype": dtype, "shape": [],
                                         "source": "encoder:text"}}})
        assert np.dtype(spec.modalities[0].dtype).kind in {"U", "S"}
    for container, dtype in (("json", "S32"), ("hdf5", "U32"), ("parquet", "U32")):
        with pytest.raises(ValueError, match="does not safely round-trip"):
            OutputSpec.from_dict({"shard_length": 2, "container": container,
                                  "modalities": {"text": {
                                      "dtype": dtype, "shape": [],
                                      "source": "encoder:text"}}})


def test_direct_output_constructors_validate_and_freeze_nested_vocabs():
    with pytest.raises(ValueError, match="reserved"):
        Modality("_n_steps", "int64", (), "field", "x")
    modality = Modality("label", "int8", (), "vocab", "label")
    spec = OutputSpec((modality,), 2, vocabs={"label": {"a": 0}})
    with pytest.raises(TypeError, match="immutable"):
        spec.vocabs["label"]["b"] = 1
    with pytest.raises(ValueError, match="unique"):
        OutputSpec((modality,), 2, vocabs={"label": {"a": 0, "b": 0}})


def test_stream_writer_rejects_a_source_that_ends_before_selected_span(tmp_path):
    spec = OutputSpec.from_dict({"shard_length": 2, "modalities": {
        "x": {"dtype": "float32", "shape": [], "source": "field:x"}}})
    ticks = [Tick(index, {0: {"x": float(index), "y": 0.0}}) for index in range(2)]
    jobs = [{"eid": 0, "span_index": 0, "start": 0, "end": 4,
             "split": "train", "name_suffix": "_seg0000"}]
    with pytest.raises(ValueError, match="ended early"):
        stream_source_shards(str(tmp_path), "mutated", iter(ticks), jobs,
                             schema=SCHEMA, spec=spec)


def test_stream_writer_rejects_source_length_or_roster_mutation(tmp_path):
    spec = OutputSpec.from_dict({"shard_length": 2, "modalities": {
        "x": {"dtype": "float32", "shape": [], "source": "field:x"}}})
    jobs = [{"eid": 0, "span_index": 0, "start": 0, "end": 2,
             "split": "train", "name_suffix": "_seg0000"}]
    ticks = [Tick(index, {0: {"x": float(index), "y": 0.0}}) for index in range(3)]
    with pytest.raises(ValueError, match="tick count"):
        stream_source_shards(str(tmp_path / "length"), "mutated", iter(ticks), jobs,
                             schema=SCHEMA, spec=spec, expected_ticks=2,
                             expected_entities=[0])
    changed_roster = [Tick(index, {1: {"x": float(index), "y": 0.0}})
                      for index in range(2)]
    with pytest.raises(ValueError, match="entity roster"):
        stream_source_shards(str(tmp_path / "roster"), "mutated", iter(changed_roster), jobs,
                             schema=SCHEMA, spec=spec, expected_ticks=2,
                             expected_entities=[0])


def test_count_output_rejects_same_shape_source_value_mutation_and_stays_uncommitted(tmp_path):
    class MutatingAdapter:
        version = 1

        def source_fingerprint(self, source):
            return f"{source}:v{self.version}"

        def ticks(self, source):
            value = self.version
            self.version = 2
            return [Tick(index, {0: {"x": float(value + index), "y": 0.0}})
                    for index in range(4)]

    adapter = MutatingAdapter()
    pipeline = (input(adapter=adapter, sources=["s"], schema=SCHEMA).window(2)
                .measure("mean_x", "mean", of="x")
                .sample("top-k", n=1, weights={"mean_x": 1.0})
                .options(on_error="raise"))
    spec = {"shard_length": 2, "modalities": {
        "x": {"dtype": "float32", "shape": [], "source": "field:x"}}}
    with pytest.raises(ValueError, match="changed between planning and output"):
        pipeline.output(tmp_path, spec=spec)
    from kuration import shard_entries
    with pytest.raises(ValueError, match="commit state"):
        shard_entries(str(tmp_path))


def test_plugin_provenance_omits_host_absolute_paths():
    assert all("path" not in entry for entry in plugin_provenance())


@pytest.mark.parametrize("kwargs", [
    {"num_cpus": -1}, {"num_gpus": float("nan")}, {"memory": 0}, {"memory": True},
])
def test_resource_spec_rejects_invalid_requests(kwargs):
    with pytest.raises(ValueError):
        ResourceSpec(**kwargs)


def test_ray_worker_receives_metric_plugin_modules_even_with_one_worker(monkeypatch):
    registry = Registry(default_registry)
    registry.register_metric("acme/ray", _window_length)
    seen = {}

    class FakeRayBackend:
        def iter_batches(self, items, fn, **kwargs):
            items = list(items)
            seen["modules"] = items[0][1][-2]
            return iter(fn(item) for item in items)

    import kuration.core.execution as execution
    monkeypatch.setattr(execution, "RayExecutionBackend", FakeRayBackend)
    ticks = [Tick(i, {0: {"x": float(i), "y": 0.0}}) for i in range(2)]

    class Adapter:
        schema = SCHEMA

        def ticks(self, source):
            return ticks

    # The fake backend runs the worker-side load_plugins(__name__) in-process, which would
    # leave this test module in the global loaded-plugins set and get forwarded to the real
    # Ray workers of later tests (where it cannot be imported) — snapshot and restore.
    import kuration.core.plugins as plugin_state
    snapshot = plugin_state._plugin_state_snapshot()
    try:
        segments, errors, _ = score_sources(
            schema=SCHEMA, adapter=Adapter(), sources=["s"], define={},
            registered=["acme/ray"], segmenter=Segmenter(2, 2), executor="ray",
            workers=1, on_error="raise", registry=registry)
    finally:
        plugin_state._restore_plugin_state(snapshot, None)
    assert segments and not errors
    assert __name__ in seen["modules"]
