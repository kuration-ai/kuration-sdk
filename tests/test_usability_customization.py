"""Usability/customization foundation: overlays, hints, presets, modes, and versions."""
import numpy as np
import pytest

from kuration import (Registry, SchemaSpec, Segment, default_registry, input, load_strategy,
                       output_preset, segment_priorities)
from kuration.core.loaders import shard_entries
from kuration.core.output.containers import read_shard, register_container, write_shard


def _schema(categorical=False):
    fields = {"x": {"kind": "scalar"}}
    if categorical:
        fields["label"] = {"kind": "categorical"}
    return SchemaSpec.from_dict({"name": "demo", "entity_axis": "entity", "fields": fields})


def test_isolated_registry_overlays_can_bind_same_name():
    a, b = Registry(default_registry), Registry(default_registry)
    a.register("method", "acme/rank", lambda segs, strategy, **kw: np.arange(len(segs)))
    b.register("method", "acme/rank", lambda segs, strategy, **kw: -np.arange(len(segs)))
    spec = {"selection": {"method": "acme/rank"}}
    sa = load_strategy(spec, _schema(), registry=a)
    sb = load_strategy(spec, _schema(), registry=b)
    segments = [Segment("s", i, i + 1, (i, i)) for i in range(3)]
    assert segment_priorities(segments, sa).tolist() == [0, 1, 2]
    assert segment_priorities(segments, sb).tolist() == [0, -1, -2]


def test_registry_has_typed_registration_helpers():
    registry = Registry(default_registry)
    registry.register_scorer("acme/scorer", lambda **kw: object())
    assert registry.descriptor("scorer", "acme/scorer").kind == "scorer"


def test_semantic_position_hint_compiles_to_explicit_schema():
    rows = [{"x": i, "y": i * 2} for i in range(8)]
    p = (input(rows).as_position("pos", dims=["x", "y"], max_rate=20).at_rate(10)
         .measure("velocity", of="pos", map="speed"))
    explicit = SchemaSpec.from_dict({
        "name": "inferred", "entity_axis": "entity", "sample_hz": 10,
        "fields": {"pos": {"kind": "vector", "dims": ["x", "y"],
                           "semantics": "position", "max_rate": 20}}})
    assert p.schema == explicit
    with pytest.raises(ValueError, match="already has a semantic hint"):
        p.as_position("pos")


def test_semantic_refinement_preserves_existing_vector_width():
    p = input([{"pos": [1.0, 2.0]}, {"pos": [3.0, 4.0]}])
    with pytest.raises(ValueError, match="must preserve that width"):
        p.as_position("pos", dims=["x", "y", "z"])


def test_output_presets_have_explicit_categorical_policy():
    spec = output_preset("numpy-timeseries", schema=_schema(True), categorical="omit")
    assert spec.container == "npz" and [m.name for m in spec.modalities] == ["x"]
    with pytest.raises(ValueError, match="categorical field"):
        output_preset("numpy-timeseries", schema=_schema(True), categorical="error")
    encoded = output_preset("numpy-timeseries", schema=_schema(True),
                            categorical="integer-vocab",
                            categorical_vocabs={"label": {"a": 0, "b": 1}})
    assert encoded.to_dict()["vocabs"]["label"] == {"a": 0, "b": 1}


def test_modes_and_explanation_are_structured():
    p = input([{"x": 1}, {"x": 2}]).options(mode="strict")
    assert p.describe()["options"]["on_error"] == "raise"
    plan = p.explain()
    assert plan.source["count"] == 1
    assert plan.to_dict()["options"]["mode"] == "strict"


def test_future_artifact_version_is_rejected():
    with pytest.raises(ValueError, match="unsupported future artifact"):
        shard_entries({"artifact_format_version": "99.0", "shards": []})


def test_namespaced_container_dispatch(tmp_path):
    def writer(path, arrays, n_steps):
        with open(path, "wb") as f:
            np.savez(f, _n_steps=n_steps, **arrays)

    def reader(path):
        z = np.load(path)
        return {k: z[k] for k in z.files if k != "_n_steps"}, int(z["_n_steps"])

    register_container("acme/test-container", extension=".acme", writer=writer, reader=reader)
    path = write_shard(str(tmp_path / "part"), "acme/test-container",
                       {"x": np.arange(3)}, 3)
    arrays, n = read_shard(path, "acme/test-container")
    assert n == 3 and arrays["x"].tolist() == [0, 1, 2]


def test_namespaced_container_is_staged_for_remote_paths():
    pytest.importorskip("fsspec")
    observed_paths: list[str] = []

    def writer(path, arrays, n_steps):
        observed_paths.append(path)
        assert "://" not in path
        with open(path, "wb") as stream:
            np.savez(stream, _n_steps=n_steps, **arrays)

    def reader(path):
        observed_paths.append(path)
        assert "://" not in path
        with np.load(path, allow_pickle=False) as archive:
            return (
                {key: np.asarray(archive[key]) for key in archive.files
                 if key != "_n_steps"},
                int(archive["_n_steps"]),
            )

    name = "acme/test-staged-remote-container"
    register_container(name, extension=".acme", writer=writer, reader=reader)
    path = write_shard(
        "memory://custom-container/part", name, {"x": np.arange(3)}, 3)
    arrays, n_steps = read_shard(path, name)

    assert arrays["x"].tolist() == [0, 1, 2]
    assert n_steps == 3
    assert len(observed_paths) == 2


def test_container_extension_cannot_escape_the_shard_directory(tmp_path):
    escaped = tmp_path / "escaped.bin"

    with pytest.raises(ValueError, match="one filename suffix"):
        register_container(
            "acme/path-escape-container",
            extension=".x/../../../escaped.bin",
            writer=lambda path, arrays, n_steps: None,
            reader=lambda path: ({}, 0),
        )

    assert not escaped.exists()
