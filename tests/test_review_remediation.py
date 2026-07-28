from __future__ import annotations

import pytest

from kuration import (
    SchemaSpec,
    Segmenter,
    Tick,
    default_registry,
    input,
    load_plugins,
    register_metric,
)
from kuration.core.adapters.huggingface import HuggingFaceAdapter
from kuration.core.adapters.timeseries import TimeseriesAdapter
from kuration.core.scoring import score_sources
from kuration.core.io import staged_write
from kuration.core.checkpoints import _publish_attempt


SCHEMA = SchemaSpec.from_dict({
    "name": "review-remediation",
    "entity_axis": "entity",
    "sample_hz": 1.0,
    "fields": {"x": {"kind": "scalar"}},
})


@pytest.mark.parametrize("kind", ["rows", "ticks"])
def test_input_rejects_source_ids_that_collide_after_normalization(kind):
    if kind == "rows":
        sources = {1: [{"x": 1.0}], "1": [{"x": 2.0}]}
        kwargs = {"fields": {"x": "x"}}
    else:
        sources = {
            1: [Tick(0, {0: {"x": 1.0}})],
            "1": [Tick(0, {0: {"x": 2.0}})],
        }
        kwargs = {}

    with pytest.raises(ValueError, match="collide after string normalization"):
        input(sources, schema=SCHEMA, **kwargs)


def test_timeseries_groups_by_normalized_time_without_retaining_history():
    adapter = TimeseriesAdapter(
        SCHEMA, fields={"x": "x"}, time_column="t", entity_column="entity")
    rows = [
        {"t": "1", "entity": "a", "x": 1.0},
        {"t": 1, "entity": "b", "x": 2.0},
    ] + [
        {"t": value, "entity": "a", "x": float(value)}
        for value in range(2, 10_002)
    ]

    ticks = list(adapter._iter_rows_to_ticks(rows))

    assert ticks[0].time == 1.0
    assert set(ticks[0].entities) == {"a", "b"}
    assert len(ticks) == 10_001


def test_timeseries_rejects_a_decreasing_normalized_time_before_emitting_it():
    adapter = TimeseriesAdapter(SCHEMA, fields={"x": "x"}, time_column="t")
    with pytest.raises(ValueError, match="not ordered"):
        list(adapter._iter_rows_to_ticks([
            {"t": "2", "x": 2.0},
            {"t": 1, "x": 1.0},
        ]))


def test_huggingface_partitioned_sources_reject_streaming_and_collisions():
    adapter = HuggingFaceAdapter(
        SCHEMA, dataset="dummy", fields={"x": "x"}, source_column="episode")
    adapter._rows = [
        {"episode": 1, "x": 1.0},
        {"episode": "1", "x": 2.0},
    ]

    assert adapter.supports_streaming is False
    with pytest.raises(ValueError, match="collide after string normalization"):
        adapter.sources()


def test_huggingface_single_source_remains_streaming_capable():
    adapter = HuggingFaceAdapter(SCHEMA, dataset="dummy", fields={"x": "x"})
    assert adapter.supports_streaming is True


def test_score_sources_consumes_an_unsized_catalog_incrementally():
    processed = []
    progress = []

    class Adapter:
        schema = SCHEMA

        def ticks(self, source):
            processed.append(source)
            return [Tick(0, {"entity": {"x": float(source[1:])}})]

    def sources():
        for index in range(5):
            source = f"s{index}"
            if index:
                assert processed[-1] == f"s{index - 1}"
            yield source

    segments, errors, _ = score_sources(
        schema=SCHEMA,
        adapter=Adapter(),
        sources=sources(),
        define={"mean_x": {"source": "entity.x", "reduce": "mean"}},
        registered=[],
        segmenter=Segmenter(1, 1),
        progress=progress.append,
    )

    assert not errors
    assert processed == [f"s{index}" for index in range(5)]
    assert len(segments) == 5
    assert progress[0]["total"] is None
    assert progress[-1]["total"] == 5


class _NonOverwritingMoveFilesystem:
    def __init__(self):
        self.objects = {"memory://bucket/final.bin": b"previous"}

    def put_file(self, local, remote):
        self.objects[remote] = __import__("pathlib").Path(local).read_bytes()

    def exists(self, path):
        return path in self.objects

    def rm_file(self, path):
        self.objects.pop(path, None)

    def _strip_protocol(self, path):
        return path

    def mv(self, source, destination):
        if destination in self.objects:
            raise FileExistsError(f"destination exists: {destination}")
        self.objects[destination] = self.objects.pop(source)

    def makedirs(self, path, exist_ok=False):
        pass


def test_remote_staged_write_keeps_previous_generation_when_publish_fails(monkeypatch):
    import kuration.core.io as io_module

    filesystem = _NonOverwritingMoveFilesystem()
    monkeypatch.setattr(io_module, "_fs", lambda path: filesystem)

    with pytest.raises(FileExistsError, match="destination exists"):
        with staged_write("memory://bucket/final.bin") as local:
            __import__("pathlib").Path(local).write_bytes(b"replacement")

    assert filesystem.objects == {"memory://bucket/final.bin": b"previous"}


def test_remote_checkpoint_publish_keeps_previous_generation_on_failure(monkeypatch):
    import kuration.core.io as io_module

    filesystem = _NonOverwritingMoveFilesystem()
    filesystem.objects["memory://bucket/attempt.bin"] = b"replacement"
    monkeypatch.setattr(io_module, "_fs", lambda path: filesystem)

    with pytest.raises(FileExistsError, match="destination exists"):
        _publish_attempt("memory://bucket/attempt.bin", "memory://bucket/final.bin")

    assert filesystem.objects["memory://bucket/final.bin"] == b"previous"


def test_explicit_plugin_failure_rolls_back_registry_and_can_retry(tmp_path, monkeypatch):
    import sys
    import kuration.core.plugins as plugin_state
    from kuration.core.dsl.registry import METRICS

    module = tmp_path / "review_broken_plugin.py"
    module.write_text(
        "from kuration import register_metric\n"
        "@register_metric('review/partial')\n"
        "def partial(window): return 1.0\n"
        "raise RuntimeError('boom')\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    for _ in range(2):
        with pytest.raises(ImportError, match="boom"):
            load_plugins(["review_broken_plugin"], include_entry_points=False)
        assert "review/partial" not in default_registry.names("metric")
        assert "review/partial" not in METRICS
        assert "review_broken_plugin" not in sys.modules
        assert "review_broken_plugin" not in plugin_state.loaded_plugin_modules()


def test_broken_ambient_entry_point_does_not_hide_healthy_plugins(monkeypatch, capsys):
    import kuration.core.plugins as plugin_state
    from kuration.core.dsl.registry import METRICS

    def broken():
        @register_metric("review/ambient-partial")
        def partial(window):
            return 1.0
        raise RuntimeError("broken initializer")

    def healthy():
        @register_metric("review/ambient-healthy")
        def metric(window):
            return 2.0

    class EntryPoint:
        def __init__(self, name, value, initializer):
            self.name, self.value, self.initializer = name, value, initializer

        def load(self):
            return self.initializer

    class EntryPoints(list):
        def select(self, **kwargs):
            return self if kwargs == {"group": "kuration.plugins"} else []

    keys = {"broken=review_broken:register", "healthy=review_healthy:register"}
    old_scanned = plugin_state._ENTRY_POINTS_SCANNED
    old_failures = dict(plugin_state._PLUGIN_FAILURES)
    plugin_state._ENTRY_POINTS_SCANNED = False
    plugin_state._PLUGIN_FAILURES.clear()
    for key in keys:
        plugin_state._LOADED_ENTRY_POINTS.discard(key)
    monkeypatch.setattr(plugin_state, "entry_points", lambda: EntryPoints([
        EntryPoint("broken", "review_broken:register", broken),
        EntryPoint("healthy", "review_healthy:register", healthy),
    ]))
    try:
        with pytest.warns(RuntimeWarning, match="broken initializer"):
            loaded = load_plugins()
        assert [entry["kind"] for entry in loaded] == ["error", "entry_point"]
        assert "review/ambient-partial" not in default_registry.names("metric")
        assert "review/ambient-partial" not in METRICS
        assert "review/ambient-healthy" in default_registry.names("metric")
        assert "review/ambient-healthy" in METRICS
        [failure] = plugin_state.plugin_failures()
        assert failure["name"] == "broken"
        assert failure["error_type"] == "builtins.RuntimeError"
        marker = next(entry for entry in plugin_state.plugin_provenance()
                      if entry.get("kind") == "error")
        assert marker["error"] == "plugin entry point failed; run kur-plugin-check broken"

        from kuration.core.cli.discovery import _report_plugin_failures
        _report_plugin_failures()
        assert "ERROR plugin broken" in capsys.readouterr().err
    finally:
        for name in ("review/ambient-partial", "review/ambient-healthy"):
            default_registry._entries["metric"].pop(name, None)
            METRICS.pop(name, None)
        for key in keys:
            plugin_state._LOADED_ENTRY_POINTS.discard(key)
        plugin_state._LOADED_MODULES.discard(healthy.__module__)
        plugin_state._PLUGIN_FAILURES.clear()
        plugin_state._PLUGIN_FAILURES.update(old_failures)
        plugin_state._ENTRY_POINTS_SCANNED = old_scanned
