"""Plugin discovery and structured progress contracts."""
from __future__ import annotations

import io
import sys

import pytest

from kuration import (
    SchemaSpec,
    Segmenter,
    Tick,
    input,
    load_plugins,
    load_strategy,
    write_dataset,
)
from kuration.core.dsl.registry import METRICS
from kuration.core.output import OutputSpec
from kuration.core.progress import (
    _notebook_progress_html,
    cli_progress,
    progress_callback,
)
from kuration.core.scoring import score_sources


SCHEMA = SchemaSpec.from_dict({
    "name": "plugins", "entity_axis": "e", "sample_hz": 1.0,
    "fields": {"x": {"kind": "scalar"}},
})


class MemAdapter:
    schema = SCHEMA

    def __init__(self):
        self.data = {f"s{i}": [Tick(t, {0: {"x": float(i + t)}}) for t in range(8)]
                     for i in range(3)}

    def ticks(self, source):
        return self.data[source]


def _plugin_file(tmp_path, name="sample_kuration_plugin"):
    path = tmp_path / f"{name}.py"
    path.write_text(
        "import numpy as np\n"
        "from kuration import register_metric\n"
        "@register_metric('plugin_mean')\n"
        "def plugin_mean(w):\n"
        "    return float(np.nanmean(w.numeric_field('x')))\n")
    return name


def test_explicit_plugin_metric_loads_and_reaches_spawned_workers(tmp_path, monkeypatch):
    name = _plugin_file(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        loaded = load_plugins([name], include_entry_points=False)
        assert loaded == [{"kind": "module", "name": name}]
        strategy = load_strategy({"selection": {"method": "top-k",
                                                 "weights": {"plugin_mean": 1.0}}}, SCHEMA)
        segs, errors, _ = score_sources(
            schema=SCHEMA, adapter=MemAdapter(), sources=["s0", "s1", "s2"], define={},
            registered=["plugin_mean"], segmenter=Segmenter(4, 4), workers=2,
            on_error="raise")
        assert not errors and len(segs) == 6
        assert all("plugin_mean" in s.features for s in segs)
        assert strategy.referenced_metrics() == {"plugin_mean"}
    finally:
        from kuration.core import plugins as plugin_state
        METRICS.pop("plugin_mean", None)
        sys.modules.pop(name, None)
        plugin_state._LOADED_MODULES.discard(name)


def test_score_progress_event_contract():
    events = []
    segs, _, _ = score_sources(
        schema=SCHEMA, adapter=MemAdapter(), sources=["s0", "s1"],
        define={"xm": {"source": "e.x", "reduce": "mean"}}, registered=[],
        segmenter=Segmenter(4, 4), progress=events.append)
    assert segs
    assert events[0] == {"phase": "scoring", "event": "start", "total": 2}
    assert [e["source"] for e in events if e["event"] == "advance"] == ["s0", "s1"]
    assert events[-1]["event"] == "complete" and events[-1]["phase"] == "scoring"


def test_entry_point_initializer_is_idempotent(monkeypatch):
    import kuration.core.plugins as plugin_state
    from kuration import default_registry
    calls = []

    def metric(window):
        return 1.0

    def initialize():
        calls.append(1)
        default_registry.register_metric("test/idempotent", metric)

    metric.__module__ = "fake_plugin"
    initialize.__module__ = "fake_plugin"

    class FakeEP:
        name, value = "fake", "fake_plugin:initialize"
        def load(self):
            return initialize

    class FakeEPs(list):
        def select(self, **kwargs):
            return self if kwargs == {"group": "kuration.plugins"} else []

    key = "fake=fake_plugin:initialize"
    snapshot = plugin_state._plugin_state_snapshot()
    old_scanned = plugin_state._ENTRY_POINTS_SCANNED
    plugin_state._LOADED_ENTRY_POINTS.discard(key)
    plugin_state._ENTRY_POINTS_SCANNED = False
    monkeypatch.setattr(plugin_state, "entry_points", lambda: FakeEPs([FakeEP()]))
    try:
        first = load_plugins()
        second = load_plugins()
        assert calls == [1]
        assert first == [{"kind": "entry_point", "name": "fake",
                          "value": "fake_plugin:initialize"}]
        assert second == []
    finally:
        plugin_state._restore_plugin_state(snapshot, None)
        plugin_state._ENTRY_POINTS_SCANNED = old_scanned


def test_entry_point_operator_is_discovered_as_a_direct_method(monkeypatch):
    import kuration.core.plugins as plugin_state
    from kuration import default_registry

    def plugin_window(pipeline, *, size):
        return pipeline.window(size)

    def initialize():
        default_registry.register_operator(
            "plugin/direct-window",
            plugin_window,
            parameters={
                "type": "object",
                "properties": {"size": {"type": "integer", "minimum": 1}},
                "required": ["size"],
                "additionalProperties": False,
            },
        )

    plugin_window.__module__ = "fake_direct_plugin"
    initialize.__module__ = "fake_direct_plugin"

    class FakeEP:
        name, value = "fake-direct", "fake_direct_plugin:initialize"

        def load(self):
            return initialize

    class FakeEPs(list):
        def select(self, **kwargs):
            return self if kwargs == {"group": "kuration.plugins"} else []

    key = "fake-direct=fake_direct_plugin:initialize"
    snapshot = plugin_state._plugin_state_snapshot()
    old_scanned = plugin_state._ENTRY_POINTS_SCANNED
    plugin_state._LOADED_ENTRY_POINTS.discard(key)
    plugin_state._ENTRY_POINTS_SCANNED = False
    monkeypatch.setattr(plugin_state, "entry_points", lambda: FakeEPs([FakeEP()]))
    try:
        # Attribute lookup itself performs the same lazy entry-point discovery as apply().
        pipeline = input([{"x": float(i)} for i in range(8)])
        expanded = pipeline.direct_window(size=4)
        assert expanded._window_desc["window"] == 4
        assert expanded._ops[-1]["args"] == ["plugin/direct-window"]
    finally:
        plugin_state._restore_plugin_state(snapshot, None)
        plugin_state._ENTRY_POINTS_SCANNED = old_scanned


def test_job_plugins_load_before_strategy_validation(tmp_path, monkeypatch):
    name = _plugin_file(tmp_path, "job_kuration_plugin")
    monkeypatch.syspath_prepend(str(tmp_path))
    job = {
        "plugins": [name],
        "schema": SCHEMA.to_dict(),
        "adapter": {"type": "timeseries", "fields": {"x": "x"}},
        "sources": ["not-read.csv"],
        "segmenter": {"window": 4, "stride": 4},
        "strategy": {"selection": {"method": "top-k", "weights": {"plugin_mean": 1.0}}},
    }
    try:
        from kuration.core.cli.validate import validate_job
        assert validate_job(job)["kind"] == "report"
    finally:
        import kuration.core.plugins as plugin_state
        METRICS.pop("plugin_mean", None)
        sys.modules.pop(name, None)
        plugin_state._LOADED_MODULES.discard(name)


def test_write_dataset_reports_all_progress_phases(tmp_path):
    events = []
    strategy = load_strategy({
        "define": {"xm": {"source": "e.x", "reduce": "mean"}},
        "selection": {"method": "top-k", "weights": {"xm": 1.0}},
    }, SCHEMA)
    spec = OutputSpec.from_dict({"shard_length": 4, "modalities": {
        "x": {"dtype": "float32", "shape": [], "source": "field:x"}}})
    write_dataset(str(tmp_path / "out"), schema=SCHEMA, adapter=MemAdapter(),
                  sources=["s0", "s1"], strategy=strategy, segmenter=Segmenter(4, 4),
                  spec=spec, train_hours=1, progress=events.append)
    completed = {e["phase"] for e in events if e["event"] == "complete"}
    assert {"provenance", "scoring", "selection", "writing"} <= completed


def test_cli_progress_is_throttled_and_human_readable():
    out = io.StringIO()
    progress = cli_progress(file=out)
    progress({"phase": "scoring", "event": "start", "total": 100})
    for i in range(1, 101):
        progress({"phase": "scoring", "event": "advance", "current": i, "total": 100,
                  "source": f"s{i}"})
    progress({"phase": "scoring", "event": "complete", "detail": "200 segments"})
    lines = out.getvalue().splitlines()
    assert lines[0] == "[scoring] starting (100 total)"
    assert lines[-1] == "[scoring] complete (200 segments)"
    assert len(lines) <= 23


def test_progress_modes_are_selectable_without_an_extra_import():
    callback = lambda event: None
    assert progress_callback(callback) is callback
    assert progress_callback("off") is None
    quiet = input([{"x": 1.0}]).options(progress="off")
    assert quiet._options["progress"] is None
    assert quiet.describe()["options"]["progress"] == "off"
    with pytest.raises(ValueError, match="progress must be"):
        progress_callback("sparkles")


def test_notebook_progress_markup_is_bounded_and_safe():
    html = _notebook_progress_html(
        {
            "phase": "<scoring>",
            "event": "advance",
            "current": 25,
            "total": 100,
            "source": "<clip>",
        }
    )
    assert "&lt;scoring&gt;" in html and "&lt;clip&gt;" in html
    assert "25/100" in html
    assert "width:25.000%" in html
