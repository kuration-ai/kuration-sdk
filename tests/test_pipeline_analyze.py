"""Fluent analyzer, stats, and quality-gate terminals on `Pipeline`.

Every capability must be reachable without a JSON job spec or a CLI, so these tests pin the
tick-stream analyzer path (`Pipeline.analyze` driving `add_batch` over segmented raw sources
with lazy media), the artifact path (`check_dataset`, identical to `run_dataset_analyzer`),
`.stats()` delegation, and `.check()` quality gates — plus the loud errors that keep an
unsupported mode from silently doing nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from kuration import (
    CuratedDatasetAdapter,
    SchemaSpec,
    SplitLeakageAnalyzer,
    StreamAnalyzer,
    default_registry,
    describe_component,
    evaluate_quality,
    import_dataset,
    input,
    inspect_dataset,
    register_analyzer,
    run_dataset_analyzer,
    run_stream_analyzer,
    stats,
)

SCHEMA = SchemaSpec.from_dict({
    "name": "analyze-stream",
    "sample_hz": 10.0,
    "fields": {"value": {"kind": "scalar"}},
})


# ---------------------------------------------------------------------------------------
# A minimal, dependency-free stream analyzer: the generic contract, no media involved.
# ---------------------------------------------------------------------------------------
class WindowMeanAnalyzer:
    """Accumulate per-source window means; the smallest possible stream analyzer."""

    def __init__(self, *, minimum: float = 0.0):
        self.minimum = float(minimum)
        self.segments: list[tuple] = []
        self.means: dict[str, list[float]] = {}
        self.batches: list[int] = []
        self.notes: list = []
        self.lookbehind_rows: list = []

    def add_batch(self, batch, ctx, *, note=None):
        token = ctx.input_token("field:value")
        values = np.asarray(batch[token])
        self.batches.append(len(batch["segment_ids"]))
        self.notes.append(note)
        self.lookbehind_rows.extend(ctx.before(token))
        for index, segment_id in enumerate(batch["segment_ids"]):
            steps = int(ctx.valid_steps[index])
            self.segments.append(tuple(segment_id))
            self.means.setdefault(segment_id[0], []).append(
                float(np.nanmean(values[index][:steps])))

    def summary(self):
        return {"sources": sorted(self.means),
                "windows": len(self.segments),
                "above": sum(mean > self.minimum
                             for means in self.means.values() for mean in means)}


class DatasetOnlyAnalyzer:
    """Artifact-only analyzer: no ``add_batch``, so the stream path must refuse it loudly."""

    def __init__(self):
        self.shards = 0

    def check_dataset(self, dataset, **options):
        from kuration import shard_entries

        entries, _container = shard_entries(dataset)
        self.shards = len(entries)
        return {"passed": True, "shards_checked": self.shards}

    def summary(self):
        return {"shards": self.shards}


class NoInputAnalyzer:
    """Stream-shaped but input-less: undrivable, and it must say so at call time."""

    def add_batch(self, batch, ctx):
        raise AssertionError("must never be driven")

    def summary(self):
        return {}


class ChattyAnalyzer:
    """Returns a per-segment mapping from ``add_batch``; the driver must refuse it."""

    def add_batch(self, batch, ctx):
        return {"oops": len(batch["segment_ids"])}

    def summary(self):
        return {}


MINIMUM_PARAMETERS = {
    "type": "object",
    "properties": {"minimum": {"type": "number"}},
    "additionalProperties": False,
}
_EMPTY_PARAMETERS = {"type": "object", "properties": {}, "additionalProperties": False}


def _register_test_analyzers() -> None:
    if "test/window-mean" in default_registry.names("analyzer"):
        return
    register_analyzer("test/window-mean", lambda *, minimum=0.0: WindowMeanAnalyzer(
        minimum=minimum), description="Accumulate window means", version="1",
        parameters=MINIMUM_PARAMETERS, inputs=("field:value",),
        lookbehind={"field:value": 2}, capabilities={"tick_stream": True})
    register_analyzer("test/dataset-only", DatasetOnlyAnalyzer,
                      description="Count manifest shards", version="1",
                      parameters=_EMPTY_PARAMETERS)
    register_analyzer("test/no-inputs", NoInputAnalyzer,
                      description="Declares no inputs", version="1",
                      parameters=_EMPTY_PARAMETERS)
    register_analyzer("test/chatty", ChattyAnalyzer, description="Returns from add_batch",
                      version="1", parameters=_EMPTY_PARAMETERS, inputs=("field:value",))


_register_test_analyzers()


def _rows(source_length: int = 12, *, start: float = 0.0):
    return [{"t": index / 10.0, "value": start + index} for index in range(source_length)]


def _pipeline(**options):
    pipeline = input({"clip-a": _rows(), "clip-b": _rows(start=100.0)}, schema=SCHEMA,
                     fields={"value": "value"}, time="t").window(4, stride=2)
    return pipeline.options(on_error="raise", **options) if options else pipeline.options(
        on_error="raise")


# ---------------------------------------------------------------------------------------
# Tick-stream mode
# ---------------------------------------------------------------------------------------
def test_analyze_drives_a_stream_analyzer_over_real_source_windows():
    result = _pipeline().analyze("test/window-mean", {"minimum": 50.0})

    assert result["name"] == "test/window-mean"
    assert result["mode"] == "tick-stream"
    assert result["sources_analyzed"] == 2
    assert result["errors"] == []
    assert result["inputs"] == ["field:value"]
    # window=4, stride=2 over 12 ticks -> starts 0,2,4,6,8 per source.
    assert result["segments_analyzed"] == 10
    assert result["summary"] == {"sources": ["clip-a", "clip-b"], "windows": 10, "above": 5}


def test_analyze_supplies_true_source_segment_ids_and_declared_lookbehind():
    analyzer = WindowMeanAnalyzer()
    result = _pipeline().analyze("test/window-mean", analyzer=analyzer)

    assert result["summary"]["windows"] == 10
    # Real (source_id, start_idx, end_idx) coordinates, in deterministic source order.
    assert analyzer.segments[:5] == [("clip-a", 0, 4), ("clip-a", 2, 6), ("clip-a", 4, 8),
                                     ("clip-a", 6, 10), ("clip-a", 8, 12)]
    assert analyzer.segments[5][0] == "clip-b"
    # Declared lookbehind context arrives per row; only a source edge is None.
    assert analyzer.lookbehind_rows[0] is None
    assert analyzer.lookbehind_rows[1] is not None
    assert analyzer.means["clip-a"][0] == pytest.approx(1.5)


def test_analyze_batches_windows_and_forwards_hook_options():
    analyzer = WindowMeanAnalyzer()
    _pipeline().analyze("test/window-mean", analyzer=analyzer, batch_size=2, note="hello")

    assert analyzer.batches == [2, 2, 1, 2, 2, 1]   # per source, bounded by batch_size
    assert set(analyzer.notes) == {"hello"}


def test_analyze_runs_under_streaming_without_a_resident_pool():
    streamed = _pipeline(streaming=True).analyze("test/window-mean", {"minimum": 50.0})
    resident = _pipeline().analyze("test/window-mean", {"minimum": 50.0})
    assert streamed == resident


def test_analyze_accepts_a_per_use_input_remapping():
    schema = SchemaSpec.from_dict({"name": "remapped", "sample_hz": 10.0,
                                  "fields": {"signal": {"kind": "scalar"}}})
    rows = [{"t": index / 10.0, "signal": float(index)} for index in range(8)]
    result = (input({"clip": rows}, schema=schema, fields={"signal": "signal"}, time="t")
              .window(4, stride=4)
              .analyze("test/window-mean", inputs=("field:signal",)))
    assert result["segments_analyzed"] == 2
    assert result["inputs"] == ["field:signal"]


def test_analyze_skips_and_logs_a_failing_source_when_asked(capsys):
    secret_source = "/private/customer-analyzer/clip.bin"

    class Exploding(WindowMeanAnalyzer):
        def add_batch(self, batch, ctx, *, note=None):
            if batch["segment_ids"][0][0] == secret_source:
                raise RuntimeError(f"{secret_source} is unreadable")
            super().add_batch(batch, ctx, note=note)

    pipeline = input(
        {"clip-a": _rows(), secret_source: _rows(start=100.0)},
        schema=SCHEMA,
        fields={"value": "value"},
        time="t",
    ).window(4, stride=2)
    result = pipeline.options(on_error="skip").analyze(
        "test/window-mean", analyzer=Exploding())
    assert result["sources_analyzed"] == 1
    assert len(result["errors"]) == 1
    assert "clip.bin-" in result["errors"][0]["source"]
    assert result["errors"][0]["error_type"] == "RuntimeError"
    assert result["errors"][0]["phase"] == "analyze"
    assert "customer-analyzer" not in str(result)
    assert "customer-analyzer" not in capsys.readouterr().err

    with pytest.raises(RuntimeError, match="customer-analyzer"):
        pipeline.options(on_error="raise").analyze(
            "test/window-mean", analyzer=Exploding())


# ---------------------------------------------------------------------------------------
# Loud errors — an unsupported mode never silently does nothing
# ---------------------------------------------------------------------------------------
def test_analyze_rejects_an_unknown_analyzer_with_the_registry_error():
    with pytest.raises(ValueError, match="unknown analyzer 'test/nope'"):
        _pipeline().analyze("test/nope")


def test_analyze_names_the_missing_hook_for_a_dataset_only_analyzer():
    with pytest.raises(ValueError, match="add_batch") as error:
        _pipeline().analyze("test/dataset-only")
    assert "check_dataset" in str(error.value)


def test_analyze_rejects_dataset_mode_without_an_artifact_backed_pipeline():
    with pytest.raises(ValueError, match="artifact-backed pipeline"):
        _pipeline().analyze("test/dataset-only", mode="dataset")


def test_analyze_rejects_an_analyzer_that_declares_no_inputs():
    with pytest.raises(ValueError, match="declares no field:/media: inputs"):
        _pipeline().analyze("test/no-inputs")


def test_analyze_rejects_a_returning_add_batch():
    with pytest.raises(ValueError, match="must accumulate state and return None"):
        _pipeline().analyze("test/chatty")


def test_analyze_refuses_a_vacuous_empty_pool():
    pipeline = input({"clip": _rows(3)}, schema=SCHEMA, fields={"value": "value"},
                     time="t").window(8, stride=8)
    with pytest.raises(ValueError, match="measured NO segments"):
        pipeline.analyze("test/window-mean")


def test_analyze_validates_its_own_arguments():
    pipeline = _pipeline()
    with pytest.raises(ValueError, match="registered analyzer name"):
        pipeline.analyze("")
    with pytest.raises(ValueError, match="must be 'dataset', 'stream', or None"):
        pipeline.analyze("test/window-mean", mode="artifact")
    with pytest.raises(ValueError, match="non-empty sequence"):
        pipeline.analyze("test/window-mean", inputs=())
    with pytest.raises(ValueError, match="parameters must be a JSON object"):
        pipeline.analyze("test/window-mean", "minimum=1")
    with pytest.raises(ValueError, match="analyzer= or parameters="):
        pipeline.analyze("test/window-mean", {"minimum": 1.0},
                         analyzer=WindowMeanAnalyzer())
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        pipeline.analyze("test/window-mean", batch_size=0)


def test_analyzer_registration_validates_declared_inputs():
    with pytest.raises(ValueError, match=r"must use field:/media:"):
        register_analyzer("test/bad-input", WindowMeanAnalyzer, inputs=("feature:x",),
                          description="bad", version="1", parameters=_EMPTY_PARAMETERS)
    with pytest.raises(ValueError, match="need declared inputs"):
        register_analyzer("test/bad-boundary", WindowMeanAnalyzer, lookahead=2,
                          description="bad", version="1", parameters=_EMPTY_PARAMETERS)
    with pytest.raises(ValueError, match="non-negative integer"):
        register_analyzer("test/bad-count", WindowMeanAnalyzer, inputs=("field:value",),
                          lookahead=-1, description="bad", version="1",
                          parameters=_EMPTY_PARAMETERS)
    with pytest.raises(ValueError, match="ownership"):
        register_analyzer("test/bad-ownership", WindowMeanAnalyzer, inputs=("field:value",),
                          ownership="leftmost", description="bad", version="1",
                          parameters=_EMPTY_PARAMETERS)
    # A rejected declaration must not leave a half-registered component behind.
    assert "test/bad-input" not in default_registry.names("analyzer")


def test_analyze_rejects_an_input_the_schema_cannot_supply():
    with pytest.raises(ValueError, match="unknown field 'absent'"):
        _pipeline().analyze("test/window-mean", inputs=("field:absent",))


def test_analyzer_methods_do_not_collide_with_registered_operators():
    from kuration.core.pipeline import operator_name_conflict

    for reserved in ("analyze", "check", "stats"):
        assert operator_name_conflict(reserved) is not None
    assert not [name for name in default_registry.names("operator")
                if operator_name_conflict(name) is not None]


def test_stream_analyzer_protocol_separates_the_two_capabilities():
    assert isinstance(WindowMeanAnalyzer(), StreamAnalyzer)
    assert not isinstance(SplitLeakageAnalyzer(), StreamAnalyzer)
    assert describe_component("analyzer", "test/window-mean")["capabilities"] == {
        "tick_stream": True}


# ---------------------------------------------------------------------------------------
# Artifact mode
# ---------------------------------------------------------------------------------------
def _written_dataset(tmp_path):
    source = tmp_path / "clip.npz"
    dataset = tmp_path / "dataset"
    np.savez(source, value=np.arange(12, dtype=np.float32))
    import_dataset(source, dataset, source_id="clip1", shard_length=4)
    return dataset


def _artifact_pipeline(dataset):
    adapter = CuratedDatasetAdapter(str(dataset))
    return input(adapter=adapter, schema=adapter.schema, sources=adapter.sources())


def test_analyze_on_an_artifact_matches_run_dataset_analyzer(tmp_path):
    dataset = _written_dataset(tmp_path)
    fluent = _artifact_pipeline(dataset).analyze("test/dataset-only")
    reference = run_dataset_analyzer("test/dataset-only", str(dataset))

    assert fluent == reference
    assert fluent["passed"] is True
    assert fluent["shards_checked"] == 3


def test_artifact_pipeline_names_its_dataset(tmp_path):
    dataset = _written_dataset(tmp_path)
    pipeline = _artifact_pipeline(dataset)
    assert pipeline._artifact_datasets() == [str(dataset)]
    # Explicit stream mode is still available over an artifact's own tick stream.
    result = pipeline.window(4, stride=4).analyze("test/window-mean", mode="stream",
                                                 inputs=("field:value",))
    assert result["mode"] == "tick-stream"
    assert result["segments_analyzed"] == 3


def test_analyze_rejects_stream_only_options_in_dataset_mode(tmp_path):
    dataset = _written_dataset(tmp_path)
    with pytest.raises(ValueError, match="tick-stream path"):
        _artifact_pipeline(dataset).analyze("test/dataset-only", batch_size=4)


def test_run_stream_analyzer_is_usable_without_the_pipeline_facade():
    from kuration import Segmenter

    pipeline = _pipeline()
    result = run_stream_analyzer(
        "test/window-mean", schema=pipeline.schema, adapter=pipeline.adapter,
        sources=pipeline.sources, segmenter=Segmenter(4, 2), parameters={"minimum": 50.0})
    assert result["summary"]["above"] == 5


# ---------------------------------------------------------------------------------------
# `.stats()` delegation
# ---------------------------------------------------------------------------------------
def test_stats_delegates_to_the_shared_entry_point():
    pipeline = _pipeline()
    assert pipeline.stats(examples=1).to_dict() == stats(pipeline, examples=1).to_dict()


def test_stats_forwards_every_keyword_the_shared_entry_point_accepts():
    events: list[dict] = []
    report = _pipeline().stats(hist_bins=4, max_sources=1, examples=1,
                              progress=events.append)
    assert report["stage"] == "uncurated"
    sources = report["sections"]["input"]["sources"]
    assert (sources["total"], sources["sampled"]) == (2, 1)
    assert len(report["sections"]["input"]["examples"]) == 1
    assert {event["phase"] for event in events}          # progress reached the feature pass

    with pytest.raises(TypeError):
        _pipeline().stats(not_a_stats_option=1)


# ---------------------------------------------------------------------------------------
# `.check()` — the fluent quality-policy path
# ---------------------------------------------------------------------------------------
def test_check_gates_input_evidence_without_raising():
    passing = _pipeline().expect(max_missing={"value": 0.0}).check()
    assert passing["passed"] is True
    assert [check["name"] for check in passing["checks"]] == ["max_missing.value"]

    failing = _pipeline().check(min_sources=5)
    assert failing["passed"] is False
    assert failing["checks"][0]["actual"] == 2


def test_check_accepts_a_policy_file_shape_and_keyword_overrides():
    result = _pipeline().expect(min_sources=1).check(
        {"expectations": {"min_sources": 2}}, min_sources=3)
    assert result["expectations"] == {"min_sources": 3}
    assert result["passed"] is False


def test_check_gates_a_written_artifact_exactly_like_the_cli(tmp_path):
    dataset = _written_dataset(tmp_path)
    policy = {"min_shards": 1, "require_modalities": ["value"]}
    fluent = _artifact_pipeline(dataset).check(policy)
    reference = evaluate_quality(inspect_dataset(str(dataset)), policy)

    assert fluent["passed"] is True
    assert fluent.to_dict() == reference.to_dict()


def test_check_is_loud_about_missing_or_unknown_expectations():
    with pytest.raises(ValueError, match="needs quality expectations"):
        _pipeline().check()
    with pytest.raises(ValueError, match="unknown quality expectation"):
        _pipeline().check(max_frames=3)
    with pytest.raises(ValueError, match="quality-policy dict"):
        _pipeline().check("policy.json")
    with pytest.raises(ValueError, match="artifact-backed pipeline"):
        _pipeline().check(min_sources=1, verify=True)


# ---------------------------------------------------------------------------------------
# The built-in gameplay analyzer, end to end on both modes
# ---------------------------------------------------------------------------------------
cv2 = pytest.importorskip("cv2")

from kuration.gameplay import action_state_consistency as asc  # noqa: E402
from kuration.gameplay import pixel_shift  # noqa: E402
from kuration.gameplay.pixel_shift import diamond_action_decode as dad  # noqa: E402
from test_pixel_shift import (  # noqa: E402  reuse the fake-video pipeline fixtures
    ACTION_DIM,
    FakeVideoProvider,
    _action,
    _shifting_noise_frames,
)

asc.register()
pixel_shift.register()

BASELINES = {"W": {"1": {"expected_sign": 1, "min": 0.0, "max": 1e9}}}
ACTION_COLUMNS = [f"a{index}" for index in range(ACTION_DIM)]
GAMEPLAY_SCHEMA = SchemaSpec.from_dict({
    "name": "action-state-stream",
    "fields": {"action": {"kind": "vector", "dims": ACTION_COLUMNS}},
})


def _gameplay_pipeline(*, press_at=3, n_frames=8, window=4, stride=2, **options):
    actions = [_action() for _ in range(n_frames)]
    actions[press_at] = _action(w=True)
    rows = [{"t": index, **{column: float(value)
                            for column, value in zip(ACTION_COLUMNS, actions[index],
                                                     strict=True)}}
            for index in range(n_frames)]
    provider = FakeVideoProvider(_shifting_noise_frames(n_frames))
    pipeline = (input({"clip1": rows}, schema=GAMEPLAY_SCHEMA,
                      fields={"action": ACTION_COLUMNS}, time="t")
                .join_video({"clip1": "fake.mp4"}, provider=provider, on="frame_index")
                .window(window, stride=stride)
                .options(on_error="raise", **options))
    return pipeline


def test_action_state_analyzer_runs_on_a_tick_stream_pipeline():
    checker = asc.action_state_consistency(anomaly_type="action_state", baselines=BASELINES)
    result = _gameplay_pipeline().analyze(asc.ANALYZER_NAME, analyzer=checker,
                                          controls=["W"])

    assert result["passed"] is True
    assert result["summary"]["consistent"] == ["clip1"]
    assert result["segments_analyzed"] == 3
    assert result["inputs"] == ["media:frame", "field:action"]
    # Tick-stream mode measures in real source frames, so the documented cross-window
    # de-duplication holds: overlapping windows [0,4) and [2,6) both see the press at
    # source frame 3 and it is recorded exactly once, at that frame.
    assert result["frame_space"] == "source"
    assert result["cross_window_dedup"] is True
    assert [record["frame"] for record in checker._clip_records["clip1"]] == [3]


def test_action_state_analyzer_reports_an_inconsistent_clip_from_a_tick_stream():
    result = _gameplay_pipeline().analyze(
        asc.ANALYZER_NAME,
        {"anomaly_type": "action_state",
         "baselines": {"W": {"1": {"expected_sign": -1, "min": 0.0, "max": 1e9}}}},
        controls=["W"])
    assert result["passed"] is False
    assert result["summary"]["inconsistent"] == ["clip1"]


def test_action_state_stream_mode_works_under_streaming():
    streamed = _gameplay_pipeline(streaming=True).analyze(
        asc.ANALYZER_NAME, {"anomaly_type": "action_state", "baselines": BASELINES},
        controls=["W"])
    resident = _gameplay_pipeline().analyze(
        asc.ANALYZER_NAME, {"anomaly_type": "action_state", "baselines": BASELINES},
        controls=["W"])
    assert streamed == resident
    assert streamed["passed"] is True


def test_action_state_analyzer_runs_on_a_written_artifact(tmp_path):
    source = tmp_path / "clip.npz"
    dataset = tmp_path / "dataset"
    rng = np.random.default_rng(11)
    base = rng.integers(0, 255, size=(40, 63, 3), dtype=np.uint8)
    frames = np.stack([base[:, index:index + 60] for index in range(3)])
    actions = np.stack([_action(), _action(w=True), _action()])
    np.savez(source, video=frames, input=actions)
    import_dataset(source, dataset, source_id="clip1")

    options = {"frames": "video", "actions": "input", "controls": ["W"]}
    parameters = {"anomaly_type": "action_state", "baselines": BASELINES}
    fluent = _artifact_pipeline(dataset).analyze(asc.ANALYZER_NAME, parameters, **options)
    reference = run_dataset_analyzer(asc.ANALYZER_NAME, str(dataset),
                                    parameters=parameters, options=options)

    assert fluent == reference
    assert fluent["passed"] is True
    # The artifact path keeps its deliberately loud frame-space caveat.
    assert fluent["frame_space"] == "shard-concatenation"
    assert fluent["cross_shard_dedup"] is False


def test_action_state_analyzer_declares_the_pixel_shift_input_contract():
    from kuration.core.analyzers import analyzer_metadata

    meta = analyzer_metadata(asc.ANALYZER_NAME,
                             default_registry.get("analyzer", asc.ANALYZER_NAME))
    transform = describe_component("transform", "aintropy/pixel-shift")
    assert meta["inputs"] == ("media:frame", "field:action")
    assert meta["lookbehind"] == {"field:action": 1}
    assert meta["lookahead"] == 3
    assert transform["capabilities"] == {}   # the roles are declared, not the capability set
    assert describe_component("analyzer", asc.ANALYZER_NAME)["capabilities"][
        "tick_stream"] is True


def test_action_state_stream_mode_does_not_decode_unused_lookbehind_frames():
    class CountingProvider(FakeVideoProvider):
        def __init__(self, frames):
            super().__init__(frames)
            self.decode_calls: list[list[int]] = []

        def decode_batch(self, refs):
            refs = list(refs)
            self.decode_calls.append([ref.frame_index for ref in refs])
            return super().decode_batch(refs)

    actions = [_action() for _ in range(8)]
    actions[5] = _action(w=True)
    rows = [{"t": index, **{column: float(value)
                            for column, value in zip(ACTION_COLUMNS, actions[index],
                                                     strict=True)}}
            for index in range(8)]
    provider = CountingProvider(_shifting_noise_frames(8))
    result = (input({"clip1": rows}, schema=GAMEPLAY_SCHEMA,
                    fields={"action": ACTION_COLUMNS}, time="t")
              .join_video({"clip1": "fake.mp4"}, provider=provider, on="frame_index")
              .window(4, stride=4)
              .options(on_error="raise")
              .analyze(asc.ANALYZER_NAME,
                       {"anomaly_type": "action_state", "baselines": BASELINES},
                       controls=["W"]))

    assert result["passed"] is True
    assert result["segments_analyzed"] == 2
    # Owned windows plus each window's declared lookahead frames — and never the [3]
    # lookbehind frame, which only the action stream declares (matching the pixel-shift
    # transform's own decode ledger, because it is the same bounded decode path).
    assert provider.decode_calls == [[0, 1, 2, 3], [4, 5, 6], [4, 5, 6, 7]]
    assert [3] not in provider.decode_calls
    assert dad.ALL_CONTROLS  # the DIAMOND vocabulary stayed importable through this path
