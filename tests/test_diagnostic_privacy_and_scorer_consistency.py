from __future__ import annotations

import pytest

from kuration import (
    Registry,
    SchemaSpec,
    Segment,
    Segmenter,
    Tick,
    curate,
    load_strategy,
)
from kuration.core.scorers import _materialize_buffers, apply_scorers
from kuration.core.scoring import _redact_diagnostic
from kuration.core.progress import emit
from kuration.core.transforms import apply_transforms


SCHEMA = SchemaSpec.from_dict({
    "name": "scorer-consistency",
    "fields": {"value": {"kind": "scalar"}},
})


class _Adapter:
    def __init__(self, ticks):
        self._ticks = ticks

    def ticks(self, _source):
        return self._ticks


def _ticks(count: int):
    return [
        Tick(time=index, entities={0: {"value": float(index)}})
        for index in range(count)
    ]


def test_scorer_buffer_materialization_rejects_a_shortened_second_read():
    segment = Segment("source", 0, 4, (0, 3))

    with pytest.raises(ValueError, match="outside the 3 ticks"):
        _materialize_buffers([segment], SCHEMA, _Adapter(_ticks(3)))


def test_scorer_buffer_materialization_rejects_changed_timestamps():
    segment = Segment("source", 0, 4, (0, 3))
    ticks = _ticks(4)
    ticks[-1].time = 30

    with pytest.raises(ValueError, match="timestamps recorded by the feature pass"):
        _materialize_buffers([segment], SCHEMA, _Adapter(ticks))


def test_persisted_diagnostics_redact_arbitrary_posix_and_windows_paths():
    rendered = _redact_diagnostic(
        "failed '/private/customer-A/raw input/data.bin' and "
        r"C:\Users\private-user\secret-dataset\session.jsonl"
    )

    assert "customer-A" not in rendered
    assert "raw input" not in rendered
    assert "private-user" not in rendered
    assert "secret-dataset" not in rendered
    assert "data.bin-" in rendered
    assert "session.jsonl-" in rendered


def test_persisted_diagnostics_redact_entire_remote_references():
    rendered = _redact_diagnostic(
        "failed s3://user:secret@internal-bucket/private/data.bin?token=value#fragment"
    )

    for secret in ("user", "secret", "internal-bucket", "private", "token", "fragment"):
        assert secret not in rendered
    assert "data.bin-" in rendered


def test_progress_events_redact_source_credentials_and_host_paths():
    events = []
    emit(
        events.append,
        "scoring",
        "advance",
        source="s3://user:secret@internal-bucket/private/data.bin?token=value#fragment",
    )

    rendered = str(events)
    for secret in ("user", "secret", "internal-bucket", "private", "token", "fragment"):
        assert secret not in rendered
    assert "data.bin-" in rendered


def test_transform_skip_policy_discards_the_whole_failed_source(capsys):
    secret_source = "/private/customer-A/failed.bin"

    class Adapter:
        schema = SCHEMA

        def ticks(self, source):
            if source == secret_source:
                raise RuntimeError(f"cannot read {secret_source}")
            return _ticks(4)

    class Transform:
        def transform_batch(self, batch, _ctx):
            return {"copy": [1.0] * len(batch["segment_ids"])}

    registry = Registry()
    registry.register_transform("test/copy", Transform)
    calls = [{
        "name": "test/copy",
        "implementation": "test/copy",
        "outputs": ["copy"],
        "inputs": ["field:value"],
        "batch_size": 1,
        "device": None,
        "params": {},
    }]
    segments = [
        Segment("good", 0, 4, (0, 3)),
        Segment(secret_source, 0, 4, (0, 3)),
    ]
    errors = []

    result = apply_transforms(
        calls,
        segments,
        adapter=Adapter(),
        registry=registry,
        on_error="skip",
        errors=errors,
    )

    assert [segment.source_id for segment in result] == ["good"]
    assert segments == result
    assert errors[0]["phase"] == "transforms"
    assert "customer-A" not in str(errors)
    assert "customer-A" not in capsys.readouterr().err


def test_transform_skip_policy_resets_component_state_after_a_failed_source():
    failed_source = "/private/customer-state/failed.bin"

    class Adapter:
        schema = SCHEMA

        def ticks(self, _source):
            return _ticks(4)

    class StatefulTransform:
        def __init__(self):
            self.poisoned = False

        def transform_batch(self, batch, _ctx):
            source = batch["segment_ids"][0][0]
            if source == failed_source:
                self.poisoned = True
                raise RuntimeError("source poisoned the component")
            if self.poisoned:
                raise AssertionError("state leaked from the skipped source")
            return {"copy": [1.0] * len(batch["segment_ids"])}

    registry = Registry()
    registry.register_transform("test/stateful", StatefulTransform)
    calls = [{
        "name": "test/stateful",
        "implementation": "test/stateful",
        "outputs": ["copy"],
        "inputs": ["field:value"],
        "batch_size": 1,
        "device": None,
        "params": {},
    }]
    segments = [
        Segment(failed_source, 0, 4, (0, 3)),
        Segment("good", 0, 4, (0, 3)),
    ]

    result = apply_transforms(
        calls,
        segments,
        adapter=Adapter(),
        registry=registry,
        on_error="skip",
    )

    assert [segment.source_id for segment in result] == ["good"]
    assert result[0].features["copy"] == 1.0


def test_scorer_second_read_skip_policy_discards_the_whole_failed_source(capsys):
    secret_source = "/private/customer-B/failed.bin"

    class Adapter:
        def ticks(self, source):
            if source == secret_source:
                raise RuntimeError(f"cannot read {secret_source}")
            return _ticks(4)

    class NeedsBuffers:
        needs_buffers = True

        def score(self, _segments, ctx):
            return [float(buffer.T) for buffer in ctx["buffers"]]

    registry = Registry()
    registry.register_scorer("test/buffers", NeedsBuffers)
    segments = [
        Segment("good", 0, 4, (0, 3)),
        Segment(secret_source, 0, 4, (0, 3)),
    ]
    errors = []

    apply_scorers(
        [("test/buffers", "buffer_size", {})],
        segments,
        schema=SCHEMA,
        adapter=Adapter(),
        registry=registry,
        on_error="skip",
        errors=errors,
    )

    assert [segment.source_id for segment in segments] == ["good"]
    assert segments[0].features["buffer_size"] == 4.0
    assert errors[0]["phase"] == "scorers"
    assert "customer-B" not in str(errors)
    assert "customer-B" not in capsys.readouterr().err


def test_curate_applies_skip_policy_to_transform_second_reads():
    secret_source = "/private/customer-C/failed.bin"

    class Adapter:
        schema = SCHEMA

        def __init__(self):
            self.reads = {}

        def ticks(self, source):
            self.reads[source] = self.reads.get(source, 0) + 1
            if source == secret_source and self.reads[source] > 1:
                raise RuntimeError(f"cannot reread {secret_source}")
            return _ticks(4)

    class Transform:
        def transform_batch(self, batch, _ctx):
            return {"copy": [1.0] * len(batch["segment_ids"])}

    registry = Registry()
    registry.register_transform("test/copy", Transform)
    calls = [{
        "name": "test/copy",
        "implementation": "test/copy",
        "outputs": ["copy"],
        "inputs": ["field:value"],
        "batch_size": 1,
        "device": None,
        "params": {},
    }]
    strategy = load_strategy({
        "define": {"mean": {"source": "value", "reduce": "mean"}},
        "selection": {"method": "top-k", "weights": {"mean": 1.0}},
    }, SCHEMA, registry=registry)

    result = curate(
        schema=SCHEMA,
        adapter=Adapter(),
        sources=["good", secret_source],
        strategy=strategy,
        segmenter=Segmenter(4, 4),
        target_size=10,
        transforms=calls,
        registry=registry,
        on_error="skip",
    )

    assert {segment.source_id for segment in result.all_segments} == {"good"}
    assert result.errors[-1]["phase"] == "transforms"
    assert "customer-C" not in str(result.errors)


def test_curate_applies_skip_policy_to_scorer_second_reads():
    secret_source = "/private/customer-D/failed.bin"

    class Adapter:
        def __init__(self):
            self.reads = {}

        def ticks(self, source):
            self.reads[source] = self.reads.get(source, 0) + 1
            if source == secret_source and self.reads[source] > 1:
                raise RuntimeError(f"cannot reread {secret_source}")
            return _ticks(4)

    class NeedsBuffers:
        needs_buffers = True

        def score(self, _segments, ctx):
            return [float(buffer.T) for buffer in ctx["buffers"]]

    registry = Registry()
    registry.register_scorer("test/buffers", NeedsBuffers)
    strategy = load_strategy({
        "scorers": [{"name": "test/buffers", "as": "buffer_size"}],
        "selection": {"method": "top-k", "weights": {"buffer_size": 1.0}},
    }, SCHEMA, registry=registry)

    result = curate(
        schema=SCHEMA,
        adapter=Adapter(),
        sources=["good", secret_source],
        strategy=strategy,
        segmenter=Segmenter(4, 4),
        target_size=10,
        registry=registry,
        on_error="skip",
    )

    assert {segment.source_id for segment in result.all_segments} == {"good"}
    assert result.errors[-1]["phase"] == "scorers"
    assert "customer-D" not in str(result.errors)
