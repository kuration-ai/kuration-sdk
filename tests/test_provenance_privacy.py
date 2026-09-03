"""Portable provenance must be strong without leaking host paths or URL credentials."""
from kuration.core.provenance import (
    publishable_source_id,
    source_descriptor,
    source_identities_allow_reuse,
)
from kuration import SchemaSpec, Segment, Tick, input
from kuration.core.curate import CurateResult
from kuration.core.review import preview_dataset, selection_review
from kuration.core.strategy import load_strategy


class _Adapter:
    pass


def test_local_source_uses_content_hash_and_hides_parent_path(tmp_path):
    source = tmp_path / "private" / "data.bin"
    source.parent.mkdir()
    source.write_bytes(b"first")
    before = source_descriptor(str(source), _Adapter())
    assert before["source"] == "data.bin"
    assert str(tmp_path) not in str(before)
    assert before["strength"] == "content"
    source.write_bytes(b"other")  # same byte length, potentially same coarse timestamp
    after = source_descriptor(str(source), _Adapter())
    assert before["metadata"]["content_sha256"] != after["metadata"]["content_sha256"]


def test_remote_source_redacts_credentials_query_and_fragment(monkeypatch):
    class _FS:
        def info(self, _path):
            return {"type": "file", "size": 1, "etag": "v1"}

    monkeypatch.setattr("kuration.core.io._fs", lambda _path: _FS())
    descriptor = source_descriptor("s3://user:secret@bucket/data?a=token#private", _Adapter())
    rendered = str(descriptor)
    assert descriptor["source"] == "s3://bucket/data"
    assert source_identities_allow_reuse([descriptor])
    assert "secret" not in rendered and "token" not in rendered and "private" not in rendered
    rotated = source_descriptor(
        "s3://other:new-secret@bucket/data?a=new-token#other-private",
        _Adapter(),
    )
    assert rotated == descriptor
    assert publishable_source_id(
        "s3://other:new-secret@bucket/data?a=new-token#other-private"
    ) == publishable_source_id(
        "s3://user:secret@bucket/data?a=token#private"
    )


def test_remote_metadata_without_a_version_token_cannot_authorize_reuse(monkeypatch):
    class _FS:
        def info(self, _path):
            return {"type": "file", "size": 1, "mtime": 123, "etag": None}

    monkeypatch.setattr("kuration.core.io._fs", lambda _path: _FS())
    descriptor = source_descriptor("s3://bucket/mutable", _Adapter())

    assert descriptor["strength"] == "remote-metadata"
    assert not source_identities_allow_reuse([descriptor])


def test_windows_source_path_is_redacted_on_every_host():
    source = r"C:\Users\private-user\secret-dataset\session.jsonl"
    descriptor = source_descriptor(source, _Adapter())

    assert descriptor["source"] == "session.jsonl"
    assert publishable_source_id(source).startswith("session.jsonl-")
    rendered = str(descriptor)
    assert "private-user" not in rendered
    assert "secret-dataset" not in rendered


def test_local_content_descriptor_survives_a_parent_directory_move(tmp_path):
    first = tmp_path / "first-private-parent" / "data.bin"
    second = tmp_path / "second-private-parent" / "data.bin"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"same immutable content")
    second.write_bytes(first.read_bytes())

    assert source_descriptor(str(first), _Adapter()) == source_descriptor(
        str(second), _Adapter()
    )


def test_preview_and_pipeline_plans_redact_host_paths(tmp_path):
    source = str(tmp_path / "private-customer" / "session.jsonl")
    schema = SchemaSpec.from_dict({
        "name": "privacy",
        "fields": {"x": {"kind": "scalar"}},
    })

    class Adapter:
        def ticks(self, _source):
            return [Tick(0, {0: {"x": 1.0}})]

    preview = preview_dataset(
        schema=schema,
        adapter=Adapter(),
        sources=[source],
        max_sources=1,
        max_ticks_per_source=1,
        examples=1,
    )
    pipeline = input(
        adapter=Adapter(), schema=schema, sources=[source]
    ).window(1).measure("score", "mean", of="x").sample(
        "top-k", n=1, weights={"score": 1.0}
    )

    rendered = str({
        "preview": preview.to_dict(),
        "describe": pipeline.describe(),
        "explain": pipeline.explain().to_dict(),
    })
    assert "private-customer" not in rendered
    assert str(tmp_path) not in rendered
    assert "session.jsonl-" in rendered


def test_selection_review_redacts_source_paths(tmp_path):
    source = str(tmp_path / "private-customer" / "session.jsonl")
    schema = SchemaSpec.from_dict({
        "name": "privacy",
        "fields": {"x": {"kind": "scalar"}},
    })
    strategy = load_strategy({
        "name": "privacy",
        "define": {"score": {"source": "x", "reduce": "mean"}},
        "selection": {"method": "top-k", "weights": {"score": 1.0}},
    }, schema)
    segment = Segment(source, 0, 1, (0, 0), features={"score": 1.0})
    result = CurateResult(
        selected=[segment],
        splits={source: "train"},
        errors=[],
        all_segments=[segment],
        schema=schema,
        strategy=strategy,
    )

    review = selection_review(result, examples=1)
    rendered = str(review.to_dict())
    assert "private-customer" not in rendered
    assert str(tmp_path) not in rendered
    assert "session.jsonl-" in rendered
