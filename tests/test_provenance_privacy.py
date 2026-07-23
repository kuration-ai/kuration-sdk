"""Portable provenance must be strong without leaking host paths or URL credentials."""
from kuration.core.provenance import source_descriptor


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
    assert "secret" not in rendered and "token" not in rendered and "private" not in rendered
