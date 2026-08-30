"""The path rule and job-document path anchoring."""
from __future__ import annotations

import pytest

from kuration_sdk_lab.paths import normalize_job_paths, resolve_path


def test_relative_resolves_against_root(tmp_path):
    (tmp_path / "data").mkdir()
    assert resolve_path("data", str(tmp_path)) == str((tmp_path / "data").resolve())


def test_absolute_allowed_by_default(tmp_path):
    assert resolve_path(str(tmp_path), "/elsewhere") == str(tmp_path.resolve())


def test_absolute_rejected_when_disabled(tmp_path):
    with pytest.raises(PermissionError):
        resolve_path(str(tmp_path), str(tmp_path), allow_absolute=False)


def test_escape_rejected_when_locked_to_root(tmp_path):
    with pytest.raises(PermissionError):
        resolve_path("../outside", str(tmp_path / "root"), allow_absolute=False)


def test_empty_path_rejected():
    with pytest.raises(ValueError):
        resolve_path("  ", "/root")


def test_normalize_job_paths_anchors_relative_only():
    job = {
        "sources": ["a.csv", "/abs/b.csv", "gs://bucket/c.csv"],
        "out": "curated",
        "schema": "schema.json",
        "strategy": {"name": "inline-stays-inline"},
        "arms": [{"name": "a", "strategy": "arm.json"}, {"name": "b", "strategy": {"x": 1}}],
    }
    normalized = normalize_job_paths(job, "/base/dir")
    assert normalized["sources"] == ["/base/dir/a.csv", "/abs/b.csv", "gs://bucket/c.csv"]
    assert normalized["out"] == "/base/dir/curated"
    assert normalized["schema"] == "/base/dir/schema.json"
    assert normalized["strategy"] == {"name": "inline-stays-inline"}
    assert normalized["arms"][0]["strategy"] == "/base/dir/arm.json"
    assert normalized["arms"][1]["strategy"] == {"x": 1}
    assert job["sources"][0] == "a.csv"      # input document is not mutated
