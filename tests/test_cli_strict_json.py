"""Every structured CLI surface emits standards-compliant JSON through one encoder."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from conftest import run_cli


def test_cli_modules_centralize_json_encoding():
    cli_root = Path(__file__).parents[1] / "src" / "kuration" / "core" / "cli"
    bypasses = []
    for path in cli_root.glob("*.py"):
        if path.name == "_common.py":
            continue
        if "json.dumps(" in path.read_text():
            bypasses.append(path.name)
    assert bypasses == []


def test_explain_json_rejects_nonfinite_output_without_a_traceback(monkeypatch):
    from kuration.core.cli import explain

    monkeypatch.setattr(explain, "read_job", lambda path: {})
    monkeypatch.setattr(explain, "explain_job", lambda job: {"score": float("nan")})

    result = run_cli("kuration.core.cli.explain", "job.json", "--json", expect=1)
    assert result.stdout == ""
    assert "strict JSON" in result.stderr
    assert "NaN" not in result.stderr and "Traceback" not in result.stderr


def test_checkpoint_json_rejects_nonfinite_scan_results(monkeypatch, tmp_path):
    from kuration.core.cli import checkpoint

    monkeypatch.setattr(
        checkpoint,
        "scan_checkpoints",
        lambda *args, **kwargs: {"invalid": 0, "score": float("inf")},
    )
    result = run_cli(
        "kuration.core.cli.checkpoint", "list", tmp_path, "--json", expect=1
    )
    assert result.stdout == ""
    assert "strict JSON" in result.stderr
    assert "Infinity" not in result.stderr


def test_analyzer_json_rejects_nonfinite_plugin_results(monkeypatch, tmp_path):
    from kuration.core import analyzers

    monkeypatch.setattr(
        analyzers,
        "run_dataset_analyzer",
        lambda *args, **kwargs: {
            "passed": True,
            "summary": {"score": float("-inf")},
        },
    )
    result = run_cli(
        "kuration.core.cli.check",
        tmp_path,
        "--analyzer",
        "test/analyzer",
        "--json",
        expect=1,
    )
    assert result.stdout == ""
    assert "strict JSON" in result.stderr
    assert "Infinity" not in result.stderr


def test_curate_review_json_is_not_published_on_nonfinite_evidence(
        monkeypatch, tmp_path):
    from kuration.core import review as review_module
    from kuration.core.cli import curate

    job = {"out": str(tmp_path / "dataset"), "train_hours": 1.0}
    resolved = SimpleNamespace(
        kind="curate",
        schema=object(),
        adapter=object(),
        sources=("source",),
        strategy=object(),
        segmenter=object(),
        output=object(),
        encoder=None,
        on_error="raise",
        registry=None,
    )
    result = {
        "admitted": [("source", "entity", 0)],
        "results": {
            "budget": {
                "emitted_seconds": {"train": 1.0, "test": 0.0},
                "requested_seconds": {"train": 1.0, "test": 0.0},
            },
            "tracks_admitted": 1,
            "tracks_total": 1,
            "shards": 1,
            "n_train_sources": 1,
            "n_test_sources": 0,
            "sources_skipped": 0,
        },
    }
    review = SimpleNamespace(
        to_dict=lambda: {"score": float("nan")},
        render=lambda: "review",
    )
    monkeypatch.setattr(curate, "read_job", lambda path: job)
    monkeypatch.setattr(curate, "resolve_job", lambda value: resolved)
    monkeypatch.setattr(curate, "write_dataset", lambda *args, **kwargs: result)
    monkeypatch.setattr(review_module, "hours_selection_review", lambda value: review)
    output = tmp_path / "review.json"

    command = run_cli(
        "kuration.core.cli.curate",
        "job.json",
        "--review-json",
        output,
        expect=1,
    )
    assert "strict JSON" in command.stderr
    assert not output.exists()


def test_validate_print_schema_remains_strict_json():
    result = run_cli("kuration.core.cli.validate", "--print-schema")
    schema = json.loads(result.stdout)
    assert schema["type"] == "object"
