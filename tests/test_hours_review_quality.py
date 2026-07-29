"""Hours-path admission review and reusable quality gates."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from kuration import (HoursSelectionReview, QualityGateError, check_quality,
                       evaluate_quality, input, inspect_dataset)


def _pipeline(*, hours=False):
    sessions = {
        f"s{source}": [{"t": tick, "entity": f"e{source}",
                         "x": float(source * 10 + tick)} for tick in range(8)]
        for source in range(2)
    }
    pipeline = (input(sessions, time="t", entity="entity", hz=1)
                .window(4, stride=2).measure("level", "mean", of="x"))
    return (pipeline.sample("top-k", hours=8 / 3600, weights={"level": 1.0})
            if hours else pipeline.sample("top-k", n=3, weights={"level": 1.0}))


def test_hours_review_reuses_writer_budget_and_decisions(tmp_path):
    review = _pipeline(hours=True).review(
        path=tmp_path / "dataset", categorical="omit", shard_length=4, span_steps=4)
    assert isinstance(review, HoursSelectionReview)
    assert review.candidate_items == 4 and review.admitted_items == 2
    assert review.budget["train"]["utilization"] == 1.0
    assert review.decision_counts["priority"] == 2
    assert review.decision_counts["budget-filled"] == 2
    assert review.examples["admitted"][0]["priority"] is not None
    assert review.examples["rejected"][0]["decision"] == "budget-filled"
    assert "requested=8.000s" in review.render()


def test_pipeline_expect_applies_to_preview_and_review():
    checked = _pipeline().expect(max_missing={"x": 0}, min_sources=2)
    assert checked.describe()["expectations"]["min_sources"] == 2
    preview = checked.preview()
    assert preview.quality["passed"] is True
    with pytest.raises(QualityGateError) as error:
        _pipeline().expect(min_sources=3).preview()
    assert error.value.result["checks"][0]["name"] == "min_sources"

    review = _pipeline().expect(max_group_fraction=1.0,
                                max_skipped_sources=0,
                                min_tag_coverage={"unused": 0}).review()
    assert review.quality["passed"] is True


def test_hours_review_quality_gates_and_direct_evaluation(tmp_path):
    review = _pipeline(hours=True).expect(
        min_budget_utilization=1.0,
        max_budget_underfill_seconds=0,
        max_skipped_sources=0,
    ).review(path=tmp_path / "dataset", categorical="omit", shard_length=4, span_steps=4)
    assert review.quality["passed"] is True
    failed = evaluate_quality(review, min_sources=3)
    assert not failed.passed
    with pytest.raises(QualityGateError):
        check_quality(review, min_sources=3)


def test_kur_check_artifact_policy_pass_and_fail(tmp_path):
    destination = tmp_path / "dataset"
    _pipeline(hours=True).output(
        destination, categorical="omit", shard_length=4, span_steps=4)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "require_verified": True,
        "require_modalities": ["x", "_present"],
        "min_sources": 1,
        "min_shards": 2,
        "min_budget_utilization": 1.0,
        "max_skipped_sources": 0,
    }))
    command = [sys.executable, "-m", "kuration.core.cli.check", str(destination),
               "--policy", str(policy), "--json"]
    passed = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(passed.stdout)["passed"] is True

    policy.write_text(json.dumps({"min_shards": 99}))
    failed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert failed.returncode == 2
    assert json.loads(failed.stdout)["passed"] is False
    assert inspect_dataset(destination).sources == 1


def test_kur_check_can_run_dataset_analyzer_without_quality_policy(tmp_path):
    destination = tmp_path / "dataset"
    destination.mkdir()
    (destination / "config.json").write_text(json.dumps({"output": {"container": "npz"}}))
    (destination / "dataset_manifest.json").write_text(json.dumps({
        "shards": [
            {"path": "train/a.npz", "source_id": "clip-1", "split": "train"},
            {"path": "test/b.npz", "source_id": "clip-1", "split": "test"},
        ]
    }))

    command = [
        sys.executable,
        "-m",
        "kuration.core.cli.check",
        str(destination),
        "--analyzer",
        "kuration/split-leakage",
        "--json",
    ]
    failed = subprocess.run(command, check=False, capture_output=True, text=True)

    assert failed.returncode == 2
    result = json.loads(failed.stdout)
    assert result["name"] == "kuration/split-leakage"
    assert result["passed"] is False


def test_quality_policy_rejects_unknown_or_invalid_expectations():
    with pytest.raises(ValueError, match="unknown quality"):
        _pipeline().expect(perfect=True)
    with pytest.raises(ValueError, match="max_missing"):
        _pipeline().expect(max_missing={"x": 2.0})
