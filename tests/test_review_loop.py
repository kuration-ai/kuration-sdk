"""Product review loop: preview, selection review, artifact diff, and HTML."""
from __future__ import annotations

import json
import subprocess
import sys

from kuration import (DatasetComparison, DatasetPreview, SelectionReview, compare_datasets,
                       input, inspect_dataset, render_html_report, write_html_report)


def _pipeline(n: int = 3):
    sessions = {
        "alpha": [{"t": i, "entity": "a", "x": float(i),
                   "kind": "late" if i >= 4 else "early"} for i in range(8)],
        "beta": [{"t": i, "entity": "b", "x": float(i + 10),
                  "kind": "late" if i >= 4 else "early"} for i in range(8)],
    }
    return (input(sessions, time="t", entity="entity", hz=1)
            .window(4, stride=2).measure("level", "mean", of="x")
            .bin("level", bins=3).sample("top-k", n=n, weights={"level": 1.0})
            .options(mode="strict", seed=5))


def test_preview_reports_quality_examples_and_bounded_sampling():
    preview = _pipeline().preview(max_sources=1, max_ticks_per_source=5, examples=2)
    assert isinstance(preview, DatasetPreview)
    assert preview.sources["sampled"] == 1 and preview.records["ticks"] == 5
    assert preview.fields["x"]["min"] == 0.0
    assert preview.fields["kind"]["top_values"] == {"early": 4, "late": 1}
    assert len(preview.examples) == 2
    assert "bounded-preview" in preview.render()


def test_selection_review_compares_metrics_coverage_and_examples():
    review = _pipeline().review(examples=2)
    assert isinstance(review, SelectionReview)
    assert review.candidates == 6 and review.selected == 3
    assert review.rationale["method"] == "top-k"
    assert review.metrics["level"]["selected"]["mean"] > \
        review.metrics["level"]["candidate"]["mean"]
    assert review.examples[0]["priority"] is not None
    assert "selection: 3/6" in review.render()


def test_compare_datasets_and_static_html(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _pipeline(3).output(left, categorical="omit", shard_length=4)
    _pipeline(2).output(right, categorical="omit", shard_length=4)
    comparison = compare_datasets(left, right, verify=True)
    assert isinstance(comparison, DatasetComparison)
    assert not comparison.identical
    assert comparison.left["shards"] == 3 and comparison.right["shards"] == 2
    assert any(change["path"] == "artifact.shards" for change in comparison.changes)

    inspection = inspect_dataset(left)
    html = render_html_report(inspection, comparison, title="Review <safe>")
    assert "<!doctype html>" in html and "Review &lt;safe&gt;" in html
    report_path = tmp_path / "review.html"
    assert write_html_report(report_path, comparison) == str(report_path)
    assert "DatasetComparison" in report_path.read_text()


def test_kur_diff_cli_json_and_html(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    html = tmp_path / "diff.html"
    _pipeline(3).output(left, categorical="omit", shard_length=4)
    _pipeline(2).output(right, categorical="omit", shard_length=4)
    command = [sys.executable, "-m", "kuration.core.cli.diff", str(left), str(right),
               "--json", "--html", str(html)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["identical"] is False
    assert html.is_file()
