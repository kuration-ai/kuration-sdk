"""In-process coverage for the artifact, comparison, and maintenance CLIs.

`test_cli.py` and `test_import_cli.py` already pin these commands' shell behavior through
subprocesses. These run the same entry points in-process so their logic is exercised under
coverage, and cover the branches a happy-path shell test does not reach.
"""
from __future__ import annotations

import csv
import json

import pytest
from conftest import run_cli


def _make_csv(tmp_path, name="d.csv"):
    path = tmp_path / name
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["t", "id", "x", "y", "act"])
        writer.writeheader()
        for step in range(12):
            for entity in range(1, 4):
                # `act` varies with time as well as entity, so window means differ and a
                # binning axis over them carries real signal.
                writer.writerow({"t": step, "id": f"e{entity}", "x": step * 0.5,
                                 "y": entity * 1.5, "act": entity + step * 0.25})
    return path


def _job(csv_path, out_dir):
    return {
        "schema": {"name": "j", "entity_axis": "e", "sample_hz": 1.0,
                   "fields": {"pos": {"kind": "vector", "dims": ["x", "y"]},
                              "act": {"kind": "scalar"}}},
        "adapter": {"type": "timeseries", "fields": {"pos": ["x", "y"], "act": "act"},
                    "time_column": "t", "entity_column": "id"},
        "sources": [str(csv_path)],
        "segmenter": {"window": 4, "stride": 4},
        "strategy": {"name": "hi", "define": {"am": {"source": "e.act", "reduce": "mean"}},
                     "selection": {"method": "top-k", "weights": {"am": 1.0}}},
        "output": {"shard_length": 4, "container": "npz", "modalities": {
            "obs": {"dtype": "float32", "shape": [2], "source": "field:pos"}}},
        "train_hours": 100, "test_hours": 0, "out": str(out_dir),
    }


@pytest.fixture()
def curated(tmp_path):
    """A written dataset plus the job that produced it."""
    source = _make_csv(tmp_path)
    out = tmp_path / "curated"
    job = tmp_path / "job.json"
    job.write_text(json.dumps(_job(source, out)))
    run_cli("kuration.core.cli.curate", job, "--quiet")
    return job, out


# ---- kur-check ----------------------------------------------------------------
def test_check_passes_and_fails_a_quality_policy(tmp_path, curated):
    _job_path, out = curated
    policy = tmp_path / "policy.json"

    policy.write_text(json.dumps({"min_shards": 1}))
    result = run_cli("kuration.core.cli.check", out, "--policy", policy)
    assert "PASS" in result.stdout

    # A policy the artifact cannot satisfy must fail the gate with exit code 2.
    policy.write_text(json.dumps({"min_shards": 10_000}))
    failed = run_cli("kuration.core.cli.check", out, "--policy", policy, expect=2)
    assert "FAIL" in failed.stdout


def test_check_emits_json_and_html(tmp_path, curated):
    _job_path, out = curated
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"expectations": {"min_shards": 1}}))
    html = tmp_path / "check.html"
    result = run_cli("kuration.core.cli.check", out, "--policy", policy,
                     "--json", "--html", html)
    # With --policy alone the JSON is the gate result itself, not a wrapper.
    assert json.loads(result.stdout)["passed"] is True
    assert html.exists() and "<html" in html.read_text().lower()


def test_check_requires_a_policy_or_analyzer(curated):
    _job_path, out = curated
    result = run_cli("kuration.core.cli.check", out, expect=1)
    assert "--policy and/or --analyzer" in result.stderr


def test_check_verify_without_policy_is_rejected_not_silently_ignored(tmp_path):
    """--verify decodes shards for the *policy* evaluation; in analyzer-only mode it used to
    be parsed and silently dropped. It must now say so rather than pretend to apply."""
    result = run_cli("kuration.core.cli.check", tmp_path / "unused",
                     "--analyzer", "kuration/split-leakage", "--verify", expect=1)
    assert "--verify requires --policy" in result.stderr
    assert "Traceback" not in result.stderr


def test_check_verify_still_works_alongside_a_policy(tmp_path, curated):
    _job_path, out = curated
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"min_shards": 1}))
    result = run_cli("kuration.core.cli.check", out, "--policy", policy, "--verify")
    assert "PASS" in result.stdout


# ---- kur-diff ------------------------------------------------------------------
def test_diff_reports_identical_and_changed_artifacts(tmp_path, curated):
    job_path, out = curated
    same = run_cli("kuration.core.cli.diff", out, out)
    assert "identical: True" in same.stdout

    spec = json.loads(job_path.read_text())
    other = tmp_path / "curated2"
    spec["out"] = str(other)
    spec["segmenter"] = {"window": 6, "stride": 6}
    changed_job = tmp_path / "job2.json"
    changed_job.write_text(json.dumps(spec))
    run_cli("kuration.core.cli.curate", changed_job, "--quiet")

    changed = run_cli("kuration.core.cli.diff", out, other, "--json")
    assert json.loads(changed.stdout)["identical"] is False


# ---- kur-report / kur-experiment -------------------------------------------------
def test_report_writes_a_metric_profile(tmp_path):
    source = _make_csv(tmp_path)
    spec = _job(source, tmp_path / "unused")
    for key in ("output", "train_hours", "test_hours", "out"):
        spec.pop(key)
    job = tmp_path / "rep.json"
    job.write_text(json.dumps(spec))
    out_json = tmp_path / "report.json"
    run_cli("kuration.core.cli.report", job, "--out", out_json, "--quiet")
    report = json.loads(out_json.read_text())
    assert report["n_segments"] > 0 and "am" in report["metrics"]


def test_experiment_compares_arms(tmp_path):
    source = _make_csv(tmp_path)
    spec = _job(source, tmp_path / "unused")
    for key in ("strategy", "output", "train_hours", "test_hours", "out"):
        spec.pop(key)
    define = {"am": {"source": "e.act", "reduce": "mean"}}
    spec.update({"target_size": 4, "baseline": "base", "arms": [
        {"name": "base", "strategy": {"name": "base", "define": define,
                                      "binning": {"axes": [{"metric": "am", "bins": 3}]},
                                      "selection": {"method": "uniform-bins"}}},
        {"name": "hi", "strategy": {"name": "hi", "define": define,
                                    "selection": {"method": "top-k",
                                                  "weights": {"am": 1.0}}}},
    ]})
    job = tmp_path / "exp.json"
    job.write_text(json.dumps(spec))
    out_json = tmp_path / "cmp.json"
    run_cli("kuration.core.cli.experiment", job, "--out", out_json, "--quiet")
    comparison = json.loads(out_json.read_text())
    assert comparison["baseline"] == "base" and "hi" in comparison["arms"]


# ---- kur-import ------------------------------------------------------------------
def test_import_detects_an_adapter_and_writes_an_artifact(tmp_path):
    import numpy as np

    source = tmp_path / "episode.npz"
    np.savez(source, obs=np.arange(40, dtype=np.float32).reshape(20, 2),
             act=np.arange(20, dtype=np.float32))
    out = tmp_path / "imported"
    result = run_cli("kuration.core.cli.import_dataset", source, "--out", out, "--json")
    assert json.loads(result.stdout)["shards"] >= 1
    assert (out / "dataset_manifest.json").exists()

    # A second import into the same directory must refuse rather than silently overwrite.
    blocked = run_cli("kuration.core.cli.import_dataset", source, "--out", out, expect=1)
    assert "force" in blocked.stderr.lower()


# ---- kur-migrate -------------------------------------------------------------------
def test_migrate_stamps_the_current_job_version(tmp_path):
    from kuration import JOB_SCHEMA_VERSION

    job = tmp_path / "old.json"
    job.write_text(json.dumps(_job(_make_csv(tmp_path), tmp_path / "out")))
    migrated = tmp_path / "new.json"
    run_cli("kuration.core.cli.migrate", job, "--out", migrated)
    assert json.loads(migrated.read_text())["job_schema_version"] == JOB_SCHEMA_VERSION


# ---- kur-plugin-check ----------------------------------------------------------------
def test_plugin_check_validates_installed_plugins():
    result = run_cli("kuration.core.cli.plugin_check", "--json")
    payload = json.loads(result.stdout)
    assert isinstance(payload, (dict, list))


# ---- kur-list / kur-describe ------------------------------------------------------
def test_discovery_lists_and_describes_components():
    listing = run_cli("kuration.core.cli.discovery:list_main", "adapters")
    assert "timeseries" in listing.stdout

    described = run_cli("kuration.core.cli.discovery:describe_main",
                        "adapter", "timeseries", "--json")
    assert json.loads(described.stdout)["name"] == "timeseries"


# ---- kur-inspect ---------------------------------------------------------------------
def test_inspect_verifies_shard_integrity(curated):
    _job_path, out = curated
    verified = run_cli("kuration.core.cli.inspect", out, "--verify")
    assert "verification: PASS" in verified.stdout

    # Corrupting a payload must be detected and reported with exit code 2.
    shard = json.loads((out / "dataset_manifest.json").read_text())["shards"][0]["path"]
    (out / shard).write_bytes(b"not an npz")
    damaged = run_cli("kuration.core.cli.inspect", out, "--verify", expect=2)
    assert "verification: FAIL" in damaged.stdout
