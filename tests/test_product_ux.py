"""End-user planning, estimation, inspection, and actionable-error contracts."""
from __future__ import annotations

import json
import subprocess
import sys

from kuration import ExecutionPlan, input, inspect_dataset


def _pipeline():
    sessions = {
        f"s{source}": [{"t": tick, "entity": f"e{source}", "x": float(source + tick),
                        "label": "high" if tick > 3 else "low"}
                       for tick in range(8)]
        for source in range(2)
    }
    return (input(sessions, time="t", entity="entity", hz=2)
            .window(4, stride=2).measure("level", "mean", of="x")
            .sample("top-k", n=3, weights={"level": 1.0})
            .options(mode="strict", seed=4))


def test_output_dry_run_resolves_output_and_estimates_without_writing(tmp_path):
    destination = tmp_path / "dataset"
    plan = _pipeline().output(destination, container="hdf5", categorical="omit",
                              shard_length=4, dry_run=True, inspect_sources=True)
    assert isinstance(plan, ExecutionPlan)
    assert plan.output["write"] is False and plan.output["container"] == "hdf5"
    assert plan.options["optional_extras"] == ["array"]
    assert plan.options["source_estimate"]["candidate_windows_estimate"] == 6
    assert plan.options["source_estimate"]["output_shards_estimate"] == 3
    assert not destination.exists()
    rendered = plan.render()
    assert "selection: top-k (count=3)" in rendered
    assert "output: hdf5; shard_length=4" in rendered
    assert "categorical-fields-omitted" in rendered


def test_explain_does_not_read_sources_unless_inspection_is_requested():
    pipeline = _pipeline()
    calls = 0
    original = pipeline.adapter.ticks

    def counted(source):
        nonlocal calls
        calls += 1
        return original(source)

    pipeline.adapter.ticks = counted
    pipeline.explain()
    assert calls == 0
    pipeline.explain(inspect_sources=True)
    assert calls == 2


def test_inspect_dataset_summarizes_and_verifies_written_artifact(tmp_path):
    destination = tmp_path / "dataset"
    _pipeline().output(destination, container="npz", categorical="omit", shard_length=4)
    report = inspect_dataset(destination, verify=True)
    assert report["shards"] == 3
    assert report["splits"]["train"]["steps"] == 12
    assert report["modalities"]["x"] == {"shape": [], "dtype": "float32"}
    assert report["verification"] == {"requested": True, "checked": 3,
                                      "valid": True, "errors": []}
    assert report["provenance_fingerprint"] is None  # count path has no resume provenance

    shard_path = next(destination.glob("train/*.npz"))
    shard_path.write_bytes(b"corrupt")
    broken = inspect_dataset(destination, verify=True)
    assert not broken["verification"]["valid"]
    assert broken["verification"]["errors"][0]["path"].endswith(".npz")


def test_kur_inspect_cli_text_and_json(tmp_path):
    destination = tmp_path / "dataset"
    _pipeline().output(destination, container="npz", categorical="omit", shard_length=4)
    command = [sys.executable, "-m", "kuration.cli.inspect", str(destination)]
    text = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "shards: 3" in text.stdout and "x()/float32" in text.stdout
    structured = subprocess.run([*command, "--json"], check=True,
                                capture_output=True, text=True)
    assert json.loads(structured.stdout)["verification"]["requested"] is False


def test_cli_empty_pool_error_includes_remedy(tmp_path):
    # Existing CLI tests cover the failure itself; this pins the new actionable suffix.
    from kuration.cli._common import format_error
    error = RuntimeError("segment pool is EMPTY: no windows")
    rendered = format_error(error)
    assert "hint:" in rendered and "window is not longer" in rendered
