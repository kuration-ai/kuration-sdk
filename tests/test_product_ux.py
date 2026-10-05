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
    return (input(sessions, time="t", entity="entity", hz=1)
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


def test_output_warns_when_shard_length_does_not_evenly_divide_the_window(tmp_path):
    """F7: a window=4 split into shard_length=3 shards silently became 3+1 shards with no
    mention in `explain()`/the dry-run preview or the written config.json's warnings. Both now
    carry a `shard-length-chops-windows` warning naming the resulting shard split."""
    pipeline = _pipeline()  # window(4, stride=2)
    destination = tmp_path / "dataset"

    plan = pipeline.output(destination, container="hdf5", categorical="omit",
                           shard_length=3, dry_run=True)
    codes = [w["code"] for w in plan.warnings]
    assert "shard-length-chops-windows" in codes
    warning = next(w for w in plan.warnings if w["code"] == "shard-length-chops-windows")
    assert "shard_length=3" in warning["message"] and "window length (4)" in warning["message"]

    written = pipeline.output(destination, container="hdf5", categorical="omit", shard_length=3)
    cfg = json.loads((destination / "config.json").read_text())
    assert "shard-length-chops-windows" in [w["code"] for w in cfg["warnings"]]
    assert written is not None


def test_output_does_not_warn_when_shard_length_evenly_divides_the_window(tmp_path):
    destination = tmp_path / "dataset"
    plan = _pipeline().output(destination, container="hdf5", categorical="omit",
                              shard_length=4, dry_run=True)
    assert "shard-length-chops-windows" not in [w["code"] for w in plan.warnings]


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


def test_shard_filenames_are_content_derived_not_rank_derived(tmp_path):
    """F1: two curations over the same source that each select a different window at the same
    output rank must not alias one shard filename. Previously both would have hashed
    (source_id, entity_id, "_seg0000") regardless of which window was actually selected."""
    rows = [{"t": float(i), "x": float(i)} for i in range(20)]

    def _pipe(weight_sign):
        return (input({"s0": rows}, time="t").window(4, stride=4)
                .measure("x_mean", "mean", of="x")
                .sample("top-k", n=1, weights={"x_mean": weight_sign})
                .options(seed=0))

    first = tmp_path / "first"
    second = tmp_path / "second"
    _pipe(1.0).output(first, container="npz", categorical="omit", shard_length=4)
    _pipe(-1.0).output(second, container="npz", categorical="omit", shard_length=4)

    first_shard = next((first / "train").glob("*.npz")).name
    second_shard = next((second / "train").glob("*.npz")).name
    assert first_shard != second_shard


def test_inspect_dataset_summarizes_and_verifies_written_artifact(tmp_path):
    destination = tmp_path / "dataset"
    _pipeline().output(destination, container="npz", categorical="omit", shard_length=4)
    report = inspect_dataset(destination, verify=True)
    assert report["shards"] == 3
    assert report["splits"]["train"]["steps"] == 12
    assert report["modalities"]["x"] == {"shape": [], "dtype": "float32"}
    assert report["verification"] == {"requested": True, "checked": 3,
                                      "valid": True, "errors": []}
    assert len(report["provenance_fingerprint"]) == 64

    shard_path = next(destination.glob("train/*.npz"))
    shard_path.write_bytes(b"corrupt")
    broken = inspect_dataset(destination, verify=True)
    assert not broken["verification"]["valid"]
    assert broken["verification"]["errors"][0]["path"].endswith(".npz")


def test_kur_inspect_cli_text_and_json(tmp_path):
    destination = tmp_path / "dataset"
    _pipeline().output(destination, container="npz", categorical="omit", shard_length=4)
    command = [sys.executable, "-m", "kuration.core.cli.inspect", str(destination)]
    text = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "shards: 3" in text.stdout and "x()/float32" in text.stdout
    structured = subprocess.run([*command, "--json"], check=True,
                                capture_output=True, text=True)
    assert json.loads(structured.stdout)["verification"]["requested"] is False


def test_cli_empty_pool_error_includes_remedy(tmp_path):
    # Existing CLI tests cover the failure itself; this pins the new actionable suffix.
    from kuration.core.cli._common import format_error
    error = RuntimeError("segment pool is EMPTY: no windows")
    rendered = format_error(error)
    assert "hint:" in rendered and "window is not longer" in rendered
