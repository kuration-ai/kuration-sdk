"""kur-curate / kur-experiment CLIs driven by a JSON job spec."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def _make_csv(tmp_path):
    p = tmp_path / "d.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "id", "x", "y", "act"])
        w.writeheader()
        for i in range(8):                    # time-major: the adapter's documented row order
            for L in range(1, 5):
                w.writerow({"t": i, "id": f"e{L}", "x": i, "y": L, "act": L})
    return p


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
        "output": {"shard_length": 4, "container": "npz",
                   "modalities": {"obs": {"dtype": "float32", "shape": [2], "source": "field:pos"}}},
        "train_hours": 100, "test_hours": 0, "out": str(out_dir),
    }


def _run(args):
    return subprocess.run([sys.executable, "-m", args[0]] + args[1:],
                          capture_output=True, text=True, timeout=60)


def test_kur_curate_help():
    p = _run(["kuration.core.cli.curate", "--help"])
    assert p.returncode == 0 and "job" in p.stdout


def test_cli_failure_is_one_line_no_traceback(tmp_path):
    """A run-time failure prints one `error:` line (exit 1), never a raw traceback."""
    csv_path = _make_csv(tmp_path)
    spec = _job(csv_path, tmp_path / "out")
    spec["adapter"]["fields"]["act"] = "activity"       # column doesn't exist -> empty pool
    job = tmp_path / "job.json"
    job.write_text(json.dumps(spec))
    p = _run(["kuration.core.cli.curate", str(job)])
    assert p.returncode == 1
    assert "Traceback" not in p.stderr, p.stderr
    assert "error: RuntimeError: segment pool is EMPTY" in p.stderr

    p = _run(["kuration.core.cli.curate", str(job), "--debug"])          # --debug re-raises
    assert p.returncode != 0 and "Traceback" in p.stderr


def test_cli_missing_job_file_is_one_line(tmp_path):
    p = _run(["kuration.core.cli.curate", str(tmp_path / "nope.json")])
    assert p.returncode == 1
    assert "Traceback" not in p.stderr and "error:" in p.stderr and "not found" in p.stderr


def test_kur_curate_runs(tmp_path):
    csv_path = _make_csv(tmp_path)
    out = tmp_path / "out"
    job = tmp_path / "job.json"
    job.write_text(json.dumps(_job(csv_path, out)))
    p = _run(["kuration.core.cli.curate", str(job)])
    assert p.returncode == 0, p.stderr
    manifest = json.loads((out / "dataset_manifest.json").read_text())
    assert manifest["count"] > 0
    assert (out / "config.json").exists()


def test_kur_curate_requires_explicit_force_to_replace(tmp_path):
    csv_path = _make_csv(tmp_path)
    out = tmp_path / "out"
    job = tmp_path / "job.json"
    job.write_text(json.dumps(_job(csv_path, out)))

    assert _run(["kuration.core.cli.curate", str(job)]).returncode == 0
    guarded = _run(["kuration.core.cli.curate", str(job)])
    assert guarded.returncode == 1
    assert "force=True" in guarded.stderr

    replaced = _run(["kuration.core.cli.curate", str(job), "--force"])
    assert replaced.returncode == 0, replaced.stderr


def test_kur_explain_is_actionable_in_text_and_json(tmp_path):
    csv_path = _make_csv(tmp_path)
    job = tmp_path / "job.json"
    job.write_text(json.dumps(_job(csv_path, tmp_path / "out")))
    text_plan = _run(["kuration.core.cli.explain", str(job)])
    assert text_plan.returncode == 0, text_plan.stderr
    assert "selection: top-k (train=100h, test=0h)" in text_plan.stdout
    assert "execution: local; workers=1" in text_plan.stdout
    assert "force=False" in text_plan.stdout
    assert "fix: Set mode='strict'" in text_plan.stdout
    json_plan = _run(["kuration.core.cli.explain", str(job), "--json"])
    plan = json.loads(json_plan.stdout)
    assert plan["selection"]["budget"] == {"kind": "hours", "train": 100, "test": 0}
    assert plan["execution"]["force"] is False
    assert plan["validation"]["n_sources"] == 1


def test_kur_report_with_scorer(tmp_path):
    """Regression: a strategy that references a scorer output (here in weights) must not push
    that output into the report's feature-pass `registered` list — doing so makes the engine
    raise 'unknown registered metric' and skip EVERY source, yielding a silent empty report."""
    csv_path = _make_csv(tmp_path)
    job = tmp_path / "rep.json"
    spec = _job(csv_path, tmp_path / "unused")
    # action_residual is registered at import and the report includes its computed output.
    spec["strategy"] = {
        "name": "withscorer",
        "define": {"am": {"source": "e.act", "reduce": "mean"}},
        "scorers": [{"name": "action_residual", "as": "resid", "obs": "pos", "action": "act"}],
        "selection": {"method": "top-k", "weights": {"resid": 1.0, "am": 0.5}},
    }
    out_json = tmp_path / "report.json"
    job.write_text(json.dumps(spec))
    p = _run(["kuration.core.cli.report", str(job), "--out", str(out_json)])
    assert p.returncode == 0, p.stderr
    assert "unknown registered metric" not in p.stderr
    assert "segment pool is" not in p.stderr.replace("\n", " ")   # the all-skipped warning
    rep = json.loads(out_json.read_text())
    assert rep["n_segments"] > 0, f"empty report — sources were skipped:\n{p.stdout}\n{p.stderr}"
    assert "resid" in rep["metrics"]


def test_kur_experiment_runs(tmp_path):
    csv_path = _make_csv(tmp_path)
    job = tmp_path / "exp.json"
    spec = _job(csv_path, tmp_path / "unused")
    for key in ("strategy", "output", "train_hours", "test_hours", "out"):
        spec.pop(key)
    spec.update({
        "target_size": 6, "baseline": "base",
        "arms": [
            {"name": "base", "strategy": {"name": "base", "define": {"am": {"source": "e.act", "reduce": "mean"}},
                                          "binning": {"axes": [{"metric": "am", "bins": 4}]},
                                          "selection": {"method": "uniform-bins"}}},
            {"name": "hi", "strategy": {"name": "hi", "define": {"am": {"source": "e.act", "reduce": "mean"}},
                                        "selection": {"method": "top-k", "weights": {"am": 1.0}}}},
        ],
    })
    job.write_text(json.dumps(spec))
    cmp_out = tmp_path / "cmp.json"
    p = _run(["kuration.core.cli.experiment", str(job), "--out", str(cmp_out)])
    assert p.returncode == 0, p.stderr
    out = json.loads(cmp_out.read_text())
    assert out["baseline"] == "base" and "hi" in out["arms"]


def test_cli_error_formatter_enforces_one_line_contract():
    from kuration.core.cli._common import format_error
    rendered = format_error(ValueError("first line\nsecond\tline"))
    assert rendered == "error: ValueError: first line second line"
    assert len(rendered.splitlines()) == 1
