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
        for L in range(1, 5):
            for i in range(8):
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
    p = _run(["kuration.cli.curate", "--help"])
    assert p.returncode == 0 and "job" in p.stdout


def test_kur_curate_runs(tmp_path):
    csv_path = _make_csv(tmp_path)
    out = tmp_path / "out"
    job = tmp_path / "job.json"
    job.write_text(json.dumps(_job(csv_path, out)))
    p = _run(["kuration.cli.curate", str(job)])
    assert p.returncode == 0, p.stderr
    manifest = json.loads((out / "dataset_manifest.json").read_text())
    assert manifest["count"] > 0
    assert (out / "config.json").exists()


def test_kur_experiment_runs(tmp_path):
    csv_path = _make_csv(tmp_path)
    job = tmp_path / "exp.json"
    spec = _job(csv_path, tmp_path / "unused")
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
    p = _run(["kuration.cli.experiment", str(job), "--out", str(cmp_out)])
    assert p.returncode == 0, p.stderr
    out = json.loads(cmp_out.read_text())
    assert out["baseline"] == "base" and "hi" in out["arms"]
