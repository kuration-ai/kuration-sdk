"""kur-validate — preflight a job spec without reading data."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from kuration.cli.validate import job_kind, validate_job


def _csv(tmp_path):
    p = tmp_path / "d.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "id", "x", "y", "act"])
        w.writeheader()
        for t in range(4):
            w.writerow({"t": t, "id": "e1", "x": t, "y": 1, "act": 1})
    return p


def _curate_job(csv_path, out_dir):
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
        "train_hours": 1, "out": str(out_dir),
    }


def test_job_kind():
    assert job_kind({"output": {}}) == "curate"
    assert job_kind({"train_hours": 1}) == "curate"
    assert job_kind({"arms": []}) == "experiment"
    assert job_kind({"strategy": {}}) == "report"


def test_validate_good_curate_job(tmp_path):
    job = _curate_job(_csv(tmp_path), tmp_path / "out")
    summary = validate_job(job)
    assert summary["kind"] == "curate" and summary["schema"] == "j" and summary["n_sources"] == 1


def test_validate_catches_bad_metric_reference(tmp_path):
    job = _curate_job(_csv(tmp_path), tmp_path / "out")
    job["strategy"]["selection"]["weights"] = {"ghost": 1.0}     # undefined metric
    import pytest
    with pytest.raises(ValueError, match="undefined metric"):
        validate_job(job)


def test_validate_catches_bad_dsl_reduce(tmp_path):
    job = _curate_job(_csv(tmp_path), tmp_path / "out")
    job["strategy"]["define"]["am"]["reduce"] = "frobnicate"     # unknown reduce -> loud at load
    import pytest
    with pytest.raises(ValueError, match="reduce"):
        validate_job(job)


def test_validate_curate_missing_output(tmp_path):
    job = _curate_job(_csv(tmp_path), tmp_path / "out")
    del job["output"]                                            # still a curate job (train_hours)
    import pytest
    with pytest.raises(ValueError, match="output"):
        validate_job(job)


def test_check_sources_flags_missing(tmp_path):
    job = _curate_job(_csv(tmp_path), tmp_path / "out")
    job["sources"] = [str(tmp_path / "nope.csv")]
    import pytest
    with pytest.raises(ValueError, match="do not exist"):
        validate_job(job, check_sources=True)


def test_check_sources_sniffs_csv_header(tmp_path):
    """A column typo in the adapter map must fail at preflight, not as N skipped sources."""
    job = _curate_job(_csv(tmp_path), tmp_path / "out")
    job["adapter"]["fields"]["act"] = "activity"                 # column doesn't exist
    import pytest
    with pytest.raises(ValueError, match=r"missing mapped column\(s\).*activity"):
        validate_job(job, check_sources=True)
    assert validate_job(job)["kind"] == "curate"                 # without the flag: no data I/O


def test_check_sources_sniffs_parquet_schema(tmp_path):
    import pytest
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq
    p = tmp_path / "d.parquet"
    pq.write_table(pa.table({"t": [0, 1], "id": ["e1", "e1"], "x": [0.0, 1.0],
                             "y": [1.0, 1.0], "act": [1.0, 1.0]}), p)
    job = _curate_job(p, tmp_path / "out")
    assert validate_job(job, check_sources=True)["sources_checked"]
    job["adapter"]["entity_column"] = "entity"                   # column doesn't exist
    with pytest.raises(ValueError, match=r"missing mapped column\(s\).*entity"):
        validate_job(job, check_sources=True)


def test_cli_missing_dep_is_error_not_invalid(tmp_path, monkeypatch):
    """--json-schema without jsonschema: an environment problem, labeled `error:` not INVALID."""
    import pytest
    try:
        import jsonschema  # noqa: F401
        pytest.skip("jsonschema installed — the missing-dep path is unreachable")
    except ImportError:
        pass
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_curate_job(_csv(tmp_path), tmp_path / "out")))
    p = _run([str(good), "--json-schema"])
    assert p.returncode == 1 and p.stderr.startswith("error:") and "INVALID" not in p.stderr


# ---- CLI wiring -------------------------------------------------------------
def _run(args):
    return subprocess.run([sys.executable, "-m", "kuration.cli.validate"] + args,
                          capture_output=True, text=True, timeout=60)


def test_cli_ok_and_invalid(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_curate_job(_csv(tmp_path), tmp_path / "out")))
    p = _run([str(good)])
    assert p.returncode == 0 and p.stdout.startswith("OK:"), (p.stdout, p.stderr)

    bad_job = _curate_job(_csv(tmp_path), tmp_path / "out")
    bad_job["strategy"]["define"]["am"]["source"] = "e.nope"     # unknown field
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(bad_job))
    p = _run([str(bad)])
    assert p.returncode == 1 and "INVALID" in p.stderr


def test_cli_print_schema():
    p = _run(["--print-schema"])
    assert p.returncode == 0
    doc = json.loads(p.stdout)
    assert doc["title"] == "kuration job spec" and "strategy" in doc["properties"]
