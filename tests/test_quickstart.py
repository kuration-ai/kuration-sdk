"""Keep the landing-page quick start executable and its claims honest."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from kuration import iter_shards


ROOT = Path(__file__).resolve().parents[1]


def test_documented_quickstart(tmp_path):
    out_dir = tmp_path / "quickstart-out"
    proc = subprocess.run(
        [sys.executable, "examples/quickstart.py", "--out", str(out_dir)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "42 candidate windows -> 12 selected" in proc.stdout
    assert "6 sessions, 288 rows" in proc.stdout
    assert (out_dir / "config.json").is_file()
    assert (out_dir / "dataset_manifest.json").is_file()

    config = json.loads((out_dir / "config.json").read_text())
    assert [op["op"] for op in config["pipeline"]] == [
        "input", "window", "measure", "measure", "bin", "bin", "split",
        "sample", "options",
    ]
    shards = list(iter_shards(out_dir))
    assert len(shards) == 12
    assert shards[0]["signal"].shape == (12,)
