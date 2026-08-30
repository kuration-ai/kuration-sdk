"""Shared fixtures: import path, jupyter server config, synthetic sources and jobs.

The package is importable either from an installed environment or straight from the source
tree (``src/`` is prepended to ``sys.path`` and ``PYTHONPATH`` so spawned worker subprocesses
can import ``kuration_sdk_lab`` too).
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
os.environ["PYTHONPATH"] = os.pathsep.join(
    [_SRC] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])

pytest_plugins = ("pytest_jupyter.jupyter_server",)


@pytest.fixture
def jp_server_config(jp_server_config, tmp_path):
    return {
        "ServerApp": {"jpserver_extensions": {"kuration_sdk_lab": True}},
        "KurationLabApp": {"runs_dir": str(tmp_path / "runs")},
    }


def make_csv(directory: Path, name: str = "d.csv", *, steps: int = 12,
             entities: int = 3) -> Path:
    path = Path(directory) / name
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["t", "id", "x", "y", "act", "team"])
        writer.writeheader()
        for step in range(steps):              # time-major, the adapter's documented order
            for entity in range(1, entities + 1):
                writer.writerow({"t": step, "id": f"e{entity}", "x": step * 0.5,
                                 "y": entity * 1.5, "act": entity,
                                 "team": "A" if entity < 3 else "B"})
    return path


def make_job(directory: Path, *, sources: list[str] | None = None,
             out: str | None = None, train_hours: float = 0.01) -> dict:
    from kuration.core.cli.init_job import build_job

    directory = Path(directory)
    if sources is None:
        sources = [str(make_csv(directory))]
    return build_job(sources, out=out or str(directory / "curated"),
                     window=4, stride=4, train_hours=train_hours, test_hours=0.0,
                     preset="numpy-timeseries", seed=0)


@pytest.fixture
def tiny_job(tmp_path) -> dict:
    return make_job(tmp_path)


@pytest.fixture
def written_dataset(tmp_path) -> str:
    """A committed dataset, produced the same way the worker produces one."""
    from kuration.core.cli.job import resolve_job
    from kuration_sdk_lab.jobs.worker import _run_curate

    job = make_job(tmp_path)
    _run_curate(job, resolve_job(job), None)
    return job["out"]
