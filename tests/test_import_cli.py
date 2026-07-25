"""Lossless NPZ-to-Kuration import API and CLI."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from kuration import import_dataset, inspect_dataset, iter_shards


def _source(path, *, offset=0, rows=5):
    frames = np.arange(rows * 2 * 3 * 3, dtype=np.uint8).reshape(rows, 2, 3, 3)
    actions = np.arange(offset, offset + rows * 4, dtype=np.float32).reshape(rows, 4)
    np.savez(path, frames=frames, actions=actions)
    return frames, actions


def test_import_dataset_preserves_npz_arrays_and_artifact_contract(tmp_path):
    source = tmp_path / "clip.npz"
    frames, actions = _source(source)
    destination = tmp_path / "dataset"

    result = import_dataset(source, destination)

    assert result == {
        "dataset": str(destination),
        "shards": 1,
        "sources": 1,
        "steps": 5,
        "sources_skipped": 0,
        "modalities": ["actions", "frames"],
    }
    arrays, entry = next(iter_shards(destination, with_meta=True))
    np.testing.assert_array_equal(arrays["frames"], frames)
    np.testing.assert_array_equal(arrays["actions"], actions)
    assert entry["source_id"] == "clip"
    report = inspect_dataset(destination, verify=True)
    assert report.verification["valid"] is True
    assert report.modalities["frames"] == {"shape": [2, 3, 3], "dtype": "uint8"}
    config = json.loads((destination / "config.json").read_text())
    assert config["import"]["sources"][0]["source"] == "clip.npz"
    assert str(tmp_path) not in json.dumps(config)


def test_import_dataset_directory_and_explicit_sharding(tmp_path):
    inputs = tmp_path / "inputs"
    (inputs / "nested").mkdir(parents=True)
    _source(inputs / "one.npz", rows=5)
    _source(inputs / "nested" / "two.npz", offset=20, rows=5)

    result = import_dataset(inputs, tmp_path / "dataset", shard_length=2)

    assert result["sources"] == 2
    assert result["steps"] == 10
    assert result["shards"] == 6
    entries = list(iter_shards(tmp_path / "dataset", with_meta=True))
    assert {entry["source_id"] for _arrays, entry in entries} == {"one", "two"}
    assert [len(arrays["actions"]) for arrays, _entry in entries] == [2, 2, 1, 2, 2, 1]


def test_import_dataset_rejects_shape_drift_and_existing_destination(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _source(first)
    np.savez(
        second,
        frames=np.zeros((5, 4, 4, 3), dtype=np.uint8),
        actions=np.zeros((5, 4), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="different modality contract"):
        import_dataset([first, second], tmp_path / "bad")

    destination = tmp_path / "dataset"
    import_dataset(first, destination)
    with pytest.raises(FileExistsError, match="force=True"):
        import_dataset(first, destination)


def test_kur_import_cli_creates_analyzable_artifact(tmp_path):
    source = tmp_path / "gameplay.npz"
    _source(source)
    destination = tmp_path / "dataset"
    command = [
        sys.executable,
        "-m",
        "kuration.core.cli.import_dataset",
        str(source),
        "--out",
        str(destination),
        "--source-id",
        "episode-7",
        "--json",
    ]

    completed = subprocess.run(command, check=True, capture_output=True, text=True)

    result = json.loads(completed.stdout)
    assert result["sources"] == 1
    assert result["modalities"] == ["actions", "frames"]
    assert inspect_dataset(destination, verify=True).verification["valid"] is True
    _arrays, entry = next(iter_shards(destination, with_meta=True))
    assert entry["source_id"] == "episode-7"


def test_kur_import_cli_failure_is_one_line(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "kuration.core.cli.import_dataset",
            str(tmp_path / "missing.npz"),
            "--out",
            str(tmp_path / "dataset"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert completed.stderr.startswith("error: FileNotFoundError:")
    assert "Traceback" not in completed.stderr
