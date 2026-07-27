"""Smart registered-adapter import API and CLI."""
from __future__ import annotations

import json
import subprocess
import sys
import csv

import numpy as np
import pytest

from kuration import (
    detect_adapter,
    import_dataset,
    inspect_dataset,
    iter_shards,
    register_adapter,
)


class _SmartArrayAdapter:
    @staticmethod
    def detect_source(source_uri):
        return 250 if str(source_uri).endswith(".smart") else 0

    @staticmethod
    def import_arrays(source_uri):
        with np.load(source_uri, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}


register_adapter("test/smart-array-import", _SmartArrayAdapter)


class _PresenceToggleAdapter:
    """Array adapter whose presence mask can be added without changing the source bytes."""

    add_present = False

    @staticmethod
    def detect_source(source_uri):
        return 250 if str(source_uri).endswith(".present") else 0

    @staticmethod
    def import_arrays(source_uri):
        with np.load(source_uri, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        if _PresenceToggleAdapter.add_present:
            rows = len(next(iter(arrays.values())))
            arrays["_present"] = np.ones(rows, dtype=bool)
        return arrays


register_adapter("test/presence-toggle-import", _PresenceToggleAdapter)


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
        "adapter": "npz",
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
    assert result["adapter"] == "npz"
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


def test_import_auto_detects_timeseries_adapter_and_infers_csv_schema(tmp_path):
    source = tmp_path / "telemetry.csv"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["t", "id", "signal", "state"])
        writer.writeheader()
        writer.writerows([
            {"t": 0, "id": "device-1", "signal": 1.25, "state": "idle"},
            {"t": 1, "id": "device-1", "signal": 2.5, "state": "active"},
            {"t": 2, "id": "device-1", "signal": 3.75, "state": "idle"},
        ])

    result = import_dataset(source, tmp_path / "dataset")

    assert detect_adapter(str(source)) == "timeseries"
    assert result["adapter"] == "timeseries"
    arrays = next(iter_shards(tmp_path / "dataset"))
    np.testing.assert_allclose(arrays["signal"], [1.25, 2.5, 3.75])
    assert arrays["state"].tolist() == [1, 0, 1]
    config = json.loads((tmp_path / "dataset" / "config.json").read_text())
    assert config["output"]["vocabs"]["state"] == {"active": 0, "idle": 1}
    assert inspect_dataset(tmp_path / "dataset", verify=True).verification["valid"] is True


def test_import_uses_registered_plugin_detector_and_array_hook(tmp_path):
    source = tmp_path / "episode.smart"
    with source.open("wb") as stream:
        np.savez(
            stream,
            frames=np.zeros((3, 2, 2, 3), dtype=np.uint8),
            actions=np.ones((3, 4), dtype=np.float32),
        )

    result = import_dataset(source, tmp_path / "dataset")

    assert detect_adapter(str(source)) == "test/smart-array-import"
    assert result["adapter"] == "test/smart-array-import"
    arrays = next(iter_shards(tmp_path / "dataset"))
    assert arrays["frames"].shape == (3, 2, 2, 3)
    assert arrays["actions"].shape == (3, 4)


def test_import_auto_detects_hdf5_and_preserves_tensor_shapes(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "episode.h5"
    with h5py.File(source, "w") as handle:
        handle["frames"] = np.zeros((3, 2, 2, 3), dtype=np.uint8)
        handle["actions"] = np.ones((3, 4), dtype=np.float32)

    result = import_dataset(source, tmp_path / "dataset")

    assert detect_adapter(str(source)) == "hdf5"
    assert result["adapter"] == "hdf5"
    arrays = next(iter_shards(tmp_path / "dataset"))
    assert arrays["frames"].shape == (3, 2, 2, 3)
    assert arrays["actions"].shape == (3, 4)


# --------------------------------------------------------------------------------------
# 2026-07 import-review fixes
# --------------------------------------------------------------------------------------
def test_import_honors_sample_rate_instead_of_dropping_it(tmp_path):
    """Finding 1: hz/time/entity/kinds were silently discarded by the array shortcut."""
    source = tmp_path / "clip.npz"
    np.savez(source, actions=np.arange(20, dtype=np.float32).reshape(5, 4))

    result = import_dataset(source, tmp_path / "dataset", hz=30.0)

    config = json.loads((tmp_path / "dataset" / "config.json").read_text())
    assert config["schema"]["sample_hz"] == 30.0
    assert "_present" in result["modalities"]          # honored via the tick path


def test_import_rejects_options_no_path_can_honor(tmp_path):
    """Finding 1: array adapters have no columns, and the escape hatch has no inference."""
    source = tmp_path / "clip.npz"
    _source(source)
    with pytest.raises(ValueError, match=r"reads named arrays.*\['entity'\]"):
        import_dataset(source, tmp_path / "a", entity="frames")
    with pytest.raises(ValueError, match=r"reads named arrays.*\['time'\]"):
        import_dataset(source, tmp_path / "b", time="frames")
    with pytest.raises(ValueError, match=r"reads named arrays.*\['kinds'\]"):
        import_dataset(source, tmp_path / "c", kinds={"frames": "categorical"})

    plugin = tmp_path / "episode.smart"
    with plugin.open("wb") as stream:
        np.savez(stream, actions=np.ones((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match=r"constructed explicitly, so \['hz'\]"):
        import_dataset(plugin, tmp_path / "d", hz=30.0)


def test_import_rejects_partially_visible_nested_hdf5(tmp_path):
    """Finding 2: a top-level dataset plus nested groups imported a silent subset."""
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "episode.h5"
    with h5py.File(source, "w") as handle:
        handle["actions"] = np.ones((3, 4), dtype=np.float32)
        handle.create_group("obs")["frames"] = np.zeros((3, 2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="nested dataset") as failure:
        import_dataset(source, tmp_path / "dataset")

    assert "obs/frames" in str(failure.value)
    assert not list((tmp_path / "dataset").glob("*/*")), "no shard may be written"


def test_import_rejects_partially_visible_nested_zarr(tmp_path):
    """Same silent-subset bug as nested HDF5: nested zarr groups vanished from imports."""
    zarr = pytest.importorskip("zarr")
    source = tmp_path / "episode.zarr"
    group = zarr.open(str(source), mode="w")
    group.create_array("actions", data=np.ones((3, 4), dtype=np.float32))
    group.create_group("obs").create_array("frames", data=np.zeros((3, 2, 2), dtype=np.uint8))

    with pytest.raises(ValueError, match="nested groups") as failure:
        import_dataset(source, tmp_path / "dataset")

    assert "obs/frames" in str(failure.value)
    assert not list((tmp_path / "dataset").glob("*/*")), "no shard may be written"


def test_import_refuses_to_drop_event_streams(tmp_path):
    """Finding 3: tracks_from_ticks cannot carry events, so importing them lost data."""
    source = tmp_path / "telemetry.csv"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["t", "id", "signal"])
        writer.writeheader()
        writer.writerows([{"t": 0, "id": "a", "signal": 1.0},
                          {"t": 1, "id": "a", "signal": 2.0}])
    schema = {
        "name": "telemetry",
        "entity_axis": "id",
        "fields": {"signal": {"kind": "scalar"}},
        "event_streams": {"hits": {"fields": {"damage": {"kind": "scalar"}}}},
    }

    with pytest.raises(ValueError, match="event stream"):
        import_dataset(source, tmp_path / "dataset", schema=schema,
                       fields={"signal": "signal"}, time="t", entity="id")


def test_import_retry_after_partial_failure_is_distinguishable_and_cleanable(tmp_path):
    """Finding 4: a failed import left an uncommitted manifest and orphan shards."""
    from kuration.core.manifest import dataset_state

    good = tmp_path / "a.npz"
    drifted = tmp_path / "b.npz"
    _source(good)
    np.savez(drifted, frames=np.zeros((5, 9, 9, 3), dtype=np.uint8),
             actions=np.zeros((5, 4), dtype=np.float32))
    destination = tmp_path / "dataset"

    with pytest.raises(ValueError, match="different modality contract"):
        import_dataset([good, drifted], destination)

    assert dataset_state(str(destination)) == "writing"
    orphans = sorted(path.name for path in (destination / "train").glob("*"))
    assert orphans, "the failed attempt should have left shard files behind"

    with pytest.raises(FileExistsError, match="failed part-way"):
        import_dataset(good, destination)

    with pytest.warns(RuntimeWarning, match="removed 1 shard file"):
        result = import_dataset(good, destination, force=True)

    assert result["replaced_shards"] == 1
    assert dataset_state(str(destination)) == "committed"
    assert inspect_dataset(destination, verify=True).verification["valid"] is True
    assert sorted(path.name for path in (destination / "train").glob("*")) == [
        entry["path"].rsplit("/", 1)[-1]
        for entry in json.loads((destination / "dataset_manifest.json").read_text())["shards"]
    ]


def test_import_committed_destination_keeps_its_own_message(tmp_path):
    """Finding 4: a committed artifact must not be described as a failed import."""
    source = tmp_path / "clip.npz"
    _source(source)
    destination = tmp_path / "dataset"
    import_dataset(source, destination)

    with pytest.raises(FileExistsError, match="already contains a committed"):
        import_dataset(source, destination)


def test_import_fingerprint_covers_source_id_and_presence_mask(tmp_path):
    """Finding 5: the provenance fingerprint ignored source_id and the presence mask."""
    source = tmp_path / "clip.npz"
    _source(source)

    def fingerprint(dataset):
        return json.loads((dataset / "config.json").read_text())["provenance"]["fingerprint"]

    first = tmp_path / "first"
    second = tmp_path / "second"
    import_dataset(source, first, source_id="episode-1")
    import_dataset(source, second, source_id="episode-2")
    assert fingerprint(first) != fingerprint(second)

    toggled = tmp_path / "episode.present"
    with toggled.open("wb") as stream:
        np.savez(stream, actions=np.ones((3, 4), dtype=np.float32))
    _PresenceToggleAdapter.add_present = False
    try:
        without = tmp_path / "without-present"
        import_dataset(toggled, without)
        _PresenceToggleAdapter.add_present = True
        with_mask = tmp_path / "with-present"
        import_dataset(toggled, with_mask)
    finally:
        _PresenceToggleAdapter.add_present = False
    assert json.loads((with_mask / "config.json").read_text())["output"]["presence_mask"] is True
    assert fingerprint(without) != fingerprint(with_mask)


def test_parquet_export_rejects_flattened_column_collision(tmp_path):
    """Finding 6: `x` (T,2) flattened over a sibling modality literally named `x_0`."""
    pytest.importorskip("pyarrow")
    from kuration.core.cli.export import export_dataset

    source = tmp_path / "clip.npz"
    np.savez(source, x=np.arange(8, dtype=np.float32).reshape(4, 2),
             x_0=np.arange(4, dtype=np.float32))
    dataset = tmp_path / "dataset"
    import_dataset(source, dataset)

    with pytest.raises(ValueError, match="claimed by both modality"):
        export_dataset(str(dataset), str(tmp_path / "arrays"), fmt="parquet")

    index = export_dataset(str(dataset), str(tmp_path / "npz-arrays"), fmt="npz")
    assert index["shards"] == 1


def test_parquet_dataset_detection_is_local_and_bounded(tmp_path, monkeypatch):
    """Finding 7: unbounded rglob, and remote directories that can never be detected."""
    from kuration.core.adapters import parquet_dataset as module

    root = tmp_path / "parts"
    (root / "date=1").mkdir(parents=True)
    (root / "date=1" / "part-0.parquet").write_bytes(b"PAR1")
    assert module.ParquetDatasetAdapter.detect_source(str(root)) == 100
    assert module.ParquetDatasetAdapter.detect_source("gs://bucket/parts") == 0

    monkeypatch.setattr(module, "_DETECT_DIR_LIMIT", 1)
    assert module.ParquetDatasetAdapter.detect_source(str(root)) == 0


def test_import_points_remote_directories_at_an_explicit_adapter(tmp_path, monkeypatch):
    """Finding 7: detection never lists a remote tree; say so instead of a bare failure."""
    from kuration.core.adapters import base

    def _no_match(source_uri, *, registry=None):
        raise ValueError(f"no registered input adapter recognized {source_uri!r};")

    monkeypatch.setattr(base, "detect_adapter", _no_match)
    with pytest.raises(ValueError, match="are not auto-detected"):
        import_dataset("gs://bucket/parts", tmp_path / "dataset")
    with pytest.raises(ValueError, match="^no registered input adapter"):
        import_dataset(tmp_path / "local.bin", tmp_path / "dataset")
