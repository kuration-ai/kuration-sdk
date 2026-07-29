"""Reader-side contract validation for recommitted or legacy artifact metadata."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from kuration import OutputSpec, SchemaSpec, read_shard_arrays, shard_entries
from kuration.core.io import write_text
from kuration.core.manifest import commit_dataset, portable_shard_entries
from kuration.core.output.shardwriter import write_track
from kuration.core.output.track import Track
from kuration.core.versioning import ARTIFACT_FORMAT_VERSION


SCHEMA = SchemaSpec.from_dict({
    "name": "reader-contract",
    "entity_axis": "entity",
    "fields": {"x": {"kind": "scalar"}},
})
SPEC = OutputSpec.from_dict({
    "container": "npz",
    "shard_length": 2,
    "modalities": {
        "x": {"dtype": "float32", "shape": [], "source": "field:x"},
    },
})


def _committed_dataset(root: Path) -> None:
    entries = write_track(
        str(root),
        Track("source", 0, [{"x": 1.0}, {"x": 2.0}]),
        SPEC,
        schema=SCHEMA,
        split="train",
    )
    manifest = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "commit_protocol": "sidecar-v1",
        "count": len(entries),
        "shards": portable_shard_entries(str(root), entries),
    }
    config = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "schema": SCHEMA.to_dict(),
        "output": SPEC.to_dict(),
        "results": {"shards": len(entries)},
    }
    write_text(root / "dataset_manifest.json", json.dumps(manifest))
    write_text(root / "config.json", json.dumps(config))
    commit_dataset(str(root))


def _metadata(root: Path) -> tuple[dict, dict]:
    return (
        json.loads((root / "dataset_manifest.json").read_text()),
        json.loads((root / "config.json").read_text()),
    )


def _recommit(root: Path, manifest: dict, config: dict) -> None:
    write_text(root / "dataset_manifest.json", json.dumps(manifest))
    write_text(root / "config.json", json.dumps(config))
    commit_dataset(str(root))


@pytest.mark.parametrize(
    "name",
    ["a/b", r"a\b", "\x00hidden", "line\nbreak", "\x7f", ".", "..",
     "_n_steps", "_present"],
)
def test_writer_uses_the_portable_modality_name_contract(name):
    with pytest.raises(ValueError, match="modality name"):
        OutputSpec.from_dict({
            "container": "npz",
            "shard_length": 1,
            "modalities": {
                name: {"dtype": "float32", "shape": [], "source": "field:x"},
            },
        })


@pytest.mark.parametrize(
    "name",
    ["../escaped", r"..\escaped", "\x00hidden", "line\nbreak", ".", "..",
     "_n_steps", "_present"],
)
def test_recommitted_sidecar_cannot_declare_an_unsafe_modality(tmp_path, name):
    _committed_dataset(tmp_path)
    manifest, config = _metadata(tmp_path)
    modality = config["output"]["modalities"].pop("x")
    config["output"]["modalities"][name] = modality
    entry = manifest["shards"][0]
    entry["shape"][name] = entry["shape"].pop("x")
    entry["dtype"][name] = entry["dtype"].pop("x")
    _recommit(tmp_path, manifest, config)

    with pytest.raises(ValueError, match="modality name"):
        shard_entries(str(tmp_path))


def test_recommitted_sidecar_validates_its_embedded_schema(tmp_path):
    _committed_dataset(tmp_path)
    manifest, config = _metadata(tmp_path)
    config["schema"]["fields"]["x"] = {"kind": "not-a-field-kind"}
    _recommit(tmp_path, manifest, config)

    with pytest.raises(ValueError, match="invalid embedded schema"):
        shard_entries(str(tmp_path))


def test_current_array_style_artifact_may_omit_an_embedded_schema(tmp_path):
    _committed_dataset(tmp_path)
    manifest, config = _metadata(tmp_path)
    config.pop("schema")
    _recommit(tmp_path, manifest, config)

    entries, container = shard_entries(str(tmp_path))
    assert len(entries) == 1 and container == "npz"


def test_legacy_config_keeps_loose_structure_but_not_unsafe_names(tmp_path):
    _committed_dataset(tmp_path)
    manifest, config = _metadata(tmp_path)
    manifest.pop("commit_protocol")
    config["output"] = {
        "container": "npz",
        "modalities": {
            "../escaped": {"legacy_extension_key": True},
        },
    }
    write_text(tmp_path / "dataset_manifest.json", json.dumps(manifest))
    write_text(tmp_path / "config.json", json.dumps(config))

    with pytest.raises(ValueError, match="path separators"):
        shard_entries(str(tmp_path))


def test_direct_legacy_shard_read_rejects_unsafe_container_keys(tmp_path):
    shard = tmp_path / "legacy.npz"
    with shard.open("wb") as stream:
        np.savez(
            stream,
            _n_steps=np.asarray(1),
            **{"../../escaped": np.asarray([1.0], dtype=np.float32)},
        )
    entry = {
        "path": str(shard),
        "n_steps": 1,
        "shape": {"../../escaped": []},
        "dtype": {"../../escaped": "float32"},
    }

    with pytest.raises(ValueError, match="path separators"):
        read_shard_arrays(entry, "npz", verify=False)


def test_direct_legacy_shard_read_preserves_safe_historical_names(tmp_path):
    shard = tmp_path / "legacy.npz"
    with shard.open("wb") as stream:
        np.savez(
            stream,
            _n_steps=np.asarray(1),
            **{"old modality": np.asarray([1.0], dtype=np.float32)},
        )
    entry = {
        "path": str(shard),
        "n_steps": 1,
        "shape": {"old modality": []},
        "dtype": {"old modality": "float32"},
    }

    arrays = read_shard_arrays(entry, "npz", verify=False)
    assert arrays["old modality"].tolist() == [1.0]


def test_verify_true_requires_complete_legacy_integrity_metadata(tmp_path):
    shard = tmp_path / "legacy.npz"
    with shard.open("wb") as stream:
        np.savez(stream, _n_steps=np.asarray(1), x=np.asarray([1.0]))
    entry = {"path": str(shard), "n_steps": 1, "shape": {"x": []},
             "dtype": {"x": "float64"}}

    with pytest.raises(ValueError, match="no complete sha256/size_bytes"):
        read_shard_arrays(entry, "npz")
    assert read_shard_arrays(entry, "npz", verify=False)["x"].tolist() == [1.0]


def test_dataset_metadata_symlink_is_rejected_before_open(tmp_path):
    dataset = tmp_path / "dataset"
    _committed_dataset(dataset)
    outside = tmp_path / "outside-manifest.json"
    (dataset / "dataset_manifest.json").replace(outside)
    try:
        (dataset / "dataset_manifest.json").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ValueError, match="must not be a symlink"):
        shard_entries(str(dataset))


def test_dataset_metadata_fifo_is_rejected_without_blocking(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFOs are unavailable")
    dataset = tmp_path / "dataset"
    _committed_dataset(dataset)
    (dataset / "config.json").unlink()
    os.mkfifo(dataset / "config.json")

    with pytest.raises(ValueError, match="must be a regular file"):
        shard_entries(str(dataset))


def test_dataset_metadata_symlink_swap_after_validation_is_rejected(
    tmp_path, monkeypatch
):
    from kuration.core import loaders as loaders_module

    dataset = tmp_path / "dataset"
    _committed_dataset(dataset)
    config_path = dataset / "config.json"
    outside = tmp_path / "outside-config.json"
    outside.write_bytes(config_path.read_bytes())
    probe = dataset / "symlink-probe"
    try:
        probe.symlink_to(outside)
        probe.unlink()
    except OSError:
        pytest.skip("symlinks are unavailable")

    real_validate = loaders_module._validate_local_dataset_file
    swapped = False

    def validate_then_swap(path, base, *, label, required=True):
        nonlocal swapped
        result = real_validate(
            path, base, label=label, required=required
        )
        if label == "dataset config" and not swapped:
            config_path.unlink()
            config_path.symlink_to(outside)
            swapped = True
        return result

    monkeypatch.setattr(
        loaders_module, "_validate_local_dataset_file", validate_then_swap
    )
    with pytest.raises(ValueError, match="must not be a symlink"):
        shard_entries(str(dataset))
    assert swapped


def test_dataset_metadata_fifo_swap_after_validation_is_nonblocking(
    tmp_path, monkeypatch
):
    from kuration.core import loaders as loaders_module

    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFOs are unavailable")
    if not loaders_module._HAS_POSIX_DESCRIPTOR_READ:
        pytest.skip("descriptor-relative nonblocking reads are unavailable")
    dataset = tmp_path / "dataset"
    _committed_dataset(dataset)
    config_path = dataset / "config.json"
    real_validate = loaders_module._validate_local_dataset_file
    real_open = loaders_module.os.open
    real_open_file = loaders_module.open_file
    swapped = False
    opened_safely = False

    def validate_then_swap(path, base, *, label, required=True):
        nonlocal swapped
        result = real_validate(
            path, base, label=label, required=required
        )
        if label == "dataset config" and not swapped:
            config_path.unlink()
            os.mkfifo(config_path)
            swapped = True
        return result

    def checked_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal opened_safely
        if path == "config.json" and dir_fd is not None:
            assert flags & os.O_NOFOLLOW
            assert flags & os.O_NONBLOCK
            opened_safely = True
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def reject_path_open(path, mode="rb"):
        if str(path) == str(config_path):
            raise AssertionError("metadata must not use a blocking pathname open")
        return real_open_file(path, mode)

    monkeypatch.setattr(
        loaders_module, "_validate_local_dataset_file", validate_then_swap
    )
    monkeypatch.setattr(loaders_module.os, "open", checked_open)
    monkeypatch.setattr(loaders_module, "open_file", reject_path_open)
    with pytest.raises(ValueError, match="must be a regular file"):
        shard_entries(str(dataset))
    assert swapped and opened_safely


def test_dataset_metadata_intermediate_ancestor_swap_is_rejected(
    tmp_path, monkeypatch
):
    from kuration.core import loaders as loaders_module

    if not loaders_module._HAS_POSIX_DESCRIPTOR_READ:
        pytest.skip("descriptor-relative no-follow reads are unavailable")
    trusted_parent = tmp_path / "namespace" / "trusted"
    dataset = trusted_parent / "dataset"
    evil_parent = tmp_path / "outside"
    _committed_dataset(dataset)
    _committed_dataset(evil_parent / "dataset")
    probe = tmp_path / "ancestor-symlink-probe"
    try:
        probe.symlink_to(evil_parent, target_is_directory=True)
        probe.unlink()
    except OSError:
        pytest.skip("symlinks are unavailable")

    real_validate = loaders_module._validate_local_dataset_file
    saved_parent = trusted_parent.with_name("trusted-before-swap")
    swapped = False

    def validate_then_swap(path, base, *, label, required=True):
        nonlocal swapped
        result = real_validate(
            path, base, label=label, required=required
        )
        if label == "dataset manifest" and not swapped:
            trusted_parent.rename(saved_parent)
            trusted_parent.symlink_to(evil_parent, target_is_directory=True)
            swapped = True
        return result

    monkeypatch.setattr(
        loaders_module, "_validate_local_dataset_file", validate_then_swap
    )
    with pytest.raises(ValueError, match="symlink.*ancestor"):
        shard_entries(str(dataset))
    assert swapped


@pytest.mark.parametrize(
    ("suffix", "match"),
    [
        (',"output":{}}', "duplicate object key 'output'"),
        (',"unexpected":NaN}', "non-finite numeric token 'NaN'"),
    ],
)
def test_committed_config_uses_strict_unambiguous_json(tmp_path, suffix, match):
    _committed_dataset(tmp_path)
    config_path = tmp_path / "config.json"
    document = config_path.read_text().rstrip()
    config_path.write_text(document[:-1] + suffix)
    commit_dataset(str(tmp_path))

    with pytest.raises(ValueError, match=match):
        shard_entries(str(tmp_path))
