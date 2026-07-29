"""Fail-closed file-output and strict-JSON contracts shared by authoring CLIs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import run_cli


def _csv(path: Path) -> None:
    path.write_text("t,id,x\n0,e1,1\n1,e1,2\n")


def test_job_and_mapping_inputs_reject_nonstandard_json_numbers(tmp_path):
    from kuration.core.cli._common import parse_mapping_option, read_job

    job = tmp_path / "job.json"
    job.write_text('{"seed": NaN}')
    with pytest.raises(ValueError, match="non-finite numeric token 'NaN'"):
        read_job(str(job))
    with pytest.raises(ValueError, match="non-finite numeric token 'Infinity'"):
        parse_mapping_option('{"x": Infinity}', label="--kinds")


def test_job_inputs_reject_duplicate_keys_in_nested_objects(tmp_path):
    from kuration.core.cli._common import read_job

    job = tmp_path / "job.json"
    job.write_text('{"adapter": {"type": "csv", "type": "npz"}}')
    with pytest.raises(ValueError, match="duplicate object key 'type'"):
        read_job(str(job))


def test_strict_json_encoding_fails_before_replacing_an_existing_file(tmp_path):
    from kuration.core.cli._common import json_document
    from kuration.core.io import write_text

    output = tmp_path / "result.json"
    output.write_text("last-good")
    with pytest.raises(ValueError, match="strict JSON"):
        encoded = json_document({"score": float("nan")}, label="test result")
        write_text(output, encoded)
    assert output.read_text() == "last-good"


@pytest.mark.parametrize(
    "module",
    ["kuration.core.cli.report", "kuration.core.cli.experiment"],
)
def test_analysis_clis_refuse_existing_outputs_and_never_replace_the_job(tmp_path, module):
    job = tmp_path / "job.json"
    job.write_text("{}")
    output = tmp_path / "result.json"
    output.write_text("last-good")

    blocked = run_cli(module, job, "--out", output, expect=1)
    assert "--force" in blocked.stderr
    assert output.read_text() == "last-good"

    same_path = run_cli(module, job, "--out", job, "--force", expect=1)
    assert "also an input path" in same_path.stderr
    assert job.read_text() == "{}"


@pytest.mark.parametrize(
    "module",
    ["kuration.core.cli.report", "kuration.core.cli.experiment"],
)
def test_analysis_clis_never_replace_a_resolved_data_source(tmp_path, monkeypatch, module):
    command = __import__(module, fromlist=["_run"])
    job_path = tmp_path / "job.json"
    job_path.write_text("{}")
    source = tmp_path / "source.csv"
    _csv(source)
    raw_job = {}
    resolved = SimpleNamespace(
        kind="report" if module.endswith("report") else "experiment",
        schema=object(),
        adapter=object(),
        segmenter=object(),
        strategy=object(),
        sources=(str(source),),
        arms=(),
    )
    monkeypatch.setattr(command, "read_job", lambda path: raw_job)
    monkeypatch.setattr(command, "resolve_job", lambda job: resolved)

    args = SimpleNamespace(
        job=str(job_path), out=str(source), force=True, quiet=True
    )
    original = source.read_bytes()
    with pytest.raises(ValueError, match="also an input path"):
        command._run(args)
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    ("module", "positional"),
    [
        ("kuration.core.cli.report", "job.json"),
        ("kuration.core.cli.experiment", "job.json"),
        ("kuration.core.cli.migrate", "job.json"),
        ("kuration.core.cli.infer_schema_cli", "source.csv"),
    ],
)
def test_file_clis_reject_force_when_there_is_no_output(module, positional):
    result = run_cli(module, positional, "--force", expect=1)
    assert "--force requires --out" in result.stderr


def test_infer_requires_force_and_cannot_overwrite_its_sample(tmp_path):
    source = tmp_path / "source.csv"
    _csv(source)
    schema = tmp_path / "schema.json"
    schema.write_text("last-good")

    blocked = run_cli(
        "kuration.core.cli.infer_schema_cli", source, "--out", schema, expect=1
    )
    assert "--force" in blocked.stderr
    assert schema.read_text() == "last-good"

    run_cli("kuration.core.cli.infer_schema_cli", source, "--out", schema, "--force")
    assert json.loads(schema.read_text())["fields"]["x"]["kind"] == "scalar"

    original = source.read_bytes()
    same_path = run_cli(
        "kuration.core.cli.infer_schema_cli",
        source,
        "--out",
        source,
        "--force",
        expect=1,
    )
    assert "also an input path" in same_path.stderr
    assert source.read_bytes() == original


def test_infer_output_does_not_publish_source_credentials(monkeypatch):
    from kuration.core import infer
    from kuration.core.cli.infer_schema_cli import infer_source

    source = "https://user:secret@example.test/private/data.csv?token=top-secret#fragment"
    monkeypatch.setattr(
        infer,
        "sample_source",
        lambda source, n: ("timeseries", [{"t": 0, "x": 1}, {"t": 1, "x": 2}]),
    )

    result = infer_source(source, time="t", hz=1)
    encoded = json.dumps(result)
    assert "secret" not in encoded and "top-secret" not in encoded
    assert result["source"].startswith("data.csv-")


def test_init_force_never_turns_an_input_source_into_the_job(tmp_path):
    source = tmp_path / "source.csv"
    _csv(source)
    original = source.read_bytes()

    blocked = run_cli(
        "kuration.core.cli.init_job",
        source,
        "--out",
        source,
        "--force",
        expect=1,
    )
    assert "also an input path" in blocked.stderr
    assert source.read_bytes() == original


def test_init_rejects_a_nonpositive_sample_before_inference(tmp_path):
    source = tmp_path / "source.csv"
    _csv(source)
    output = tmp_path / "job.json"
    result = run_cli(
        "kuration.core.cli.init_job",
        source,
        "--out",
        output,
        "--sample",
        "0",
        expect=1,
    )
    assert "--sample must be a positive integer" in result.stderr
    assert not output.exists()


def test_init_rejects_a_dataset_destination_inside_a_source_tree(tmp_path):
    source = tmp_path / "parts"
    source.mkdir()
    output = tmp_path / "job.json"
    dataset = source / "curated"
    result = run_cli(
        "kuration.core.cli.init_job",
        source,
        "--out",
        output,
        "--dataset-out",
        dataset,
        expect=1,
    )
    assert "overlaps output" in result.stderr
    assert not output.exists() and not dataset.exists()


def test_migrate_requires_force_for_replacement_and_preserves_output_on_bad_json(tmp_path):
    source = tmp_path / "old.json"
    source.write_text('{"command": "curate"}')
    output = tmp_path / "migrated.json"
    output.write_text("last-good")

    blocked = run_cli("kuration.core.cli.migrate", source, "--out", output, expect=1)
    assert "--force" in blocked.stderr
    assert output.read_text() == "last-good"

    run_cli("kuration.core.cli.migrate", source, "--out", output, "--force")
    assert "job_schema_version" in json.loads(output.read_text())

    source.write_text('{"command": "curate", "seed": NaN}')
    output.write_text("last-good")
    invalid = run_cli(
        "kuration.core.cli.migrate", source, "--out", output, "--force", expect=1
    )
    assert "non-finite numeric token" in invalid.stderr
    assert output.read_text() == "last-good"


def test_migrate_allows_explicit_atomic_in_place_upgrade(tmp_path):
    path = tmp_path / "job.json"
    path.write_text('{"command": "curate"}')
    run_cli("kuration.core.cli.migrate", path, "--out", path, "--force")
    assert "job_schema_version" in json.loads(path.read_text())


def test_migrate_api_does_not_replace_output_with_non_json_values(tmp_path):
    from kuration.core.versioning import migrate

    output = tmp_path / "job.json"
    output.write_text("last-good")
    with pytest.raises(ValueError, match="strict JSON"):
        migrate({"command": "curate", "value": float("nan")}, out=output)
    assert output.read_text() == "last-good"


def test_migrate_module_execution_exposes_the_same_help_as_other_clis():
    result = subprocess.run(
        [sys.executable, "-m", "kuration.core.cli.migrate", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "Migrate a kuration job or config JSON" in result.stdout
    assert "--force" in result.stdout


def test_export_rejects_any_destination_overlapping_its_input_dataset(tmp_path):
    from kuration.core.cli.export import export_dataset

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    with pytest.raises(ValueError, match="overlaps output"):
        export_dataset(str(dataset), str(dataset / "plain-arrays"))
    assert not (dataset / "plain-arrays").exists()


def test_export_requires_force_for_an_existing_directory(tmp_path, monkeypatch):
    from kuration.core import loaders
    from kuration.core.cli.export import export_dataset

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    destination = tmp_path / "arrays"
    destination.mkdir()
    (destination / "unrelated.txt").write_text("keep")
    entry = {"path": "train/clip_shard0000.npz", "source_id": "s", "entity_id": "e",
             "split": "train", "shard_index": 0, "n_steps": 2}
    monkeypatch.setattr(
        loaders,
        "iter_shards",
        lambda *args, **kwargs: iter([({"x": np.asarray([1, 2])}, entry)]),
    )

    with pytest.raises(FileExistsError, match="--force"):
        export_dataset(str(dataset), str(destination))
    index = export_dataset(str(dataset), str(destination), force=True)
    assert index["shards"] == 1
    assert (destination / "unrelated.txt").read_text() == "keep"


def test_export_accepts_an_existing_empty_directory(tmp_path, monkeypatch):
    from kuration.core import loaders
    from kuration.core.cli.export import export_dataset

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    destination = tmp_path / "arrays"
    destination.mkdir()
    entry = {"path": "train/clip_shard0000.npz", "source_id": "s", "entity_id": "e",
             "split": "train", "shard_index": 0, "n_steps": 1}
    monkeypatch.setattr(
        loaders,
        "iter_shards",
        lambda *args, **kwargs: iter([({"x": np.asarray([1])}, entry)]),
    )
    assert export_dataset(str(dataset), str(destination))["shards"] == 1


def test_export_names_are_split_aware_and_do_not_overwrite_each_other(tmp_path, monkeypatch):
    from kuration.core import loaders
    from kuration.core.cli.export import export_dataset

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    entries = [
        ({"x": np.asarray([value])},
         {"path": f"{split}/clip_shard0000.npz", "source_id": split, "entity_id": "e",
          "split": split, "shard_index": 0, "n_steps": 1})
        for value, split in enumerate(("train", "test"), start=1)
    ]
    monkeypatch.setattr(
        loaders, "iter_shards", lambda *args, **kwargs: iter(entries)
    )

    index = export_dataset(str(dataset), str(tmp_path / "arrays"))
    paths = [tmp_path / "arrays" / record["files"][0] for record in index["entries"]]
    assert len(set(paths)) == 2
    assert {Path(path).name.split("__", 1)[0] for path in paths} == {"train", "test"}
    assert {int(np.load(path)["x"][0]) for path in paths} == {1, 2}


def test_npy_export_modality_names_cannot_escape_the_destination(tmp_path, monkeypatch):
    from kuration.core import loaders
    from kuration.core.cli.export import export_dataset

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    destination = tmp_path / "arrays"
    entry = {"path": "train/clip_shard0000.npz", "source_id": "s", "entity_id": "e",
             "split": "train", "shard_index": 0, "n_steps": 1}
    monkeypatch.setattr(
        loaders,
        "iter_shards",
        lambda *args, **kwargs: iter([
            ({"../../escaped": np.asarray([1]), "..\\escaped": np.asarray([2])}, entry)
        ]),
    )

    index = export_dataset(str(dataset), str(destination), fmt="npy")
    written = [destination / path for path in index["entries"][0]["files"]]
    assert len(set(written)) == 2
    assert all(path.is_relative_to(destination / ".kuration-export-generations")
               for path in written)
    assert not (tmp_path / "escaped.npy").exists()


def test_export_index_is_portable_and_does_not_publish_source_credentials(
        tmp_path, monkeypatch):
    from kuration.core import loaders
    from kuration.core.cli import export as command

    destination = tmp_path / "arrays"
    destination.mkdir()
    source = "https://user:secret@example.test/private/data?token=top-secret#fragment"
    entry = {"path": "train/clip_shard0000.npz", "source_id": source, "entity_id": "e",
             "split": "train", "shard_index": 0, "n_steps": 1}
    monkeypatch.setattr(
        loaders,
        "iter_shards",
        lambda *args, **kwargs: iter([({"x": np.asarray([1])}, entry)]),
    )
    monkeypatch.setattr(
        command, "_prepare_export_destination",
        lambda dataset, out, force: str(destination),
    )

    index = command.export_dataset(source, str(destination))
    encoded = json.dumps(index)
    assert "secret" not in encoded and "top-secret" not in encoded
    assert all(not Path(path).is_absolute() for path in index["entries"][0]["files"])
    assert (destination / index["entries"][0]["files"][0]).exists()


def test_export_index_commits_one_generation_with_integrity_for_every_payload(
        tmp_path, monkeypatch):
    from kuration.core import loaders
    from kuration.core.cli.export import export_dataset

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    destination = tmp_path / "arrays"
    entries = [
        (
            {"x": np.asarray([value]), "y": np.asarray([value + 10])},
            {
                "path": f"{split}/clip_shard0000.npz",
                "source_id": split,
                "entity_id": "e",
                "split": split,
                "shard_index": 0,
                "n_steps": 1,
            },
        )
        for value, split in enumerate(("train", "test"), start=1)
    ]
    monkeypatch.setattr(
        loaders, "iter_shards", lambda *args, **kwargs: iter(entries)
    )

    index = export_dataset(str(dataset), str(destination), fmt="npy")

    referenced = {
        path for entry in index["entries"] for path in entry["files"]
    }
    assert index["state"] == "committed"
    assert index["index_version"] == 1
    assert referenced == set(index["integrity"])
    assert all(
        path.startswith(
            f".kuration-export-generations/{index['generation']}/"
        )
        for path in referenced
    )
    for relative, recorded in index["integrity"].items():
        payload = (destination / relative).read_bytes()
        assert recorded == {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    assert json.loads((destination / "export_index.json").read_text()) == index


def test_failed_forced_export_leaves_prior_generation_and_index_unchanged(
        tmp_path, monkeypatch):
    from kuration.core import io, loaders
    from kuration.core.cli.export import export_dataset

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    destination = tmp_path / "arrays"
    old_entry = {
        "path": "train/clip_shard0000.npz",
        "source_id": "old",
        "entity_id": "e",
        "split": "train",
        "shard_index": 0,
        "n_steps": 1,
    }
    monkeypatch.setattr(
        loaders,
        "iter_shards",
        lambda *args, **kwargs: iter([({"x": np.asarray([1])}, old_entry)]),
    )
    old_index = export_dataset(str(dataset), str(destination))
    old_index_bytes = (destination / "export_index.json").read_bytes()
    old_payloads = {
        path: (destination / path).read_bytes() for path in old_index["integrity"]
    }
    old_generations = set(
        (destination / ".kuration-export-generations").iterdir()
    )

    new_entry = dict(old_entry, source_id="new")
    monkeypatch.setattr(
        loaders,
        "iter_shards",
        lambda *args, **kwargs: iter([({"x": np.asarray([99])}, new_entry)]),
    )
    real_write_text = io.write_text

    def fail_index(path, text):
        if str(path).endswith("/export_index.json"):
            raise RuntimeError("simulated commit interruption")
        return real_write_text(path, text)

    monkeypatch.setattr(io, "write_text", fail_index)
    with pytest.raises(RuntimeError, match="commit interruption"):
        export_dataset(str(dataset), str(destination), force=True)

    assert (destination / "export_index.json").read_bytes() == old_index_bytes
    assert {
        path: (destination / path).read_bytes() for path in old_index["integrity"]
    } == old_payloads
    assert set((destination / ".kuration-export-generations").iterdir()) == old_generations


def test_export_lease_rejects_a_concurrent_writer_before_reading_sources(
        tmp_path, monkeypatch):
    from kuration.core import loaders
    from kuration.core.cli.export import export_dataset
    from kuration.core.manifest import dataset_writer_lease

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    destination = tmp_path / "arrays"
    reads = 0

    def shards(*args, **kwargs):
        nonlocal reads
        reads += 1
        return iter([])

    monkeypatch.setattr(loaders, "iter_shards", shards)

    with dataset_writer_lease(str(destination)):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                export_dataset, str(dataset), str(destination)
            )
            with pytest.raises(FileExistsError, match="active writer lease"):
                future.result()

    assert reads == 0
