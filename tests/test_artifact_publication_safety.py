"""Artifact publication is explicit and starts only after selection succeeds."""
from __future__ import annotations

import builtins
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from kuration import (OutputSpec, SchemaSpec, Segmenter, Tick, input, load_strategy,
                       shard_entries, write_dataset)
from kuration.core.manifest import begin_dataset, clear_dataset_shards, dataset_state


SCHEMA = SchemaSpec.from_dict({
    "name": "publication",
    "entity_axis": "entity",
    "sample_hz": 1.0,
    "fields": {"x": {"kind": "scalar"}},
})
SPEC = OutputSpec.from_dict({
    "container": "npz",
    "shard_length": 2,
    "modalities": {
        "x": {"dtype": "float32", "shape": [], "source": "field:x"},
    },
})
FILTER = load_strategy({
    "name": "all",
    "define": {"mean_x": {"source": "entity.x", "reduce": "mean"}},
    "selection": {
        "method": "filter",
        "predicates": [{"metric": "mean_x", "op": ">=", "value": 0.0}],
    },
}, SCHEMA)


@dataclass(frozen=True)
class UnsupportedEntity:
    value: int


class NonJsonFingerprint:
    pass


class MemAdapter:
    def __init__(self, data):
        self.schema = SCHEMA
        self.data = data
        self.reads = 0

    def ticks(self, source):
        self.reads += 1
        value = self.data[source]
        if isinstance(value, BaseException):
            raise value
        return value

    def source_fingerprint(self, source):
        from kuration.core.provenance import _state_hash
        return _state_hash(self.data[source])


def _ticks(values):
    return [Tick(index, {0: {"x": float(value)}})
            for index, value in enumerate(values)]


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _classic_write(destination, adapter, *, force=False, resume=False):
    return write_dataset(
        str(destination),
        schema=SCHEMA,
        adapter=adapter,
        sources=["source"],
        strategy=FILTER,
        segmenter=Segmenter(2, 2),
        spec=SPEC,
        train_hours=4 / 3600,
        on_error="raise",
        force=force,
        resume=resume,
    )


def test_count_selection_failure_preserves_committed_artifact_bytes(tmp_path):
    destination = tmp_path / "count"
    base = (
        input([{"x": float(value)} for value in range(6)], hz=1)
        .window(2, stride=2)
        .measure("mean_x", "mean", of="x")
        .sample("top-k", n=2, weights={"mean_x": 1.0})
    )
    base.output(destination, categorical="omit")
    before = _artifact_bytes(destination)

    empty = (
        input([{"x": float(value)} for value in range(6)], hz=1)
        .window(2, stride=2)
        .measure("mean_x", "mean", of="x")
        .filter("mean_x", ">", 100)
        .sample("filter", n=1)
    )
    with pytest.raises(ValueError, match="selection is empty"):
        empty.output(destination, categorical="omit", force=True)

    assert dataset_state(str(destination)) == "committed"
    assert _artifact_bytes(destination) == before
    assert shard_entries(str(destination))[0]


def test_classic_selection_failure_preserves_committed_artifact_bytes(tmp_path):
    destination = tmp_path / "hours"
    _classic_write(destination, MemAdapter({"source": _ticks([1, 2, 3, 4])}))
    before = _artifact_bytes(destination)

    failing = MemAdapter({"source": RuntimeError("selection failed")})
    with pytest.raises(RuntimeError, match="selection failed"):
        _classic_write(destination, failing, force=True)

    assert failing.reads == 1
    assert dataset_state(str(destination)) == "committed"
    assert _artifact_bytes(destination) == before
    assert shard_entries(str(destination))[0]


def test_unsupported_entity_identity_fails_before_forced_replacement(tmp_path):
    destination = tmp_path / "entity-identity"
    _classic_write(destination, MemAdapter({"source": _ticks([1, 2, 3, 4])}))
    before = _artifact_bytes(destination)
    ticks = [
        Tick(index, {UnsupportedEntity(7): {"x": float(index)}})
        for index in range(4)
    ]

    with pytest.raises(ValueError, match="cannot derive a stable shard identity"):
        _classic_write(destination, MemAdapter({"source": ticks}), force=True)

    assert dataset_state(str(destination)) == "committed"
    assert _artifact_bytes(destination) == before


def test_nonzero_hours_budget_rejects_empty_sources_without_publishing(tmp_path):
    destination = tmp_path / "empty-sources"

    with pytest.raises(ValueError, match="no sources.*non-zero budget"):
        write_dataset(
            str(destination),
            schema=SCHEMA,
            adapter=MemAdapter({}),
            sources=[],
            strategy=FILTER,
            segmenter=Segmenter(2, 2),
            spec=SPEC,
            train_hours=1 / 3600,
            on_error="raise",
        )

    assert dataset_state(str(destination)) == "absent"


def test_committed_artifact_redacts_signed_source_credentials_and_host_paths(tmp_path):
    class SignedSourceAdapter(MemAdapter):
        def source_fingerprint(self, _source):
            return "immutable-test-source"

    source = (
        "s3://private-user:super-secret@bucket.example/private/session.npz"
        "?token=do-not-publish#fragment"
    )
    destination = tmp_path / "redacted-source"
    result = write_dataset(
        str(destination),
        schema=SCHEMA,
        adapter=SignedSourceAdapter({source: _ticks([1, 2, 3, 4])}),
        sources=[source],
        strategy=FILTER,
        segmenter=Segmenter(2, 2),
        spec=SPEC,
        train_hours=4 / 3600,
        on_error="raise",
    )

    published = json.dumps({
        "manifest": json.loads((destination / "dataset_manifest.json").read_text()),
        "config": json.loads((destination / "config.json").read_text()),
        "paths": [Path(entry["path"]).name for entry in result["shards"]],
    })
    for secret in ("private-user", "super-secret", "do-not-publish", "fragment",
                   str(tmp_path)):
        assert secret not in published
    assert all(entry["source_id"].startswith("session.npz-")
               for entry in result["shards"])


def test_bounded_artifact_redacts_checkpoint_paths_and_partition_source_ids(tmp_path):
    pytest.importorskip("pyarrow")

    class SignedSourceAdapter(MemAdapter):
        def source_fingerprint(self, _source):
            return "immutable-bounded-source"

        def __repr__(self):
            return "SignedSourceAdapter(immutable-bounded-source)"

    source = (
        "s3://private-user:super-secret@bucket.example/private/session.npz"
        "?token=do-not-publish#fragment"
    )
    strategy = load_strategy({
        "name": "bounded",
        "define": {"mean_x": {"source": "entity.x", "reduce": "mean"}},
        "selection": {"method": "top-k", "weights": {"mean_x": 1.0}},
    }, SCHEMA)
    fingerprints = []
    for name in ("first", "second"):
        destination = tmp_path / name
        write_dataset(
            str(destination),
            schema=SCHEMA,
            adapter=SignedSourceAdapter({source: _ticks([1, 2, 3, 4])}),
            sources=[source],
            strategy=strategy,
            segmenter=Segmenter(2, 2),
            spec=SPEC,
            train_hours=4 / 3600,
            on_error="raise",
        )
        config = json.loads((destination / "config.json").read_text())
        serialized = json.dumps(config)
        for secret in ("private-user", "super-secret", "do-not-publish", "fragment",
                       str(destination)):
            assert secret not in serialized
        checkpoint = config["execution"]["checkpoint"]
        assert checkpoint["partition_count"] > 0
        assert all("data_file" not in entry and "partition_id" not in entry
                   for entry in checkpoint["partitions"])
        assert config["execution"]["physical_plan"]["spill_path"] is None
        fingerprints.append(checkpoint["identity_fingerprint"])

    assert fingerprints[0] == fingerprints[1]


def test_non_json_source_fingerprint_fails_before_forced_replacement(tmp_path):
    class InvalidFingerprintAdapter(MemAdapter):
        def source_fingerprint(self, _source):
            return NonJsonFingerprint()

    destination = tmp_path / "invalid-fingerprint"
    _classic_write(destination, MemAdapter({"source": _ticks([1, 2, 3, 4])}))
    before = _artifact_bytes(destination)

    with pytest.raises(ValueError, match=r"source_fingerprint.*strict JSON"):
        _classic_write(
            destination,
            InvalidFingerprintAdapter({"source": _ticks([1, 2, 3, 4])}),
            force=True,
        )

    assert dataset_state(str(destination)) == "committed"
    assert _artifact_bytes(destination) == before


def test_zero_shard_count_selection_fails_before_forced_replacement(tmp_path):
    destination = tmp_path / "count-drop"
    base = (
        input([{"x": float(value)} for value in range(6)], hz=1)
        .window(2, stride=2)
        .measure("mean_x", "mean", of="x")
        .sample("top-k", n=1, weights={"mean_x": 1.0})
    )
    base.output(destination, shard_length=2, categorical="omit")
    before = _artifact_bytes(destination)

    with pytest.raises(ValueError, match="emit zero shards"):
        base.output(
            destination,
            shard_length=4,
            short_policy="drop",
            categorical="omit",
            force=True,
        )

    assert dataset_state(str(destination)) == "committed"
    assert _artifact_bytes(destination) == before


def test_existing_destination_requires_force_before_reading_sources(tmp_path):
    destination = tmp_path / "guarded"
    _classic_write(destination, MemAdapter({"source": _ticks([1, 2, 3, 4])}))
    unread = MemAdapter({"source": AssertionError("destination guard must run first")})

    with pytest.raises(FileExistsError, match="committed.*force=True"):
        _classic_write(destination, unread)
    assert unread.reads == 0

    begin_dataset(str(destination))
    with pytest.raises(FileExistsError, match="partially written.*resume=True"):
        _classic_write(destination, unread)
    assert unread.reads == 0


@pytest.mark.parametrize(
    ("sentinel", "contents"),
    [
        ("dataset.commit.json", b"{}"),
        ("config.json", b"{}"),
        (".kuration_output_spec.json", b"{}"),
        ("train/orphan_shard0000.npz", b"orphan"),
    ],
)
def test_damaged_destination_requires_force_before_source_reads(
        tmp_path, sentinel, contents):
    destination = tmp_path / "damaged"
    path = destination / sentinel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    before = _artifact_bytes(destination)
    unread = MemAdapter({"source": AssertionError("damaged destination guard must run first")})

    assert dataset_state(str(destination)) == "corrupt"
    with pytest.raises(FileExistsError, match="damaged or incomplete.*force=True"):
        _classic_write(destination, unread)

    assert unread.reads == 0
    assert _artifact_bytes(destination) == before


def test_missing_manifest_is_not_treated_as_an_empty_destination(tmp_path):
    destination = tmp_path / "missing-manifest"
    _classic_write(destination, MemAdapter({"source": _ticks([1, 2, 3, 4])}))
    (destination / "dataset_manifest.json").unlink()
    before = _artifact_bytes(destination)
    unread = MemAdapter({"source": AssertionError("damaged destination guard must run first")})

    assert dataset_state(str(destination)) == "corrupt"
    with pytest.raises(FileExistsError, match="damaged or incomplete.*force=True"):
        _classic_write(destination, unread)

    assert unread.reads == 0
    assert _artifact_bytes(destination) == before


def test_duplicate_commit_keys_make_a_destination_corrupt(tmp_path):
    destination = tmp_path / "duplicate-commit"
    destination.mkdir()
    (destination / "dataset_manifest.json").write_text("{}")
    (destination / "dataset.commit.json").write_text(
        '{"state":"committed","state":"writing"}'
    )

    assert dataset_state(str(destination)) == "corrupt"


def test_force_cleanup_discovers_registered_container_orphans(tmp_path):
    from kuration.core.manifest import clear_dataset_shards

    destination = tmp_path / "custom-orphan"
    orphan = destination / "train" / "source_shard0000.acme"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"sensitive partial payload")

    assert dataset_state(str(destination)) == "corrupt"
    assert clear_dataset_shards(str(destination)) == [str(orphan)]
    assert not orphan.exists()


@pytest.mark.parametrize("budget", ["count", "hours"])
def test_output_rejects_a_source_inside_its_destination_before_publication(
        tmp_path, budget):
    destination = tmp_path / f"source-tree-{budget}"
    destination.mkdir()
    source = destination / "source.jsonl"
    source.write_text(
        "".join(json.dumps({"t": index, "x": float(index)}) + "\n" for index in range(6))
    )
    before = source.read_bytes()
    pipeline = (
        input([str(source)], time="t", hz=1)
        .window(2, stride=2)
        .measure("mean_x", "mean", of="x")
    )
    pipeline = (
        pipeline.sample("top-k", n=2, weights={"mean_x": 1.0})
        if budget == "count"
        else pipeline.sample("top-k", hours=4 / 3600, weights={"mean_x": 1.0})
    )

    with pytest.raises(ValueError, match="overlaps output.*separate output path"):
        pipeline.output(destination, categorical="omit")

    assert source.read_bytes() == before
    assert dataset_state(str(destination)) == "absent"
    assert not (destination / "dataset_manifest.json").exists()


@pytest.mark.parametrize(
    "destination",
    [
        "memory://overlap/data//source/output",
        "memory://overlap/data/tmp/../source/output",
    ],
)
def test_remote_source_output_separation_normalizes_or_rejects_ambiguous_paths(destination):
    pytest.importorskip("fsspec")
    from kuration.core.manifest import validate_source_destination_separation

    with pytest.raises(ValueError):
        validate_source_destination_separation(
            destination, ["memory://overlap/data/source"])


def test_unrelated_shard_like_file_is_not_owned_or_removed(tmp_path):
    destination = tmp_path / "mixed-directory"
    note = destination / "train" / "notes_shard_backup.txt"
    note.parent.mkdir(parents=True)
    note.write_bytes(b"user note")

    assert dataset_state(str(destination)) == "absent"
    assert clear_dataset_shards(str(destination)) == []
    assert note.read_bytes() == b"user note"


def test_symlinked_split_cannot_redirect_cleanup_or_fresh_writes(tmp_path):
    destination = tmp_path / "symlinked"
    destination.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    victim_file = victim / "keep_shard0000.npz"
    victim_file.write_bytes(b"victim")
    (destination / "train").symlink_to(victim, target_is_directory=True)

    with pytest.raises(ValueError, match="contains symlink child"):
        clear_dataset_shards(str(destination))

    pipeline = (
        input([{"x": float(value)} for value in range(6)], hz=1)
        .window(2, stride=2)
        .measure("mean_x", "mean", of="x")
        .sample("top-k", n=2, weights={"mean_x": 1.0})
    )
    with pytest.raises(ValueError, match="contains symlink child"):
        pipeline.output(destination, categorical="omit")

    assert victim_file.read_bytes() == b"victim"
    assert not (destination / "dataset_manifest.json").exists()


@pytest.mark.parametrize(
    ("literal_name", "matching_sibling"),
    [
        ("run*", "run-victim"),
        ("run?", "runX"),
        ("run[abc]", "runa"),
    ],
)
def test_forced_shard_cleanup_never_expands_destination_as_a_glob(
        tmp_path, literal_name, matching_sibling):
    destination = tmp_path / literal_name
    target = destination / "train" / "drop_shard0000.npz"
    victim = tmp_path / matching_sibling / "train" / "keep_shard0000.npz"
    target.parent.mkdir(parents=True)
    victim.parent.mkdir(parents=True)
    target.write_bytes(b"target")
    victim.write_bytes(b"victim")

    removed = clear_dataset_shards(str(destination))

    assert removed == [str(target)]
    assert not target.exists()
    assert victim.read_bytes() == b"victim"


def test_remote_damaged_destination_and_literal_glob_path_are_root_bounded():
    pytest.importorskip("fsspec")
    from kuration.core.io import _fs, exists

    root = "memory://publication-safety/run*"
    target = f"{root}/train/drop_shard0000.npz"
    victim = "memory://publication-safety/run-victim/train/keep_shard0000.npz"
    fs = _fs(root)
    fs.pipe(fs._strip_protocol(target), b"target")
    fs.pipe(fs._strip_protocol(victim), b"victim")
    unread = MemAdapter({"source": AssertionError("damaged destination guard must run first")})

    assert dataset_state(root) == "corrupt"
    with pytest.raises(ValueError, match="glob metacharacter"):
        _classic_write(root, unread)
    assert unread.reads == 0

    assert clear_dataset_shards(root) == ["memory:///publication-safety/run*/train/"
                                          "drop_shard0000.npz"]
    assert not exists(target)
    assert exists(victim)


def test_remote_manifest_recorded_shard_is_deleted_as_a_literal_key():
    from kuration.core.io import _fs, exists

    root = "memory://publication-safety/literal-record"
    literal = f"{root}/train/payload*"
    sibling = f"{root}/train/payload-private.txt"
    fs = _fs(root)
    fs.pipe(fs._strip_protocol(literal), b"owned")
    fs.pipe(fs._strip_protocol(sibling), b"private")
    fs.pipe(
        fs._strip_protocol(f"{root}/dataset_manifest.json"),
        json.dumps({
            "artifact_format_version": "1.2",
            "count": 1,
            "shards": [{"path": "train/payload*", "split": "train"}],
        }).encode(),
    )

    assert clear_dataset_shards(root) == [literal]
    assert not exists(literal)
    assert exists(sibling)


def test_sidecar_manifest_without_commit_is_corrupt_not_legacy(tmp_path):
    destination = tmp_path / "missing-sidecar"
    destination.mkdir()
    (destination / "dataset_manifest.json").write_text(json.dumps({
        "artifact_format_version": "1.2",
        "commit_protocol": "sidecar-v1",
        "count": 0,
        "shards": [],
    }))

    assert dataset_state(str(destination)) == "corrupt"


def test_resume_continues_writing_state_but_cannot_be_combined_with_force(tmp_path):
    destination = tmp_path / "resume"
    adapter = MemAdapter({"source": _ticks([1, 2, 3, 4])})
    _classic_write(destination, adapter)
    begin_dataset(str(destination))
    assert dataset_state(str(destination)) == "writing"

    resumed = _classic_write(
        destination, MemAdapter({"source": _ticks([1, 2, 3, 4])}), resume=True)
    assert resumed.shards
    assert dataset_state(str(destination)) == "committed"

    with pytest.raises(ValueError, match="mutually exclusive"):
        _classic_write(
            destination, MemAdapter({"source": _ticks([1, 2, 3, 4])}),
            force=True, resume=True)


def test_resume_rejects_existing_artifact_without_run_fingerprint(tmp_path):
    destination = tmp_path / "legacy"
    _classic_write(destination, MemAdapter({"source": _ticks([1, 2, 3, 4])}))
    (destination / ".kuration_output_spec.json").unlink()
    unread = MemAdapter({"source": AssertionError("unsafe resume must fail before source I/O")})

    with pytest.raises(ValueError, match="no complete run fingerprint"):
        _classic_write(destination, unread, resume=True)

    assert unread.reads == 0
    assert dataset_state(str(destination)) == "committed"


def test_resume_rejects_marker_not_pinned_by_commit(tmp_path):
    destination = tmp_path / "old"
    replacement = tmp_path / "replacement"
    _classic_write(destination, MemAdapter({"source": _ticks([1, 2, 3, 4])}))
    _classic_write(replacement, MemAdapter({"source": _ticks([101, 102, 103, 104])}))

    commit_path = destination / "dataset.commit.json"
    commit = json.loads(commit_path.read_text())
    commit["files"].pop(".kuration_output_spec.json")
    commit_path.write_text(json.dumps(commit))
    (destination / ".kuration_output_spec.json").write_bytes(
        (replacement / ".kuration_output_spec.json").read_bytes())
    unread = MemAdapter({"source": AssertionError("unpinned marker must fail before source I/O")})

    with pytest.raises(ValueError, match="not integrity-pinned"):
        _classic_write(destination, unread, resume=True)
    assert unread.reads == 0


def test_forced_hours_to_count_replacement_removes_resume_marker(tmp_path):
    destination = tmp_path / "switch"
    hours = (
        input([{"x": float(value)} for value in range(6)], hz=1)
        .window(2, stride=2)
        .measure("mean_x", "mean", of="x")
        .sample("top-k", hours=4 / 3600, weights={"mean_x": 1.0})
    )
    hours.output(destination, categorical="omit")
    assert (destination / ".kuration_output_spec.json").exists()

    count = (
        input([{"x": float(value)} for value in range(6)], hz=1)
        .window(2, stride=2)
        .measure("mean_x", "mean", of="x")
        .sample("top-k", n=2, weights={"mean_x": 1.0})
    )
    count.output(destination, categorical="omit", force=True)

    assert not (destination / ".kuration_output_spec.json").exists()
    commit = json.loads((destination / "dataset.commit.json").read_text())
    assert ".kuration_output_spec.json" not in commit["files"]
    assert shard_entries(str(destination))[0]


def test_pipeline_hours_output_commits_enriched_config_once(tmp_path, monkeypatch):
    import kuration.core.manifest as manifest_module

    commits = []
    real_commit = manifest_module.commit_dataset

    def observed_commit(path):
        commits.append(path)
        return real_commit(path)

    monkeypatch.setattr(manifest_module, "commit_dataset", observed_commit)
    destination = tmp_path / "hours-once"
    pipeline = (
        input([{"x": float(value)} for value in range(6)], hz=1)
        .window(2, stride=2)
        .measure("mean_x", "mean", of="x")
        .sample("top-k", hours=4 / 3600, weights={"mean_x": 1.0})
    )

    pipeline.output(destination, categorical="omit")

    assert commits == [str(destination)]
    config = json.loads((destination / "config.json").read_text())
    assert config["pipeline"] and config["output_request"]["op"] == "output"
    assert shard_entries(str(destination))[0]


def test_pipeline_output_config_redacts_input_output_and_adapter_runtime_identity(tmp_path):
    source = (
        "s3://private-user:super-secret@bucket.example/private/session.npz"
        "?token=do-not-publish#fragment"
    )
    adapter = MemAdapter({source: _ticks([1, 2, 3, 4])})
    destination = tmp_path / "private-output-parent" / "dataset"
    pipeline = (
        input(adapter=adapter, schema=SCHEMA, sources=[source])
        .window(2, stride=2)
        .measure("mean_x", "mean", of="x")
        .sample("top-k", hours=4 / 3600, weights={"mean_x": 1.0})
    )

    pipeline.output(destination, categorical="omit")

    config = json.loads((destination / "config.json").read_text())
    rendered = json.dumps({
        "pipeline": config["pipeline"],
        "output_request": config["output_request"],
    })
    for secret in (
        "private-user",
        "super-secret",
        "do-not-publish",
        "fragment",
        str(tmp_path),
        "private-output-parent",
        " object at 0x",
    ):
        assert secret not in rendered
    input_op = config["pipeline"][0]
    assert input_op["kwargs"]["sources"][0].startswith("session.npz-")
    assert input_op["kwargs"]["adapter"]["component"]["type"].endswith(".MemAdapter")
    assert config["output_request"]["args"][0].startswith("dataset-")


def test_missing_container_dependency_fails_before_forced_publication(
        tmp_path, monkeypatch):
    destination = tmp_path / "container"
    pipeline = (
        input([{"x": float(value)} for value in range(6)], hz=1)
        .window(2, stride=2)
        .measure("mean_x", "mean", of="x")
        .sample("top-k", n=2, weights={"mean_x": 1.0})
    )
    pipeline.output(destination, categorical="omit")
    before = _artifact_bytes(destination)
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "h5py" or name.startswith("h5py."):
            raise ImportError("simulated missing h5py")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match=r"kuration-sdk\[array\]"):
        pipeline.output(
            destination, categorical="omit", container="hdf5", force=True)

    assert dataset_state(str(destination)) == "committed"
    assert _artifact_bytes(destination) == before


def test_count_memory_preflight_preserves_forced_destination(tmp_path):
    destination = tmp_path / "memory"
    pipeline = (
        input([{"x": float(value)} for value in range(6)], hz=1)
        .window(2, stride=2)
        .measure("mean_x", "mean", of="x")
        .sample("top-k", n=2, weights={"mean_x": 1.0})
    )
    pipeline.output(destination, categorical="omit")
    before = _artifact_bytes(destination)

    with pytest.raises(ValueError, match="cannot hold one count-output shard"):
        pipeline.options(memory_budget=1).output(
            destination, categorical="omit", force=True)

    assert dataset_state(str(destination)) == "committed"
    assert _artifact_bytes(destination) == before


def test_writer_lease_rejects_a_concurrent_publisher_and_releases(tmp_path):
    from kuration.core.manifest import dataset_writer_lease

    destination = str(tmp_path / "leased")

    def contend():
        with dataset_writer_lease(destination):
            return "unexpected"

    with dataset_writer_lease(destination):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(contend)
            with pytest.raises(FileExistsError, match="active writer lease"):
                future.result()

    with dataset_writer_lease(destination):
        pass


def test_remote_writer_lease_is_inside_destination_and_normalizes_equivalent_urls():
    from kuration.core.io import _fs
    from kuration.core.manifest import dataset_writer_lease

    destination = "memory://publication-safety/remote-lease"
    equivalent = "memory:///publication-safety/remote-lease"
    fs = _fs(destination)

    with dataset_writer_lease(destination):
        assert fs.exists("/publication-safety/remote-lease/.kuration-writer.lock")
        with pytest.raises(FileExistsError, match="active writer lease"):
            with dataset_writer_lease(equivalent):
                pass

    assert not fs.exists("/publication-safety/remote-lease/.kuration-writer.lock")
    assert not fs.exists("/publication-safety/remote-lease.kuration-writer.lock")


def test_remote_publication_refuses_glob_metacharacters_before_writing():
    from kuration.core.io import _fs

    destination = "memory://publication-safety/refuse*"
    pipeline = (
        input([{"x": float(value)} for value in range(6)], hz=1)
        .window(2, stride=2)
        .measure("mean_x", "mean", of="x")
        .sample("top-k", n=2, weights={"mean_x": 1.0})
    )

    with pytest.raises(ValueError, match="glob metacharacter"):
        pipeline.output(destination, categorical="omit")

    assert not _fs(destination).find("/publication-safety/refuse*")


def test_progress_callback_cannot_reenter_same_destination_writer(tmp_path):
    destination = tmp_path / "callback-reentry"
    pipeline = (
        input([{"x": float(value)} for value in range(6)], hz=1)
        .window(2, stride=2)
        .measure("mean_x", "mean", of="x")
        .sample("top-k", n=2, weights={"mean_x": 1.0})
    )
    attempts: list[BaseException] = []

    def nested_writer(_event):
        if attempts:
            return
        try:
            pipeline.output(destination, categorical="omit")
        except FileExistsError as error:
            attempts.append(error)
        else:  # pragma: no cover - the lease must reject this path
            attempts.append(AssertionError("nested writer unexpectedly acquired the lease"))

    pipeline.options(progress=nested_writer).output(destination, categorical="omit")

    assert len(attempts) == 1
    assert isinstance(attempts[0], FileExistsError)
    assert "active writer lease" in str(attempts[0])
    assert dataset_state(str(destination)) == "committed"
    assert shard_entries(str(destination))[0]
