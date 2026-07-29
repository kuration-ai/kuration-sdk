"""Regression coverage for the July 2026 I/O and output correctness review."""
from __future__ import annotations

import builtins
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import pytest

from kuration import (
    OutputSpec,
    SchemaSpec,
    Segment,
    Tick,
    compare_datasets,
    iter_shards,
    stats,
)
from kuration.core.adapters.curated import CuratedDatasetAdapter
from kuration.core.adapters.parquet_dataset import ParquetDatasetAdapter
from kuration.core.adapters.timeseries import TimeseriesAdapter
from kuration.core.buffer import build_window
from kuration.core.external import iter_external_hash_dedup
from kuration.core.io import write_text
from kuration.core.loaders import read_shard_arrays, rebase_entries, shard_entries
from kuration.core.manifest import begin_dataset, commit_dataset, portable_shard_entries
from kuration.core.media import FrameRef, JoinedMediaAdapter, VideoAdapter, VideoMetadata
from kuration.core.output.external_fill import external_fill_to_hours
from kuration.core.output.shardwriter import track_name, write_track
from kuration.core.output.stream import stream_source_shards
from kuration.core.output.track import Track
from kuration.core.provenance import source_descriptor
from kuration.core.versioning import ARTIFACT_FORMAT_VERSION


SCHEMA = SchemaSpec.from_dict({
    "name": "io-review", "entity_axis": "entity", "sample_hz": 1.0,
    "fields": {"x": {"kind": "scalar"}},
})
SPEC = OutputSpec.from_dict({
    "shard_length": 2, "container": "npz",
    "modalities": {"x": {"dtype": "float32", "shape": [], "source": "field:x"}},
})


class _IdentityOnlyId:
    def __str__(self):
        return "same-text"


def test_track_identity_is_type_stable_and_does_not_overwrite(tmp_path):
    assert track_name("source", 1) != track_name("source", "1")
    numeric = write_track(str(tmp_path), Track("source", 1, [{"x": 1.0}]), SPEC,
                          schema=SCHEMA)
    textual = write_track(str(tmp_path), Track("source", "1", [{"x": 2.0}]), SPEC,
                          schema=SCHEMA)
    assert numeric[0]["path"] != textual[0]["path"]
    assert numeric[0]["entity_id"] != textual[0]["entity_id"]
    assert np.load(numeric[0]["path"])["x"].tolist() == [1.0]
    assert np.load(textual[0]["path"])["x"].tolist() == [2.0]

    class UnsupportedId:
        def __reduce__(self):
            raise TypeError("no stable serialization")

    with pytest.raises(ValueError, match="stable shard identity"):
        track_name("source", UnsupportedId())
    # Pickle is not a canonical identity contract: two unequal empty instances serialize to the
    # same state. Reject extension-defined IDs instead of letting them alias a shard path.
    with pytest.raises(ValueError, match="stable shard identity"):
        track_name("source", _IdentityOnlyId())


def test_output_and_manifest_paths_reject_split_traversal(tmp_path):
    with pytest.raises(ValueError, match="split"):
        write_track(str(tmp_path / "dataset"), Track("s", 0, [{"x": 1.0}]), SPEC,
                    schema=SCHEMA, split="../../outside")
    assert not (tmp_path / "outside").exists()

    jobs = [{"eid": 0, "span_index": 0, "start": 0, "end": 0,
             "split": "../outside", "name_suffix": ""}]
    with pytest.raises(ValueError, match="split"):
        stream_source_shards(str(tmp_path / "stream"), "s", iter(()), jobs,
                             schema=SCHEMA, spec=SPEC)

    with pytest.raises(ValueError, match="safe path component"):
        rebase_entries([{"path": "ignored/secret.npz", "split": "../../victim"}],
                       str(tmp_path / "dataset"))


def test_shard_loader_verifies_integrity_by_default_with_opt_out(tmp_path):
    entry = write_track(str(tmp_path), Track("s", 0, [{"x": 1.0}]), SPEC,
                        schema=SCHEMA)[0]
    np.savez(entry["path"], _n_steps=np.asarray(1),
             x=np.asarray([999.0], dtype=np.float32),
             _present=np.asarray([True]))
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_shard_arrays(entry, "npz", modalities=["x"])
    assert read_shard_arrays(entry, "npz", modalities=["x"], verify=False)["x"].tolist() \
        == [999.0]


def test_verified_shard_decode_uses_the_exact_snapshot_that_was_hashed(
        tmp_path, monkeypatch):
    from kuration.core import io as io_module

    entry = write_track(str(tmp_path), Track("s", 0, [{"x": 1.0}]), SPEC,
                        schema=SCHEMA)[0]
    replacement = tmp_path / "replacement.npz"
    np.savez(
        replacement,
        _n_steps=np.asarray(1),
        x=np.asarray([999.0], dtype=np.float32),
        _present=np.asarray([True]),
    )
    real_snapshot = io_module.local_snapshot

    @contextmanager
    def replace_after_snapshot(path):
        with real_snapshot(path) as snapshot:
            Path(path).write_bytes(replacement.read_bytes())
            yield snapshot

    monkeypatch.setattr(io_module, "local_snapshot", replace_after_snapshot)
    arrays = read_shard_arrays(entry, "npz", modalities=["x"])

    assert arrays["x"].tolist() == [1.0]
    assert np.load(entry["path"])["x"].tolist() == [999.0]


def _committed_dataset(root: Path, *, values=(1.0,)) -> None:
    entries = []
    for index, value in enumerate(values):
        entries.extend(
            write_track(
                str(root),
                Track(f"s{index}", 0, [{"x": value}]),
                SPEC,
                schema=SCHEMA,
                split="train",
            )
        )
    manifest = {"artifact_format_version": ARTIFACT_FORMAT_VERSION,
                "commit_protocol": "sidecar-v1", "count": len(entries),
                "shards": portable_shard_entries(str(root), entries)}
    config = {"artifact_format_version": ARTIFACT_FORMAT_VERSION,
              "schema": SCHEMA.to_dict(), "output": SPEC.to_dict()}
    write_text(root / "dataset_manifest.json", json.dumps(manifest))
    write_text(root / "config.json", json.dumps(config))
    commit_dataset(str(root))


def _replace_config_generation(root: Path, generation: str) -> None:
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    config["generation_probe"] = generation
    write_text(config_path, json.dumps(config))
    commit_dataset(str(root))


def test_metadata_reader_rejects_a_generation_switch_mid_snapshot(
        tmp_path, monkeypatch):
    from kuration.core import loaders as loaders_module

    first = tmp_path / "first"
    second = tmp_path / "second"
    _committed_dataset(first)
    _committed_dataset(second)
    second_manifest = json.loads((second / "dataset_manifest.json").read_text())
    second_manifest["generation_probe"] = "second"
    (second / "dataset_manifest.json").write_text(json.dumps(second_manifest))
    commit_dataset(str(second))
    real_read = loaders_module._read_bytes
    switched = False

    def switch_after_manifest(path):
        nonlocal switched
        payload = real_read(path)
        if not switched and str(path) == str(first / "dataset_manifest.json"):
            switched = True
            for name in ("dataset_manifest.json", "config.json", "dataset.commit.json"):
                (first / name).write_bytes((second / name).read_bytes())
        return payload

    monkeypatch.setattr(loaders_module, "_read_bytes", switch_after_manifest)
    with pytest.raises(ValueError, match="not atomically committed"):
        shard_entries(str(first))


def test_iter_shards_never_crosses_force_replacement_generations(tmp_path):
    _committed_dataset(tmp_path, values=(1.0, 2.0))
    shards = iter_shards(str(tmp_path), verify=False)

    assert next(shards)["x"].tolist() in ([1.0], [2.0])
    _replace_config_generation(tmp_path, "replacement")

    with pytest.raises(ValueError, match="changed generation during shard iteration"):
        next(shards)


def test_stats_rejects_generation_switch_during_content_scan(tmp_path, monkeypatch):
    from kuration.core import loaders as loaders_module

    _committed_dataset(tmp_path, values=(1.0, 2.0))
    real_read = loaders_module.read_shard_arrays
    switched = False

    def switch_after_first_read(entry, container, **kwargs):
        nonlocal switched
        arrays = real_read(entry, container, **kwargs)
        if not switched:
            switched = True
            _replace_config_generation(tmp_path, "replacement")
        return arrays

    monkeypatch.setattr(loaders_module, "read_shard_arrays", switch_after_first_read)
    with pytest.raises(ValueError, match="changed generation during artifact statistics"):
        stats(str(tmp_path), verify=False)


def test_compare_rejects_generation_switch_after_one_side_is_inspected(
        tmp_path, monkeypatch):
    from kuration.core import inspection as inspection_module

    left = tmp_path / "left"
    right = tmp_path / "right"
    _committed_dataset(left)
    _committed_dataset(right)
    real_inspect = inspection_module._inspect_resolved_dataset
    switched = False

    def switch_after_left(snapshot, **kwargs):
        nonlocal switched
        result = real_inspect(snapshot, **kwargs)
        if not switched and snapshot.base == str(left):
            switched = True
            _replace_config_generation(left, "replacement")
        return result

    monkeypatch.setattr(
        inspection_module, "_inspect_resolved_dataset", switch_after_left
    )
    with pytest.raises(ValueError, match="changed generation during dataset comparison"):
        compare_datasets(str(left), str(right))


def test_curated_adapter_rejects_generation_switch_before_shard_reuse(tmp_path):
    _committed_dataset(tmp_path)
    adapter = CuratedDatasetAdapter(str(tmp_path))
    shard = adapter.sources()[0]

    _replace_config_generation(tmp_path, "replacement")

    with pytest.raises(ValueError, match="changed generation during curated shard read"):
        adapter.ticks(shard)


def test_current_format_shard_symlink_cannot_escape_dataset_root(tmp_path):
    dataset = tmp_path / "dataset"
    _committed_dataset(dataset)
    adapter = CuratedDatasetAdapter(str(dataset))
    adapter_shard = adapter.sources()[0]
    manifest = json.loads((dataset / "dataset_manifest.json").read_text())
    shard = dataset / manifest["shards"][0]["path"]
    victim = tmp_path / "outside.npz"
    victim.write_bytes(shard.read_bytes())
    shard.unlink()
    try:
        shard.symlink_to(victim)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ValueError, match="shard path .* is a symlink"):
        shard_entries(str(dataset))
    with pytest.raises(ValueError, match="shard path .* is a symlink"):
        CuratedDatasetAdapter(str(dataset))
    with pytest.raises(ValueError, match="shard path .* is a symlink"):
        adapter.ticks(adapter_shard)


def test_current_format_shard_must_be_a_regular_file(tmp_path):
    _committed_dataset(tmp_path)
    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text())
    shard = tmp_path / manifest["shards"][0]["path"]
    shard.unlink()
    shard.mkdir()

    with pytest.raises(ValueError, match="shard path .* is not a regular file"):
        shard_entries(str(tmp_path))


def test_current_format_manifest_dtype_must_match_pinned_output_contract(tmp_path):
    _committed_dataset(tmp_path)
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"][0]["dtype"]["x"] = "float64"
    write_text(manifest_path, json.dumps(manifest))
    commit_dataset(str(tmp_path))

    with pytest.raises(ValueError, match="dtype .* does not match config.output dtype"):
        list(iter_shards(str(tmp_path)))


def test_sidecar_commit_requires_integrity_on_every_shard(tmp_path):
    _committed_dataset(tmp_path)
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"][0].pop("sha256")
    manifest_path.write_text(json.dumps(manifest))
    commit_dataset(str(tmp_path))

    with pytest.raises(ValueError, match="missing a lowercase SHA-256"):
        shard_entries(str(tmp_path))


def test_sidecar_manifest_count_must_match_entries(tmp_path):
    _committed_dataset(tmp_path)
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["count"] = 2
    manifest_path.write_text(json.dumps(manifest))
    commit_dataset(str(tmp_path))

    with pytest.raises(ValueError, match="count=2.*1 shard entries"):
        shard_entries(str(tmp_path))


def test_sidecar_manifest_rejects_duplicate_shard_paths(tmp_path):
    _committed_dataset(tmp_path)
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"].append(dict(manifest["shards"][0]))
    manifest["count"] = 2
    manifest_path.write_text(json.dumps(manifest))
    commit_dataset(str(tmp_path))

    with pytest.raises(ValueError, match="repeats shard path"):
        shard_entries(str(tmp_path))


def test_sidecar_manifest_rejects_path_alias_for_same_semantic_shard(tmp_path):
    _committed_dataset(tmp_path)
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    duplicate = dict(manifest["shards"][0])
    original = tmp_path / duplicate["path"]
    duplicate["path"] = duplicate["path"].replace(
        duplicate["track_id"], f"duplicate_{duplicate['track_id']}")
    copied = tmp_path / duplicate["path"]
    copied.write_bytes(original.read_bytes())
    manifest["shards"].append(duplicate)
    manifest["count"] = 2
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["results"] = {"shards": 2}
    config_path.write_text(json.dumps(config))
    manifest_path.write_text(json.dumps(manifest))
    commit_dataset(str(tmp_path))

    with pytest.raises(ValueError, match="track_id.*does not match"):
        shard_entries(str(tmp_path))


def test_sidecar_manifest_rejects_gapped_track_indices(tmp_path):
    _committed_dataset(tmp_path)
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["shards"][0]
    entry["path"] = entry["path"].replace("_shard0000.", "_shard0001.")
    entry["shard_index"] = 1
    manifest_path.write_text(json.dumps(manifest))
    commit_dataset(str(tmp_path))

    with pytest.raises(ValueError, match="repeated or gapped shard indices"):
        shard_entries(str(tmp_path))


def test_sidecar_config_shard_count_must_match_manifest(tmp_path):
    _committed_dataset(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["results"] = {"shards": 2}
    config_path.write_text(json.dumps(config))
    commit_dataset(str(tmp_path))

    with pytest.raises(ValueError, match=r"results\.shards=2.*1 entries"):
        shard_entries(str(tmp_path))


def test_curated_adapter_rejects_incomplete_commit(tmp_path):
    _committed_dataset(tmp_path)
    begin_dataset(str(tmp_path))
    with pytest.raises(ValueError, match="commit state"):
        CuratedDatasetAdapter(str(tmp_path))


def test_timeseries_preserves_large_integer_times_and_missing_categories():
    adapter = TimeseriesAdapter(SCHEMA, fields={"x": "x"}, time_column="time")
    first = 2**53
    ticks = adapter.from_rows([{"time": str(first), "x": 1},
                               {"time": str(first + 1), "x": 2}])
    assert [tick.time for tick in ticks] == [first, first + 1]

    categorical = SchemaSpec.from_dict({
        "name": "category", "entity_axis": "entity",
        "fields": {"label": {"kind": "categorical"}},
    })
    cat_adapter = TimeseriesAdapter(categorical, fields={"label": "label"})
    window = build_window(categorical, cat_adapter.from_rows([{"label": None}]))
    assert window.codes["label"].tolist() == [[-1]]
    assert window.vocabs["label"] == []


def test_parquet_dataset_glob_preflight_and_fingerprint_track_parts(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    path = tmp_path / "part-0.parquet"
    pq.write_table(pa.table({"x": [1.0]}), path)
    adapter = ParquetDatasetAdapter(
        SCHEMA, fields={"x": "x"}, path=str(tmp_path / "*.parquet"),
        source_per="dataset")
    source = adapter.sources()[0]
    adapter.check_source(source)
    before = source_descriptor(source, adapter)
    pq.write_table(pa.table({"x": [2.0]}), path)
    after = source_descriptor(source, adapter)
    assert before != after


class _MediaProvider:
    capabilities = {"streaming": True, "variable_frame_rate": True}

    def __init__(self, rates=None, shapes=None):
        self.rates = rates or {}
        self.shapes = shapes or {}

    def probe(self, uri):
        height, width = self.shapes.get(uri, (2, 2))
        return VideoMetadata(uri, width, height, 1, 1.0, self.rates.get(uri, 10.0))

    def iter_refs(self, uri, *, source_id=None):
        yield FrameRef(uri, source_id or uri, 0, 0.0)

    def decode_batch(self, refs):
        refs = list(refs)
        return np.zeros((len(refs), 2, 2, 3), dtype=np.uint8)


class _FingerprintAdapter:
    def __init__(self):
        self.schema = SCHEMA
        self.version = 1

    def source_fingerprint(self, _source):
        return {"version": self.version}

    def ticks(self, _source):
        return [Tick(0, {0: {"x": 1.0}})]


def test_joined_media_fingerprint_composes_wrapped_source_and_video(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"first")
    wrapped = _FingerprintAdapter()
    joined = JoinedMediaAdapter(wrapped, {"logical": str(video)},
                                provider=_MediaProvider(), on="frame_index")
    first = source_descriptor("logical", joined)
    wrapped.version = 2
    second = source_descriptor("logical", joined)
    assert second != first
    video.write_bytes(b"second generation")
    third = source_descriptor("logical", joined)
    assert third != second


def test_video_adapter_rejects_mixed_frame_rates():
    provider = _MediaProvider({"a.mp4": 30.0, "b.mp4": 60.0})
    with pytest.raises(ValueError, match="frame rate"):
        VideoAdapter(["a.mp4", "b.mp4"], provider=provider)


def test_joined_media_adapter_validates_every_mapped_video_contract():
    provider = _MediaProvider(shapes={"second.mp4": (3, 4)})
    with pytest.raises(ValueError, match=r"second\.mp4.*frame shape.*does not match"):
        JoinedMediaAdapter(
            _FingerprintAdapter(),
            {"first": "first.mp4", "second": "second.mp4"},
            provider=provider,
            on="frame_index",
        )


def test_external_fill_uses_python_equality_for_duplicate_ids(tmp_path):
    items = [
        {"id": 1, "source_id": "s", "priority": 2.0, "seconds": 1.0,
         "group": None},
        {"id": True, "source_id": "s", "priority": 1.0, "seconds": 1.0,
         "group": None},
    ]
    with pytest.raises(ValueError, match="duplicate fill item id"):
        external_fill_to_hours(items, train_hours=1, test_hours=0,
                               spill_dir=str(tmp_path))


def test_columnar_rejects_duplicate_bins_reads_all_batches_and_hashes_streaming(
        tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    from kuration.core.columnar import SegmentBatch

    segments = [Segment("a", 0, 1, (0, 1), {"x": 1.0}, bin=(1, 2))]
    with pytest.raises(ValueError, match="bin_names must be unique"):
        SegmentBatch.from_segments(segments, bin_names=("same", "same"))

    first = SegmentBatch.from_segments([Segment("a", 0, 1, (0, 1), {"x": 1.0})])
    second = SegmentBatch.from_segments([Segment("b", 1, 2, (1, 2), {"x": 2.0})])
    multi = tmp_path / "multi.arrow"
    with pa.OSFile(str(multi), "wb") as sink:
        with pa.ipc.new_file(sink, first.schema) as writer:
            writer.write_batch(first.record_batch)
            writer.write_batch(second.record_batch)
    restored = SegmentBatch.read_ipc(str(multi))
    assert [segment.source_id for segment in restored.to_segments()] == ["a", "b"]

    target = tmp_path / "hashed.arrow"
    real_open = builtins.open
    read_sizes = []

    class _ReadProxy:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def read(self, size=-1):
            read_sizes.append(size)
            return self.file.read(size)

        def __getattr__(self, name):
            return getattr(self.file, name)

    def tracked_open(file, mode="r", *args, **kwargs):
        opened = real_open(file, mode, *args, **kwargs)
        return _ReadProxy(opened) if mode == "rb" else opened

    monkeypatch.setattr(builtins, "open", tracked_open)
    first.write_ipc(str(target))
    assert read_sizes and -1 not in read_sizes


def test_external_hash_dedup_rejects_partition_feature_drift(tmp_path):
    pytest.importorskip("pyarrow")
    from kuration.core.columnar import SegmentBatch

    first = tmp_path / "first.arrow"
    second = tmp_path / "second.arrow"
    SegmentBatch.from_segments([
        Segment("s", 0, 1, (0, 1), {"x": 1.0})]).write_ipc(str(first))
    SegmentBatch.from_segments([
        Segment("s", 1, 2, (1, 2), {"x": 1.0, "y": 99.0})]).write_ipc(str(second))
    with pytest.raises(ValueError, match="feature schema differs"):
        list(iter_external_hash_dedup([str(first), str(second)]))


def test_noncallable_batch_hook_falls_back_to_timestep_encoder(tmp_path):
    class Encoder:
        encode_batch = None

        def timestep(self, _record, _ctx):
            return {"value": 7.0}

    spec = OutputSpec.from_dict({
        "shard_length": 1, "container": "npz",
        "modalities": {"encoded": {"dtype": "float32", "shape": [],
                                     "source": "encoder:value"}},
    })
    entry = write_track(str(tmp_path), Track("s", 0, [{}]), spec,
                        encoder=Encoder())[0]
    assert np.load(entry["path"])["encoded"].tolist() == [7.0]
