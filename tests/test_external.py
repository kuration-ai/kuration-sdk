"""Disk-backed exact quantiles and stable top-k selection."""
from contextlib import contextmanager
import os
import uuid

import numpy as np
import pytest

import kuration.core.external as external_module
from kuration import (PartitionCommit, Segment, SegmentBatch, external_quantile_edges,
                       external_top_k, external_uniform_bins, external_weighted,
                       iter_external_span_items, iter_external_hash_dedup,
                       iter_external_top_k_priorities, iter_external_uniform_priorities,
                       iter_external_weighted_priorities)
from kuration.core.output.external_fill import external_fill_to_hours
from kuration.core.output.fill import fill_to_hours

pytest.importorskip("pyarrow")


def _partitions(tmp_path):
    segments = [
        Segment("a", index, index + 1, (index, index),
                {"x": value, "y": float(7 - index)})
        for index, value in enumerate([4.0, 1.0, 8.0, 2.0, 9.0, 3.0])
    ]
    paths = []
    for index in range(0, len(segments), 2):
        path = tmp_path / f"part-{index // 2}.arrow"
        SegmentBatch.from_segments(segments[index:index + 2]).write_ipc(str(path))
        paths.append(str(path))
    return segments, paths


def _reference_scores(segments, weights, normalize):
    score = np.zeros(len(segments))
    for name, weight in weights.items():
        values = np.asarray([segment.features[name] for segment in segments])
        if normalize == "quantile":
            order = values.argsort(kind="stable")
            norm = np.empty(len(values))
            norm[order] = np.arange(len(values)) / max(1, len(values) - 1)
        elif normalize == "zscore":
            norm = (values - values.mean()) / values.std()
        elif normalize == "minmax":
            norm = (values - values.min()) / (values.max() - values.min())
        else:
            norm = values
        score += weight * norm
    return score


def _track_ipc_reads(monkeypatch):
    original = SegmentBatch.read_ipc.__func__
    calls = []

    def tracked(cls, path, *, columns=None):
        calls.append((str(path), None if columns is None else tuple(columns)))
        return original(cls, path, columns=columns)

    monkeypatch.setattr(SegmentBatch, "read_ipc", classmethod(tracked))
    return calls


def test_external_quantiles_match_numpy_exactly(tmp_path):
    segments, paths = _partitions(tmp_path)
    values = np.asarray([segment.features["x"] for segment in segments])
    expected = np.quantile(values, np.linspace(0, 1, 5)[1:-1])
    actual = external_quantile_edges(paths, "x", 4, spill_dir=str(tmp_path / "spill"))
    assert np.array_equal(actual, expected)
    assert not list((tmp_path / "spill").glob("*.sqlite3"))


@pytest.mark.parametrize("normalize", ["quantile", "zscore", "minmax", "none"])
def test_external_top_k_matches_exact_stable_reference(tmp_path, normalize):
    segments, paths = _partitions(tmp_path)
    weights = {"x": 0.7, "y": -0.2}
    scores = _reference_scores(segments, weights, normalize)
    expected = np.argsort(-scores, kind="stable")[:3]
    selected = external_top_k(paths, weights=weights, target_size=3,
                              normalize=normalize, spill_dir=str(tmp_path / "spill"))
    assert [(segment.source_id, segment.start_idx) for segment in selected] == [
        (segments[index].source_id, segments[index].start_idx) for index in expected]


def test_external_top_k_materializes_each_selected_partition_once(tmp_path, monkeypatch):
    segments, paths = _partitions(tmp_path)
    calls = _track_ipc_reads(monkeypatch)
    weights = {"x": 0.7, "y": -0.2}
    scores = _reference_scores(segments, weights, "quantile")
    expected = np.argsort(-scores, kind="stable")

    selected = external_top_k(
        paths, weights=weights, target_size=len(segments), normalize="quantile",
        spill_dir=str(tmp_path / "spill"))

    assert [segment.start_idx for segment in selected] == [
        segments[index].start_idx for index in expected]
    full_reads = [path for path, columns in calls if columns is None]
    assert len(full_reads) == len(paths)
    assert len(set(full_reads)) == len(paths)
    assert not set(full_reads) & set(paths)


def test_external_reducers_reject_nonfinite_features(tmp_path):
    path = tmp_path / "bad.arrow"
    SegmentBatch.from_segments([
        Segment("a", 0, 1, (0, 0), {"x": float("nan")})
    ]).write_ipc(str(path))
    with pytest.raises(ValueError, match="non-finite"):
        external_quantile_edges([str(path)], "x", 2)
    with pytest.raises(ValueError, match="non-finite"):
        external_top_k([str(path)], weights={"x": 1}, target_size=1)


def test_external_uniform_bins_is_balanced_bounded_and_deterministic(tmp_path):
    segments = [Segment("a", index, index + 1, (index, index), {"x": float(index)})
                for index in range(40)]
    paths = []
    for offset in range(0, 40, 7):
        path = tmp_path / f"bins-{offset}.arrow"
        SegmentBatch.from_segments(segments[offset:offset + 7]).write_ipc(str(path))
        paths.append(str(path))
    first = external_uniform_bins(paths, axes=[("x", 4)], target_size=12, seed=9,
                                  spill_dir=str(tmp_path / "spill"))
    second = external_uniform_bins(paths, axes=[("x", 4)], target_size=12, seed=9,
                                   spill_dir=str(tmp_path / "spill"))
    assert [segment.start_idx for segment in first] == [segment.start_idx for segment in second]
    counts = {code: sum(segment.bin == (code,) for segment in first) for code in range(4)}
    assert sorted(counts.values()) == [3, 3, 3, 3]


def test_external_uniform_bins_matches_seeded_reference_exactly(tmp_path):
    import random
    from kuration.core.binning import bin_segments
    from kuration.core.select import sample_uniform_bins
    segments, paths = _partitions(tmp_path)
    reference = [Segment(segment.source_id, segment.start_idx, segment.end_idx,
                         segment.time_range, dict(segment.features)) for segment in segments]
    bin_segments(reference, [("x", 3), ("y", 2)])
    expected = sample_uniform_bins(reference, target_size=5, rng=random.Random(17))
    actual = external_uniform_bins(paths, axes=[("x", 3), ("y", 2)], target_size=5,
                                   seed=17, spill_dir=str(tmp_path / "spill"))
    assert [(s.source_id, s.start_idx) for s in actual] == [
        (s.source_id, s.start_idx) for s in expected]


def test_external_uniform_materializes_each_selected_partition_once(tmp_path, monkeypatch):
    import random
    from kuration.core.binning import bin_segments
    from kuration.core.select import sample_uniform_bins

    segments, paths = _partitions(tmp_path)
    reference = [Segment(segment.source_id, segment.start_idx, segment.end_idx,
                         segment.time_range, dict(segment.features)) for segment in segments]
    axes = [("x", 3), ("y", 2)]
    bin_segments(reference, axes)
    expected = sample_uniform_bins(
        reference, target_size=len(reference), rng=random.Random(17))
    calls = _track_ipc_reads(monkeypatch)

    actual = external_uniform_bins(
        paths, axes=axes, target_size=len(segments), seed=17,
        spill_dir=str(tmp_path / "spill"))

    assert [(segment.start_idx, segment.bin) for segment in actual] == [
        (segment.start_idx, segment.bin) for segment in expected]
    full_reads = [path for path, columns in calls if columns is None]
    assert len(full_reads) == len(paths)
    assert len(set(full_reads)) == len(paths)
    assert not set(full_reads) & set(paths)


def test_external_reducer_keeps_validated_generation_after_path_swap(
        tmp_path, monkeypatch):
    original_path = tmp_path / "partition.arrow"
    original = SegmentBatch.from_segments([
        Segment("original", 0, 1, (0, 0), {"x": 9.0})
    ])
    metadata = original.write_ipc(str(original_path))
    commit = PartitionCommit(
        partition_id="source:a",
        identity="a" * 64,
        data_file=str(original_path),
        rows=len(original),
        sha256=str(metadata["sha256"]),
        columns=original.column_names,
        size_bytes=int(metadata["size_bytes"]),
    )
    replacement_path = tmp_path / "replacement.arrow"
    SegmentBatch.from_segments([
        Segment("replacement", 0, 1, (0, 0), {"x": -1.0})
    ]).write_ipc(str(replacement_path))
    real_pins = external_module.pinned_partition_files

    @contextmanager
    def swap_after_pin(partitions):
        with real_pins(partitions) as paths:
            os.replace(replacement_path, original_path)
            yield paths

    monkeypatch.setattr(external_module, "pinned_partition_files", swap_after_pin)

    selected = external_top_k([commit], weights={"x": 1.0}, target_size=1)

    assert [segment.source_id for segment in selected] == ["original"]


def test_external_bare_path_pins_intermediate_ancestor_before_swap(
        tmp_path, monkeypatch):
    from kuration.core import io as io_module

    if not io_module._HAS_POSIX_DESCRIPTOR_OPEN:
        pytest.skip("descriptor-relative no-follow reads are unavailable")
    trusted = tmp_path / "trusted"
    attacker = tmp_path / "attacker"
    trusted.mkdir()
    attacker.mkdir()
    original_path = trusted / "partition.arrow"
    SegmentBatch.from_segments([
        Segment("original", 0, 1, (0, 0), {"x": 9.0})
    ]).write_ipc(str(original_path))
    SegmentBatch.from_segments([
        Segment("redirected", 0, 1, (0, 0), {"x": -1.0})
    ]).write_ipc(str(attacker / "partition.arrow"))
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(attacker, target_is_directory=True)
        probe.unlink()
    except OSError:
        pytest.skip("symlinks are unavailable")

    saved = tmp_path / "trusted-before-swap"
    real_open = io_module.os.open
    swapped = False

    def swap_ancestor():
        nonlocal swapped
        trusted.rename(saved)
        trusted.symlink_to(attacker, target_is_directory=True)
        swapped = True

    def open_then_swap(path, flags, mode=0o777, *, dir_fd=None):
        if not swapped and path == "trusted" and dir_fd is not None:
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
            swap_ancestor()
            return descriptor
        # This branch makes the regression fail under a leaf-only full-path open.
        if not swapped and str(path) == str(original_path) and dir_fd is None:
            swap_ancestor()
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(io_module.os, "open", open_then_swap)

    selected = external_top_k(
        [str(original_path)], weights={"x": 1.0}, target_size=1)

    assert swapped
    assert [segment.source_id for segment in selected] == ["original"]


def test_external_non_posix_fallback_requires_checksum_bound_commit(
        tmp_path, monkeypatch):
    from kuration.core import io as io_module

    path = tmp_path / "partition.arrow"
    original = SegmentBatch.from_segments([
        Segment("original", 0, 1, (0, 0), {"x": 9.0})
    ])
    metadata = original.write_ipc(str(path))
    commit = PartitionCommit(
        partition_id="source:a",
        identity="a" * 64,
        data_file=str(path),
        rows=len(original),
        sha256=str(metadata["sha256"]),
        columns=original.column_names,
        size_bytes=int(metadata["size_bytes"]),
    )
    monkeypatch.setattr(io_module, "_HAS_POSIX_DESCRIPTOR_OPEN", False)

    with pytest.raises(ValueError, match="checksum-bound PartitionCommit"):
        external_top_k([str(path)], weights={"x": 1.0}, target_size=1)

    selected = external_top_k([commit], weights={"x": 1.0}, target_size=1)
    assert [segment.source_id for segment in selected] == ["original"]


def test_external_reducer_rejects_changed_commit_and_local_symlink(tmp_path):
    original_path = tmp_path / "partition.arrow"
    original = SegmentBatch.from_segments([
        Segment("original", 0, 1, (0, 0), {"x": 9.0})
    ])
    metadata = original.write_ipc(str(original_path))
    commit = PartitionCommit(
        partition_id="source:a",
        identity="a" * 64,
        data_file=str(original_path),
        rows=len(original),
        sha256=str(metadata["sha256"]),
        columns=original.column_names,
        size_bytes=int(metadata["size_bytes"]),
    )
    replacement_path = tmp_path / "replacement.arrow"
    SegmentBatch.from_segments([
        Segment("replacement", 0, 1, (0, 0), {"x": -1.0})
    ]).write_ipc(str(replacement_path))
    os.replace(replacement_path, original_path)

    with pytest.raises(ValueError, match="integrity"):
        external_top_k([commit], weights={"x": 1.0}, target_size=1)

    target = tmp_path / "target.arrow"
    os.replace(original_path, target)
    original_path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink|regular file"):
        external_quantile_edges([str(original_path)], "x", 2)


def test_external_reducer_preserves_remote_partition_commit_behavior():
    remote_path = f"memory://external-snapshot-{uuid.uuid4().hex}/partition.arrow"
    original = SegmentBatch.from_segments([
        Segment("remote", 0, 1, (0, 0), {"x": 3.0})
    ])
    metadata = original.write_ipc(remote_path)
    commit = PartitionCommit(
        partition_id="source:remote",
        identity="a" * 64,
        data_file=remote_path,
        rows=len(original),
        sha256=str(metadata["sha256"]),
        columns=original.column_names,
        size_bytes=int(metadata["size_bytes"]),
    )

    selected = external_top_k([commit], weights={"x": 1.0}, target_size=1)

    assert [segment.source_id for segment in selected] == ["remote"]


def test_external_uniform_priorities_match_hours_reference_exactly(tmp_path):
    import random
    from kuration.core.binning import bin_segments
    segments, paths = _partitions(tmp_path)
    reference = [Segment(segment.source_id, segment.start_idx, segment.end_idx,
                         segment.time_range, dict(segment.features)) for segment in segments]
    axes = [("x", 3), ("y", 2)]
    bin_segments(reference, axes)
    rng = random.Random(17)
    members = {}
    for index, segment in enumerate(reference):
        members.setdefault(segment.bin, []).append(index)
    expected = np.empty(len(reference))
    for indices in members.values():
        order = indices[:]
        rng.shuffle(order)
        for rank, index in enumerate(order):
            expected[index] = -(rank + 1)
    actual = list(iter_external_uniform_priorities(
        paths, axes=axes, seed=17, spill_dir=str(tmp_path / "spill")))
    assert [(segment.start_idx, segment.bin) for segment, _ in actual] == [
        (segment.start_idx, segment.bin) for segment in reference]
    assert [priority for _, priority in actual] == pytest.approx(expected)


def test_external_hours_fill_matches_reference_and_retains_only_admitted_metadata(tmp_path):
    items = [
        {"id": (f"s{i // 3}", i % 2, i), "source_id": f"s{i // 3}",
         "priority": float(20 - i), "seconds": 7.0 + i % 4,
         "group": f"g{i % 3}", "tags": {"rare"} if i in {7, 13} else {"common"},
         "start": i * 10, "end": i * 10 + 5, "meta": {"score": float(i)}}
        for i in range(18)
    ]
    kwargs = dict(train_hours=0.025, test_hours=0.012, max_group_overshoot=2.0,
                  tag_coverage={"min_tag_frac": 0.2}, seed=11)
    expected_ids, expected_split, expected_stats = fill_to_hours(
        items, return_stats=True, **kwargs)
    ids, split, stats, details = external_fill_to_hours(
        (item for item in items), spill_dir=str(tmp_path / "spill"), **kwargs)
    assert ids == expected_ids
    assert split == expected_split
    assert stats == expected_stats
    assert set(details) == ids
    for identity, detail in details.items():
        original = next(item for item in items if item["id"] == identity)
        assert detail == {"start": original["start"], "end": original["end"],
                          "meta": original["meta"]}
    assert not list((tmp_path / "spill").glob("*.sqlite3"))


def test_external_priority_and_span_stream_matches_in_memory_reference(tmp_path):
    segments, paths = _partitions(tmp_path)
    weights = {"x": 0.7, "y": -0.2}
    expected_priorities = _reference_scores(segments, weights, "quantile")
    prioritized = list(iter_external_top_k_priorities(
        paths, weights=weights, normalize="quantile", spill_dir=str(tmp_path / "spill")))
    assert [(segment.start_idx, priority) for segment, priority in prioritized] == pytest.approx(
        [(segment.start_idx, expected_priorities[index])
         for index, segment in enumerate(segments)])

    spans = list(iter_external_span_items(
        iter(prioritized), {"a": ((0, 6),)}, sample_hz=2, span_len=3,
        clip_aggregate="max", spill_dir=str(tmp_path / "spill")))
    assert [(item["start"], item["end"]) for item in spans] == [(0, 3), (3, 6)]
    assert spans[0]["priority"] == pytest.approx(max(expected_priorities[:3]))
    assert spans[1]["priority"] == pytest.approx(max(expected_priorities[3:]))
    assert spans[0]["meta"]["x"] == 8.0


def test_external_weighted_priorities_and_selection_match_reference(tmp_path):
    segments, paths = _partitions(tmp_path)
    weights, seed = {"x": 0.7, "y": -0.2}, 31
    scores = _reference_scores(segments, weights, "quantile")
    rng = np.random.default_rng(seed)
    adjusted = scores - scores.min() + 1e-9
    expected_priorities = np.log(adjusted) + (-np.log(-np.log(rng.random(len(scores)))))
    priorities = list(iter_external_weighted_priorities(
        paths, weights=weights, normalize="quantile", seed=seed,
        spill_dir=str(tmp_path / "spill")))
    assert [priority for _segment, priority in priorities] == pytest.approx(expected_priorities)
    expected = np.argsort(-expected_priorities, kind="stable")[:3]
    selected = external_weighted(paths, weights=weights, target_size=3,
                                 normalize="quantile", seed=seed,
                                 spill_dir=str(tmp_path / "spill"))
    assert [segment.start_idx for segment in selected] == [segments[i].start_idx for i in expected]


def test_external_weighted_priorities_build_one_reducer_index(tmp_path, monkeypatch):
    _segments, paths = _partitions(tmp_path)
    calls = 0
    original = external_module._database

    def tracked(spill_dir):
        nonlocal calls
        calls += 1
        return original(spill_dir)

    monkeypatch.setattr(external_module, "_database", tracked)
    priorities = list(iter_external_weighted_priorities(
        paths, weights={"x": 0.7, "y": -0.2}, normalize="quantile", seed=31,
        spill_dir=str(tmp_path / "spill")))

    assert len(priorities) == 6
    assert calls == 1


def test_external_exact_hash_dedup_is_canonical_and_group_scoped(tmp_path):
    segments = [
        Segment("a", 0, 1, (0, 0), {"x": 1.0, "y": 2.0}, group="g1"),
        Segment("a", 1, 2, (1, 1), {"x": 1.0, "y": 2.0}, group="g1"),
        Segment("a", 2, 3, (2, 2), {"x": 1.0, "y": 2.0}, group="g2"),
        Segment("a", 3, 4, (3, 3), {"x": 2.0, "y": 2.0}, group="g1"),
    ]
    paths = []
    for index in range(0, 4, 2):
        path = tmp_path / f"dedup-{index}.arrow"
        SegmentBatch.from_segments(segments[index:index + 2]).write_ipc(str(path))
        paths.append(str(path))
    global_kept = list(iter_external_hash_dedup(
        paths, feature_names=["x", "y"], spill_dir=str(tmp_path / "spill")))
    grouped_kept = list(iter_external_hash_dedup(
        paths, feature_names=["x", "y"], per_group=True,
        spill_dir=str(tmp_path / "spill")))
    assert [segment.start_idx for segment in global_kept] == [0, 3]
    assert [segment.start_idx for segment in grouped_kept] == [0, 2, 3]
