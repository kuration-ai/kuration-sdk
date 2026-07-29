"""Disk-backed exact reducers over checkpointed segment feature tables."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import heapq
import json
import math
import os
import pickle
import random
import sqlite3
import struct
import tempfile
from typing import Any, Iterable

import numpy as np

from .checkpoints import PartitionCommit, pinned_partition_files
from .columnar import SegmentBatch
from .segment import Segment


def _database(spill_dir: str | None):
    if spill_dir is not None:
        from .io import is_remote
        if is_remote(spill_dir):
            raise ValueError("disk-backed reducers currently require a local spill_dir")
        os.makedirs(spill_dir, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(suffix=".sqlite3", dir=spill_dir, delete=False)
    handle.close()
    connection = sqlite3.connect(handle.name)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    return connection, handle.name


def _materialize_selected_rows(selected):
    """Load each selected Arrow partition once while retaining reducer result order."""
    grouped: dict[Any, list[Any]] = {}
    for output_index, (path, row_idx, *metadata) in enumerate(selected):
        grouped.setdefault(path, []).append((output_index, row_idx, metadata))

    materialized: list[Any] = [None] * len(selected)
    for path, requests in grouped.items():
        batch = SegmentBatch.read_ipc(path)
        for output_index, row_idx, metadata in requests:
            segment = batch.slice(row_idx, 1).to_segments()[0]
            materialized[output_index] = (segment, *metadata)
    return materialized


def _shuffle_slots(connection: sqlite3.Connection, rng: random.Random) -> None:
    """Shuffle each canonical bin in memory, then persist it in one batch."""
    bins = connection.execute(
        "SELECT b.bin_key FROM bins b JOIN slots s ON s.bin_key=b.bin_key "
        "GROUP BY b.bin_key,b.bin_order ORDER BY b.bin_order").fetchall()
    for bin_key, in bins:
        candidate_ids = [row[0] for row in connection.execute(
            "SELECT candidate_id FROM slots WHERE bin_key=? ORDER BY position",
            (bin_key,))]
        rng.shuffle(candidate_ids)
        connection.executemany(
            "UPDATE slots SET candidate_id=? WHERE bin_key=? AND position=?",
            ((candidate_id, bin_key, position)
             for position, candidate_id in enumerate(candidate_ids)))


def _create_canonical_uniform_slots(connection: sqlite3.Connection) -> None:
    """Index uniform-bin members in the same first-seen order as a canonical local pool."""
    connection.executescript(
        "CREATE TABLE bins (bin_key TEXT PRIMARY KEY,bin_order INTEGER NOT NULL);"
        "INSERT INTO bins "
        "SELECT bin_key,ROW_NUMBER() OVER "
        "(ORDER BY source_id,start_idx,end_idx)-1 "
        "FROM ("
        "SELECT bin_key,source_id,start_idx,end_idx,"
        "ROW_NUMBER() OVER (PARTITION BY bin_key "
        "ORDER BY source_id,start_idx,end_idx) AS member_order "
        "FROM candidates"
        ") WHERE member_order=1;"
        "CREATE TABLE slots (bin_key TEXT NOT NULL,position INTEGER NOT NULL,"
        "candidate_id INTEGER NOT NULL,PRIMARY KEY(bin_key,position));"
        "INSERT INTO slots "
        "SELECT bin_key,ROW_NUMBER() OVER (PARTITION BY bin_key "
        "ORDER BY source_id,start_idx,end_idx)-1,id FROM candidates;")


def _external_quantile_edges_paths(
    paths: Iterable[str],
    feature: str,
    n_bins: int,
    *,
    spill_dir: str | None = None,
) -> np.ndarray:
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    column = feature if feature.startswith("feature:") else f"feature:{feature}"
    connection, db_path = _database(spill_dir)
    try:
        connection.execute("CREATE TABLE values_table (value REAL NOT NULL)")
        count = 0
        for path in paths:
            batch = SegmentBatch.read_ipc(path, columns=[column])
            values = batch.numpy(column, zero_copy_only=False).astype(np.float64, copy=False)
            if not np.isfinite(values).all():
                raise ValueError(f"quantile feature {feature!r} contains non-finite values")
            connection.executemany("INSERT INTO values_table VALUES (?)",
                                   ((float(value),) for value in values))
            count += len(values)
        connection.execute("CREATE INDEX values_order ON values_table(value)")
        connection.commit()
        if count == 0:
            return np.zeros(max(0, n_bins - 1), dtype=np.float64)
        edges = []
        for quantile in np.linspace(0.0, 1.0, n_bins + 1)[1:-1]:
            position = float(quantile) * (count - 1)
            lower, upper = math.floor(position), math.ceil(position)
            low = connection.execute(
                "SELECT value FROM values_table ORDER BY value LIMIT 1 OFFSET ?", (lower,)
            ).fetchone()[0]
            high = connection.execute(
                "SELECT value FROM values_table ORDER BY value LIMIT 1 OFFSET ?", (upper,)
            ).fetchone()[0]
            edges.append(low + (high - low) * (position - lower))
        return np.asarray(edges, dtype=np.float64)
    finally:
        connection.close()
        if os.path.exists(db_path):
            os.unlink(db_path)


def external_quantile_edges(partitions: Iterable[PartitionCommit | str], feature: str,
                            n_bins: int, *, spill_dir: str | None = None) -> np.ndarray:
    """Compute NumPy-compatible exact quantile edges from pinned partition generations."""
    with pinned_partition_files(partitions) as paths:
        return _external_quantile_edges_paths(
            paths, feature, n_bins, spill_dir=spill_dir)


def external_top_k(partitions: Iterable[PartitionCommit | str], *, weights: dict[str, float],
                   target_size: int, normalize: str = "quantile",
                   spill_dir: str | None = None, predicates=()) -> list:
    """Exact stable top-k using a disk-backed feature index.

    Only the selected ``k`` Python ``Segment`` objects are materialized. Equal feature values
    receive their shared average rank, matching the local tie-aware quantile normalization.
    """
    if not isinstance(target_size, int) or isinstance(target_size, bool) or target_size < 1:
        raise ValueError("target_size must be a positive integer")
    with _top_k_priority_index(
            partitions, weights=weights, normalize=normalize,
            spill_dir=spill_dir) as (connection, total):
        if not total:
            return []
        selected = connection.execute(
            "SELECT data_file,row_idx FROM candidates ORDER BY score DESC,source_id,start_idx,end_idx "
        ).fetchall()
        if not predicates:
            selected = selected[:target_size]
            return [segment for segment, in _materialize_selected_rows(selected)]
        from .strategy import _passes
        out = []
        for segment, in _materialize_selected_rows(selected):
            if _passes(segment, predicates):
                out.append(segment)
                if len(out) == target_size:
                    break
        return out


def _external_uniform_bins_paths(paths: Iterable[str], *,
                                 axes: Iterable[tuple[str, int]], target_size: int,
                                 seed: int = 0, spill_dir: str | None = None,
                                 predicates=()) -> list:
    axes = tuple(axes)
    if not axes:
        raise ValueError("external_uniform_bins requires at least one axis")
    if not isinstance(target_size, int) or isinstance(target_size, bool) or target_size < 1:
        raise ValueError("target_size must be a positive integer")
    paths = tuple(paths)
    edges = {}
    for name, bins in axes:
        values = _external_quantile_edges_paths(
            paths, name, bins, spill_dir=spill_dir)
        edges[name] = np.unique(values)
    columns = [f"feature:{name}" for name, _ in axes]
    connection, db_path = _database(spill_dir)
    try:
        connection.executescript(
            "CREATE TABLE candidates (id INTEGER PRIMARY KEY,source_id TEXT NOT NULL,"
            "start_idx INTEGER NOT NULL,"
            "end_idx INTEGER NOT NULL,data_file TEXT NOT NULL,row_idx INTEGER NOT NULL,"
            "bin_key TEXT NOT NULL,"
            "UNIQUE(source_id,start_idx,end_idx));")
        candidate_id = 0
        for path in paths:
            batch = SegmentBatch.read_ipc(path, columns=None if predicates else columns)
            raw = batch.record_batch.to_pydict()
            rows = []
            for row_idx in range(len(batch)):
                codes = []
                for name, _ in axes:
                    value = raw[f"feature:{name}"][row_idx]
                    if value is None:
                        raise ValueError(
                            f"binning feature {name!r} is missing for segment "
                            f"{raw['source_id'][row_idx]!r} "
                            f"[{raw['start_idx'][row_idx]}, {raw['end_idx'][row_idx]})")
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError(f"binning feature {name!r} contains non-finite values")
                    codes.append(int(np.searchsorted(edges[name], value, side="left")))
                bin_key = json.dumps(codes, separators=(",", ":"))
                source_id = raw["source_id"][row_idx]
                start_idx, end_idx = raw["start_idx"][row_idx], raw["end_idx"][row_idx]
                if predicates:
                    from .strategy import _passes
                    segment = batch.slice(row_idx, 1).to_segments()[0]
                    if not _passes(segment, predicates):
                        continue
                rows.append((candidate_id, source_id, start_idx, end_idx, path, row_idx,
                             bin_key))
                candidate_id += 1
            try:
                connection.executemany("INSERT INTO candidates VALUES (?,?,?,?,?,?,?)", rows)
            except sqlite3.IntegrityError as error:
                raise ValueError("duplicate stable segment key in external feature table") from error
        _create_canonical_uniform_slots(connection)
        _shuffle_slots(connection, random.Random(seed))
        connection.commit()
        selected = connection.execute(
            "SELECT c.data_file,c.row_idx,c.bin_key FROM slots s "
            "JOIN candidates c ON c.id=s.candidate_id JOIN bins b ON b.bin_key=s.bin_key "
            "ORDER BY s.position,b.bin_order LIMIT ?", (target_size,)).fetchall()
        out = []
        for segment, bin_key in _materialize_selected_rows(selected):
            segment.bin = tuple(json.loads(bin_key))
            out.append(segment)
        return out
    finally:
        connection.close()
        if os.path.exists(db_path):
            os.unlink(db_path)


def external_uniform_bins(partitions: Iterable[PartitionCommit | str], *,
                          axes: Iterable[tuple[str, int]], target_size: int, seed: int = 0,
                          spill_dir: str | None = None, predicates=()) -> list:
    """Select exact uniform bins while holding immutable partition snapshots."""
    with pinned_partition_files(partitions) as paths:
        return _external_uniform_bins_paths(
            paths, axes=axes, target_size=target_size, seed=seed,
            spill_dir=spill_dir, predicates=predicates)


def _iter_external_uniform_priorities_paths(paths: Iterable[str], *,
                                            axes: Iterable[tuple[str, int]], seed: int = 0,
                                            spill_dir: str | None = None):
    axes = tuple(axes)
    if not axes:
        raise ValueError("uniform-bin priorities require at least one axis")
    paths = tuple(paths)
    edges = {name: np.unique(_external_quantile_edges_paths(
        paths, name, bins, spill_dir=spill_dir)) for name, bins in axes}
    columns = [f"feature:{name}" for name, _ in axes]
    connection, db_path = _database(spill_dir)
    try:
        connection.executescript(
            "CREATE TABLE candidates (id INTEGER PRIMARY KEY,source_id TEXT NOT NULL,"
            "start_idx INTEGER NOT NULL,end_idx INTEGER NOT NULL,data_file TEXT NOT NULL,"
            "row_idx INTEGER NOT NULL,bin_key TEXT NOT NULL,"
            "UNIQUE(source_id,start_idx,end_idx));")
        candidate_id = 0
        for path in paths:
            batch = SegmentBatch.read_ipc(path, columns=columns)
            raw = batch.record_batch.to_pydict()
            rows = []
            for row_idx in range(len(batch)):
                codes = []
                for name, _ in axes:
                    value = raw[f"feature:{name}"][row_idx]
                    if value is None:
                        raise ValueError(
                            f"binning feature {name!r} is missing for segment "
                            f"{raw['source_id'][row_idx]!r} "
                            f"[{raw['start_idx'][row_idx]}, {raw['end_idx'][row_idx]})")
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError(f"binning feature {name!r} contains non-finite values")
                    codes.append(int(np.searchsorted(edges[name], value, side="left")))
                bin_key = json.dumps(codes, separators=(",", ":"))
                rows.append((candidate_id, raw["source_id"][row_idx],
                             raw["start_idx"][row_idx], raw["end_idx"][row_idx],
                             path, row_idx, bin_key))
                candidate_id += 1
            try:
                connection.executemany("INSERT INTO candidates VALUES (?,?,?,?,?,?,?)", rows)
            except sqlite3.IntegrityError as error:
                raise ValueError("duplicate stable segment key in external feature table") from error
        _create_canonical_uniform_slots(connection)
        _shuffle_slots(connection, random.Random(seed))
        connection.commit()
        cached_path, cached_batch = None, None
        query = ("SELECT c.data_file,c.row_idx,c.bin_key,s.position FROM candidates c "
                 "JOIN slots s ON s.candidate_id=c.id ORDER BY c.source_id,c.start_idx,c.end_idx")
        for path, row_idx, bin_key, position in connection.execute(query):
            if path != cached_path:
                cached_path, cached_batch = path, SegmentBatch.read_ipc(path)
            assert cached_batch is not None
            segment = cached_batch.slice(row_idx, 1).to_segments()[0]
            segment.bin = tuple(json.loads(bin_key))
            yield segment, -float(position + 1)
    finally:
        connection.close()
        if os.path.exists(db_path):
            os.unlink(db_path)


def iter_external_uniform_priorities(partitions: Iterable[PartitionCommit | str], *,
                                     axes: Iterable[tuple[str, int]], seed: int = 0,
                                     spill_dir: str | None = None):
    """Yield exact uniform-bin priorities from pinned partition generations."""
    with pinned_partition_files(partitions) as paths:
        yield from _iter_external_uniform_priorities_paths(
            paths, axes=axes, seed=seed, spill_dir=spill_dir)


@contextmanager
def _top_k_priority_index_paths(paths: Iterable[str], *,
                                weights: dict[str, float], normalize: str,
                                spill_dir: str | None):
    """Build one reusable disk-backed normalized-priority index."""
    if not weights:
        raise ValueError("top-k priorities require at least one feature weight")
    if normalize not in {"quantile", "minmax", "zscore", "none"}:
        raise ValueError("normalize must be quantile, minmax, zscore, or none")
    names = tuple(weights)
    columns = [f"feature:{name}" for name in names]
    connection, db_path = _database(spill_dir)
    try:
        connection.executescript(
            "CREATE TABLE candidates (id INTEGER PRIMARY KEY,source_id TEXT NOT NULL,"
            "start_idx INTEGER NOT NULL,end_idx INTEGER NOT NULL,data_file TEXT NOT NULL,"
            "row_idx INTEGER NOT NULL,score REAL NOT NULL DEFAULT 0,"
            "UNIQUE(source_id,start_idx,end_idx));"
            "CREATE TABLE features (candidate_id INTEGER NOT NULL,name TEXT NOT NULL,"
            "value REAL NOT NULL,PRIMARY KEY(candidate_id,name));")
        candidate_id = 0
        for path in paths:
            batch = SegmentBatch.read_ipc(path, columns=columns)
            raw = batch.record_batch.to_pydict()
            candidates, features = [], []
            for row_idx in range(len(batch)):
                candidates.append((candidate_id, raw["source_id"][row_idx],
                                   raw["start_idx"][row_idx], raw["end_idx"][row_idx],
                                   path, row_idx))
                for name in names:
                    value = raw[f"feature:{name}"][row_idx]
                    if value is None:
                        raise ValueError(
                            f"top-k feature {name!r} is missing for segment "
                            f"{raw['source_id'][row_idx]!r} "
                            f"[{raw['start_idx'][row_idx]}, {raw['end_idx'][row_idx]})")
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError(f"top-k feature {name!r} contains non-finite values")
                    features.append((candidate_id, name, value))
                candidate_id += 1
            try:
                connection.executemany(
                    "INSERT INTO candidates(id,source_id,start_idx,end_idx,data_file,row_idx) "
                    "VALUES (?,?,?,?,?,?)", candidates)
            except sqlite3.IntegrityError as error:
                raise ValueError("duplicate stable segment key in external feature table") from error
            connection.executemany("INSERT INTO features VALUES (?,?,?)", features)
        connection.execute("CREATE INDEX feature_lookup ON features(name,value)")
        total = candidate_id
        if total:
            for name in sorted(weights):
                weight = float(weights[name])
                if not math.isfinite(weight):
                    raise ValueError("top-k weights must be finite")
                if normalize == "quantile":
                    low, high = connection.execute(
                        "SELECT MIN(value),MAX(value) FROM features WHERE name=?", (name,)
                    ).fetchone()
                    if total > 1 and high > low:
                        connection.execute(
                            "WITH ranked AS (SELECT f.candidate_id AS id,("
                            "CAST(RANK() OVER (ORDER BY f.value)-1 AS REAL)"
                            "+(COUNT(*) OVER (PARTITION BY f.value)-1)/2.0)/? AS norm "
                            "FROM features f WHERE f.name=?) "
                            "UPDATE candidates SET score=score+?*(SELECT norm FROM ranked WHERE "
                            "ranked.id=candidates.id)", (total - 1, name, weight))
                elif normalize in {"minmax", "zscore"}:
                    low, high, mean = connection.execute(
                        "SELECT MIN(value),MAX(value),AVG(value) FROM features "
                        "WHERE name=?", (name,)).fetchone()
                    if normalize == "minmax":
                        center, scale = low, high - low
                    else:
                        # Center before squaring.  AVG(x*x)-AVG(x)**2 catastrophically
                        # cancels for low-variance values with a large offset, causing the
                        # bounded path to silently drop a feature that NumPy retains.
                        variance = connection.execute(
                            "SELECT AVG((value-?)*(value-?)) FROM features WHERE name=?",
                            (mean, mean, name)).fetchone()[0]
                        center, scale = mean, math.sqrt(max(0.0, variance))
                    connection.execute(
                        "UPDATE candidates SET score=score+?*COALESCE((SELECT CASE WHEN ?>0 THEN "
                        "(value-?)/? ELSE 0 END FROM features WHERE candidate_id=candidates.id "
                        "AND name=?),0)", (weight, scale, center, scale or 1.0, name))
                else:
                    connection.execute(
                        "UPDATE candidates SET score=score+?*(SELECT value FROM features WHERE "
                        "candidate_id=candidates.id AND name=?)", (weight, name))
        connection.commit()
        yield connection, total
    finally:
        connection.close()
        if os.path.exists(db_path):
            os.unlink(db_path)


@contextmanager
def _top_k_priority_index(partitions: Iterable[PartitionCommit | str], *,
                          weights: dict[str, float], normalize: str,
                          spill_dir: str | None):
    """Build an index while retaining one private snapshot of each partition."""
    with pinned_partition_files(partitions) as paths:
        with _top_k_priority_index_paths(
                paths, weights=weights, normalize=normalize,
                spill_dir=spill_dir) as indexed:
            yield indexed


def _iter_indexed_priorities(connection):
    """Materialize one partition at a time from a normalized-priority index."""
    cached_path, cached_batch = None, None
    query = ("SELECT data_file,row_idx,score FROM candidates "
             "ORDER BY source_id,start_idx,end_idx")
    for path, row_idx, priority in connection.execute(query):
        if path != cached_path:
            cached_path, cached_batch = path, SegmentBatch.read_ipc(path)
        assert cached_batch is not None
        segment = cached_batch.slice(row_idx, 1).to_segments()[0]
        yield segment, float(priority)


def iter_external_top_k_priorities(partitions: Iterable[PartitionCommit | str], *,
                                   weights: dict[str, float], normalize: str = "quantile",
                                   spill_dir: str | None = None):
    """Yield every segment and its exact normalized top-k priority in canonical order."""
    with _top_k_priority_index(
            partitions, weights=weights, normalize=normalize,
            spill_dir=spill_dir) as (connection, _total):
        yield from _iter_indexed_priorities(connection)


def iter_external_weighted_priorities(partitions: Iterable[PartitionCommit | str], *,
                                      weights: dict[str, float], normalize: str = "quantile",
                                      seed: int = 0, spill_dir: str | None = None):
    """Yield exact reference-compatible Gumbel weighted priorities canonically."""
    with _top_k_priority_index(
            partitions, weights=weights, normalize=normalize,
            spill_dir=spill_dir) as (connection, total):
        if not total:
            return
        minimum = float(connection.execute("SELECT MIN(score) FROM candidates").fetchone()[0])
        rng = np.random.default_rng(seed)
        for segment, score in _iter_indexed_priorities(connection):
            weight = score - minimum + 1e-9
            gumbel = -math.log(-math.log(float(rng.random())))
            yield segment, math.log(weight) + gumbel


def external_weighted(partitions: Iterable[PartitionCommit | str], *, weights: dict[str, float],
                      target_size: int, normalize: str = "quantile", seed: int = 0,
                      spill_dir: str | None = None) -> list:
    """Select the highest exact Gumbel priorities with a fixed-size heap.

    This matches the hours-path weighted priority contract. Count selection intentionally uses
    NumPy probability sampling and is not routed through this helper.
    """
    if not isinstance(target_size, int) or isinstance(target_size, bool) or target_size < 1:
        raise ValueError("target_size must be a positive integer")
    heap: list[tuple[float, int, Segment]] = []
    for ordinal, (segment, priority) in enumerate(iter_external_weighted_priorities(
            partitions, weights=weights, normalize=normalize, seed=seed,
            spill_dir=spill_dir)):
        ranked = (priority, -ordinal, segment)
        if len(heap) < target_size:
            heapq.heappush(heap, ranked)
        elif ranked[:2] > heap[0][:2]:
            heapq.heapreplace(heap, ranked)
    heap.sort(key=lambda item: item[:2], reverse=True)
    return [item[2] for item in heap]


def iter_external_span_items(prioritized_segments, source_tracks: dict, *, sample_hz: float,
                             span_len: int | None = None, clip_aggregate: str = "max",
                             group_of=None, spill_dir: str | None = None):
    """Aggregate a prioritized segment stream into bounded per-span fill items."""
    if clip_aggregate not in {"max", "min", "mean"}:
        raise ValueError("clip_aggregate must be max, min, or mean")
    connection, db_path = _database(spill_dir)
    try:
        connection.execute(
            "CREATE TABLE segments (source_id TEXT NOT NULL,start_idx INTEGER NOT NULL,"
            "end_idx INTEGER NOT NULL,priority REAL NOT NULL,tags BLOB NOT NULL,"
            "features BLOB NOT NULL)")
        pending = []
        for segment, priority in prioritized_segments:
            pending.append((str(segment.source_id), segment.start_idx, segment.end_idx,
                            float(priority), pickle.dumps(segment.tags, protocol=5),
                            pickle.dumps(segment.features, protocol=5)))
            if len(pending) >= 1024:
                connection.executemany("INSERT INTO segments VALUES (?,?,?,?,?,?)", pending)
                pending = []
        if pending:
            connection.executemany("INSERT INTO segments VALUES (?,?,?,?,?,?)", pending)
        connection.execute("CREATE INDEX overlap ON segments(source_id,start_idx,end_idx)")
        connection.commit()

        def reduce_values(values):
            if clip_aggregate == "max":
                return max(values)
            if clip_aggregate == "min":
                return min(values)
            return sum(values) / len(values)

        for source, tracks in source_tracks.items():
            group = group_of(source) if group_of else None
            for entity_id, length in tracks:
                bounds = ((start, min(start + span_len, length))
                          for start in range(0, length, span_len)) if span_len else ((0, length),)
                for span_index, (start, end) in enumerate(bounds):
                    priorities: list[float] = []
                    tags: set = set()
                    feature_values: dict[str, list[float]] = {}
                    query = ("SELECT priority,tags,features FROM segments WHERE source_id=? "
                             "AND start_idx<? AND end_idx>? ORDER BY start_idx,end_idx")
                    for priority, tags_blob, features_blob in connection.execute(
                            query, (str(source), end, start)):
                        priorities.append(float(priority))
                        tags.update(pickle.loads(tags_blob))
                        for name, value in pickle.loads(features_blob).items():
                            feature_values.setdefault(name, []).append(float(value))
                    meta = {name: reduce_values(values)
                            for name, values in feature_values.items()}
                    yield {"id": (source, entity_id, span_index), "source_id": source,
                           "priority": reduce_values(priorities) if priorities else float("-inf"),
                           "seconds": (end - start) / sample_hz, "group": group,
                           "tags": frozenset(tags), "start": start, "end": end, "meta": meta}
    finally:
        connection.close()
        if os.path.exists(db_path):
            os.unlink(db_path)


def _iter_external_hash_dedup_paths(paths: Iterable[str], *,
                                    feature_names: Iterable[str] | None = None,
                                    per_group: bool = False,
                                    spill_dir: str | None = None):
    connection, db_path = _database(spill_dir)
    try:
        connection.execute(
            "CREATE TABLE seen (scope BLOB NOT NULL,digest BLOB NOT NULL,payload BLOB NOT NULL,"
            "PRIMARY KEY(scope,digest,payload))")
        infer_names = feature_names is None
        resolved_names = tuple(feature_names) if feature_names is not None else None
        inferred_schema: set[str] | None = None
        for path in paths:
            batch = SegmentBatch.read_ipc(path)
            current_names = tuple(name.split(":", 1)[1] for name in batch.column_names
                                  if name.startswith("feature:"))
            if infer_names and resolved_names is None:
                resolved_names = current_names
                inferred_schema = set(current_names)
            elif infer_names and set(current_names) != inferred_schema:
                raise ValueError(
                    "hash dedup feature schema differs across partitions: "
                    f"expected {sorted(inferred_schema or ())}, got {sorted(current_names)} "
                    f"in {path!r}")
            assert resolved_names is not None
            unknown = [name for name in resolved_names
                       if f"feature:{name}" not in batch.column_names]
            if unknown:
                raise ValueError(f"hash dedup feature(s) missing from partition: {unknown}")
            for segment in batch.to_segments():
                from .strategy import _feature_value
                values = [float(_feature_value(segment, name)) for name in resolved_names]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("hash dedup features must be finite")
                payload = b"".join(struct.pack(">d", value) for value in values)
                digest = hashlib.sha256(payload).digest()
                scope = pickle.dumps(segment.group, protocol=5) if per_group else b""
                inserted = connection.execute(
                    "INSERT OR IGNORE INTO seen VALUES (?,?,?)", (scope, digest, payload))
                if inserted.rowcount:
                    yield segment
        connection.commit()
    finally:
        connection.close()
        if os.path.exists(db_path):
            os.unlink(db_path)


def iter_external_hash_dedup(partitions: Iterable[PartitionCommit | str], *,
                             feature_names: Iterable[str] | None = None,
                             per_group: bool = False, spill_dir: str | None = None):
    """Yield canonical exact-vector first occurrences from pinned partitions.

    SHA-256 narrows lookup, while the complete encoded vector remains in the primary key for
    collision resolution. This is exact identity deduplication, not cosine/near deduplication.
    """
    with pinned_partition_files(partitions) as paths:
        yield from _iter_external_hash_dedup_paths(
            paths, feature_names=feature_names, per_group=per_group,
            spill_dir=spill_dir)
