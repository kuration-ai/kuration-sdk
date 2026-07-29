"""Disk-backed hours-budget item fill with reference-compatible semantics."""
from __future__ import annotations

from collections import Counter
import math
import os
import pickle
import sqlite3
import tempfile
from typing import Any

from ..select import _stable_value_order, assign_splits


def _dump(value) -> bytes:
    return pickle.dumps(value, protocol=5)


def _load(value: bytes):
    return pickle.loads(value)


def external_fill_to_hours(items, *, train_hours: float, test_hours: float,
                           max_group_overshoot=None, tag_coverage=None, seed: int = 0,
                           spill_dir: str | None = None):
    """Fill hours from an iterable while storing candidate metadata on disk.

    Items accept the ordinary fill fields plus optional ``start``, ``end``, and ``meta``.
    Returns admitted ids, source splits, budget accounting, and metadata only for admitted ids.
    """
    for name, value in (("train_hours", train_hours), ("test_hours", test_hours)):
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    if max_group_overshoot is not None and (
            not isinstance(max_group_overshoot, (int, float))
            or isinstance(max_group_overshoot, bool)
            or not math.isfinite(max_group_overshoot) or max_group_overshoot <= 0):
        raise ValueError("max_group_overshoot must be a finite positive number or None")
    if tag_coverage is not None:
        fraction = tag_coverage.get("min_tag_frac", 0.005)
        if (not isinstance(fraction, (int, float)) or isinstance(fraction, bool)
                or not math.isfinite(fraction) or not 0 <= fraction <= 1):
            raise ValueError("tag_coverage.min_tag_frac must be finite and in [0, 1]")
    if spill_dir is not None:
        os.makedirs(spill_dir, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(suffix=".sqlite3", dir=spill_dir, delete=False)
    handle.close()
    connection = sqlite3.connect(handle.name)
    try:
        connection.executescript(
            "PRAGMA journal_mode=OFF;PRAGMA synchronous=OFF;PRAGMA temp_store=FILE;"
            "CREATE TABLE items (id_blob BLOB PRIMARY KEY,id_hash TEXT NOT NULL,"
            "id_sort TEXT NOT NULL,"
            "source_id TEXT NOT NULL,priority REAL NOT NULL,seconds REAL NOT NULL,"
            "group_key INTEGER NOT NULL,group_blob BLOB NOT NULL,tags_blob BLOB NOT NULL,"
            "start_idx INTEGER,end_idx INTEGER,meta_blob BLOB NOT NULL,split TEXT);"
            "CREATE TABLE groups (value_key INTEGER PRIMARY KEY,value_hash TEXT NOT NULL,"
            "value_blob BLOB NOT NULL);"
            "CREATE TABLE tags (value_key INTEGER PRIMARY KEY,value_hash TEXT NOT NULL,"
            "value_blob BLOB NOT NULL);"
            "CREATE TABLE item_tags (id_blob BLOB NOT NULL,tag_key INTEGER NOT NULL,"
            "PRIMARY KEY(id_blob,tag_key));"
            "CREATE INDEX group_equality ON groups(value_hash);"
            "CREATE INDEX tag_equality ON tags(value_hash);"
            "CREATE INDEX item_equality ON items(id_hash);")

        def equality_key(table: str, value, label: str) -> int:
            try:
                value_hash = str(hash(value))
            except TypeError as error:
                raise ValueError(f"{label} values must be hashable, got {value!r}") from error
            for key, blob in connection.execute(
                    f"SELECT value_key,value_blob FROM {table} WHERE value_hash=?",
                    (value_hash,)):
                if _load(blob) == value:
                    return int(key)
            cursor = connection.execute(
                f"INSERT INTO {table} (value_hash,value_blob) VALUES (?,?)",
                (value_hash, _dump(value)),
            )
            if cursor.lastrowid is None:
                raise RuntimeError(f"failed to register {label} equality value")
            return int(cursor.lastrowid)

        required = {"id", "source_id", "priority", "seconds", "group"}
        count = 0
        for count, item in enumerate(items, 1):
            if not isinstance(item, dict) or not required <= set(item):
                missing = sorted(required - set(item)) if isinstance(item, dict) else sorted(required)
                raise ValueError(f"item {count - 1} is missing required key(s) {missing}")
            seconds = item["seconds"]
            if (not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
                    or not math.isfinite(seconds) or seconds <= 0):
                raise ValueError(f"item {count - 1} seconds must be a finite positive number, "
                                 f"got {seconds!r}")
            try:
                priority = float(item["priority"])
                hash(item["id"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"item {count - 1} needs a numeric priority and hashable id") \
                    from error
            identity = _dump(item["id"])
            identity_hash = str(hash(item["id"]))
            # Pickle bytes are serialization identity, not Python equality: ``1`` and ``True``
            # have different pickles but are duplicate set/dict keys. Compare only the tiny
            # same-hash bucket so duplicate detection remains disk-backed.
            for (existing_blob,) in connection.execute(
                    "SELECT id_blob FROM items WHERE id_hash=?", (identity_hash,)):
                if _load(existing_blob) == item["id"]:
                    raise ValueError(f"duplicate fill item id {item['id']!r}")
            group_key = equality_key("groups", item["group"], "group")
            tags = tuple(_stable_value_order(item.get("tags", ())))
            tag_keys = [(equality_key("tags", tag, "tag"), tag) for tag in tags]
            try:
                connection.execute(
                    "INSERT INTO items VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL)",
                    (identity, identity_hash, str(item["id"]), str(item["source_id"]), priority,
                     float(seconds), group_key, _dump(item["group"]), _dump(tags),
                     item.get("start"), item.get("end"),
                     _dump(dict(item.get("meta") or {}))))
            except sqlite3.IntegrityError as error:
                raise ValueError(f"duplicate fill item id {item['id']!r}") from error
            connection.executemany(
                "INSERT INTO item_tags VALUES (?,?)",
                ((identity, tag_key) for tag_key, _tag in tag_keys),
            )
        connection.execute("CREATE INDEX item_priority ON items(priority DESC,source_id,id_sort)")
        connection.execute("CREATE INDEX tag_lookup ON item_tags(tag_key,id_blob)")
        connection.commit()

        train_target, test_target = train_hours * 3600.0, test_hours * 3600.0
        total = train_target + test_target
        source_groups = {}
        for source_id, group_blob in connection.execute(
                "SELECT source_id,group_blob FROM items GROUP BY source_id"):
            source_groups[source_id] = _load(group_blob)
        split = assign_splits(source_groups, test_target / total if total > 0 else 0.0, seed)
        connection.executemany("UPDATE items SET split=? WHERE source_id=?",
                               ((value, source) for source, value in split.items()))
        connection.commit()

        n_groups = max(connection.execute("SELECT COUNT(*) FROM groups").fetchone()[0], 1)
        max_per_group = (max_group_overshoot * total / n_groups
                         if max_group_overshoot is not None and total > 0 else float("inf"))
        seconds = {"train": 0.0, "test": 0.0}
        target = {"train": train_target, "test": test_target}
        group_seconds: Counter = Counter()
        admitted: set = set()
        admission_reasons: dict[bytes, str] = {}

        def decode(row):
            identity, source_id, priority, duration, group_key, tags_blob = row
            return identity, source_id, priority, duration, group_key, _load(tags_blob)

        def try_admit(row, reason: str = "priority") -> bool:
            identity, source_id, priority, duration, group_key, _tags = decode(row)
            if not math.isfinite(priority) or identity in admitted:
                return False
            item_split = split.get(source_id)
            if item_split is None or seconds[item_split] >= target[item_split]:
                return False
            if group_seconds[group_key] + duration > max_per_group:
                return False
            seconds[item_split] += duration
            group_seconds[group_key] += duration
            admitted.add(identity)
            admission_reasons[identity] = reason
            return True

        select_rows = ("SELECT id_blob,source_id,priority,seconds,group_key,tags_blob "
                       "FROM items ORDER BY priority DESC,source_id,id_sort")
        if tag_coverage and total > 0:
            reserve_cap = tag_coverage.get("min_tag_frac", 0.005) * total
            reserved = 0.0
            covered: set = set()
            all_tags = [
                (int(tag_key), _load(tag_blob))
                for tag_key, tag_blob in connection.execute(
                    "SELECT value_key,value_blob FROM tags"
                )
            ]
            all_tags.sort(key=lambda pair: (
                f"{type(pair[1]).__module__}.{type(pair[1]).__qualname__}",
                repr(pair[1]),
            ))
            for tag_key, tag in all_tags:
                if tag in covered or reserved >= reserve_cap:
                    continue
                query = ("SELECT i.id_blob,i.source_id,i.priority,i.seconds,i.group_key,i.tags_blob "
                         "FROM items i JOIN item_tags t ON t.id_blob=i.id_blob WHERE t.tag_key=? "
                         "ORDER BY i.priority DESC,i.source_id,i.id_sort")
                for row in connection.execute(query, (tag_key,)):
                    if try_admit(row, "tag-reserve"):
                        reserved += row[3]
                        covered.update(_load(row[5]))
                        break
        for row in connection.execute(select_rows):
            if seconds["train"] >= train_target and seconds["test"] >= test_target:
                break
            try_admit(row)

        admitted_details = {}
        admitted_seconds = {"train": 0.0, "test": 0.0}
        candidate_seconds = {"train": 0.0, "test": 0.0}
        nonfinite = 0
        rejected_group = 0
        decision_counts: Counter = Counter()
        decision_examples: dict[str, list[dict]] = {"admitted": [], "rejected": []}
        priority_stats: dict[str, dict[str, Any]] = {
            "admitted": {"count": 0, "min": None, "max": None, "sum": 0.0},
            "rejected": {"count": 0, "min": None, "max": None, "sum": 0.0},
        }

        def jsonable(value):
            if isinstance(value, tuple):
                return [jsonable(item) for item in value]
            if isinstance(value, dict):
                return {str(key): jsonable(item) for key, item in value.items()}
            if isinstance(value, (set, frozenset)):
                return [jsonable(item) for item in sorted(value, key=str)]
            if isinstance(value, list):
                return [jsonable(item) for item in value]
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            return str(value)

        def add_priority(bucket: str, priority: float) -> None:
            if not math.isfinite(priority):
                return
            stats = priority_stats[bucket]
            stats["count"] += 1
            stats["sum"] += priority
            stats["min"] = priority if stats["min"] is None else min(stats["min"], priority)
            stats["max"] = priority if stats["max"] is None else max(stats["max"], priority)

        for row in connection.execute(
                "SELECT id_blob,source_id,priority,seconds,group_key,group_blob,tags_blob,"
                "start_idx,end_idx,meta_blob FROM items "
                "ORDER BY priority DESC,source_id,id_sort"):
            (identity, source_id, priority, duration, group_key, group_blob, tags_blob,
             start, end, meta_blob) = row
            item_split = split[source_id]
            if math.isfinite(priority):
                candidate_seconds[item_split] += duration
            else:
                nonfinite += 1
            if identity in admitted:
                admitted_seconds[item_split] += duration
                admitted_details[_load(identity)] = {"start": start, "end": end,
                                                      "meta": _load(meta_blob)}
                reason = admission_reasons.get(identity, "priority")
                bucket = "admitted"
            elif math.isfinite(max_per_group) and math.isfinite(priority):
                if group_seconds[group_key] + duration > max_per_group:
                    rejected_group += 1
                    reason = "group-cap"
                elif seconds[item_split] >= target[item_split]:
                    reason = "budget-filled"
                else:
                    reason = "lower-priority"
                bucket = "rejected"
            elif not math.isfinite(priority):
                reason = "nonfinite-priority"
                bucket = "rejected"
            elif seconds[item_split] >= target[item_split]:
                reason = "budget-filled"
                bucket = "rejected"
            else:
                reason = "lower-priority"
                bucket = "rejected"
            decision_counts[reason] += 1
            add_priority(bucket, priority)
            if len(decision_examples[bucket]) < 20:
                from ..provenance import publishable_source_id
                published_source = publishable_source_id(source_id)

                def redact_source(value):
                    if value == source_id:
                        return published_source
                    if isinstance(value, list):
                        return [redact_source(child) for child in value]
                    if isinstance(value, dict):
                        return {key: redact_source(child) for key, child in value.items()}
                    return value

                decision_examples[bucket].append({
                    "id": redact_source(jsonable(_load(identity))),
                    "source": published_source,
                    "split": item_split, "priority": priority if math.isfinite(priority) else None,
                    "seconds": duration,
                    "group": redact_source(jsonable(_load(group_blob))),
                    "tags": redact_source(jsonable(_load(tags_blob))),
                    "start": start, "end": end,
                    "decision": reason,
                })
        admitted_ids = set(admitted_details)
        for priority_summary in priority_stats.values():
            priority_count = priority_summary.pop("count")
            total_priority = priority_summary.pop("sum")
            priority_summary["count"] = priority_count
            priority_summary["mean"] = (total_priority / priority_count
                                        if priority_count else None)
        stats = {
            "requested_seconds": {"train": train_target, "test": test_target},
            "admitted_seconds": admitted_seconds,
            "underfill_seconds": {name: max(0.0, target[name] - admitted_seconds[name])
                                  for name in ("train", "test")},
            "candidate_seconds": candidate_seconds, "items_total": count,
            "items_admitted": len(admitted_ids), "items_rejected_nonfinite": nonfinite,
            "items_rejected_group_cap": rejected_group,
            "review": {"decision_counts": dict(sorted(decision_counts.items())),
                       "priority": priority_stats, "examples": decision_examples}}
        return admitted_ids, split, stats, admitted_details
    finally:
        connection.close()
        if os.path.exists(handle.name):
            os.unlink(handle.name)
