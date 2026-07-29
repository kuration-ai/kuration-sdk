"""Hours-as-budget fill over output items (the piece deferred from Phase C).

Now that the output unit exists (a track), curation can *drive* the emitted volume instead of
selecting a fixed count: rank tracks by the priority of their source's segments and fill the
train/test hour budgets in descending order. Generalizes the reference `fill_to_hours`:
items instead of clips, source-level (not match-level) leakage-free split, schema `sample_hz`
instead of a fixed 64 Hz, `group` instead of `map`. Budget honored to within one item.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Optional

from ..select import _stable_value_order, assign_splits


def fill_to_hours(items: list[dict], *, train_hours: float, test_hours: float,
                  max_group_overshoot: Optional[float] = None, tag_coverage: dict = None,
                  seed: int = 0, return_stats: bool = False):
    """Admit items (each {id, source_id, priority, seconds, group, tags?}) to the train/test
    hour budgets in descending priority. Sources are pre-partitioned into split pools (seeded,
    stratified by group) so the two budgets are decoupled — the fix that stops a spread-out
    priority from committing every source to train and starving test. With `tag_coverage`
    ({"min_tag_frac": f}), a reserve pass first admits the highest-priority item carrying each
    tag (bounded by f x total budget) so a rare label isn't dropped by the priority cut.
    Returns `(admitted_ids, source->split)`. With `return_stats=True`, appends a third budget
    accounting dict without changing the long-standing default return shape."""
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
        frac = tag_coverage.get("min_tag_frac", 0.005)
        if (not isinstance(frac, (int, float)) or isinstance(frac, bool)
                or not math.isfinite(frac) or not 0 <= frac <= 1):
            raise ValueError("tag_coverage.min_tag_frac must be finite and in [0, 1]")
    required = {"id", "source_id", "priority", "seconds", "group"}
    normalized = []
    seen_ids = set()
    for i, it in enumerate(items):
        if not isinstance(it, dict) or not required <= set(it):
            missing = sorted(required - set(it)) if isinstance(it, dict) else sorted(required)
            raise ValueError(f"item {i} is missing required key(s) {missing}")
        seconds = it["seconds"]
        if (not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
                or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError(f"item {i} seconds must be a finite positive number, "
                             f"got {seconds!r}")
        try:
            priority = float(it["priority"])
            hash(it["id"])
        except (TypeError, ValueError) as e:
            raise ValueError(f"item {i} needs a numeric priority and hashable id") from e
        if it["id"] in seen_ids:
            raise ValueError(f"duplicate fill item id {it['id']!r}")
        seen_ids.add(it["id"])
        item = dict(it)
        item["priority"] = priority
        normalized.append(item)
    items = normalized
    train_t, test_t = train_hours * 3600.0, test_hours * 3600.0
    total = train_t + test_t
    source_groups = {it["source_id"]: it["group"] for it in items}
    test_frac = test_t / total if total > 0 else 0.0
    split = assign_splits(source_groups, test_frac, seed)

    n_groups = max(len({it["group"] for it in items}), 1)
    max_per_group = (max_group_overshoot * total / n_groups
                     if max_group_overshoot is not None and total > 0 else float("inf"))

    secs = {"train": 0.0, "test": 0.0}
    target = {"train": train_t, "test": test_t}
    group_secs: Counter = Counter()
    admitted: set = set()
    admission_reasons: dict = {}

    def _try_admit(it, reason: str = "priority") -> bool:
        if not math.isfinite(it["priority"]):
            return False   # -inf marks "never admit" (failed filter / no overlapping segments)
        if it["id"] in admitted:
            return False
        sp = split.get(it["source_id"])
        if sp is None or secs[sp] >= target[sp]:
            return False
        g = it["group"]
        if group_secs[g] + it["seconds"] > max_per_group:
            return False
        secs[sp] += it["seconds"]
        group_secs[g] += it["seconds"]
        admitted.add(it["id"])
        admission_reasons[it["id"]] = reason
        return True

    by_prio = sorted(items, key=lambda x: (-x["priority"], x["source_id"], str(x["id"])))

    # tag-coverage reserve: each tag's highest-priority item, bounded by min_tag_frac of budget.
    if tag_coverage and total > 0:
        reserve_cap = tag_coverage.get("min_tag_frac", 0.005) * total
        reserved, covered = 0.0, set()
        for tag in _stable_value_order(
                t for it in items for t in it.get("tags", ())):
            if tag in covered or reserved >= reserve_cap:
                continue
            for it in by_prio:
                if tag in it.get("tags", ()) and _try_admit(it, "tag-reserve"):
                    reserved += it["seconds"]
                    covered |= set(it.get("tags", ()))
                    break

    for it in by_prio:
        if secs["train"] >= train_t and secs["test"] >= test_t:
            break
        _try_admit(it)

    admitted_seconds = {sp: sum(it["seconds"] for it in items
                                if it["id"] in admitted and split.get(it["source_id"]) == sp)
                        for sp in ("train", "test")}
    candidate_seconds = {sp: sum(it["seconds"] for it in items
                                 if math.isfinite(it["priority"])
                                 and split.get(it["source_id"]) == sp)
                         for sp in ("train", "test")}
    rejected_group_cap = 0
    if math.isfinite(max_per_group):
        for it in items:
            g = it["group"]
            if (it["id"] not in admitted and math.isfinite(it["priority"])
                    and group_secs[g] + it["seconds"] > max_per_group):
                rejected_group_cap += 1
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

    for item in by_prio:
        priority = float(item["priority"])
        item_split = split[item["source_id"]]
        if item["id"] in admitted:
            reason, bucket = admission_reasons.get(item["id"], "priority"), "admitted"
        elif not math.isfinite(priority):
            reason, bucket = "nonfinite-priority", "rejected"
        else:
            group = item["group"]
            if math.isfinite(max_per_group) and group_secs[group] + item["seconds"] > max_per_group:
                reason = "group-cap"
            elif secs[item_split] >= target[item_split]:
                reason = "budget-filled"
            else:
                reason = "lower-priority"
            bucket = "rejected"
        decision_counts[reason] += 1
        if math.isfinite(priority):
            summary = priority_stats[bucket]
            summary["count"] += 1
            summary["sum"] += priority
            summary["min"] = priority if summary["min"] is None else min(summary["min"], priority)
            summary["max"] = priority if summary["max"] is None else max(summary["max"], priority)
        if len(decision_examples[bucket]) < 20:
            from ..provenance import publishable_source_id
            published_source = publishable_source_id(item["source_id"])

            def redact_source(value):
                if value == item["source_id"]:
                    return published_source
                if isinstance(value, list):
                    return [redact_source(child) for child in value]
                if isinstance(value, dict):
                    return {key: redact_source(child) for key, child in value.items()}
                return value

            decision_examples[bucket].append({
                "id": redact_source(jsonable(item["id"])), "source": published_source,
                "split": item_split, "priority": priority if math.isfinite(priority) else None,
                "seconds": item["seconds"], "group": redact_source(jsonable(item["group"])),
                "tags": redact_source(jsonable(item.get("tags", ()))),
                "start": item.get("start"),
                "end": item.get("end"), "decision": reason,
            })
    for summary in priority_stats.values():
        count = summary.pop("count")
        total_priority = summary.pop("sum")
        summary["count"] = count
        summary["mean"] = total_priority / count if count else None
    stats = {
        "requested_seconds": {"train": train_t, "test": test_t},
        "admitted_seconds": admitted_seconds,
        "underfill_seconds": {sp: max(0.0, target[sp] - admitted_seconds[sp])
                              for sp in ("train", "test")},
        "candidate_seconds": candidate_seconds,
        "items_total": len(items),
        "items_admitted": len(admitted),
        "items_rejected_nonfinite": sum(not math.isfinite(it["priority"]) for it in items),
        "items_rejected_group_cap": rejected_group_cap,
        "review": {"decision_counts": dict(sorted(decision_counts.items())),
                   "priority": priority_stats, "examples": decision_examples},
    }
    return (admitted, split, stats) if return_stats else (admitted, split)
