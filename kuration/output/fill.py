"""Hours-as-budget fill over output items (the piece deferred from Phase C).

Now that the output unit exists (a track), curation can *drive* the emitted volume instead of
selecting a fixed count: rank tracks by the priority of their source's segments and fill the
train/test hour budgets in descending order. Generalizes the reference `fill_to_hours`:
items instead of clips, source-level (not match-level) leakage-free split, schema `sample_hz`
instead of a fixed 64 Hz, `group` instead of `map`. Budget honored to within one item.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from ..select import assign_splits


def aggregate_source_priorities(all_segments, priorities, clip_aggregate: str = "max") -> dict:
    """One priority per source = aggregate of its segments' priorities (whole-track spans
    overlap all of a source's segments). `clip_aggregate`: max (inclusion) / mean / min
    (avoidance) — same rationale as the reference fill."""
    reduce = {"max": max, "min": min, "mean": lambda x: sum(x) / len(x)}.get(clip_aggregate)
    if reduce is None:
        raise ValueError(f"unknown clip_aggregate {clip_aggregate!r}; expected max/mean/min")
    by_src: dict = defaultdict(list)
    for i, s in enumerate(all_segments):
        by_src[s.source_id].append(float(priorities[i]))
    return {src: reduce(ps) for src, ps in by_src.items()}


def fill_to_hours(items: list[dict], *, train_hours: float, test_hours: float,
                  max_group_overshoot: float = 1.5, seed: int = 0) -> tuple[set, dict]:
    """Admit items (each {id, source_id, priority, seconds, group}) to the train/test hour
    budgets in descending priority. Sources are pre-partitioned into split pools (seeded,
    stratified by group) so the two budgets are decoupled — the fix that stops a spread-out
    priority from committing every source to train and starving test. Returns
    (admitted_ids, source->split)."""
    train_t, test_t = train_hours * 3600.0, test_hours * 3600.0
    total = train_t + test_t
    source_groups = {it["source_id"]: it["group"] for it in items}
    test_frac = test_t / total if total > 0 else 0.0
    split = assign_splits(source_groups, test_frac, seed)

    n_groups = max(len({it["group"] for it in items}), 1)
    max_per_group = (max_group_overshoot * total / n_groups) if total > 0 else float("inf")

    secs = {"train": 0.0, "test": 0.0}
    target = {"train": train_t, "test": test_t}
    group_secs: Counter = Counter()
    admitted: set = set()

    for it in sorted(items, key=lambda x: (-x["priority"], x["source_id"], str(x["id"]))):
        if secs["train"] >= train_t and secs["test"] >= test_t:
            break
        sp = split.get(it["source_id"])
        if sp is None or secs[sp] >= target[sp]:
            continue
        g = it["group"] if it["group"] is not None else "\x00"
        if group_secs[g] + it["seconds"] > max_per_group:
            continue
        secs[sp] += it["seconds"]
        group_secs[g] += it["seconds"]
        admitted.add(it["id"])
    return admitted, split
