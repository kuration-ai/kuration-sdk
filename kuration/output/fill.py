"""Hours-as-budget fill over output items (the piece deferred from Phase C).

Now that the output unit exists (a track), curation can *drive* the emitted volume instead of
selecting a fixed count: rank tracks by the priority of their source's segments and fill the
train/test hour budgets in descending order. Generalizes the reference `fill_to_hours`:
items instead of clips, source-level (not match-level) leakage-free split, schema `sample_hz`
instead of a fixed 64 Hz, `group` instead of `map`. Budget honored to within one item.
"""
from __future__ import annotations

from collections import Counter

from ..select import assign_splits


def fill_to_hours(items: list[dict], *, train_hours: float, test_hours: float,
                  max_group_overshoot: float = 1.5, tag_coverage: dict = None,
                  seed: int = 0) -> tuple[set, dict]:
    """Admit items (each {id, source_id, priority, seconds, group, tags?}) to the train/test
    hour budgets in descending priority. Sources are pre-partitioned into split pools (seeded,
    stratified by group) so the two budgets are decoupled — the fix that stops a spread-out
    priority from committing every source to train and starving test. With `tag_coverage`
    ({"min_tag_frac": f}), a reserve pass first admits the highest-priority item carrying each
    tag (bounded by f x total budget) so a rare label isn't dropped by the priority cut.
    Returns (admitted_ids, source->split)."""
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

    def _try_admit(it) -> bool:
        if it["id"] in admitted:
            return False
        sp = split.get(it["source_id"])
        if sp is None or secs[sp] >= target[sp]:
            return False
        g = it["group"] if it["group"] is not None else "\x00"
        if group_secs[g] + it["seconds"] > max_per_group:
            return False
        secs[sp] += it["seconds"]
        group_secs[g] += it["seconds"]
        admitted.add(it["id"])
        return True

    by_prio = sorted(items, key=lambda x: (-x["priority"], x["source_id"], str(x["id"])))

    # tag-coverage reserve: each tag's highest-priority item, bounded by min_tag_frac of budget.
    if tag_coverage and total > 0:
        reserve_cap = tag_coverage.get("min_tag_frac", 0.005) * total
        reserved, covered = 0.0, set()
        for tag in sorted({t for it in items for t in it.get("tags", ())}):
            if tag in covered or reserved >= reserve_cap:
                continue
            for it in by_prio:
                if tag in it.get("tags", ()) and _try_admit(it):
                    reserved += it["seconds"]
                    covered |= set(it.get("tags", ()))
                    break

    for it in by_prio:
        if secs["train"] >= train_t and secs["test"] >= test_t:
            break
        _try_admit(it)
    return admitted, split
