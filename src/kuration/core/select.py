"""Selection primitives: per-bin sampling + the shared tag-coverage / group-cap post-passes.

Generalized from the CS2-specific original: "object coverage" -> tag coverage (over a
segment's categorical `tags`), "map cap" -> group cap (over a segment's `group` key). The
chaining order is unchanged and deliberate: a diversity baseline gives the bulk of the set,
tag coverage patches the long tail (a rare label the base pass would miss), and the group
cap is a last guard so one group can't dominate.

Source-level train/test splitting lives here too (`assign_splits`): leakage-free by
construction (a whole source goes to one split), seeded + sorted for determinism.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Iterable

from .segment import Segment


def _stable_value_order(values: Iterable) -> list:
    """Return Python-equality-unique values in a deterministic mixed-type order.

    A plain ``sorted(set(values))`` cannot compare values such as ``1`` and ``"1"``.
    Stringifying first is also incorrect because Python considers ``1 == True`` while
    distinguishing ``1`` from ``"1"``. Dict insertion performs the intended equality
    comparisons; the type-qualified display key is used only to order representatives.
    """
    unique = dict.fromkeys(values)
    return sorted(
        unique,
        key=lambda value: (
            f"{type(value).__module__}.{type(value).__qualname__}",
            repr(value),
        ),
    )


def group_by_bin(segments: list[Segment]) -> dict[tuple, list[Segment]]:
    out: dict[tuple, list[Segment]] = defaultdict(list)
    for s in segments:
        out[s.bin].append(s)
    return out


def sample_uniform_bins(segments: list[Segment], *, target_size: int,
                        rng: random.Random) -> list[Segment]:
    """Pick min(target_size, n) segments spread as evenly as possible across occupied bins:
    one from every bin, then a second, ... — never over budget (a naive per-bin quota of
    max(1, target//n_bins) selects one PER BIN, overshooting whenever bins outnumber the
    target), and uneven bins spill their shortfall into fuller ones."""
    bins = group_by_bin(segments)
    if not bins:
        return []
    k = min(target_size, len(segments))
    pools = []
    for members in bins.values():
        p = members[:]
        rng.shuffle(p)
        pools.append(p)
    selected: list[Segment] = []
    rank = 0
    while len(selected) < k:
        for p in pools:
            if rank < len(p):
                selected.append(p[rank])
                if len(selected) >= k:
                    break
        rank += 1
    return selected


def enforce_tag_coverage(selected: list[Segment], all_segments: list[Segment], *,
                         tag_vocab: Iterable, target_size: int,
                         min_tag_frac: float) -> list[Segment]:
    """Meet tag floors by adding or swapping segments without exceeding the count budget."""
    tag_order = tuple(tag_vocab)
    selected = list(selected)
    if min_tag_frac == 0:
        return selected
    limit = max(target_size, len(selected))  # ``filter`` intentionally selects every match.
    selected_ids = set(id(s) for s in selected)
    counts: Counter = Counter()
    for s in selected:
        for t in s.tags:
            counts[t] += 1
    # ``min_tag_frac`` is a lower bound: a fractional required count must round upward.
    # Python's round() would under-deliver at .5 boundaries (for example 5 * .5 -> 2).
    floor = max(1, math.ceil(target_size * min_tag_frac))
    required_tags = set(tag_order)
    protected = {tag for tag in required_tags if counts[tag] == floor}
    unselected = [s for s in all_segments if id(s) not in selected_ids]
    bin_counts: Counter = Counter(s.bin for s in selected)
    for tag in tag_order:
        deficit = floor - counts[tag]
        if deficit <= 0:
            continue
        cands = [s for s in unselected if tag in s.tags]
        cands.sort(key=lambda s: bin_counts[s.bin])
        for s in cands:
            if counts[tag] >= floor:
                break
            if len(selected) < limit:
                selected.append(s)
                affected = set(s.tags)
            else:
                # Scan once instead of sorting the full selection for every candidate; max()
                # preserves the densest-bin/reverse-index deterministic preference.
                eligible = ((index, victim) for index, victim in enumerate(selected)
                            if tag not in victim.tags
                            and not (protected & (victim.tags - s.tags)))
                victim_pair = max(
                    eligible, key=lambda pair: (bin_counts[pair[1].bin], pair[0]), default=None)
                if victim_pair is None:
                    continue
                victim_index, victim = victim_pair
                selected_ids.discard(id(victim))
                bin_counts[victim.bin] -= 1
                for covered in victim.tags:
                    counts[covered] -= 1
                selected[victim_index] = s
                unselected.append(victim)
                affected = set(victim.tags) | set(s.tags)
            selected_ids.add(id(s))
            bin_counts[s.bin] += 1
            for covered in s.tags:
                counts[covered] += 1
            # Only tags exactly at their floor can be harmed by the next swap. Updating the
            # touched tags keeps victim checks independent of total vocabulary size.
            for covered in affected & required_tags:
                if counts[covered] == floor:
                    protected.add(covered)
                else:
                    protected.discard(covered)
        unselected = [s for s in unselected if id(s) not in selected_ids]
    return selected


def enforce_group_cap(selected: list[Segment], *, max_group_overshoot: float) -> list[Segment]:
    """Drop excess from any group exceeding `max_group_overshoot` x its fair share,
    cutting its most over-represented bins first. Segments with group=None are pooled
    under a single key, so the cap is a no-op when no groups are set."""
    gkey = lambda s: s.group
    group_counts: Counter = Counter(gkey(s) for s in selected)
    if len(group_counts) <= 1:
        return selected
    bin_counts: Counter = Counter(s.bin for s in selected)
    fair_share = len(selected) / len(group_counts)
    # This is an inclusive population ceiling. Flooring a fractional fair share makes
    # overshoot=1.0 stricter than the fair share itself (for example, 2.5 becomes 2).
    max_per_group = math.ceil(fair_share * max_group_overshoot)
    for g, cnt in list(group_counts.items()):
        if cnt <= max_per_group:
            continue
        excess = cnt - max_per_group
        segs = [s for s in selected if gkey(s) == g]
        segs.sort(key=lambda s: bin_counts[s.bin], reverse=True)
        drop = set(id(s) for s in segs[:excess])
        selected = [s for s in selected if id(s) not in drop]
        for s in segs[:excess]:
            bin_counts[s.bin] -= 1
    return selected


def assign_splits(source_groups: dict, test_frac: float, seed: int) -> dict:
    """Partition sources into 'train'/'test', stratified by `group` at `test_frac`.

    `source_groups`: {source_id -> group_key}. Seeded + sorted so the split is identical
    regardless of the order sources arrive in. Leakage-free: a whole source -> one split."""
    if not isinstance(test_frac, (int, float)) or not 0 <= test_frac <= 1:
        raise ValueError(f"test_frac must be in [0, 1], got {test_frac!r}")
    rng = random.Random(seed)
    by_group: dict = defaultdict(list)
    for sid, g in sorted(source_groups.items()):
        by_group[g].append(sid)
    groups = sorted(by_group, key=lambda x: (x is None, str(type(x)), str(x)))
    # Allocate the global test count across strata by largest remainder. Independently
    # rounding every small group would assign singleton strata entirely to train.
    total_test = int(round(test_frac * len(source_groups)))
    exact = {g: test_frac * len(by_group[g]) for g in groups}
    quota = {g: int(exact[g]) for g in groups}
    remaining = total_test - sum(quota.values())
    order = sorted(range(len(groups)), key=lambda i: (-(exact[groups[i]] - quota[groups[i]]), i))
    for i in order:
        if remaining <= 0:
            break
        g = groups[i]
        if quota[g] < len(by_group[g]):
            quota[g] += 1
            remaining -= 1

    split: dict = {}
    for g in groups:
        ids = by_group[g][:]
        rng.shuffle(ids)
        n_test = quota[g]
        for i, sid in enumerate(ids):
            split[sid] = "test" if i < n_test else "train"
    return split
