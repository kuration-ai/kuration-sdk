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

import random
from collections import Counter, defaultdict
from typing import Iterable, Optional

from .segment import Segment


def group_by_bin(segments: list[Segment]) -> dict[tuple, list[Segment]]:
    out: dict[tuple, list[Segment]] = defaultdict(list)
    for s in segments:
        out[s.bin].append(s)
    return out


def sample_uniform_bins(segments: list[Segment], *, target_size: int,
                        rng: random.Random) -> list[Segment]:
    """Pick ~target_size segments by sampling each occupied bin uniformly."""
    bins = group_by_bin(segments)
    if not bins:
        return []
    target_per_bin = max(1, target_size // len(bins))
    selected: list[Segment] = []
    for members in bins.values():
        if len(members) <= target_per_bin:
            selected.extend(members)
        else:
            selected.extend(rng.sample(members, target_per_bin))
    return selected


def enforce_tag_coverage(selected: list[Segment], all_segments: list[Segment], *,
                         tag_vocab: Iterable, target_size: int,
                         min_tag_frac: float) -> list[Segment]:
    """Add segments until every tag in the vocab reaches a floor count, preferring
    underrepresented bins so the patch-up doesn't pile onto already-dense bins."""
    selected_ids = set(id(s) for s in selected)
    counts: Counter = Counter()
    for s in selected:
        for t in s.tags:
            counts[t] += 1
    floor = max(1, int(round(target_size * min_tag_frac)))
    unselected = [s for s in all_segments if id(s) not in selected_ids]
    bin_counts: Counter = Counter(s.bin for s in selected)
    for tag in tag_vocab:
        deficit = floor - counts[tag]
        if deficit <= 0:
            continue
        cands = [s for s in unselected if tag in s.tags]
        cands.sort(key=lambda s: bin_counts[s.bin])
        for s in cands[:deficit]:
            selected.append(s)
            selected_ids.add(id(s))
            bin_counts[s.bin] += 1
            for t in s.tags:
                counts[t] += 1
        unselected = [s for s in unselected if id(s) not in selected_ids]
    return selected


def enforce_group_cap(selected: list[Segment], *, max_group_overshoot: float) -> list[Segment]:
    """Drop excess from any group exceeding `max_group_overshoot` x its fair share,
    cutting its most over-represented bins first. Segments with group=None are pooled
    under a single key, so the cap is a no-op when no groups are set."""
    gkey = lambda s: s.group if s.group is not None else "\x00"
    group_counts: Counter = Counter(gkey(s) for s in selected)
    if len(group_counts) <= 1:
        return selected
    bin_counts: Counter = Counter(s.bin for s in selected)
    fair_share = len(selected) / len(group_counts)
    max_per_group = int(fair_share * max_group_overshoot)
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
    rng = random.Random(seed)
    by_group: dict = defaultdict(list)
    for sid, g in sorted(source_groups.items()):
        by_group[g].append(sid)
    split: dict = {}
    for g in sorted(by_group, key=lambda x: (x is None, x)):
        ids = by_group[g][:]
        rng.shuffle(ids)
        n_test = int(round(test_frac * len(ids)))
        for i, sid in enumerate(ids):
            split[sid] = "test" if i < n_test else "train"
    return split
