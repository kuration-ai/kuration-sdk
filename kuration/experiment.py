"""Multi-arm comparison — score once, select per arm, compare distributions.

Run several strategies (+ a baseline) over the same sources and see how differently they
shape the curated set. The cost-saving trick (and the fairness guarantee) is **score once over
the union** of every arm's metrics: features are computed a single time, so every arm — and
the baseline — is compared on *every* metric, even ones only one arm defined. Each arm then
bins + selects on its own axes/method; comparison reports per-metric mean and Jensen-Shannon
divergence vs the baseline over a shared reference grid.

numpy-only (no sklearn) — this is the distribution-comparison harness, distinct from the
feature-space *selection* methods (a later phase).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .binning import bin_segments
from .features import FeatureEngine
from .schema import SchemaSpec
from .strategy import Strategy, select


@dataclass
class Arm:
    name: str
    strategy: Strategy


def union_define(arms: list[Arm]) -> dict:
    """Merge every arm's `define` specs. A shared metric name must mean one thing — conflicting
    specs for the same name raise (the score-once contract depends on it)."""
    out: dict = {}
    for arm in arms:
        for name, spec in arm.strategy.define.items():
            if name in out and out[name] != spec:
                raise ValueError(f"metric {name!r} has conflicting specs across arms "
                                 f"(score-once needs one meaning per name)")
            out[name] = spec
    return out


def _js(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (bits) between two probability vectors."""
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def run_experiment(*, schema: SchemaSpec, adapter, sources, arms: list[Arm], segmenter,
                   target_size: int, seed: int = 0, baseline: str = None,
                   group_of=None, tags_of=None) -> dict:
    union = union_define(arms)
    referenced: set = set()
    for arm in arms:
        referenced |= arm.strategy.referenced_metrics()
    registered = [r for r in referenced if r not in union]
    engine = FeatureEngine(schema, define=union, registered=registered)

    all_segments = []
    for src in sources:
        ticks = list(adapter.ticks(src))
        segs = segmenter.segments(src, ticks)
        if group_of:
            g = group_of(src)
            for s in segs:
                s.group = g
        engine.compute(ticks, segs)
        if tags_of:
            for s in segs:
                s.tags = frozenset(tags_of(s))
        all_segments.extend(segs)
    all_segments.sort(key=lambda s: (s.source_id, s.start_idx))

    tag_vocab = sorted({t for s in all_segments for t in s.tags}, key=str)
    selected: dict = {}
    for arm in arms:
        pool = all_segments
        if arm.strategy.filters:
            from .methods import apply_filters
            pool = apply_filters(all_segments, arm.strategy, seed=seed)
        bin_segments(pool, arm.strategy.axes)                    # sequential; bin overwritten per arm
        selected[arm.name] = select(pool, arm.strategy, target_size=target_size,
                                    tag_vocab=tag_vocab, seed=seed)

    metrics = sorted(set(union) | referenced)
    edges = {m: np.histogram_bin_edges(
                 np.array([s.features.get(m, 0.0) for s in all_segments] or [0.0]), bins=20)
             for m in metrics}

    def hist(segs, m):
        v = np.array([s.features.get(m, 0.0) for s in segs] or [0.0])
        c, _ = np.histogram(v, bins=edges[m])
        total = c.sum()
        return c / total if total > 0 else np.ones(len(c)) / len(c)

    base = baseline if baseline in selected else None
    base_hist = {m: hist(selected[base], m) for m in metrics} if base else None

    arms_out: dict = {}
    for name, segs in selected.items():
        per_metric: dict = {}
        for m in metrics:
            v = np.array([s.features.get(m, 0.0) for s in segs])
            entry = {"mean": float(v.mean()) if v.size else 0.0}
            if base_hist is not None and name != base:
                entry["js_vs_baseline"] = round(_js(hist(segs, m), base_hist[m]), 4)
            per_metric[m] = entry
        arms_out[name] = {"n_selected": len(segs), "metrics": per_metric}

    return {"baseline": base, "n_total_segments": len(all_segments),
            "metrics": metrics, "arms": arms_out}
