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
from .schema import SchemaSpec
from .scoring import score_sources
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
                   group_of=None, tags_of=None, workers: int = 1, on_error: str = "skip",
                   streaming: bool = False, progress=None) -> dict:
    from .strategy import validate_target_size
    target_size = validate_target_size(target_size)
    if not arms:
        raise ValueError("run_experiment needs at least one arm")
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate arm names "
                         f"{sorted(n for n in set(names) if names.count(n) > 1)} — a later "
                         f"arm would silently overwrite the earlier one's selection")
    union = union_define(arms)
    referenced: set = set()
    for arm in arms:
        referenced |= arm.strategy.referenced_metrics()
    scorer_outputs = {out for arm in arms for (_, out, _) in (arm.strategy.scorers or [])}
    registered = [r for r in referenced if r not in union and r not in scorer_outputs]
    # Tags are derived once over the union of every arm's tag sources.
    tag_froms = sorted({s for arm in arms for s in ((arm.strategy.tags or {}).get("from") or [])})
    union_tags = {"from": tag_froms} if tag_froms else None

    # Score once over the union (parallel); each arm then bins + selects on the shared pool.
    all_segments, _, _ = score_sources(
        schema=schema, adapter=adapter, sources=sources, define=union, registered=registered,
        segmenter=segmenter, tags_spec=union_tags, group_of=group_of, tags_of=tags_of,
        workers=workers, on_error=on_error, streaming=streaming, progress=progress)

    # Whole-pool scorers, unioned across arms (a shared output name must mean one scorer,
    # with one parameterization — otherwise the last arm's params would silently win).
    union_scorers: dict = {}
    for arm in arms:
        for (name, out, params) in (arm.strategy.scorers or []):
            if out in union_scorers and union_scorers[out] != (name, out, params):
                raise ValueError(f"arms disagree on scorer output {out!r}: "
                                 f"{union_scorers[out]!r} vs {(name, out, params)!r}")
            union_scorers[out] = (name, out, params)
    if union_scorers:
        from .scorers import apply_scorers
        apply_scorers(list(union_scorers.values()), all_segments, schema=schema, adapter=adapter,
                      seed=seed, progress=progress)

    tag_vocab = sorted({t for s in all_segments for t in s.tags}, key=str)
    selected: dict = {}
    from .progress import emit
    emit(progress, "selection", "start", total=len(arms))
    for arm in arms:
        pool = all_segments
        if arm.strategy.filters:
            from .methods import apply_filters
            pool = apply_filters(all_segments, arm.strategy, seed=seed)
        bin_segments(pool, arm.strategy.axes)                    # sequential; bin overwritten per arm
        selected[arm.name] = select(pool, arm.strategy, target_size=target_size,
                                    tag_vocab=tag_vocab, seed=seed)
        emit(progress, "selection", "advance", current=len(selected), total=len(arms),
             source=arm.name, status="ok")
    emit(progress, "selection", "complete", current=len(arms), total=len(arms),
         detail=f"{len(arms)} arms")

    metrics = sorted(set(union) | referenced)
    edges = {m: np.histogram_bin_edges(
                 np.array([s.features.get(m, 0.0) for s in all_segments] or [0.0]), bins=20)
             for m in metrics}

    def hist(segs, m):
        v = np.array([s.features.get(m, 0.0) for s in segs] or [0.0])
        c, _ = np.histogram(v, bins=edges[m])
        total = c.sum()
        return c / total if total > 0 else np.ones(len(c)) / len(c)

    if baseline is not None and baseline not in selected:
        raise ValueError(f"baseline {baseline!r} is not an arm name "
                         f"(arms: {sorted(selected)})")
    base = baseline
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

    from .results import ExperimentResult
    return ExperimentResult(baseline=base, n_total_segments=len(all_segments),
                            metrics=metrics, arms=arms_out)
