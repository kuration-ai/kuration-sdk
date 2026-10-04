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
from typing import Any

import numpy as np

from .binning import bin_segments
from .results import ExperimentResult
from .schema import SchemaSpec
from .scoring import score_sources
from .strategy import Strategy, _feature_value, select


@dataclass
class Arm:
    name: str
    strategy: Strategy
    transforms: tuple[dict, ...] = ()
    registry: Any = None


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
                   streaming: bool = False, progress=None,
                   executor: str = "local") -> ExperimentResult:
    """Score a shared source pool once and compare selections from multiple strategy arms.

    Returns an ``ExperimentResult`` containing ``baseline``, ``n_total_segments``, ``metrics``,
    and per-arm counts/distribution comparisons. Empty or duplicate arms, conflicting metric or
    scorer definitions, an unknown baseline, and ordinary scoring errors raise ``ValueError``.
    """
    from .strategy import require_target_size
    target_size = require_target_size(target_size)
    if not arms:
        raise ValueError("run_experiment needs at least one arm")
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate arm names "
                         f"{sorted(n for n in set(names) if names.count(n) > 1)} — a later "
                         f"arm would silently overwrite the earlier one's selection")
    union = union_define(arms)
    from .registry import Registry, default_registry
    registries = [arm.registry or arm.strategy.registry or default_registry for arm in arms]
    registry = Registry.merged(*registries)

    # Transform outputs share the same score-once contract as DSL metrics.  Identical calls
    # are executed once; two meanings for the same output fail before any source is read.
    union_transforms: list[dict] = []
    transform_by_output: dict[str, dict] = {}
    for arm in arms:
        for call in arm.transforms:
            conflicts = [output for output in call["outputs"]
                         if output in transform_by_output
                         and transform_by_output[output] != call]
            if conflicts:
                raise ValueError(f"transform output(s) {conflicts} have conflicting calls "
                                 "across experiment arms")
            if all(output in transform_by_output for output in call["outputs"]):
                continue
            overlap = [output for output in call["outputs"] if output in transform_by_output]
            if overlap:
                raise ValueError(f"transform call {call['name']!r} partially overlaps existing "
                                 f"outputs {overlap}; transform output sets must agree")
            union_transforms.append(call)
            for output in call["outputs"]:
                transform_by_output[output] = call

    # Whole-pool scorers are also shared work. Validate their score-once identity before
    # reading sources, and retain the normalized union for the execution pass below.
    union_scorers: dict[str, tuple] = {}
    for arm in arms:
        for name, out, params in (arm.strategy.scorers or []):
            scorer = (name, out, params)
            if out in union_scorers and union_scorers[out] != scorer:
                raise ValueError(f"arms disagree on scorer output {out!r}: "
                                 f"{union_scorers[out]!r} vs {scorer!r}")
            union_scorers[out] = scorer

    # A shared feature column must have exactly one producer kind. Without this check a
    # transform or scorer overwrites a DSL/registered metric in every arm, invalidating the
    # score-once comparison while still returning plausible-looking distributions.
    metric_outputs = set(union)
    for arm, arm_registry in zip(arms, registries, strict=True):
        arm_transforms = {output for call in arm.transforms for output in call["outputs"]}
        arm_scorers = arm.strategy.scorer_outputs()
        arm_dependencies = arm.strategy.referenced_metrics() | {
            token.split(":", 1)[1]
            for call in arm.transforms for token in call["inputs"]
            if token.startswith("feature:")
        }
        metric_outputs.update(
            name for name in arm_dependencies
            if name not in arm.strategy.define and name not in arm_transforms
            and name not in arm_scorers and name in arm_registry.names("metric"))
    producer_outputs = {
        "metric": metric_outputs,
        "transform": set(transform_by_output),
        "scorer": set(union_scorers),
    }
    producer_conflicts = {
        output: [kind for kind, outputs in producer_outputs.items() if output in outputs]
        for output in set().union(*producer_outputs.values())
    }
    producer_conflicts = {
        output: kinds for output, kinds in producer_conflicts.items() if len(kinds) > 1}
    if producer_conflicts:
        detail = ", ".join(f"{output!r} ({'/'.join(kinds)})"
                           for output, kinds in sorted(producer_conflicts.items()))
        raise ValueError(f"experiment feature output name(s) have multiple producer kinds: "
                         f"{detail}")

    referenced: set = set()
    for arm in arms:
        referenced |= arm.strategy.referenced_metrics()
    referenced |= {
        token.split(":", 1)[1]
        for call in union_transforms for token in call["inputs"]
        if token.startswith("feature:")
    }
    scorer_outputs = set(union_scorers)
    transform_outputs = set(transform_by_output)
    registered = sorted(
        r for r in referenced
        if r not in union and r not in scorer_outputs and r not in transform_outputs
    )
    # Tags are derived once over the union of every arm's tag sources.
    tag_froms = sorted({s for arm in arms for s in ((arm.strategy.tags or {}).get("from") or [])})
    union_tags = {"from": tag_froms} if tag_froms else None

    # Score once over the union (parallel); each arm then bins + selects on the shared pool.
    all_segments, errors, _ = score_sources(
        schema=schema, adapter=adapter, sources=sources, define=union, registered=registered,
        segmenter=segmenter, tags_spec=union_tags, group_of=group_of, tags_of=tags_of,
        workers=workers, on_error=on_error, streaming=streaming, progress=progress,
        executor=executor, registry=registry,
        required_fields={token.split(":", 1)[1]
                         for call in union_transforms for token in call["inputs"]
                         if token.startswith("field:")})

    if union_transforms:
        from .transforms import apply_transforms
        all_segments = apply_transforms(
            union_transforms, all_segments, adapter=adapter,
            registry=registry, progress=progress, on_error=on_error, errors=errors)

    if union_scorers:
        from .scorers import apply_scorers
        apply_scorers(list(union_scorers.values()), all_segments, schema=schema, adapter=adapter,
                      seed=seed, progress=progress, registry=registry,
                      on_error=on_error, errors=errors)

    selected: dict = {}
    from .progress import emit
    emit(progress, "selection", "start", total=len(arms))
    for arm in arms:
        pool = all_segments
        if arm.strategy.filters:
            from .methods import apply_filters
            pool = apply_filters(all_segments, arm.strategy, seed=seed, workers=workers)
        pool_tag_vocab = sorted({tag for segment in pool for tag in segment.tags}, key=str)
        if tags_of is not None:
            # The callback replaces declarative tags during scoring, so its vocabulary is the
            # only meaningful one for this arm's filtered pool.
            arm_tag_vocab = pool_tag_vocab
        else:
            tag_sources = tuple((arm.strategy.tags or {}).get("from") or ())
            prefixes = tuple(f"{source}:" for source in tag_sources)
            arm_tag_vocab = [
                tag for tag in pool_tag_vocab
                if isinstance(tag, str) and tag.startswith(prefixes)
            ] if prefixes else []
        bin_segments(pool, arm.strategy.axes)                    # sequential; bin overwritten per arm
        selected[arm.name] = select(pool, arm.strategy, target_size=target_size,
                                    tag_vocab=arm_tag_vocab, seed=seed, workers=workers)
        emit(progress, "selection", "advance", current=len(selected), total=len(arms),
             source=arm.name, status="ok")
    emit(progress, "selection", "complete", current=len(arms), total=len(arms),
         detail=f"{len(arms)} arms")

    metrics = sorted(set(union) | referenced)
    edges = {m: np.histogram_bin_edges(
                 np.array([_feature_value(s, m) for s in all_segments] or [0.0]), bins=20)
             for m in metrics}

    def hist(segs, m):
        v = np.array([_feature_value(s, m) for s in segs] or [0.0])
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
            v = np.array([_feature_value(s, m) for s in segs])
            entry = {"mean": float(v.mean()) if v.size else 0.0}
            if base_hist is not None and name != base:
                entry["js_vs_baseline"] = round(_js(hist(segs, m), base_hist[m]), 4)
            per_metric[m] = entry
        arms_out[name] = {"n_selected": len(segs), "metrics": per_metric}

    return ExperimentResult(baseline=base, n_total_segments=len(all_segments),
                            metrics=metrics, arms=arms_out)
