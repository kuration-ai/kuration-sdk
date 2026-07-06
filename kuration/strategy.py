"""Declarative curation strategies — the malleable core.

A strategy is a small JSON document describing *how* to turn scored segments into a curated
set: inline metric definitions (`define`, DSL specs compiled against the schema), optional
bin axes, and a selection method with its knobs. It is pure data, so the fully-resolved spec
lands verbatim in `config.json` and any run reproduces from the artifact alone.

Base methods (uniform-bins / weighted / top-k / target-distribution / filter) live here;
feature-space methods (cluster/k-center/facility/dsir/rare/typical) + pre-pass filters are in
`methods.py`. Count selection is `select()`; the hours-over-output-spans budget fill lives in
the output layer (`output/fill.py`), where the output unit (a track/span) is defined.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .dsl.grammar import compile_metric
from .dsl.registry import METRICS
from .schema import SchemaSpec
from .segment import Segment
from .select import enforce_group_cap, enforce_tag_coverage, group_by_bin, sample_uniform_bins

# Feature-space methods live in methods.py (optional scikit-learn), dispatched from here.
_METHOD_MODULE_METHODS = {"cluster-balanced", "k-center", "facility-location",
                         "dsir", "rare", "typical"}
_BASE_METHODS = {"uniform-bins", "weighted", "target-distribution", "top-k", "filter"}
_METHODS = _BASE_METHODS | _METHOD_MODULE_METHODS
_FILTER_TYPES = {"anomaly", "dedup"}
_PRED_OPS = {">", ">=", "<", "<=", "==", "!="}


def _resolve_filters(raw) -> list:
    """Validate the optional `filters` block (list of {type, ...knobs}) at load time."""
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ValueError("strategy 'filters' must be a list of {\"type\": ...} objects")
    out = []
    for f in raw:
        if not isinstance(f, dict) or "type" not in f:
            raise ValueError(f"each filter needs a 'type' (offending: {f!r})")
        if f["type"] not in _FILTER_TYPES:
            raise ValueError(f"unknown filter type {f['type']!r}; expected {sorted(_FILTER_TYPES)}")
        fs = f.get("feature_space")
        if fs is not None and not (isinstance(fs, list) and all(isinstance(m, str) for m in fs)):
            raise ValueError(f"filter 'feature_space' must be a list of metric names (got {fs!r})")
        out.append(dict(f))
    return out


def _cmp(a: float, op: str, b: float) -> bool:
    return {">": a > b, ">=": a >= b, "<": a < b, "<=": a <= b,
            "==": a == b, "!=": a != b}[op]


def _passes(seg, predicates: list) -> bool:
    """True iff a segment satisfies every DSL-metric predicate (the `filter` method)."""
    return all(_cmp(seg.features.get(p["metric"], 0.0), p["op"], p["value"]) for p in predicates)


@dataclass
class Strategy:
    name: str
    define: dict                          # name -> DSL spec
    axes: list[tuple[str, int]]
    method: str
    weights: dict[str, float]
    normalize: str
    target: Any                           # "flat" | {"metric": name} | None
    tag_coverage: Optional[dict]
    group_cap: Optional[dict]
    resolved: dict
    predicates: list = None              # `filter` method: [{metric, op, value}, ...]
    feature_space: Optional[list] = None  # metric names spanning the vector space (methods.py)
    method_params: dict = None           # extra selection knobs (clusters/dsir_bins/knn_k/...)
    filters: list = None                 # pre-pass filters [{type: anomaly|dedup, ...}]
    tags: Optional[dict] = None          # declarative tag derivation for tag_coverage ({"from": [...]})
    scorers: list = None                 # parsed [(name, output, params)] — learned/whole-pool scorers

    def scorer_outputs(self) -> set[str]:
        """Feature names produced by the strategy's scorers (referenceable like any metric)."""
        return {out for (_, out, _) in (self.scorers or [])}

    def referenced_metrics(self) -> set[str]:
        """Every metric name the strategy reads (axes/weights/target/predicates/feature_space/
        filters) — drives the feature pass and the load-time defined-or-registered check."""
        names = {m for m, _ in self.axes} | set(self.weights)
        if isinstance(self.target, dict):
            names |= {self.target[k] for k in ("metric", "anchor_metric") if k in self.target}
        names |= {p["metric"] for p in (self.predicates or [])}
        names |= set(self.feature_space or [])
        for f in (self.filters or []):
            names |= set(f.get("feature_space") or [])
        return names


def _resolve(spec: dict) -> Strategy:
    sel = dict(spec.get("selection") or {})
    method = sel.get("method", "uniform-bins")
    if method not in _METHODS:
        raise ValueError(f"unknown selection method {method!r}; expected {sorted(_METHODS)}")

    raw_axes = (spec.get("binning") or {}).get("axes") or []
    axes = [(a["metric"], int(a["bins"])) for a in raw_axes]
    if method in ("uniform-bins", "target-distribution") and not axes:
        raise ValueError(f"method {method!r} requires binning.axes")

    weights = {k: float(v) for k, v in (sel.get("weights") or {}).items()}
    if method in ("weighted", "top-k") and not weights:
        raise ValueError(f"method {method!r} requires selection.weights")
    normalize = sel.get("normalize", "quantile")
    target = sel.get("target", "flat")
    if method in ("target-distribution", "dsir"):
        # A dict target missing its metric key would otherwise KeyError only at selection
        # time — after the full (potentially hours-long) feature pass.
        if not (target is None or target == "flat" or isinstance(target, dict)):
            raise ValueError(f"method {method!r}: target must be 'flat' or a dict, got {target!r}")
        want = {"metric"} if method == "target-distribution" else {"metric", "anchor_metric"}
        if isinstance(target, dict) and not (set(target) & want):
            raise ValueError(f"method {method!r}: dict target needs one of {sorted(want)} "
                             f"(got keys {sorted(target)})")
    tag_coverage = sel.get("tag_coverage", None)
    group_cap = sel.get("group_cap", None)
    define = dict(spec.get("define") or {})

    predicates = list(sel.get("predicates") or [])
    if method == "filter" and not predicates:
        raise ValueError("method 'filter' requires selection.predicates")
    for p in predicates:
        if not {"metric", "op", "value"} <= set(p):
            raise ValueError(f"predicate needs metric/op/value: {p!r}")
        if p["op"] not in _PRED_OPS:
            raise ValueError(f"predicate op must be one of {sorted(_PRED_OPS)}, got {p['op']!r}")

    feature_space = spec.get("feature_space")
    filters = _resolve_filters(spec.get("filters"))
    tags = spec.get("tags")
    if tags is not None:
        # Accept {"from": [...]} or a bare list, but normalize so every consumer
        # (compile_tagger, experiment's union over arms) sees the dict form.
        froms = tags.get("from") if isinstance(tags, dict) else tags
        if not (isinstance(froms, (list, tuple)) and froms
                and all(isinstance(t, str) for t in froms)):
            raise ValueError(f"strategy 'tags' must be a list of sources or "
                             f"{{'from': [...]}}; got {tags!r}")
        tags = {"from": list(froms)}
    from .scorers import parse_scorers
    scorers = parse_scorers(spec.get("scorers"))
    _KNOWN_SEL = {"method", "weights", "normalize", "target", "tag_coverage", "group_cap",
                  "predicates"}
    _KNOWN_PARAMS = {"feature_normalize", "dsir_bins", "knn_k", "rare_sample", "clusters"}
    method_params = {k: v for k, v in sel.items() if k not in _KNOWN_SEL}
    unknown = set(method_params) - _KNOWN_PARAMS
    if unknown:                                   # a typo'd knob must not be silently ignored
        raise ValueError(f"unknown selection key(s) {sorted(unknown)}; expected "
                         f"{sorted(_KNOWN_SEL)} or method params {sorted(_KNOWN_PARAMS)}")

    resolved = {
        "name": spec.get("name", "unnamed"),
        "define": define,
        "feature_space": feature_space,
        "filters": filters,
        "tags": tags,
        "scorers": spec.get("scorers"),
        "binning": {"axes": [{"metric": m, "bins": b} for m, b in axes]},
        "selection": {"method": method, "weights": weights, "normalize": normalize,
                      "target": target, "tag_coverage": tag_coverage, "group_cap": group_cap,
                      "predicates": predicates, **method_params},
    }
    return Strategy(name=resolved["name"], define=define, axes=axes, method=method,
                    weights=weights, normalize=normalize, target=target,
                    tag_coverage=tag_coverage, group_cap=group_cap, resolved=resolved,
                    predicates=predicates, feature_space=feature_space,
                    method_params=method_params, filters=filters, tags=tags, scorers=scorers)


def load_strategy(name_or_spec, schema: SchemaSpec) -> Strategy:
    """Load a strategy from a dict, a JSON file path, or a built-in name under strategies/.
    Validates `define` (compiles each DSL metric against `schema`) and that every referenced
    metric is either defined or registered — all loudly, before any data is read."""
    if isinstance(name_or_spec, dict):
        spec = name_or_spec
    else:
        p = Path(name_or_spec)
        if p.suffix == ".json" and p.exists():
            spec = json.loads(p.read_text())
        else:
            builtin = Path(__file__).resolve().parent / "strategies" / f"{name_or_spec}.json"
            if not builtin.exists():
                raise FileNotFoundError(f"strategy {name_or_spec!r} not found (no dict, file, "
                                        f"or built-in)")
            spec = json.loads(builtin.read_text())

    strat = _resolve(spec)
    for nm, msp in strat.define.items():
        compile_metric(msp, schema)  # validate tokens now; discard the fn
    if strat.tags:
        from .tags import compile_tagger
        compile_tagger(strat.tags, schema)  # validate tag sources at load
    if strat.scorers:
        from .scorers import has_scorer
        for name, _out, _p in strat.scorers:
            if not has_scorer(name):
                raise ValueError(f"strategy {strat.name!r} references unregistered scorer {name!r}")
    # scorer outputs are produced at runtime, so they count as available metric references.
    available = set(strat.define) | set(METRICS) | strat.scorer_outputs()
    missing = [m for m in strat.referenced_metrics() if m not in available]
    if missing:
        raise ValueError(f"strategy {strat.name!r} references undefined metrics {sorted(missing)} "
                         f"(not in 'define', registered, nor a scorer output)")
    return strat


# ---- scoring ----------------------------------------------------------------
def _normalize(arr: np.ndarray, method: str) -> np.ndarray:
    n = arr.size
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    if method == "minmax":
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo) if hi > lo else np.zeros(n, dtype=np.float64)
    if method == "zscore":
        mu, sd = arr.mean(), arr.std()
        return (arr - mu) / sd if sd > 0 else np.zeros(n, dtype=np.float64)
    order = arr.argsort(kind="stable")        # quantile (default): ordinal rank -> [0,1]
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / (n - 1)


def _weighted_score(segments: list[Segment], weights: dict[str, float], normalize: str) -> np.ndarray:
    n = len(segments)
    score = np.zeros(n, dtype=np.float64)
    for name, w in weights.items():
        vals = np.fromiter((s.features.get(name, 0.0) for s in segments), dtype=np.float64, count=n)
        score += w * _normalize(vals, normalize)
    return score


def segment_priorities(segments: list[Segment], strategy: Strategy, *, seed: int = 0,
                       workers: int = 1) -> np.ndarray:
    """Per-segment priority (higher = admit sooner), aligned with `segments`. Consumed by the
    hours-budget fill and by the set-based `select` below."""
    n = len(segments)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if strategy.method in _METHOD_MODULE_METHODS:
        from .methods import method_priorities
        return method_priorities(segments, strategy, seed=seed, workers=workers)
    if strategy.method == "filter":
        # passing clips share one priority; failing clips are excluded from any budget fill.
        return np.array([0.0 if _passes(s, strategy.predicates) else float("-inf")
                         for s in segments], dtype=np.float64)
    if strategy.method in ("weighted", "top-k"):
        score = _weighted_score(segments, strategy.weights, strategy.normalize)
        if strategy.method == "top-k":
            return score
        rng = np.random.default_rng(seed)            # weighted: Gumbel-top-k (sample w/o replacement ∝ score)
        w = score - score.min() + 1e-9
        return np.log(w) + (-np.log(-np.log(rng.random(n))))
    # bin-based: rare-bin round-robin (take the 1st of every bin, then 2nd, ...)
    rng = random.Random(seed)
    bins: dict[tuple, list[int]] = {}
    for i, s in enumerate(segments):
        bins.setdefault(s.bin, []).append(i)
    if strategy.method == "target-distribution" and isinstance(strategy.target, dict):
        name = strategy.target["metric"]
        binw = {b: max(1e-9, sum(segments[i].features.get(name, 0.0) for i in idxs))
                for b, idxs in bins.items()}
    else:
        binw = {b: 1.0 for b in bins}
    prio = np.empty(n, dtype=np.float64)
    for b, idxs in bins.items():
        order = idxs[:]
        rng.shuffle(order)
        w = binw[b]
        for rank, i in enumerate(order):
            prio[i] = -(rank + 1) / w
    return prio


# ---- set-based selection ----------------------------------------------------
def _select_top_k(segments, *, weights, normalize, target_size) -> list[Segment]:
    score = _weighted_score(segments, weights, normalize)
    order = np.argsort(-score, kind="stable")[:min(target_size, len(segments))]
    return [segments[i] for i in order]


def _select_weighted(segments, *, weights, normalize, target_size, np_rng) -> list[Segment]:
    n = len(segments)
    k = min(target_size, n)
    if k <= 0:
        return []
    score = _weighted_score(segments, weights, normalize)
    probs = score - score.min() + 1e-9
    probs = probs / probs.sum()
    idx = np_rng.choice(n, size=k, replace=False, p=probs)
    return [segments[i] for i in idx]


def _select_target_distribution(segments, *, target, target_size, py_rng) -> list[Segment]:
    bins = group_by_bin(segments)
    keys = list(bins)
    if not keys:
        return []
    if target == "flat" or target is None:
        w = {k: 1.0 for k in keys}
    else:
        name = target["metric"]
        w = {k: max(0.0, sum(s.features.get(name, 0.0) for s in bins[k])) for k in keys}
    total = sum(w.values()) or 1.0
    # Proportional quotas capped by bin population, largest-remainder for the residue, then
    # spill capped bins' shortfall into bins with room — lands on exactly min(target, n)
    # (the old max(1, round(...)) per bin overshot whenever bins outnumbered the target).
    k_total = min(target_size, len(segments))
    exact = {k: k_total * w[k] / total for k in keys}
    quota = {k: min(int(exact[k]), len(bins[k])) for k in keys}
    n_sel = sum(quota.values())
    for k in sorted(keys, key=lambda b: (-(exact[b] - int(exact[b])), str(b))):
        if n_sel >= k_total:
            break
        if quota[k] < len(bins[k]):
            quota[k] += 1
            n_sel += 1
    while n_sel < k_total:                        # spill (k_total <= n, so this terminates)
        for k in keys:
            if n_sel >= k_total:
                break
            if quota[k] < len(bins[k]):
                quota[k] += 1
                n_sel += 1
    selected: list[Segment] = []
    for k in keys:
        members, q = bins[k], quota[k]
        selected.extend(members if len(members) <= q else py_rng.sample(members, q))
    return selected


def select(segments: list[Segment], strategy: Strategy, *, target_size: int,
           tag_vocab: Iterable = (), seed: int = 0, workers: int = 1) -> list[Segment]:
    """Run the strategy's method, then its enabled post-passes (tag coverage, group cap).
    Callers pre-sort `segments` by (source_id, start_idx) so seeded RNGs are
    worker-count-independent."""
    if not segments:
        return []
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if strategy.method in _METHOD_MODULE_METHODS:
        # feature-space methods: take the top target_size of the same priority the hours fill
        # uses, so set- and hours-selection agree by construction.
        prio = segment_priorities(segments, strategy, seed=seed, workers=workers)
        k = min(target_size, len(segments))
        order = np.argsort(-prio, kind="stable")[:k]
        selected = [segments[i] for i in order]
    elif strategy.method == "filter":
        selected = [s for s in segments if _passes(s, strategy.predicates)]
    elif strategy.method == "uniform-bins":
        selected = sample_uniform_bins(segments, target_size=target_size, rng=py_rng)
    elif strategy.method == "top-k":
        selected = _select_top_k(segments, weights=strategy.weights,
                                 normalize=strategy.normalize, target_size=target_size)
    elif strategy.method == "weighted":
        selected = _select_weighted(segments, weights=strategy.weights,
                                    normalize=strategy.normalize, target_size=target_size,
                                    np_rng=np_rng)
    else:  # target-distribution
        selected = _select_target_distribution(segments, target=strategy.target,
                                               target_size=target_size, py_rng=py_rng)
    if not selected:
        return []
    if strategy.tag_coverage:
        selected = enforce_tag_coverage(
            selected, segments, tag_vocab=tag_vocab, target_size=target_size,
            min_tag_frac=strategy.tag_coverage.get("min_tag_frac", 0.005))
    if strategy.group_cap:
        selected = enforce_group_cap(
            selected, max_group_overshoot=strategy.group_cap.get("max_group_overshoot", 1.5))
    return selected
