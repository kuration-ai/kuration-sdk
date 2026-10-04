"""Declarative curation strategies — the malleable core.

A strategy is a small JSON document describing *how* to turn scored segments into a curated
set: inline metric definitions (`define`, DSL specs compiled against the schema), optional
bin axes, and a selection method with its knobs. It is pure data, so the fully-resolved spec
lands verbatim in ``config.json`` as part of an auditable run contract. Replay also requires the
same immutable or content-addressed sources and compatible component implementations.

Base methods (uniform-bins / weighted / top-k / target-distribution / filter) live here;
feature-space methods (cluster/k-center/facility/dsir/rare/typical) + pre-pass filters are in
`methods.py`. Count selection is `select()`; the hours-over-output-spans budget fill lives in
the output layer (`output/fill.py`), where the output unit (a track/span) is defined.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .dsl.grammar import compile_metric
from .schema import SchemaSpec
from .segment import Segment
from .select import enforce_group_cap, enforce_tag_coverage, group_by_bin, sample_uniform_bins
from .registry import default_registry as _default_registry

# Feature-space methods live in methods.py (optional scikit-learn), dispatched from here.
_METHOD_MODULE_METHODS = {"cluster-balanced", "k-center", "facility-location",
                         "dsir", "rare", "typical"}
_BASE_METHODS = {"uniform-bins", "weighted", "target-distribution", "top-k", "filter", "none"}
_METHODS = _BASE_METHODS | _METHOD_MODULE_METHODS
_FILTER_TYPES = {"anomaly", "dedup"}
_PRED_OPS = {">", ">=", "<", "<=", "==", "!="}
_NORMALIZATIONS = {"quantile", "minmax", "zscore", "none"}
_FILTER_KEYS = {
    "anomaly": {"type", "feature_space", "per_group", "contamination", "normalize"},
    "dedup": {"type", "feature_space", "per_group", "threshold", "n_clusters", "normalize"},
}


def validate_target_size(target_size: Optional[int]) -> Optional[int]:
    """Return a positive integer selection count, or None (meaning "every scored and
    filtered segment, no selection method applied"); reject bools and lossy coercions."""
    if target_size is None:
        return None
    if (not isinstance(target_size, int) or isinstance(target_size, bool)
            or target_size < 1):
        raise ValueError(f"target_size must be a positive integer or None, got {target_size!r}")
    return target_size


def require_target_size(target_size: int) -> int:
    """Like `validate_target_size`, for callers (`select`, `run_experiment`,
    `Pipeline.experiment`) that have no "every segment" mode and always need a real count."""
    validated = validate_target_size(target_size)
    if validated is None:
        raise ValueError("target_size must be a positive integer, got None")
    return validated


def _resolve_filters(raw, registry=None) -> list:
    """Validate the optional `filters` block (list of {type, ...knobs}) at load time."""
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ValueError("strategy 'filters' must be a list of {\"type\": ...} objects")
    out = []
    for f in raw:
        if not isinstance(f, dict) or "type" not in f:
            raise ValueError(f"each filter needs a 'type' (offending: {f!r})")
        from .plugins import load_plugins
        from .registry import default_registry, validate_parameters
        registry = registry or default_registry
        load_plugins()
        if f["type"] not in _FILTER_TYPES:
            if f["type"] not in registry.names("filter"):
                raise ValueError(f"unknown filter type {f['type']!r}; registered: "
                                 f"{registry.names('filter')}")
            validate_parameters(registry.descriptor("filter", f["type"]),
                                {k: v for k, v in f.items() if k not in {"type", "feature_space",
                                                                         "per_group"}})
            out.append(dict(f))
            continue
        fs = f.get("feature_space")
        if fs is not None and not (isinstance(fs, list) and all(isinstance(m, str) for m in fs)):
            raise ValueError(f"filter 'feature_space' must be a list of metric names (got {fs!r})")
        unknown = set(f) - _FILTER_KEYS[f["type"]]
        if unknown:
            raise ValueError(f"filter {f['type']!r}: unknown key(s) {sorted(unknown)}; expected "
                             f"{sorted(_FILTER_KEYS[f['type']])}")
        normalize = f.get("normalize", "zscore")
        if normalize not in {"zscore", "quantile", "none"}:
            raise ValueError(f"filter {f['type']!r}: normalize must be zscore/quantile/none")
        if f["type"] == "anomaly":
            contamination = f.get("contamination", 0.02)
            if not isinstance(contamination, (int, float)) or not 0 < contamination <= 0.5:
                raise ValueError("filter 'anomaly': contamination must be in (0, 0.5]")
        if f["type"] == "dedup":
            threshold = f.get("threshold", 0.98)
            if not isinstance(threshold, (int, float)) or not -1 <= threshold <= 1:
                raise ValueError("filter 'dedup': threshold must be in [-1, 1]")
            clusters = f.get("n_clusters", 2048)
            if not isinstance(clusters, int) or isinstance(clusters, bool) or clusters < 1:
                raise ValueError("filter 'dedup': n_clusters must be a positive integer")
        out.append(dict(f))
    return out


def _cmp(a: float, op: str, b: float) -> bool:
    return {">": a > b, ">=": a >= b, "<": a < b, "<=": a <= b,
            "==": a == b, "!=": a != b}[op]


def _feature_value(seg, metric: str) -> float:
    """Return a required selection feature, with one loud missing-feature policy."""
    try:
        return seg.features[metric]
    except KeyError as error:
        raise ValueError(
            f"segment {seg.source_id!r} [{seg.start_idx}, {seg.end_idx}) has no feature "
            f"{metric!r}; ensure every metric/transform/scorer populates every candidate"
        ) from error


def _passes(seg, predicates: Optional[list]) -> bool:
    """True iff a segment satisfies every DSL-metric predicate. Predicates are a universal
    pre-filter: with `method: filter` they ARE the selection; with any other method the
    failing segments are dropped from the pool (set path) / priced -inf (hours path)."""
    return all(_cmp(_feature_value(seg, p["metric"]), p["op"], p["value"])
               for p in (predicates or []))


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
    predicates: Optional[list] = None    # universal pre-filter: [{metric, op, value}, ...]
    feature_space: Optional[list] = None  # metric names spanning the vector space (methods.py)
    method_params: Optional[dict] = None # extra selection knobs (clusters/dsir_bins/knn_k/...)
    filters: Optional[list] = None       # pre-pass filters [{type: anomaly|dedup, ...}]
    tags: Optional[dict] = None          # declarative tag derivation for tag_coverage ({"from": [...]})
    scorers: Optional[list] = None       # parsed [(name, output, params)] — learned/whole-pool scorers
    registry: Any = None                 # explicit overlay registry (not serialized)

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


def _resolve(spec: dict, registry=None) -> Strategy:
    sel = dict(spec.get("selection") or {})
    method = sel.get("method", "uniform-bins")
    from .plugins import load_plugins
    from .registry import default_registry
    registry = registry or default_registry
    if registry is default_registry:
        load_plugins()
    if method not in _METHODS and method not in registry.names("method"):
        raise ValueError(f"unknown selection method {method!r}; registered: "
                         f"{sorted(set(_METHODS) | set(registry.names('method')))}")

    raw_axes = (spec.get("binning") or {}).get("axes") or []
    axes = []
    for a in raw_axes:
        if not isinstance(a, dict) or set(a) != {"metric", "bins"}:
            raise ValueError(f"each binning axis must contain exactly metric/bins, got {a!r}")
        if not isinstance(a["metric"], str) or not a["metric"]:
            raise ValueError(f"binning axis metric must be a non-empty string, got {a['metric']!r}")
        if not isinstance(a["bins"], int) or isinstance(a["bins"], bool) or a["bins"] < 1:
            raise ValueError(f"binning axis {a['metric']!r}: bins must be a positive integer")
        axes.append((a["metric"], a["bins"]))
    axis_metrics = [metric for metric, _bins in axes]
    duplicate_axes = sorted({metric for metric in axis_metrics if axis_metrics.count(metric) > 1})
    if duplicate_axes:
        raise ValueError(f"duplicate bin axis metric(s) {duplicate_axes}")
    if method in ("uniform-bins", "target-distribution") and not axes:
        raise ValueError(f"method {method!r} requires binning.axes")

    weights = {k: float(v) for k, v in (sel.get("weights") or {}).items()}
    if any(not math.isfinite(v) for v in weights.values()):
        raise ValueError("selection.weights values must be finite numbers")
    if method in ("weighted", "top-k") and not weights:
        raise ValueError(f"method {method!r} requires selection.weights")
    normalize = sel.get("normalize", "quantile")
    if normalize not in _NORMALIZATIONS:
        raise ValueError(f"selection.normalize must be one of {sorted(_NORMALIZATIONS)}, "
                         f"got {normalize!r}")
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
    if tag_coverage is not None:
        if not isinstance(tag_coverage, dict):
            raise ValueError("selection.tag_coverage must be an object or null")
        unknown = set(tag_coverage) - {"min_tag_frac"}
        if unknown:
            raise ValueError(f"selection.tag_coverage: unknown key(s) {sorted(unknown)}")
        frac = tag_coverage.get("min_tag_frac", 0.005)
        if not isinstance(frac, (int, float)) or not 0 <= frac <= 1:
            raise ValueError("selection.tag_coverage.min_tag_frac must be in [0, 1]")
    if group_cap is not None:
        if not isinstance(group_cap, dict):
            raise ValueError("selection.group_cap must be an object or null")
        unknown = set(group_cap) - {"max_group_overshoot"}
        if unknown:
            raise ValueError(f"selection.group_cap: unknown key(s) {sorted(unknown)}")
        cap = group_cap.get("max_group_overshoot", 1.5)
        if (not isinstance(cap, (int, float)) or isinstance(cap, bool)
                or not math.isfinite(cap) or cap <= 0):
            raise ValueError("selection.group_cap.max_group_overshoot must be finite and > 0")
    define = dict(spec.get("define") or {})

    predicates = list(sel.get("predicates") or [])
    if method == "filter" and not predicates:
        raise ValueError("method 'filter' requires selection.predicates")
    for p in predicates:
        if not {"metric", "op", "value"} <= set(p):
            raise ValueError(f"predicate needs metric/op/value: {p!r}")
        if p["op"] not in _PRED_OPS:
            raise ValueError(f"predicate op must be one of {sorted(_PRED_OPS)}, got {p['op']!r}")
        if (not isinstance(p["value"], (int, float)) or isinstance(p["value"], bool)
                or not math.isfinite(p["value"])):
            raise ValueError(f"predicate value must be a finite number, got {p['value']!r}")

    feature_space = spec.get("feature_space")
    filters = _resolve_filters(spec.get("filters"), registry)
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
    outputs = [out for _, out, _ in scorers]
    duplicate_outputs = sorted({out for out in outputs if outputs.count(out) > 1})
    if duplicate_outputs:
        raise ValueError(f"duplicate scorer output feature(s) {duplicate_outputs}")
    collisions = sorted(set(outputs) & set(define))
    if collisions:
        raise ValueError(f"scorer output feature(s) {collisions} also appear in 'define'")
    _KNOWN_SEL = {"method", "weights", "normalize", "target", "tag_coverage", "group_cap",
                  "predicates"}
    _KNOWN_PARAMS = {"feature_normalize", "dsir_bins", "knn_k", "rare_sample", "clusters",
                     "floor"}
    method_params = {k: v for k, v in sel.items() if k not in _KNOWN_SEL}
    unknown = set(method_params) - _KNOWN_PARAMS
    if method not in _METHODS:
        from .registry import validate_parameters
        validate_parameters(registry.descriptor("method", method), method_params)
        unknown = set()
    if unknown:                                   # a typo'd knob must not be silently ignored
        raise ValueError(f"unknown selection key(s) {sorted(unknown)}; expected "
                         f"{sorted(_KNOWN_SEL)} or method params {sorted(_KNOWN_PARAMS)}")
    for key in ("dsir_bins", "knn_k", "rare_sample", "clusters"):
        if key in method_params and (not isinstance(method_params[key], int)
                                     or isinstance(method_params[key], bool)
                                     or method_params[key] < 1):
            raise ValueError(f"selection.{key} must be a positive integer")
    if "feature_normalize" in method_params and method_params["feature_normalize"] not in {
            "zscore", "quantile", "none"}:
        raise ValueError("selection.feature_normalize must be zscore/quantile/none")
    if "floor" in method_params:
        floor_value = method_params["floor"]
        if (not isinstance(floor_value, (int, float)) or isinstance(floor_value, bool)
                or not math.isfinite(floor_value) or floor_value < 0):
            raise ValueError("selection.floor must be a finite number >= 0")

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
                    method_params=method_params, filters=filters, tags=tags, scorers=scorers,
                    registry=registry)


def load_strategy(name_or_spec, schema: SchemaSpec, *, registry=None, available_metrics=()) -> Strategy:
    """Load a strategy from a dict or JSON file path.
    Validates `define` (compiles each DSL metric against `schema`) and that every referenced
    metric is either defined or registered — all loudly, before any data is read."""
    from .plugins import load_plugins
    from .registry import default_registry
    if registry is None or registry is default_registry:
        load_plugins()
    if isinstance(name_or_spec, dict):
        spec = name_or_spec
    else:
        p = Path(name_or_spec)
        if not p.exists():
            raise FileNotFoundError(f"strategy JSON file {name_or_spec!r} not found")
        if p.suffix.lower() != ".json":
            raise ValueError(f"strategy file must have a .json suffix, got {str(p)!r}")
        spec = json.loads(p.read_text())

    strat = _resolve(spec, registry)
    for nm, msp in strat.define.items():
        compile_metric(msp, schema)  # validate tokens now; discard the fn
    if strat.tags:
        from .tags import compile_tagger
        compile_tagger(strat.tags, schema)  # validate tag sources at load
    active_registry = strat.registry or default_registry
    if strat.scorers:
        from .registry import validate_parameters
        for name, _out, params in strat.scorers:
            if name not in active_registry.names("scorer"):
                raise ValueError(f"strategy {strat.name!r} references unregistered scorer {name!r}")
            validate_parameters(active_registry.descriptor("scorer", name), params)
    # Scorer outputs are produced at runtime, so they count as available metric references.
    # Registry overlays are the authoritative extension namespace; consulting only the legacy
    # process-global mapping makes ``Pipeline.using(registry)`` fail during strategy loading.
    available = (set(strat.define) | set(active_registry.names("metric"))
                 | strat.scorer_outputs() | set(available_metrics))
    missing = [m for m in strat.referenced_metrics() if m not in available]
    if missing:
        raise ValueError(f"strategy {strat.name!r} references undefined metrics {sorted(missing)} "
                         f"(not in 'define', registered, nor a scorer output)")
    return strat


# ---- scoring ----------------------------------------------------------------
def _normalize(arr: np.ndarray, method: str) -> np.ndarray:
    n = arr.size
    if not np.isfinite(arr).all():
        raise ValueError("selection features must be finite before normalization")
    if method == "none":
        # Raw scale, no transform -- unlike the other methods, well-defined for n<=1 too.
        return arr.astype(np.float64, copy=True)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    if method == "minmax":
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo) if hi > lo else np.zeros(n, dtype=np.float64)
    if method == "zscore":
        mu, sd = arr.mean(), arr.std()
        return (arr - mu) / sd if sd > 0 else np.zeros(n, dtype=np.float64)
    # Quantile normalization uses the average ordinal rank for ties.  Assigning a different
    # rank to equal values invents signal from input order; in particular, a constant feature
    # must contribute nothing to a weighted selector.
    order = arr.argsort(kind="stable")
    sorted_values = arr[order]
    if sorted_values[0] == sorted_values[-1]:
        return np.zeros(n, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    ranks = np.empty(n, dtype=np.float64)
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end - 1) / 2.0
    return ranks / (n - 1)


def _weighted_score(segments: list[Segment], weights: dict[str, float], normalize: str) -> np.ndarray:
    n = len(segments)
    score = np.zeros(n, dtype=np.float64)
    for name in sorted(weights):
        w = weights[name]
        vals = np.fromiter((_feature_value(s, name) for s in segments),
                           dtype=np.float64, count=n)
        score += w * _normalize(vals, normalize)
    return score


def segment_priorities(segments: list[Segment], strategy: Strategy, *, seed: int = 0,
                       workers: int = 1) -> np.ndarray:
    """Per-segment priority (higher = admit sooner), aligned with `segments`. Consumed by the
    hours-budget fill and by the set-based `select` below. Predicate-failing segments get
    -inf under every method (fill_to_hours never admits non-finite priorities)."""
    n = len(segments)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    prio = _raw_priorities(segments, strategy, seed=seed, workers=workers)
    if strategy.predicates and strategy.method != "filter":
        ok = np.fromiter((_passes(s, strategy.predicates) for s in segments), dtype=bool, count=n)
        prio = np.where(ok, prio, float("-inf"))
    return prio


def _filter_priorities(segments: list[Segment], strategy: Strategy, *, seed: int,
                       workers: int) -> np.ndarray:
    """Passing clips share one priority; failing clips are excluded from any budget fill."""
    return np.array([0.0 if _passes(s, strategy.predicates) else float("-inf")
                     for s in segments], dtype=np.float64)


def _none_priorities(segments: list[Segment], strategy: Strategy, *, seed: int,
                     workers: int) -> np.ndarray:
    """No real ranking signal: every segment shares one priority. Unlike "filter", this
    carries no implicit predicate requirement, so `segment_priorities()` applies the
    universal predicate pre-filter on top, same as every other non-"filter" method."""
    return np.zeros(len(segments), dtype=np.float64)


def _top_k_priorities(segments: list[Segment], strategy: Strategy, *, seed: int,
                      workers: int) -> np.ndarray:
    return _weighted_score(segments, strategy.weights, strategy.normalize)


def _weighted_priorities(segments: list[Segment], strategy: Strategy, *, seed: int,
                         workers: int) -> np.ndarray:
    """Gumbel-top-k: sample without replacement, proportional to the weighted score.

    Shifting by the pool's own minimum so every weight is non-negative means the single
    lowest-scoring segment always gets a weight near ``floor`` -- a near-zero chance of
    being picked, by construction, regardless of how close its real score was to the rest
    of the pool. ``floor`` (``selection.floor``, default ``1e-9``) exists so a caller who
    wants the lowest scorer to keep a real, non-negligible relative chance can raise it
    (scaled to the weighted score's own spread); it does not change behavior unset."""
    n = len(segments)
    score = _weighted_score(segments, strategy.weights, strategy.normalize)
    rng = np.random.default_rng(seed)
    floor = (strategy.method_params or {}).get("floor", 1e-9)
    w = score - score.min() + floor
    return np.log(w) + (-np.log(-np.log(rng.random(n))))


def _bin_round_robin_priorities(segments: list[Segment], strategy: Strategy, *, seed: int,
                                workers: int) -> np.ndarray:
    """Shared by uniform-bins/target-distribution: rare-bin round-robin (take the 1st of
    every bin, then 2nd, ...), weighted by the target metric's per-bin sum when
    target-distribution names one."""
    n = len(segments)
    py_rng = random.Random(seed)
    bins: dict[tuple, list[int]] = {}
    for i, s in enumerate(segments):
        bins.setdefault(s.bin, []).append(i)
    if strategy.method == "target-distribution" and isinstance(strategy.target, dict):
        name = strategy.target["metric"]
        binw = {b: max(1e-9, sum(_feature_value(segments[i], name) for i in idxs))
                for b, idxs in bins.items()}
    else:
        binw = {b: 1.0 for b in bins}
    prio = np.empty(n, dtype=np.float64)
    for b, idxs in bins.items():
        order = idxs[:]
        py_rng.shuffle(order)
        w = binw[b]
        for rank, i in enumerate(order):
            prio[i] = -(rank + 1) / w
    return prio


#: One dispatch table entry per base method, every entry sharing the exact
#: `(segments, strategy, *, seed, workers) -> ndarray` signature the module-method and
#: custom-registry-method branches below already use -- previously these 5 were computed
#: inline in `_raw_priorities` with no real function call at all, a third, inconsistent shape.
_BASE_METHOD_PRIORITIES = {
    "filter": _filter_priorities,
    "none": _none_priorities,
    "top-k": _top_k_priorities,
    "weighted": _weighted_priorities,
    "uniform-bins": _bin_round_robin_priorities,
    "target-distribution": _bin_round_robin_priorities,
}


def _raw_priorities(segments: list[Segment], strategy: Strategy, *, seed: int,
                    workers: int) -> np.ndarray:
    n = len(segments)
    if strategy.method in _METHOD_MODULE_METHODS:
        from .methods import method_priorities
        return method_priorities(segments, strategy, seed=seed, workers=workers)
    if strategy.method not in _BASE_METHODS:
        from .registry import default_registry
        registry = strategy.registry or default_registry
        values = registry.get("method", strategy.method)(
            segments, strategy, seed=seed, workers=workers)
        out = np.asarray(values, dtype=np.float64)
        if out.shape != (n,) or not np.isfinite(out).all():
            raise ValueError(f"method {strategy.method!r} must return {n} finite priorities; "
                             f"got shape {out.shape}")
        return out
    return _BASE_METHOD_PRIORITIES[strategy.method](segments, strategy, seed=seed,
                                                     workers=workers)


# ---- set-based selection ----------------------------------------------------
def _select_top_k(segments, *, weights, normalize, target_size) -> list[Segment]:
    score = _weighted_score(segments, weights, normalize)
    order = np.argsort(-score, kind="stable")[:min(target_size, len(segments))]
    return [segments[i] for i in order]


def _select_weighted(segments, *, weights, normalize, target_size, np_rng,
                     floor: float = 1e-9) -> list[Segment]:
    n = len(segments)
    k = min(target_size, n)
    if k <= 0:
        return []
    score = _weighted_score(segments, weights, normalize)
    probs = score - score.min() + floor
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
        w = {k: max(0.0, sum(_feature_value(s, name) for s in bins[k])) for k in keys}
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
    target_size = require_target_size(target_size)
    if not segments:
        return []
    if strategy.predicates:
        segments = [s for s in segments if _passes(s, strategy.predicates)]
        if not segments:
            return []
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if strategy.method in _METHOD_MODULE_METHODS or strategy.method not in _BASE_METHODS:
        # feature-space methods: take the top target_size of the same priority the hours fill
        # uses, so set- and hours-selection agree by construction.
        prio = segment_priorities(segments, strategy, seed=seed, workers=workers)
        k = min(target_size, len(segments))
        order = np.argsort(-prio, kind="stable")[:k]
        selected = [segments[i] for i in order]
    elif strategy.method in ("filter", "none"):
        selected = list(segments)
    elif strategy.method == "uniform-bins":
        selected = sample_uniform_bins(segments, target_size=target_size, rng=py_rng)
    elif strategy.method == "top-k":
        selected = _select_top_k(segments, weights=strategy.weights,
                                 normalize=strategy.normalize, target_size=target_size)
    elif strategy.method == "weighted":
        selected = _select_weighted(segments, weights=strategy.weights,
                                    normalize=strategy.normalize, target_size=target_size,
                                    np_rng=np_rng,
                                    floor=(strategy.method_params or {}).get("floor", 1e-9))
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


for _name in _BASE_METHODS:
    if _name not in _default_registry.names("method"):
        _default_registry.register("method", _name, _raw_priorities,
                                   capabilities={"streaming": True})
