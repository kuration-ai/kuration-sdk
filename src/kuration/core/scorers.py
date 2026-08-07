"""Segment scorers — the first-class seam for learned / action-aware / influence metrics.

The closed-grammar DSL is per-segment, single-pass, stateless, model-free — the right shape
for geometric/statistical features, the wrong shape for the robotics-style estimators
(obs→action mutual information, datamodel/validation-loss influence, learned quality) that the
literature argues actually rank trajectory quality. Those are **whole-pool, often two-stage,
model-backed**. So they get their own stage rather than being bent through `@register_metric`.

A `SegmentScorer.score(segments, ctx)` sees the entire pool at once, may load a model / use a
GPU / a reference set, and returns one value per segment. The pipeline writes those into
`seg.features[<output name>]`, after which the *existing* selection machinery (top-k / weighted
/ k-center / predicates) consumes them like any other feature — so a learned quality score
**composes with** geometric coverage instead of replacing it.

`needs_buffers=True` asks the pipeline for each segment's `WindowBuffer` (the raw obs+action
sequences) — supplied by re-extracting windows once per source (a second read of the data; the
honest cost of sequence-level scoring). Scorers that work on already-computed features or a
precomputed embedding field leave it False and pay nothing.

Reproducibility note: a model-backed scorer is only as deterministic as its pinned model + seed
— the config records the scorer name/params, so repro holds *given the same model*.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from .buffer import build_window


@runtime_checkable
class SegmentScorer(Protocol):
    needs_buffers: bool
    def score(self, segments: list, ctx: dict) -> np.ndarray: ...


_SCORERS: dict[str, Callable[..., SegmentScorer]] = {}


def register_scorer(name: str, factory: Callable[..., SegmentScorer], **metadata) -> None:
    """Register a whole-pool scorer factory under a unique name.

    Duplicate names raise ``ValueError``; ``metadata`` becomes its component descriptor.
    """
    if name in _SCORERS:
        raise ValueError(f"scorer {name!r} already registered")
    _SCORERS[name] = factory
    from .registry import default_registry
    default_registry.register("scorer", name, factory, replace=True, **metadata)


def validate_scorer_parameters(name: str, params: dict, *, registry=None) -> None:
    """Validate scorer factory kwargs against its component descriptor."""
    from .registry import default_registry, validate_parameters
    active = registry or default_registry
    if active is default_registry and name not in active.names("scorer"):
        from .plugins import load_plugins
        load_plugins()
    if name not in active.names("scorer"):
        raise ValueError(f"unknown scorer {name!r}; registered: {active.names('scorer')}")
    validate_parameters(active.descriptor("scorer", name), params)


def get_scorer(name: str, *, registry=None, **kwargs) -> SegmentScorer:
    """Construct a named scorer, loading plugins on demand.

    ``kwargs`` are passed to the factory. An unknown name raises ``ValueError``.
    """
    from .registry import default_registry
    active = registry or default_registry
    validate_scorer_parameters(name, kwargs, registry=active)
    return active.get("scorer", name)(**kwargs)


def has_scorer(name: str) -> bool:
    from .registry import default_registry
    if name not in default_registry.names("scorer"):
        from .plugins import load_plugins
        load_plugins()
    return name in default_registry.names("scorer")


def parse_scorers(raw) -> list[tuple]:
    """Normalize a strategy `scorers` block to [(name, output_feature, params)]."""
    out: list[tuple[str, str, dict]] = []
    for e in raw or []:
        if isinstance(e, str):
            if not e:
                raise ValueError("scorer name must be a non-empty string")
            out.append((e, e, {}))
        elif isinstance(e, dict) and "name" in e:
            params = {k: v for k, v in e.items() if k not in ("name", "as")}
            output = e.get("as", e["name"])
            if not isinstance(e["name"], str) or not e["name"]:
                raise ValueError(f"scorer name must be a non-empty string, got {e['name']!r}")
            if not isinstance(output, str) or not output:
                raise ValueError(f"scorer output 'as' must be a non-empty string, got {output!r}")
            out.append((e["name"], output, params))
        else:
            raise ValueError(f"scorer entry must be a name or {{'name': ...}}, got {e!r}")
    return out


def _materialize_buffers(segments, schema, adapter) -> list:
    """Re-extract each segment's WindowBuffer, reading every source once. Aligned to `segments`."""
    by_src: dict = defaultdict(list)
    for i, s in enumerate(segments):
        by_src[s.source_id].append(i)
    bufs: list[Any] = [None] * len(segments)
    for src, idxs in by_src.items():
        ticks = list(adapter.ticks(src))
        for i in idxs:
            s = segments[i]
            if (not isinstance(s.start_idx, int) or isinstance(s.start_idx, bool)
                    or not isinstance(s.end_idx, int) or isinstance(s.end_idx, bool)
                    or s.start_idx < 0 or s.end_idx <= s.start_idx
                    or s.end_idx > len(ticks)):
                from .provenance import publishable_source_id
                raise ValueError(
                    f"source {publishable_source_id(src)!r} changed before scorer buffer "
                    f"materialization: segment [{s.start_idx}, {s.end_idx}) is outside the "
                    f"{len(ticks)} ticks available on the second read")
            observed_range = (ticks[s.start_idx].time, ticks[s.end_idx - 1].time)
            if observed_range != tuple(s.time_range):
                from .provenance import publishable_source_id
                raise ValueError(
                    f"source {publishable_source_id(src)!r} changed before scorer buffer "
                    f"materialization: segment [{s.start_idx}, {s.end_idx}) no longer has "
                    "the timestamps recorded by the feature pass")
            bufs[i] = build_window(schema, ticks[s.start_idx:s.end_idx])
    return bufs


def _materialize_buffers_with_policy(segments, schema, adapter, *, on_error, errors) -> list:
    """Materialize per source so the ordinary skip policy remains an honest isolation unit."""
    by_src: dict[Any, list] = defaultdict(list)
    for segment in segments:
        by_src[segment.source_id].append(segment)
    retained_segments = []
    retained_buffers = []
    failures = 0
    for source_segments in by_src.values():
        source = source_segments[0].source_id
        try:
            source_buffers = _materialize_buffers(source_segments, schema, adapter)
        except Exception as error:
            if on_error == "raise":
                raise
            import sys
            from .scoring import _redact_diagnostic

            failures += 1
            safe_source = _redact_diagnostic(source)
            safe_error = _redact_diagnostic(repr(error))
            print(f"  [skip] scorer source {safe_source!r}: {safe_error}", file=sys.stderr)
            if errors is not None:
                errors.append({
                    "source": safe_source,
                    "error": safe_error,
                    "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
                    "message": _redact_diagnostic(str(error)),
                    "phase": "scorers",
                })
            continue
        retained_segments.extend(source_segments)
        retained_buffers.extend(source_buffers)
    if failures:
        segments[:] = retained_segments
    return retained_buffers


def apply_scorers(entries: list[tuple], segments: list, *, schema, adapter, seed: int = 0,
                  progress=None, registry=None, on_error="raise", errors=None) -> None:
    """Run each scorer over the whole pool and write its output into `seg.features[output]`.
    Buffers are materialized once and shared across buffer-needing scorers. ``on_error="skip"``
    applies at source granularity to second-read buffer failures; scorer-model failures remain
    whole-pool errors because they cannot be attributed to one source."""
    if not entries or not segments:
        return
    if on_error not in {"raise", "skip"}:
        raise ValueError("on_error must be 'raise' or 'skip'")
    from .progress import emit
    emit(progress, "scorers", "start", total=len(entries))
    buffers = None
    for scorer_i, (name, output, params) in enumerate(entries, 1):
        scorer = get_scorer(name, registry=registry, **params)
        ctx = {"schema": schema, "seed": seed}
        if getattr(scorer, "needs_buffers", False):
            if buffers is None:
                buffers = _materialize_buffers_with_policy(
                    segments, schema, adapter, on_error=on_error, errors=errors)
                if not segments:
                    raise RuntimeError(
                        "segment pool is EMPTY: every source failed scorer buffer "
                        "materialization")
            ctx["buffers"] = buffers
        vals = np.asarray(scorer.score(segments, ctx), dtype=np.float64)
        if vals.shape != (len(segments),) or not np.isfinite(vals).all():
            raise ValueError(
                f"scorer {name!r} must return a finite 1-D vector with shape "
                f"({len(segments)},), got shape {vals.shape}"
                + (" containing non-finite values" if not np.isfinite(vals).all() else ""))
        for s, v in zip(segments, vals):
            s.features[output] = float(v)
        emit(progress, "scorers", "advance", current=scorer_i,
             total=len(entries), source=output, status="ok")
    emit(progress, "scorers", "complete", current=len(entries), total=len(entries),
         detail=f"{len(entries)} scorers")


# --------------------------------------------------------------------------------------
# Reference scorer (demonstrator): action-conditioned predictive residual.
# Whole-pool, buffer-using, action-aware — the class the closed grammar can't express. It is
# NOT DemInf/DataMIL (those need the user's task model); it shows the seam end-to-end and how a
# learned-ish score composes with coverage. Needs scikit-learn (the [methods] extra).
# --------------------------------------------------------------------------------------
class ActionResidualScorer:
    needs_buffers = True

    def __init__(self, *, obs: str, action: str, k: int = 8):
        self.obs, self.action, self.k = obs, action, int(k)

    def score(self, segments, ctx) -> np.ndarray:
        try:
            from sklearn.neighbors import KNeighborsRegressor
        except ImportError as e:
            raise ImportError("scorer 'action_residual' needs scikit-learn: "
                              "pip install 'kuration-sdk[methods]'") from e
        bufs = ctx["buffers"]

        def pooled(w, field):
            a = w.numeric_field(field)                       # (T, E, D)
            m = np.nanmean(a.reshape(-1, a.shape[-1]), axis=0)
            return np.nan_to_num(m)

        X = np.array([pooled(w, self.obs) for w in bufs])     # per-segment mean obs
        Y = np.array([pooled(w, self.action) for w in bufs])  # per-segment mean action
        k = max(1, min(self.k + 1, len(X)))                   # +1: a point is its own neighbor
        knn = KNeighborsRegressor(n_neighbors=k).fit(X, Y)
        # residual of the obs->action map = how atypical this action is given its obs (informative)
        return np.linalg.norm(Y - knn.predict(X), axis=1)


register_scorer(
    "action_residual",
    ActionResidualScorer,
    description="Action-conditioned k-nearest-neighbor predictive residual.",
    parameters={
        "type": "object",
        "properties": {
            "obs": {"type": "string", "minLength": 1},
            "action": {"type": "string", "minLength": 1},
            "k": {"type": "integer", "minimum": 1},
        },
        "required": ["obs", "action"],
        "additionalProperties": False,
    },
)
