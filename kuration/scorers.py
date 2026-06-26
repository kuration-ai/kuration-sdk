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
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from .buffer import build_window


@runtime_checkable
class SegmentScorer(Protocol):
    needs_buffers: bool
    def score(self, segments: list, ctx: dict) -> np.ndarray: ...


_SCORERS: dict[str, Callable[..., SegmentScorer]] = {}


def register_scorer(name: str, factory: Callable[..., SegmentScorer]) -> None:
    if name in _SCORERS:
        raise ValueError(f"scorer {name!r} already registered")
    _SCORERS[name] = factory


def get_scorer(name: str, **kwargs) -> SegmentScorer:
    if name not in _SCORERS:
        raise ValueError(f"unknown scorer {name!r}; registered: {sorted(_SCORERS)}")
    return _SCORERS[name](**kwargs)


def has_scorer(name: str) -> bool:
    return name in _SCORERS


def parse_scorers(raw) -> list[tuple]:
    """Normalize a strategy `scorers` block to [(name, output_feature, params)]."""
    out = []
    for e in raw or []:
        if isinstance(e, str):
            out.append((e, e, {}))
        elif isinstance(e, dict) and "name" in e:
            params = {k: v for k, v in e.items() if k not in ("name", "as")}
            out.append((e["name"], e.get("as", e["name"]), params))
        else:
            raise ValueError(f"scorer entry must be a name or {{'name': ...}}, got {e!r}")
    return out


def _materialize_buffers(segments, schema, adapter) -> list:
    """Re-extract each segment's WindowBuffer, reading every source once. Aligned to `segments`."""
    by_src: dict = defaultdict(list)
    for i, s in enumerate(segments):
        by_src[s.source_id].append(i)
    bufs = [None] * len(segments)
    for src, idxs in by_src.items():
        ticks = list(adapter.ticks(src))
        for i in idxs:
            s = segments[i]
            bufs[i] = build_window(schema, ticks[s.start_idx:s.end_idx])
    return bufs


def apply_scorers(entries: list[tuple], segments: list, *, schema, adapter, seed: int = 0) -> None:
    """Run each scorer over the whole pool and write its output into `seg.features[output]`.
    Buffers are materialized once and shared across buffer-needing scorers."""
    if not entries or not segments:
        return
    buffers = None
    for name, output, params in entries:
        scorer = get_scorer(name, **params)
        ctx = {"schema": schema, "seed": seed}
        if getattr(scorer, "needs_buffers", False):
            if buffers is None:
                buffers = _materialize_buffers(segments, schema, adapter)
            ctx["buffers"] = buffers
        vals = np.asarray(scorer.score(segments, ctx), dtype=np.float64)
        if vals.shape[0] != len(segments):
            raise ValueError(f"scorer {name!r} returned {vals.shape[0]} values for "
                             f"{len(segments)} segments")
        for s, v in zip(segments, vals):
            s.features[output] = float(v)


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
                              "pip install -e '.[methods]'") from e
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


register_scorer("action_residual", ActionResidualScorer)
