"""The closed metric grammar, schema-driven.

A metric is a JSON spec describing a reduction over one segment's window:

    SOURCE  ->  MAP (optional)  ->  FILTER (optional)  ->  REDUCE  [x SCOPE]

`compile_metric(spec, schema)` validates every token against the supplied `SchemaSpec`
(unknown tokens fail loudly) and returns `fn(WindowBuffer) -> float`. The grammar is
*closed* — every token is enumerated — so a spec is safe, JSON-serializable and
reproducible. Anything it can't express drops to a registered Python metric (the escape
hatch in `registry.py`).

What is universal vs schema-gated:
  - SOURCE  `<entity>.<field>[.<dim>]` | `<event_stream>.<field>` | `<global>` — validated
            against the schema's declared namespaces/fields.
  - MAP     `abs`/`delta` (any), `speed` (a `position` vector), `angle_delta` (an `angular`
            field), `bit:<NAME>` (a `bitmask` field; NAME in that field's `bits`). `delta`
            on an angular field wraps; `speed` is scaled by `sample_hz` and clamped by the
            field's `max_rate`.
  - FILTER  `<field><op><value>` (op in >=,<=,==,>,<) | `<field>:bit:<NAME>` — over an
            entity field, producing a per-cell mask.
  - REDUCE  universal numeric set (mean/std/var/min/max/sum/count/rate/entropy/nunique/jerk/
            p<q>) + spatial (dispersion/min_pair_dist/hull_area, require a `position` source)
            + categorical (entropy/nunique/count).
  - SCOPE   `pooled` | `per_entity_mean` (single-stream schemas allow only `pooled`).

This is a faithful generalization of the CS2-specific original: the reduce machinery is
ported verbatim (it was already domain-agnostic); only source/map/filter resolution became
schema-driven.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from ..buffer import WindowBuffer
from ..schema import FieldSpec, SchemaSpec

_SCALAR_REDUCES = {"mean", "std", "var", "min", "max", "sum", "count",
                   "rate", "entropy", "nunique", "jerk"}  # plus p<q>
# Reduces `_agg_1d` (+ jerk) actually implement over a numeric stream. `nunique` is
# categorical-only; spatial reduces need a position plane — both excluded so an ill-typed
# numeric/global reduce fails loudly at LOAD, not mid-run (closed-grammar contract).
_NUMERIC_REDUCES = {"mean", "std", "var", "min", "max", "sum", "count", "rate", "entropy", "jerk"}
_SPATIAL_REDUCES = {"dispersion", "min_pair_dist", "hull_area"}
_CATEGORICAL_REDUCES = {"entropy", "nunique", "count"}
_SCOPES = {"pooled", "per_entity_mean"}
_DEG_PERIOD = 360.0  # angular wrap period (degrees — the supported angular unit in v1)


# ======================================================================================
# reduce helpers (domain-agnostic; ported from the reference implementation)
# ======================================================================================
def _shannon_counts(counts) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            pr = c / total
            h -= pr * math.log(pr, 2)
    return h


def _numeric_entropy(v: np.ndarray, bins: int = 16) -> float:
    if v.size == 0:
        return 0.0
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return 0.0
    counts, _ = np.histogram(v, bins=bins, range=(lo, hi))
    return _shannon_counts(counts.tolist())


def _agg_1d(name: str, v: np.ndarray) -> float:
    if v.size == 0:
        return 0.0
    if name == "mean":
        return float(v.mean())
    if name == "std":
        return float(v.std())
    if name == "var":
        return float(v.var())
    if name == "min":
        return float(v.min())
    if name == "max":
        return float(v.max())
    if name == "sum":
        return float(v.sum())
    if name == "count":
        return float(v.size)
    if name == "rate":
        return float((v != 0).mean())
    if name == "entropy":
        return _numeric_entropy(v)
    if name.startswith("p"):
        return float(np.percentile(v, float(name[1:])))
    raise ValueError(f"reduce {name!r} not applicable to a numeric stream")


def _reduce_scalar(name: str, grid: np.ndarray, scope: str, mask: Optional[np.ndarray]) -> float:
    g = grid.astype(float)
    if mask is not None:
        g = np.where(mask, g, np.nan)
    if scope == "per_entity_mean":
        out = []
        for j in range(g.shape[1]):
            col = g[:, j]
            col = col[~np.isnan(col)]
            if col.size:
                out.append(_agg_1d(name, col))
        return float(np.mean(out)) if out else 0.0
    flat = g[~np.isnan(g)]
    return _agg_1d(name, flat)


def _reduce_jerk(grid: np.ndarray, mask: Optional[np.ndarray]) -> float:
    """Mean over entities of the mean absolute consecutive change — acceleration/jitter.
    Inherently per-entity temporal, so it ignores scope. Diff BEFORE dropping NaN: a gap
    (absent tick / filtered cell) must break adjacency, not splice its neighbors together."""
    g = grid.astype(float)
    if mask is not None:
        g = np.where(mask, g, np.nan)
    out = []
    for j in range(g.shape[1]):
        d = np.diff(g[:, j])
        d = d[~np.isnan(d)]
        if d.size:
            out.append(float(np.mean(np.abs(d))))
    return float(np.mean(out)) if out else 0.0


def _convex_hull_area(pts: np.ndarray) -> float:
    """2D convex hull area via Andrew's monotone chain (no scipy dep)."""
    if pts.shape[0] < 3:
        return 0.0
    P = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(points):
        h: list[np.ndarray] = []
        for p in points:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - \
                    (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(p)
        return h
    hull = half(P)[:-1] + half(P[::-1])[:-1]
    if len(hull) < 3:
        return 0.0
    h = np.array(hull)
    x, y = h[:, 0], h[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _spatial_reduce(name: str, points: np.ndarray, valid: np.ndarray) -> float:
    """Per-tick aggregate across present entities, averaged over ticks.

    Dispersion and pair distance operate in every declared position dimension. Hull area is
    planar and its caller supplies the established first-two-dimensions projection.
    """
    per_tick = []
    for ti in range(points.shape[0]):
        m = valid[ti]
        coordinates = points[ti, m]
        coordinates = coordinates[~np.isnan(coordinates).any(axis=1)]
        if coordinates.shape[0] < 2:
            continue
        if name == "dispersion":
            center = coordinates.mean(axis=0)
            per_tick.append(float(np.sqrt(
                np.mean(np.sum((coordinates - center) ** 2, axis=1)))))
        elif name == "min_pair_dist":
            d = np.sqrt(((coordinates[:, None, :] - coordinates[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            per_tick.append(float(d.min()))
        elif name == "hull_area":
            per_tick.append(_convex_hull_area(coordinates))
    return float(np.mean(per_tick)) if per_tick else 0.0


def _reduce_categorical(reduce: str, codes: np.ndarray, scope: str,
                        mask: Optional[np.ndarray]) -> float:
    c = codes.copy()
    if mask is not None:
        c = np.where(mask, c, -1)

    def aggregate(valid: np.ndarray) -> float:
        if reduce == "nunique":
            return float(np.unique(valid).size)
        if reduce == "count":
            return float(valid.size)
        if reduce == "entropy":
            if valid.size == 0:
                return 0.0
            _, counts = np.unique(valid, return_counts=True)
            return _shannon_counts(counts.tolist())
        raise ValueError(f"reduce {reduce!r} not applicable to a categorical source "
                         f"(use {sorted(_CATEGORICAL_REDUCES)})")

    if scope == "per_entity_mean":
        per_entity = [aggregate(column[column >= 0]) for column in c.T
                      if np.any(column >= 0)]
        return float(np.mean(per_entity)) if per_entity else 0.0
    return aggregate(c[c >= 0])


def _wrap_deg(d: np.ndarray) -> np.ndarray:
    return (d + _DEG_PERIOD / 2) % _DEG_PERIOD - _DEG_PERIOD / 2


def _extract_bit(arr: np.ndarray, bitval: int) -> np.ndarray:
    """Extract a bit from a NaN-padded int-valued float array -> 0.0/1.0, NaN preserved for
    absent cells (so a `rate` doesn't count an absent entity as 'off')."""
    valid = ~np.isnan(arr)
    gi = np.where(valid, arr, 0.0).astype(np.int64)
    out = ((gi & bitval) != 0).astype(np.float64)
    out[~valid] = np.nan
    return out


# ======================================================================================
# validation + compilation
# ======================================================================================
def _validate_reduce(reduce: str) -> str:
    if reduce in _SCALAR_REDUCES or reduce in _SPATIAL_REDUCES:
        return reduce
    if reduce.startswith("p"):
        try:
            float(reduce[1:])
            return reduce
        except ValueError:
            pass
    raise ValueError(f"unknown reduce {reduce!r}")


def _validate_numeric_reduce(reduce: str) -> str:
    """Reduce must be applicable to a numeric/global scalar stream. Catches `nunique`
    (categorical-only), spatial reduces, and out-of-range percentiles at load time."""
    if reduce in _NUMERIC_REDUCES:
        return reduce
    if reduce.startswith("p"):
        try:
            q = float(reduce[1:])
        except ValueError:
            pass
        else:
            if 0.0 <= q <= 100.0:
                return reduce
            raise ValueError(f"percentile reduce {reduce!r} out of range (need p0..p100)")
    raise ValueError(f"reduce {reduce!r} not applicable to a numeric stream "
                     f"(use {sorted(_NUMERIC_REDUCES)} or p<q>)")


def _parse_filter(filt: str):
    """Return ('bit', field, name) or ('cmp', field, op, value_str)."""
    if ":bit:" in filt:
        field, name = filt.split(":bit:", 1)
        return ("bit", field, name)
    for op in (">=", "<=", "==", ">", "<"):
        if op in filt:
            field, val = filt.split(op, 1)
            return ("cmp", field, op, val)
    raise ValueError(f"unrecognized filter {filt!r} (expected '<field><op><value>' or "
                     f"'<field>:bit:<NAME>')")


def _numeric_filter_value(filt: str, value: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"filter '{filt}': value {value!r} is not numeric") from error
    if not math.isfinite(numeric):
        raise ValueError(f"filter '{filt}': value {value!r} must be finite")
    return numeric


def _filter_mask(filt: Optional[str], w: WindowBuffer, schema: SchemaSpec,
                 offset: int) -> Optional[np.ndarray]:
    if filt is None:
        return None
    parsed = _parse_filter(filt)
    field = parsed[1]
    fs = schema.fields.get(field)
    if fs is None:
        raise ValueError(f"filter references unknown field {field!r}")
    if parsed[0] == "bit":
        if fs.semantics != "bitmask":
            raise ValueError(f"filter '{filt}': field {field!r} is not a bitmask")
        name = parsed[2]
        if name not in fs.bits:
            raise ValueError(f"filter '{filt}': unknown bit {name!r} for {field!r}")
        grid = w.numeric_field(field)[..., 0]
        mask = _extract_bit(grid, fs.bits[name]) == 1.0
    else:  # cmp
        _, _, op, val = parsed
        if fs.kind == "categorical":
            if op != "==":
                raise ValueError(f"filter '{filt}': categorical field {field!r} supports only ==")
            codes, vocab = w.categorical_field(field)
            code = vocab.index(val) if val in vocab else -2  # -2: never matches
            mask = codes == code
        else:
            grid = w.numeric_field(field)[..., 0]
            fv = _numeric_filter_value(filt, val)
            ops = {">": grid > fv, ">=": grid >= fv, "<": grid < fv,
                   "<=": grid <= fv, "==": grid == fv}
            mask = ops[op]
    return mask[offset:] if offset else mask


def _apply_map(mapop: str, grid: np.ndarray, is_vec: bool, fs: FieldSpec,
               schema: SchemaSpec):
    """Return (grid, is_vec, t_offset)."""
    if mapop == "abs":
        return np.abs(grid), is_vec, 0
    if mapop.startswith("bit:"):
        if fs.semantics != "bitmask":
            raise ValueError(f"map '{mapop}' requires a bitmask field, got {fs.name!r}")
        name = mapop[4:]
        if name not in fs.bits:
            raise ValueError(f"map '{mapop}': unknown bit {name!r} for {fs.name!r}")
        return _extract_bit(grid, fs.bits[name]), False, 0
    if mapop == "delta":
        d = grid[1:] - grid[:-1]
        if fs.semantics == "angular":
            d = _wrap_deg(d)
        return d, is_vec, 1
    if mapop == "speed":
        if not (is_vec and fs.semantics == "position"):
            raise ValueError("map 'speed' requires a 'position' vector source")
        d = grid[1:] - grid[:-1]
        hz = schema.sample_hz
        if hz is None:
            raise ValueError("map 'speed' requires schema.sample_hz")
        speed = np.sqrt((d ** 2).sum(axis=-1)) * hz
        if fs.max_rate is not None:
            # Reject teleport/outlier spikes (a per-step rate above the field's declared
            # max) rather than let one dominate std/jerk/mean. NaN cells stay NaN.
            speed[speed > fs.max_rate] = np.nan
        return speed, False, 1
    if mapop == "angle_delta":
        if fs.semantics != "angular":
            raise ValueError(f"map 'angle_delta' requires an 'angular' field, got {fs.name!r}")
        d = _wrap_deg(grid[1:] - grid[:-1])
        if is_vec:
            return np.abs(d).sum(axis=-1), False, 1
        return np.abs(d), False, 1
    raise ValueError(f"unknown map {mapop!r}")


def _resolve_entity_source(rest: str, schema: SchemaSpec):
    """rest = '<field>' or '<field>.<dim>'. Return (fieldspec, dim_index|None)."""
    parts = rest.split(".")
    field = parts[0]
    fs = schema.fields.get(field)
    if fs is None:
        raise ValueError(f"unknown field {field!r} (declared: {sorted(schema.fields)})")
    if len(parts) == 1:
        return fs, None
    if len(parts) != 2:
        raise ValueError(f"source {rest!r}: at most one '.<dim>' component")
    dim = parts[1]
    if fs.kind != "vector":
        raise ValueError(f"source {rest!r}: only vector fields take a .<dim> component")
    if dim not in fs.dims:
        raise ValueError(f"source {rest!r}: dim {dim!r} not in {field!r} dims {list(fs.dims)}")
    return fs, fs.dims.index(dim)


def _validate_map(mapop: str, fs: FieldSpec, is_vec: bool, schema: SchemaSpec) -> bool:
    """Compile-time check of a map token against the field's semantics. Returns the
    resulting is_vec (whether the stream is still a vector after the map). Raises loudly."""
    if mapop == "abs" or mapop == "delta":
        return is_vec
    if mapop.startswith("bit:"):
        if fs.semantics != "bitmask":
            raise ValueError(f"map '{mapop}' requires a bitmask field, got {fs.name!r}")
        if mapop[4:] not in fs.bits:
            raise ValueError(f"map '{mapop}': unknown bit for {fs.name!r}")
        return False
    if mapop == "speed":
        if not (is_vec and fs.semantics == "position"):
            raise ValueError("map 'speed' requires a 'position' vector source")
        if schema.sample_hz is None:
            raise ValueError("map 'speed' requires schema.sample_hz")
        return False
    if mapop == "angle_delta":
        if fs.semantics != "angular":
            raise ValueError(f"map 'angle_delta' requires an 'angular' field, got {fs.name!r}")
        return False
    raise ValueError(f"unknown map {mapop!r}")


def _validate_filter(filt: str, schema: SchemaSpec) -> None:
    parsed = _parse_filter(filt)
    field = parsed[1]
    fs = schema.fields.get(field)
    if fs is None:
        raise ValueError(f"filter references unknown field {field!r}")
    if parsed[0] == "bit":
        if fs.semantics != "bitmask":
            raise ValueError(f"filter '{filt}': field {field!r} is not a bitmask")
        if parsed[2] not in fs.bits:
            raise ValueError(f"filter '{filt}': unknown bit {parsed[2]!r} for {field!r}")
    else:
        _, _, op, val = parsed
        if fs.kind == "categorical":
            if op != "==":
                raise ValueError(f"filter '{filt}': categorical field {field!r} supports only ==")
        else:
            if fs.kind == "vector":
                raise ValueError(
                    f"filter '{filt}': field {field!r} is a vector — comparison filters take "
                    f"scalar fields only (there is no .<dim> filter syntax)")
            _numeric_filter_value(filt, val)


def _compile_entity(rest: str, spec: dict, schema: SchemaSpec) -> Callable[[WindowBuffer], float]:
    fs, dim_idx = _resolve_entity_source(rest, schema)
    mapop = spec.get("map")
    filt = spec.get("filter")
    reduce = _validate_reduce(spec["reduce"])
    scope = spec.get("scope", "pooled")
    if scope not in _SCOPES:
        raise ValueError(f"unknown scope {scope!r} (expected {sorted(_SCOPES)})")

    # Spatial reduces consume the position plane directly (no map); a filter narrows which
    # (tick, entity) cells count as present.
    if reduce in _SPATIAL_REDUCES:
        if not (fs.kind == "vector" and fs.semantics == "position" and dim_idx is None):
            raise ValueError(f"reduce {reduce!r} requires a whole 'position' vector source")
        if mapop is not None:
            raise ValueError(f"spatial reduce {reduce!r} takes no map (got {mapop!r})")
        if filt is not None:
            _validate_filter(filt, schema)
        field = fs.name

        def fn_spatial(w: WindowBuffer) -> float:
            valid = w.valid
            if filt is not None:
                valid = valid & _filter_mask(filt, w, schema, 0)
            points = w.numeric_field(field)
            # Hull area is the established planar projection contract. The other spatial
            # reducers are dimension-agnostic and must retain every position component.
            if reduce == "hull_area":
                points = points[..., :2]
            return _spatial_reduce(reduce, points, valid)
        return fn_spatial

    field = fs.name
    is_cat = fs.kind == "categorical"

    # Compile-time validation (closed-grammar contract: unknown/ill-typed tokens fail at
    # load, not mid-run). Determine the post-map vector-ness here so a vector stream fed to
    # a scalar reduce is caught now.
    is_vec = fs.kind == "vector" and dim_idx is None
    if mapop is not None:
        if is_cat:
            raise ValueError(f"categorical source {fs.name!r} takes no map")
        is_vec = _validate_map(mapop, fs, is_vec, schema)
    if filt is not None:
        _validate_filter(filt, schema)
    if is_cat and reduce not in _CATEGORICAL_REDUCES:
        raise ValueError(f"categorical source {fs.name!r}: use {sorted(_CATEGORICAL_REDUCES)}")
    if not is_cat and is_vec:
        raise ValueError(
            f"reduce {reduce!r} needs a scalar stream; source {spec['source']!r} is a vector "
            f"— add a map (speed/angle_delta) or select a .<dim>")
    if not is_cat:                       # numeric scalar stream — reject categorical/ill-typed reduces now
        _validate_numeric_reduce(reduce)

    def fn(w: WindowBuffer) -> float:
        if is_cat:
            codes, _ = w.categorical_field(field)
            mask = _filter_mask(filt, w, schema, 0)
            return _reduce_categorical(reduce, codes, scope, mask)
        arr = w.numeric_field(field)
        if dim_idx is not None:
            grid, is_vec = arr[..., dim_idx], False
        elif fs.kind == "scalar":
            grid, is_vec = arr[..., 0], False
        else:
            grid, is_vec = arr, True
        offset = 0
        if mapop is not None:
            grid, is_vec, offset = _apply_map(mapop, grid, is_vec, fs, schema)
        if is_vec:
            raise ValueError(
                f"reduce {reduce!r} needs a scalar stream; source {spec['source']!r} is a "
                f"vector — add a map (speed/angle_delta) or select a .<dim>")
        mask = _filter_mask(filt, w, schema, offset)
        if reduce == "jerk":
            return _reduce_jerk(grid, mask)
        return _reduce_scalar(reduce, grid, scope, mask)
    return fn


def _compile_global(name: str, spec: dict, schema: SchemaSpec) -> Callable[[WindowBuffer], float]:
    if spec.get("map") or spec.get("filter"):
        raise ValueError(f"global source {name!r} takes no map/filter")
    if "scope" in spec:
        raise ValueError(f"global source {name!r} takes no scope (a global is a single stream)")
    reduce = _validate_numeric_reduce(spec["reduce"])   # a global is a 1-D scalar stream

    def fn(w: WindowBuffer) -> float:
        v = w.globals[name].astype(float)
        if reduce == "jerk":                     # diff first: a NaN gap breaks adjacency
            d = np.diff(v)
            d = d[~np.isnan(d)]
            return float(np.mean(np.abs(d))) if d.size else 0.0
        return _agg_1d(reduce, v[~np.isnan(v)])
    return fn


def _compile_event(stream: str, rest: str, spec: dict, schema: SchemaSpec) -> Callable[[WindowBuffer], float]:
    sfields = schema.event_streams[stream]["fields"]
    field = rest
    if field not in sfields:
        raise ValueError(f"unknown event field {stream}.{field!r}")
    if spec.get("map"):
        raise ValueError("event sources take no map")
    if "scope" in spec:
        raise ValueError("event sources take no scope")
    reduce = spec["reduce"]
    if reduce not in _CATEGORICAL_REDUCES and reduce != "rate":
        raise ValueError(f"event reduce must be rate/{sorted(_CATEGORICAL_REDUCES)}, got {reduce!r}")
    if sfields[field].kind == "vector" and reduce in {"nunique", "entropy"}:
        raise ValueError(
            f"event reduce {reduce!r} needs hashable scalar/categorical values; "
            f"source {stream}.{field!r} is a vector (use count/rate instead)")
    filt = spec.get("filter")
    want = None
    if filt is not None:
        if filt.startswith("=="):
            want = filt[2:]
        elif filt.startswith("eq:"):
            want = filt[3:]
        else:
            raise ValueError(f"event filter must be '==<value>', got {filt!r}")

    def matches(ev: dict) -> bool:
        v = ev.get(field)
        if v is None:                    # an event without this field is not a category value
            return False
        # The filter literal is a string; compare its str form too so a numeric event value
        # still matches ('==5' vs code=5) instead of silently counting zero.
        return want is None or v == want or str(v) == want

    def fn(w: WindowBuffer) -> float:
        per_tick = w.events.get(stream, [])
        if reduce == "rate":
            hits = sum(1 for evs in per_tick if any(matches(e) for e in evs))
            return hits / max(len(per_tick), 1)
        vals = [e.get(field) for evs in per_tick for e in evs if matches(e)]
        if reduce == "count":
            return float(len(vals))
        if reduce == "nunique":
            return float(len(set(vals)))
        from collections import Counter
        return _shannon_counts(list(Counter(vals).values()))
    return fn


_SPEC_KEYS = {"source", "reduce", "map", "filter", "scope"}


def compile_metric(spec: dict, schema: SchemaSpec) -> Callable[[WindowBuffer], float]:
    """Compile a metric spec to `fn(WindowBuffer) -> float` against `schema`. Raises
    ValueError on any unknown token (loud failure — no silent fallbacks)."""
    if "source" not in spec or "reduce" not in spec:
        raise ValueError(f"metric spec needs 'source' and 'reduce': {spec!r}")
    unknown = set(spec) - _SPEC_KEYS
    if unknown:                                   # a typo'd key must not drop a constraint
        raise ValueError(f"unknown metric spec key(s) {sorted(unknown)}; "
                         f"expected {sorted(_SPEC_KEYS)}")
    src = spec["source"]
    if "." not in src:
        if src in schema.globals:
            return _compile_global(src, spec, schema)
        if schema.entity_axis is None and src in schema.fields:
            return _compile_entity(src, spec, schema)
        expected = sorted(set(schema.globals) | (set(schema.fields) if schema.entity_axis is None
                                                 else set()))
        raise ValueError(f"unknown source {src!r} (declared unqualified sources: {expected})")
    ns, rest = src.split(".", 1)
    if schema.entity_axis and ns == schema.entity_axis:
        return _compile_entity(rest, spec, schema)
    if schema.entity_axis is None and ns in schema.fields:
        return _compile_entity(src, spec, schema)
    if ns in schema.event_streams:
        return _compile_event(ns, rest, spec, schema)
    raise ValueError(
        f"unknown source namespace {ns!r} in {src!r}; expected the entity axis "
        f"{schema.entity_axis!r} or an event stream {sorted(schema.event_streams)}")
