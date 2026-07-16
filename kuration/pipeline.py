"""Fluent pipeline facade — `input(...)` -> chained operators -> terminals.

`input(src)` sniffs the source (file/glob/dir/DataFrame/rows/ticks/columnar dict), infers a
`SchemaSpec` (via `kuration.infer`) unless one is given, and builds the right adapter — all
eagerly, so a bad source fails at `input()` time. Every operator returns a NEW `Pipeline`
(immutable chaining) and validates its arguments loudly against the schema and the declared
measures. Terminals compile the chain into the classic raw-strategy dict (`load_strategy`)
and run the existing engine: `collect()` -> `curate`, `output()` -> `write_dataset` (hours)
or a selected-segment write (count), `report()` -> `dataset_report`, `experiment()` ->
`run_experiment`. The full operator chain (`_ops`) lands in `config.json` under "pipeline",
so a written dataset records how it was asked for, not just what was resolved.

The factory intentionally shadows `builtins.input` inside this module.
"""
from __future__ import annotations

import json
import math
import os
import copy
from itertools import islice
from typing import Any, Callable, Optional

import numpy as np

from .adapters.base import get_adapter
from .adapters.hdf5 import HDF5Adapter
from .adapters.jsonl import JsonlAdapter
from .adapters.npz import NpzAdapter
from .adapters.parquet_dataset import ParquetDatasetAdapter
from .adapters.timeseries import TimeseriesAdapter
from .buffer import Tick
from .dsl.grammar import compile_metric
from .dsl.registry import register_compiled
from .schema import SchemaSpec
from .segment import Segmenter

__all__ = ["Pipeline", "input"]

_GLOB_CHARS = set("*?[")
_FILTER_TYPES = {"anomaly", "dedup"}
_OPTION_KEYS = {"workers", "on_error", "seed", "streaming", "resume", "sample_hz", "progress",
                "mode", "executor"}
_TABULAR_KINDS = {"timeseries", "jsonl", "json-rows", "parquet-dataset"}
_SUFFIX_CONTAINER = {".hdf5": "hdf5", ".h5": "hdf5", ".npz": "npz",
                     ".parquet": "parquet", ".json": "json"}

# predicate-op aliases (matched case-insensitively) -> canonical strategy op
_OP_ALIASES = {
    "<=": "<=", "le": "<=", "lte": "<=", "lessorequal": "<=", "leq": "<=",
    ">=": ">=", "ge": ">=", "gte": ">=", "greaterorequal": ">=", "geq": ">=",
    "<": "<", "lt": "<", "less": "<",
    ">": ">", "gt": ">", "greater": ">",
    "==": "==", "eq": "==", "equal": "==", "=": "==",
    "!=": "!=", "ne": "!=", "neq": "!=", "notequal": "!=",
}

# sample() method aliases (matched case-insensitively) -> canonical strategy method
_METHOD_ALIASES = {
    "equal": "uniform-bins", "uniform": "uniform-bins", "uniform-bins": "uniform-bins",
    "top": "top-k", "top-k": "top-k", "topk": "top-k",
    "target": "target-distribution", "target-distribution": "target-distribution",
    "cluster": "cluster-balanced", "cluster-balanced": "cluster-balanced",
    "facility": "facility-location", "facility-location": "facility-location",
    "weighted": "weighted", "k-center": "k-center", "dsir": "dsir",
    "rare": "rare", "typical": "typical", "filter": "filter",
}


# ======================================================================================
# provenance helpers
# ======================================================================================
def _jsonable(v):
    """JSON-safe view of an operator argument for the `_ops` provenance list; anything not
    JSON-native is repr()'d (truncated — provenance, not a data store)."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    r = repr(v)
    return r if len(r) <= 200 else r[:200] + "..."


def _describe_src(src) -> Any:
    """Compact provenance form of input()'s src (in-memory data is summarized, not embedded)."""
    if isinstance(src, str):
        return src
    if isinstance(src, (list, tuple)) and src and all(isinstance(x, str) for x in src):
        return list(src)
    return f"<in-memory {type(src).__name__}>"


# ======================================================================================
# schema helpers
# ======================================================================================
def _auto_qualify(tok: str, schema: SchemaSpec) -> str:
    """Resolve a bare measure token to a DSL source. Loud on anything unknown."""
    if "." in tok:
        return tok
    if tok in schema.fields:
        if not schema.entity_axis:
            raise ValueError(f"field {tok!r}: schema {schema.name!r} declares no entity_axis, "
                             f"so entity fields cannot be referenced")
        return f"{schema.entity_axis}.{tok}"
    if tok in schema.globals:
        return tok
    if tok in schema.event_streams:
        raise ValueError(
            f"{tok!r} is an event stream — name a field: '{tok}.<field>' "
            f"(fields: {sorted(schema.event_streams[tok]['fields'])})")
    raise ValueError(f"unknown metric source {tok!r}; schema fields: "
                     f"{sorted(schema.fields)}, globals: {sorted(schema.globals)}")


def _default_field_map(schema: SchemaSpec, *, tabular: bool) -> dict:
    """Identity column mapping for an explicit schema with no fields= given. Tabular sources
    address vector components as '<field>.<i>' columns; array stores map the whole array."""
    out: dict = {}
    for name, fs in schema.fields.items():
        if tabular and fs.kind == "vector":
            out[name] = [f"{name}.{i}" for i in range(len(fs.dims))]
        else:
            out[name] = name
    return out


# ======================================================================================
# in-memory adapters (module-level so they pickle; still not worker-rebuildable — the
# pipeline raises loudly on .options(workers>1) with in-memory data)
# ======================================================================================
def _expand_row(row: dict) -> dict:
    """Expand list/tuple/1-D-array cells into '<col>.<i>' keys (how vector fields are
    addressed by the inferred column mapping). Scalars pass through."""
    out = dict(row)
    for k, v in row.items():
        if isinstance(v, np.ndarray) and v.ndim == 1:
            v = v.tolist()
        if isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                out[f"{k}.{i}"] = x
    return out


class _RowsAdapter(TimeseriesAdapter):
    """In-memory {source_id -> [row dict, ...]} as curation sources."""
    in_memory = True

    def __init__(self, schema, *, data: dict, fields: dict, time_column=None,
                 entity_column=None, globals=None):
        super().__init__(schema, fields=fields, time_column=time_column,
                         entity_column=entity_column, globals=globals)
        self._data = data

    def ticks(self, source_uri):
        if source_uri not in self._data:
            raise ValueError(f"unknown in-memory source {source_uri!r} "
                             f"(have: {sorted(self._data)})")
        return self._rows_to_ticks([_expand_row(r) for r in self._data[source_uri]])


class _TickAdapter:
    """In-memory {source_id -> [Tick, ...]} as curation sources."""
    in_memory = True

    def __init__(self, schema, data: dict):
        self.schema = schema
        self._data = data

    def ticks(self, source_uri):
        if source_uri not in self._data:
            raise ValueError(f"unknown in-memory source {source_uri!r} "
                             f"(have: {sorted(self._data)})")
        return self._data[source_uri]


class _JsonRowsAdapter(TimeseriesAdapter):
    """Plain .json sources (array-of-rows or columnar dict), loaded lazily per source."""

    def ticks(self, source_uri):
        from .infer import load_rows
        return self._rows_to_ticks([_expand_row(r) for r in load_rows(source_uri)])


# ======================================================================================
# input() — the factory (intentionally shadows builtins.input in this module)
# ======================================================================================
def input(src=None, *, schema=None, adapter=None, sources=None, fields=None,
          time=None, entity=None, hz=None, kinds=None, sample=200, globals=None,
          **adapter_kw) -> "Pipeline":
    """Open a source as a `Pipeline`. Sniffs the source kind, infers the schema (unless
    `schema=` is given) from a sample, and builds the adapter — eagerly and loudly.

    Accepted `src`: file path / glob / list of paths / parquet-parts directory; a pandas
    DataFrame (duck-typed via .to_dict); list[dict] rows; list[Tick]; dict of columns
    (columnar in-memory); {source_id: rows|ticks} (multi-source in-memory). Escape hatch:
    `input(adapter=<InputAdapter or registry name>, sources=[...], schema=...)`.
    """
    op = {"op": "input",
          "args": [] if src is None else [_describe_src(src)],
          "kwargs": {k: _jsonable(v) for k, v in dict(
              schema=schema, adapter=adapter, sources=sources, fields=fields, time=time,
              entity=entity, hz=hz, kinds=kinds, sample=sample, globals=globals,
              **adapter_kw).items() if v is not None}}
    if isinstance(schema, dict):
        schema = SchemaSpec.from_dict(schema)

    # -- escape hatch: explicit adapter -----------------------------------------------
    if adapter is not None:
        if schema is None:
            raise ValueError("input(adapter=...) requires schema= (SchemaSpec or dict) — "
                             "there is no inference on the escape hatch")
        if isinstance(adapter, str):
            kw = dict(adapter_kw)
            if fields is not None:
                kw["fields"] = fields
            if globals is not None:
                kw["globals"] = globals
            adapter = get_adapter(adapter, schema=schema, **kw)
        srcs = list(sources) if sources is not None else (
            list(adapter.sources()) if hasattr(adapter, "sources") else [])
        if not srcs:
            raise ValueError("input(adapter=...) requires sources=[...] "
                             "(or an adapter exposing .sources())")
        return Pipeline(schema=schema, adapter=adapter, sources=srcs,
                        in_memory=bool(getattr(adapter, "in_memory", False)), ops=[op])

    if src is None:
        raise ValueError("input() needs a source (path/glob/list/DataFrame/rows/ticks/dict) "
                         "or the adapter= escape hatch")

    # -- pandas DataFrame (duck-typed; no pandas import) -------------------------------
    if not isinstance(src, (str, list, tuple, dict)) and hasattr(src, "to_dict"):
        src = src.to_dict("records")

    # -- file paths ---------------------------------------------------------------------
    if isinstance(src, str) or (isinstance(src, (list, tuple)) and src
                                and all(isinstance(x, str) for x in src)):
        return _input_files(src, op, schema=schema, fields=fields, time=time, entity=entity,
                            hz=hz, kinds=kinds, sample=sample, globals=globals, **adapter_kw)

    if adapter_kw:
        raise ValueError(f"unknown input() argument(s) {sorted(adapter_kw)} — adapter "
                         f"keywords only apply to file-backed sources")

    # -- in-memory lists ----------------------------------------------------------------
    if isinstance(src, (list, tuple)):
        if not src:
            raise ValueError("input(): empty list — nothing to curate")
        if all(isinstance(x, Tick) for x in src):
            return _input_ticks({"mem": list(src)}, op, schema=schema, time=time,
                                entity=entity, hz=hz, kinds=kinds, sample=sample)
        if all(isinstance(x, dict) for x in src):
            return _input_rows({"mem": list(src)}, op, schema=schema, fields=fields,
                               time=time, entity=entity, hz=hz, kinds=kinds, sample=sample,
                               globals=globals)
        raise ValueError("input(): a list must be all paths (str), all row dicts, or all "
                         f"Ticks; got element types {sorted({type(x).__name__ for x in src})}")

    # -- in-memory dicts: columnar vs multi-source --------------------------------------
    if isinstance(src, dict):
        if not src:
            raise ValueError("input(): empty dict — nothing to curate")

        def _is_rows(v):
            return isinstance(v, (list, tuple)) and len(v) > 0 and \
                all(isinstance(r, dict) for r in v)

        def _is_ticks(v):
            return isinstance(v, (list, tuple)) and len(v) > 0 and \
                all(isinstance(r, Tick) for r in v)

        def _is_column(v):
            if isinstance(v, np.ndarray):
                return True
            return isinstance(v, (list, tuple)) and len(v) > 0 and \
                all(not isinstance(x, (dict, Tick)) for x in v)

        vals = list(src.values())
        if all(_is_ticks(v) for v in vals):
            return _input_ticks({str(k): list(v) for k, v in src.items()}, op, schema=schema,
                                time=time, entity=entity, hz=hz, kinds=kinds, sample=sample)
        if all(_is_rows(v) for v in vals):
            return _input_rows({str(k): list(v) for k, v in src.items()}, op, schema=schema,
                               fields=fields, time=time, entity=entity, hz=hz, kinds=kinds,
                               sample=sample, globals=globals)
        if all(_is_column(v) for v in vals):
            lens = {len(v) for v in vals}
            if len(lens) != 1:
                raise ValueError(f"input(): columnar dict has mismatched column lengths "
                                 f"{sorted(lens)}")
            n = lens.pop()

            def _cell(x):
                if isinstance(x, np.ndarray):
                    return x.tolist()
                if isinstance(x, np.generic):
                    return x.item()
                return x
            rows = [{k: _cell(v[i]) for k, v in src.items()} for i in range(n)]
            return _input_rows({"mem": rows}, op, schema=schema, fields=fields, time=time,
                               entity=entity, hz=hz, kinds=kinds, sample=sample,
                               globals=globals)
        raise ValueError(
            "input(): ambiguous dict — values must be ALL row-lists / Tick-lists "
            "({source_id: rows|ticks} multi-source) or ALL scalar columns (columnar); "
            f"got value types {sorted({type(v).__name__ for v in vals})} with mixed contents")

    raise ValueError(f"input(): unsupported source type {type(src).__name__}")


def _infer_or_explicit(sample_data, *, schema, fields, time, entity, hz, kinds,
                       tabular: bool):
    """Return (schema, field_map, time_column, entity_column). Explicit schema skips
    inference entirely; fields= overrides the inferred column mapping."""
    if schema is not None:
        fmap = dict(fields) if fields is not None else _default_field_map(schema,
                                                                          tabular=tabular)
        return schema, fmap, time, entity
    from .infer import infer_schema
    inferred, hints = infer_schema(sample_data, name="inferred", time=time, entity=entity,
                                   hz=hz, kinds=kinds)
    fmap = dict(fields) if fields is not None else dict(hints["fields"])
    tc = time if time is not None else hints.get("time_column")
    ec = entity if entity is not None else hints.get("entity_column")
    return inferred, fmap, tc, ec


def _expand_paths(src) -> tuple[list[str], str]:
    """Resolve a path/glob/list-of-paths to (sorted sources, sniffed kind). Loud on an empty
    glob or mixed kinds."""
    from .infer import sniff
    if isinstance(src, str):
        if any(c in src for c in _GLOB_CHARS):
            from .io import glob as _glob
            paths = _glob(src)
            if not paths:
                raise ValueError(f"input(): glob {src!r} matched no files")
        else:
            return [src], sniff(src)
    else:
        paths = list(src)
    kinds_ = {p: sniff(p) for p in paths}
    if len(set(kinds_.values())) > 1:
        raise ValueError(f"input(): mixed source kinds {kinds_} — all sources must be the "
                         f"same kind")
    return paths, next(iter(kinds_.values()))


def _input_files(src, op, *, schema, fields, time, entity, hz, kinds, sample, globals,
                 **adapter_kw) -> "Pipeline":
    paths, kind = _expand_paths(src)
    if kind == "video":
        if schema is not None or fields is not None or time is not None or entity is not None:
            raise ValueError("video input probes its media schema; schema/fields/time/entity "
                             "overrides are not supported (use input(adapter=...) instead)")
        from .media import VideoAdapter
        video_ad = VideoAdapter(paths, **adapter_kw)
        return Pipeline(schema=video_ad.schema, adapter=video_ad, sources=paths,
                        in_memory=False, ops=[op])
    sample_data = None
    if schema is None:
        from .infer import sample_source
        _k, sample_data = sample_source(paths[0], n=sample)
    schema, fmap, tc, ec = _infer_or_explicit(
        sample_data, schema=schema, fields=fields, time=time, entity=entity, hz=hz,
        kinds=kinds, tabular=kind in _TABULAR_KINDS)
    ad: Any
    if kind == "timeseries":
        ad = TimeseriesAdapter(schema, fields=fmap, time_column=tc, entity_column=ec,
                               globals=globals, **adapter_kw)
    elif kind == "jsonl":
        ad = JsonlAdapter(schema, fields=fmap, time_column=tc, entity_column=ec,
                          globals=globals, **adapter_kw)
    elif kind == "json-rows":
        ad = _JsonRowsAdapter(schema, fields=fmap, time_column=tc, entity_column=ec,
                              globals=globals, **adapter_kw)
    elif kind == "npz":
        ad = NpzAdapter(schema, fields=fmap, globals=globals, **adapter_kw)
    elif kind == "hdf5":
        ad = HDF5Adapter(schema, fields=fmap, globals=globals, **adapter_kw)
    elif kind == "parquet-dataset":
        ad = ParquetDatasetAdapter(schema, fields=fmap, path=paths[0], time_column=tc,
                                   entity_column=ec, globals=globals, **adapter_kw)
        paths = ad.sources()
    else:
        raise ValueError(f"input(): unsupported sniffed source kind {kind!r}")
    return Pipeline(schema=schema, adapter=ad, sources=paths, in_memory=False, ops=[op])


def _input_rows(data: dict, op, *, schema, fields, time, entity, hz, kinds, sample,
                globals) -> "Pipeline":
    first = next(iter(data.values()))
    schema, fmap, tc, ec = _infer_or_explicit(
        first[:sample], schema=schema, fields=fields, time=time, entity=entity, hz=hz,
        kinds=kinds, tabular=True)
    ad = _RowsAdapter(schema, data=data, fields=fmap, time_column=tc, entity_column=ec,
                      globals=globals)
    return Pipeline(schema=schema, adapter=ad, sources=list(data), in_memory=True, ops=[op])


def _input_ticks(data: dict, op, *, schema, time, entity, hz, kinds, sample) -> "Pipeline":
    if schema is None:
        from .infer import infer_schema
        first = next(iter(data.values()))
        schema, _hints = infer_schema(first[:sample], name="inferred", time=time,
                                      entity=entity, hz=hz, kinds=kinds)
    ad = _TickAdapter(schema, data)
    return Pipeline(schema=schema, adapter=ad, sources=list(data), in_memory=True, ops=[op])


# ======================================================================================
# Pipeline
# ======================================================================================
class Pipeline:
    """Immutable fluent curation pipeline. Build via `kuration.pipeline.input(...)`;
    every operator returns a new Pipeline."""

    def __init__(self, *, schema: SchemaSpec, adapter, sources: list, in_memory: bool,
                 ops: list):
        self.schema = schema
        self.adapter = adapter
        self.sources = list(sources)
        self._in_memory = in_memory
        self._ops = list(ops)
        self._segmenter: Optional[Segmenter] = None
        self._window_desc = {"window": 64, "stride": 64, "unit": "steps"}
        self._define: dict = {}          # measure name -> DSL spec (insertion-ordered)
        self._callables: dict = {}       # measure name -> fn(WindowBuffer) -> float
        self._predicates: list = []      # [{"metric","op","value"}]
        self._prepass: list = []         # [{"type": "anomaly"|"dedup", ...}]
        self._axes: list = []            # [(metric, bins)]
        self._tags: list = []            # tag sources for the tags block
        self._tag_coverage: Optional[dict] = None
        self._scorers: list = []         # [{"name", "as", ...params}]
        self._transforms: list = []      # resolved BatchTransform calls
        self._method: Optional[str] = None
        self._weights: Optional[dict] = None
        self._normalize: Optional[str] = None
        self._target: Any = None
        self._group_cap: Any = None
        self._method_params: dict = {}
        self._feature_space: Optional[list] = None
        self._n: Optional[int] = None
        self._hours: Optional[float] = None
        self._test_hours: float = 0.0
        self._test_hours_explicit = False
        self._split_test: Optional[float] = None
        self._group_of: Optional[Callable] = None
        self._options: dict[str, Any] = {
            "workers": 1, "on_error": "skip", "seed": 0, "streaming": False,
            "resume": False, "sample_hz": None, "progress": None, "executor": "local"}
        self._mode: Optional[str] = None
        self._explicit_options: set[str] = set()
        from .registry import default_registry
        self._registry = default_registry

    # ---- immutability plumbing -------------------------------------------------------
    def _clone(self) -> "Pipeline":
        p = object.__new__(Pipeline)
        p.__dict__.update(self.__dict__)
        for attr in ("_ops", "_define", "_callables", "_predicates", "_prepass", "_axes",
                     "_tags", "_scorers", "_transforms", "_method_params", "_options",
                     "_window_desc"):
            cur = getattr(self, attr)
            setattr(p, attr, type(cur)(cur))
        p._explicit_options = set(self._explicit_options)
        return p

    def _next(self, opname: str, *args, **kwargs) -> "Pipeline":
        p = self._clone()
        p._ops.append({"op": opname, "args": [_jsonable(a) for a in args],
                       "kwargs": {k: _jsonable(v) for k, v in kwargs.items()
                                  if v is not None}})
        return p

    def _metric_names(self) -> set:
        return (set(self._define) | set(self._callables) |
                {s["as"] for s in self._scorers} |
                {o for t in self._transforms for o in t["outputs"]})

    def using(self, registry) -> "Pipeline":
        """Return a pipeline resolving extension names through an explicit overlay registry."""
        from .registry import Registry
        if not isinstance(registry, Registry):
            raise ValueError("using() needs a Registry")
        p = self._next("using", "<registry>")
        p._registry = registry
        return p

    def apply(self, name: str, **params) -> "Pipeline":
        """Expand a registered compositional operator over this pipeline.

        The operator receives a clone and must return a derived ``Pipeline`` containing at
        least one additional ordinary operator. Its descriptor validates parameters before
        expansion. Provenance records the named call, descriptor, and nested expansion.
        """
        from .plugins import load_plugins
        from .registry import validate_parameters
        if not isinstance(name, str) or not name:
            raise ValueError("apply() needs a non-empty registered operator name")
        if name not in self._registry.names("operator"):
            load_plugins()
        descriptor = self._registry.descriptor("operator", name)
        validate_parameters(descriptor, params)
        operator = self._registry.get("operator", name)
        result = operator(self._clone(), **params)
        if not isinstance(result, Pipeline):
            raise ValueError(
                f"operator {name!r} must return a Pipeline, got {type(result).__name__}")
        prefix = result._ops[:len(self._ops)]
        if prefix != self._ops:
            raise ValueError(
                f"operator {name!r} must derive from the Pipeline it receives")
        expansion = result._ops[len(self._ops):]
        if not expansion:
            raise ValueError(f"operator {name!r} must add at least one pipeline operator")
        p = result._clone()
        p._ops = list(self._ops) + [{
            "op": "apply",
            "args": [name],
            "kwargs": {k: _jsonable(v) for k, v in params.items()},
            "component": descriptor.to_dict(),
            "expansion": expansion,
        }]
        return p

    def _with_schema(self, schema: SchemaSpec, opname: str, *args, **kwargs) -> "Pipeline":
        p = self._next(opname, *args, **kwargs)
        p.schema = schema
        p.adapter = copy.copy(self.adapter)
        p.adapter.schema = schema
        return p

    def semantic(self, field: str, semantics: str, *, dims=None, bits=None,
                 max_rate=None) -> "Pipeline":
        """Refine inferred fields with a load-bearing core semantic."""
        if semantics not in {"position", "angular", "bitmask"}:
            raise ValueError("semantic kind must be position, angular, or bitmask")
        raw = self.schema.to_dict()
        fields = raw["fields"]
        if field in fields:
            if fields[field].get("semantics") is not None:
                raise ValueError(f"field {field!r} already has a semantic hint")
            fd = dict(fields[field])
            if dims is not None:
                if fd["kind"] != "vector":
                    raise ValueError(f"dims can only refine an existing vector field, got {field!r}")
                if len(dims) != len(fd.get("dims") or []):
                    raise ValueError(
                        f"field {field!r} has {len(fd.get('dims') or [])} vector components; "
                        f"semantic dims must preserve that width (got {len(dims)})")
                fd["dims"] = list(dims)
        elif semantics == "position" and dims:
            missing = [d for d in dims if d not in fields]
            if missing:
                raise ValueError(f"position {field!r} references unknown component fields {missing}")
            bad = [d for d in dims if fields[d]["kind"] != "scalar"]
            if bad:
                raise ValueError(f"position components must be scalar fields, got {bad}")
            fd = {"kind": "vector", "dims": list(dims)}
            for d in dims:
                fields.pop(d)
            if hasattr(self.adapter, "fields"):
                mapped = []
                for d in dims:
                    value = self.adapter.fields[d]
                    if isinstance(value, (list, tuple)):
                        raise ValueError(f"position component {d!r} is already vector-mapped")
                    mapped.append(value)
        else:
            raise ValueError(f"semantic field {field!r} is not in schema")
        fd["semantics"] = semantics
        if bits is not None:
            fd["bits"] = dict(bits)
        if max_rate is not None:
            fd["max_rate"] = max_rate
        fields[field] = fd
        schema = SchemaSpec.from_dict(raw)
        p = self._with_schema(schema, "semantic", field, semantics, dims=dims, bits=bits,
                              max_rate=max_rate)
        if field not in self.schema.fields and hasattr(p.adapter, "fields"):
            p.adapter.fields = {k: v for k, v in p.adapter.fields.items() if k not in set(dims)}
            p.adapter.fields[field] = mapped
        return p

    def as_position(self, field: str, *, dims=None, max_rate=None) -> "Pipeline":
        return self.semantic(field, "position", dims=dims, max_rate=max_rate)

    def as_angular(self, field: str, *, max_rate=None) -> "Pipeline":
        return self.semantic(field, "angular", max_rate=max_rate)

    def as_bitmask(self, field: str, *, bits) -> "Pipeline":
        return self.semantic(field, "bitmask", bits=bits)

    def at_rate(self, hz: float) -> "Pipeline":
        if self.schema.sample_hz is not None:
            raise ValueError(f"schema already has sample_hz={self.schema.sample_hz}; repeated hints fail")
        raw = self.schema.to_dict()
        raw["sample_hz"] = hz
        return self._with_schema(SchemaSpec.from_dict(raw), "at_rate", hz)

    # ---- operators ---------------------------------------------------------------------
    def window(self, window, stride=None, unit: str = "steps") -> "Pipeline":
        """Set the segment window (and stride, default = window) in 'steps' or 'seconds'."""
        p = self._next("window", window, stride=stride, unit=unit)
        p._segmenter = Segmenter(
            window, stride if stride is not None else window, unit=unit,
            sample_hz=self.schema.sample_hz if unit == "seconds" else None)
        p._window_desc = {"window": window,
                          "stride": stride if stride is not None else window, "unit": unit}
        return p

    def join_video(self, videos, *, column: str = "frame", on: str = "timestamp",
                   tolerance=None, mismatch: str = "error", provider=None) -> "Pipeline":
        """Attach a lazily decoded video media column to every ordinary source."""
        if not isinstance(column, str) or not column:
            raise ValueError("join_video column must be a non-empty string")
        if callable(videos):
            mapping = {src: videos(src) for src in self.sources}
        elif isinstance(videos, dict):
            mapping = {src: videos[src] for src in self.sources if src in videos}
        else:
            raise ValueError("join_video videos= must be a {source: video} mapping or callable")
        missing = [src for src in self.sources if src not in mapping]
        if missing:
            raise ValueError(f"join_video has no video for source(s) {missing}")
        if any(not isinstance(uri, str) or not uri for uri in mapping.values()):
            raise ValueError("join_video mappings must resolve to non-empty video URI strings")
        from .media import JoinedMediaAdapter
        p = self._next("join_video", videos=mapping, column=column, on=on,
                       tolerance=tolerance, mismatch=mismatch)
        p.adapter = JoinedMediaAdapter(self.adapter, mapping, provider=provider, column=column,
                                       on=on, tolerance=tolerance, mismatch=mismatch)
        p.schema = p.adapter.schema
        return p

    def transform(self, name: str, *, using=None, inputs=None, batch_size=None,
                  device="cpu", **params) -> "Pipeline":
        """Add a registered bounded transform that produces scalar segment features."""
        from .transforms import resolve_transform
        call = resolve_transform(name, using=using, inputs=inputs, batch_size=batch_size,
                                 device=device, params=params, schema=self.schema,
                                 registry=self._registry)
        collisions = set(call["outputs"]) & self._metric_names()
        if collisions:
            raise ValueError(f"transform output(s) collide with existing features {sorted(collisions)}")
        p = self._next("transform", name, using=using, inputs=inputs,
                       batch_size=batch_size, device=device, **params)
        p._transforms.append(call)
        return p

    def measure(self, name: str, spec=None, *, of=None, map=None, filter=None,
                scope=None) -> "Pipeline":
        """Declare a per-segment metric. Forms:
        measure("hp")                      -> mean of the (auto-qualified) field
        measure("x_p95", "p95", of="x")    -> reduce string + of/map/filter/scope kwargs
        measure(name, {full DSL dict})     -> passed through
        measure(name, callable)            -> registered Python metric (escape hatch)
        """
        if name in self._define or name in self._callables:
            raise ValueError(f"measure {name!r} already declared — names must be unique")
        p = self._next("measure", name, spec, of=of, map=map, filter=filter, scope=scope)
        if callable(spec):
            if of is not None or map is not None or filter is not None or scope is not None:
                raise ValueError(f"measure {name!r}: a callable metric takes no "
                                 f"of/map/filter/scope kwargs")
            p._callables[name] = spec
            return p
        if isinstance(spec, dict):
            if of is not None or map is not None or filter is not None or scope is not None:
                raise ValueError(f"measure {name!r}: a full DSL dict takes no extra kwargs "
                                 f"— put map/filter/scope in the dict")
            mspec = dict(spec)
        elif spec is None or isinstance(spec, str):
            mspec = {"source": _auto_qualify(of if of is not None else name, self.schema),
                     "reduce": spec if isinstance(spec, str) else "mean"}
            if map is not None:
                mspec["map"] = map
            if filter is not None:
                mspec["filter"] = filter
            if scope is not None:
                mspec["scope"] = scope
        else:
            raise ValueError(f"measure {name!r}: spec must be a reduce string, a DSL dict, "
                             f"or a callable (got {type(spec).__name__})")
        compile_metric(mspec, self.schema)   # eager, loud
        p._define[name] = mspec
        return p

    def filter(self, what, op=None, value=None, **kw) -> "Pipeline":
        """`.filter("anomaly"/"dedup", **knobs)` adds a pre-pass filter; `.filter(metric,
        op, value)` adds a universal predicate (applies under every selection method)."""
        is_registered_filter = (isinstance(what, str) and
                                what in self._registry.names("filter"))
        if isinstance(what, str) and (what in _FILTER_TYPES or is_registered_filter) \
                and op is None and value is None:
            entry = {"type": what, **kw}
            from .strategy import _resolve_filters
            _resolve_filters([entry], self._registry)  # eager knob/descriptor validation
            unknown_features = sorted(set(entry.get("feature_space") or [])
                                      - self._metric_names())
            if unknown_features:
                raise ValueError(f"filter {what!r} references undeclared metrics "
                                 f"{unknown_features}; known: {sorted(self._metric_names())}")
            p = self._next("filter", what, **kw)
            p._prepass.append(entry)
            return p
        if kw:
            raise ValueError(f"filter({what!r}, ...): keyword knobs only apply to the "
                             f"{sorted(_FILTER_TYPES)} pre-pass filters (got {sorted(kw)})")
        if op is None or value is None:
            raise ValueError("predicate filter needs .filter(metric, op, value)")
        known = self._metric_names()
        if what not in known:
            raise ValueError(f"filter metric {what!r} is not a declared measure, callable "
                             f"metric, or scorer output (known: {sorted(known)})")
        canon = _OP_ALIASES.get(str(op).lower())
        if canon is None:
            raise ValueError(
                f"unknown filter op {op!r}; canonical ops {sorted(set(_OP_ALIASES.values()))}"
                f", aliases {sorted(k for k in _OP_ALIASES if k.isalpha())}")
        if isinstance(value, bool):
            raise ValueError(f"filter predicate value must be a finite number, got {value!r}")
        try:
            value = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"filter predicate value must be a finite number, got {value!r}") from e
        if not math.isfinite(value):
            raise ValueError(f"filter predicate value must be a finite number, got {value!r}")
        p = self._next("filter", what, op, value)
        p._predicates.append({"metric": what, "op": canon, "value": value})
        return p

    def organize(self, metric: str, *args, bins=None) -> "Pipeline":
        """Compatibility alias for :meth:`bin`."""
        return self._add_bin("organize", metric, *args, bins=bins)

    def bin(self, metric: str, *args, bins=None) -> "Pipeline":
        """Add a quantile-bin axis over a declared metric."""
        return self._add_bin("bin", metric, *args, bins=bins)

    def _add_bin(self, operator: str, metric: str, *args, bins=None) -> "Pipeline":
        known = self._metric_names()
        if metric not in known:
            raise ValueError(f"{operator} metric {metric!r} is not a declared measure, "
                             f"callable metric, or scorer output (known: {sorted(known)})")
        if any(metric == m for m, _ in self._axes):
            raise ValueError(f"duplicate bin axis {metric!r}")
        if args:
            if bins is not None:
                raise ValueError(f"{operator}(): pass bins positionally or as bins=, not both")
            if len(args) == 1 and isinstance(args[0], int):
                bins = args[0]
            elif len(args) == 2 and args[0] == "bin":
                bins = args[1]
            else:
                raise ValueError(f"{operator}(): unsupported positional form {args!r}; use "
                                 f"{operator}(metric, bins=n)")
        bins = bins if bins is not None else 4
        if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
            raise ValueError(f"{operator}(): bins must be a positive integer, got {bins!r}")
        p = self._next(operator, metric, bins=bins)
        p._axes.append((metric, bins))
        return p

    def score(self, name: str, out: Optional[str] = None, **params) -> "Pipeline":
        """Add a whole-pool SegmentScorer; its output feature is `out` (default: name)."""
        from .scorers import get_scorer, has_scorer
        if name not in self._registry.names("scorer") and not has_scorer(name):
            get_scorer(name)                 # raises the canonical loud message
        out_name = out or name
        if out_name in self._metric_names():
            raise ValueError(f"scorer output {out_name!r} collides with a declared metric")
        p = self._next("score", name, out=out, **params)
        p._scorers.append({"name": name, "as": out_name, **params})
        return p

    def tag(self, *sources: str, min_frac: Optional[float] = None) -> "Pipeline":
        """Derive segment tags from categorical fields / 'stream.field' event sources;
        `min_frac` turns on tag-coverage at that floor fraction."""
        if not sources:
            raise ValueError("tag() needs at least one source (a categorical field or "
                             "'stream.field')")
        from .tags import compile_tagger
        compile_tagger({"from": list(sources)}, self.schema)   # eager, loud
        if min_frac is not None:
            if (not isinstance(min_frac, (int, float)) or isinstance(min_frac, bool)
                    or not math.isfinite(min_frac) or not 0 <= min_frac <= 1):
                raise ValueError("tag(min_frac=...) must be a finite number in [0, 1]")
        p = self._next("tag", *sources, min_frac=min_frac)
        p._tags = list(dict.fromkeys(list(self._tags) + list(sources)))
        if min_frac is not None:
            p._tag_coverage = {"min_tag_frac": float(min_frac)}
        return p

    def split(self, test: float = 0.2) -> "Pipeline":
        """Hold out a source-level test split: count path -> test_frac; hours path ->
        test share of the hours budget (unless test_hours was set explicitly)."""
        test = float(test)
        if not (0.0 <= test < 1.0):
            raise ValueError(f"split(test=...) must be in [0, 1), got {test}")
        p = self._next("split", test=test)
        p._split_test = test
        return p

    def group(self, fn: Callable) -> "Pipeline":
        """Set the source grouping key (group cap + split strata): fn(source_id) -> key."""
        if not callable(fn):
            raise ValueError("group() takes a callable source_id -> group key")
        p = self._next("group", fn)
        p._group_of = fn
        return p

    def options(self, **kw) -> "Pipeline":
        """Execution options: workers, executor ('local'|'ray'), on_error, seed, streaming,
        resume, sample_hz, and progress callback. Unknown options are loud."""
        unknown = set(kw) - _OPTION_KEYS
        if unknown:
            raise ValueError(f"unknown option(s) {sorted(unknown)}; "
                             f"expected {sorted(_OPTION_KEYS)}")
        mode = kw.pop("mode", None)
        if mode is not None and mode not in {"explore", "strict"}:
            raise ValueError("mode must be 'explore' or 'strict'")
        if "on_error" in kw and kw["on_error"] not in ("skip", "raise"):
            raise ValueError(f"on_error must be 'skip' or 'raise', got {kw['on_error']!r}")
        if "workers" in kw and (not isinstance(kw["workers"], int)
                                or isinstance(kw["workers"], bool) or kw["workers"] < 1):
            raise ValueError("workers must be a positive integer")
        if "seed" in kw and (not isinstance(kw["seed"], int) or isinstance(kw["seed"], bool)):
            raise ValueError("seed must be an integer")
        for flag in ("streaming", "resume"):
            if flag in kw and not isinstance(kw[flag], bool):
                raise ValueError(f"{flag} must be true or false")
        if "sample_hz" in kw and kw["sample_hz"] is not None:
            hz = kw["sample_hz"]
            if (not isinstance(hz, (int, float)) or isinstance(hz, bool)
                    or not math.isfinite(hz) or hz <= 0):
                raise ValueError("sample_hz must be a finite positive number")
        if "progress" in kw and kw["progress"] is not None and not callable(kw["progress"]):
            raise ValueError("progress must be callable or None")
        if "executor" in kw and kw["executor"] not in {"local", "ray"}:
            raise ValueError("executor must be 'local' or 'ray'")
        p = self._next("options", mode=mode, **kw)
        if mode is not None:
            p._mode = mode
            defaults = {"on_error": "raise"} if mode == "strict" else {"on_error": "skip"}
            for key, value in defaults.items():
                if key not in self._explicit_options and key not in kw:
                    p._options[key] = value
        p._options.update(kw)
        p._explicit_options.update(kw)
        return p

    def curate(self, method: Optional[str] = None, *, n: Optional[int] = None,
               hours: Optional[float] = None, test_hours: Optional[float] = None,
               weights: Optional[dict] = None, target=None, normalize: Optional[str] = None,
               group_cap=None, **method_params) -> "Pipeline":
        """Configure selection. Exactly one of n= (count path -> collect/output) or
        hours= (budget path -> output). Method defaults to uniform-bins when axes exist,
        top-k when weights are given; otherwise it must be named."""
        if (n is None) == (hours is None):
            raise ValueError("curate() needs exactly one of n= (count) or hours= (budget)")
        if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n < 1):
            raise ValueError("curate(n=...) needs a positive integer")
        if hours is not None and (not isinstance(hours, (int, float))
                                  or isinstance(hours, bool) or not math.isfinite(hours)
                                  or hours <= 0):
            raise ValueError("curate(hours=...) needs a finite positive number")
        if test_hours is not None and (not isinstance(test_hours, (int, float))
                                       or isinstance(test_hours, bool)
                                       or not math.isfinite(test_hours) or test_hours < 0):
            raise ValueError("test_hours must be a finite non-negative number")
        known_metrics = self._metric_names()
        if weights is not None:
            if not isinstance(weights, dict):
                raise ValueError("curate(weights=...) must be a metric-to-weight dict")
            unknown_weights = sorted(set(weights) - known_metrics)
            if unknown_weights:
                raise ValueError(f"curate weights reference undeclared metrics {unknown_weights}; "
                                 f"known: {sorted(known_metrics)}")
            for metric, weight in weights.items():
                if (not isinstance(weight, (int, float)) or isinstance(weight, bool)
                        or not math.isfinite(weight)):
                    raise ValueError(f"curate weight for {metric!r} must be finite")
        if normalize is not None and normalize not in {"quantile", "minmax", "zscore"}:
            raise ValueError("curate normalize must be quantile, minmax, or zscore")
        if isinstance(target, dict):
            target_metrics: set[str] = set()
            for key in ("metric", "anchor_metric"):
                if key not in target:
                    continue
                metric = target[key]
                if not isinstance(metric, str):
                    raise ValueError("curate target metric references must be strings")
                target_metrics.add(metric)
            unknown_target = sorted(target_metrics - known_metrics)
            if unknown_target:
                raise ValueError(f"curate target references undeclared metrics {unknown_target}; "
                                 f"known: {sorted(known_metrics)}")
        fs = method_params.get("feature_space")
        if fs is not None:
            if not isinstance(fs, (list, tuple)) or not all(isinstance(x, str) for x in fs):
                raise ValueError("curate feature_space must be a list of metric names")
            unknown_fs = sorted(set(fs) - known_metrics)
            if unknown_fs:
                raise ValueError(f"curate feature_space references undeclared metrics "
                                 f"{unknown_fs}; known: {sorted(known_metrics)}")
        if method is not None:
            canon = _METHOD_ALIASES.get(str(method).lower())
            if canon is None:
                from .plugins import load_plugins
                load_plugins()
                if method not in self._registry.names("method"):
                    raise ValueError(f"unknown curation method {method!r}; registered: "
                                     f"{self._registry.names('method')}")
                canon = method
        elif self._axes:
            canon = "uniform-bins"
        elif weights:
            canon = "top-k"
        else:
            raise ValueError("curate(): no selection method resolvable — add .organize(...) "
                             "axes, pass weights=, or name a method")
        p = self._next("curate", method, n=n, hours=hours, test_hours=test_hours,
                       weights=weights, target=target, normalize=normalize,
                       group_cap=group_cap, **method_params)
        p._method = canon
        if weights is not None:
            p._weights = {k: float(v) for k, v in weights.items()}
        if target is not None:
            p._target = target
        if normalize is not None:
            p._normalize = normalize
        if group_cap is not None:
            p._group_cap = group_cap
        fs = method_params.pop("feature_space", None)
        if fs is not None:
            p._feature_space = list(fs)
        p._method_params = dict(method_params)
        p._n, p._hours = n, hours
        if test_hours is not None:
            p._test_hours = float(test_hours)
            p._test_hours_explicit = True
        # Validate the fully assembled selection now, while the operator is being added.
        # Metric DSL entries were already compiled by measure(); _resolve covers method-
        # specific knobs, targets, normalization, caps, predicates, filters, and scorers
        # without touching data or registering callable metrics globally.
        from .strategy import _resolve
        _resolve(p._raw_strategy(), p._registry)
        return p

    def sample(self, method: Optional[str] = None, *, n: Optional[int] = None,
               hours: Optional[float] = None, test_hours: Optional[float] = None,
               weights: Optional[dict] = None, target=None, normalize: Optional[str] = None,
               group_cap=None, **method_params) -> "Pipeline":
        """Configure selection; preferred fluent alias for :meth:`curate`."""
        p = self.curate(method, n=n, hours=hours, test_hours=test_hours, weights=weights,
                        target=target, normalize=normalize, group_cap=group_cap,
                        **method_params)
        p._ops[-1] = {**p._ops[-1], "op": "sample"}
        return p

    # ---- compilation -------------------------------------------------------------------
    def _raw_strategy(self, *, require_method: bool = True) -> dict:
        method = self._method
        if method is None:
            if self._axes:
                method = "uniform-bins"
            elif self._weights:
                method = "top-k"
            elif require_method:
                raise ValueError("no selection method resolvable — add .organize(...) axes, "
                                 "pass weights= to .curate(), or name a method")
        d: dict = {"name": "pipeline", "define": dict(self._define)}
        if self._feature_space:
            d["feature_space"] = list(self._feature_space)
        if self._prepass:
            d["filters"] = [dict(f) for f in self._prepass]
        if self._tags:
            d["tags"] = {"from": list(self._tags)}
        if self._scorers:
            d["scorers"] = [dict(s) for s in self._scorers]
        if self._axes:
            d["binning"] = {"axes": [{"metric": m, "bins": b} for m, b in self._axes]}
        sel: dict = {}
        if method is not None:
            sel["method"] = method
        if self._weights:
            sel["weights"] = dict(self._weights)
        if self._normalize is not None:
            sel["normalize"] = self._normalize
        if self._target is not None:
            sel["target"] = self._target
        if self._tag_coverage:
            sel["tag_coverage"] = dict(self._tag_coverage)
        if self._group_cap is not None:
            sel["group_cap"] = (self._group_cap if isinstance(self._group_cap, dict)
                                else {"max_group_overshoot": float(self._group_cap)})
        if self._predicates:
            sel["predicates"] = [dict(pr) for pr in self._predicates]
        sel.update(self._method_params)
        if sel:
            d["selection"] = sel
        return d

    def _compile(self):
        """Assemble the raw strategy dict, register callable metrics, and load (validates
        everything loudly against the schema)."""
        from .strategy import load_strategy
        for nm, fn in self._callables.items():
            register_compiled(nm, fn)
        transform_outputs = {o for t in self._transforms for o in t["outputs"]}
        return load_strategy(self._raw_strategy(), self.schema, registry=self._registry,
                             available_metrics=transform_outputs)

    def _segmenter_or_default(self) -> Segmenter:
        return self._segmenter if self._segmenter is not None else Segmenter(64, 64)

    def _check_exec(self) -> None:
        if self._options["workers"] and self._options["workers"] > 1 and self._in_memory:
            raise ValueError(
                "workers>1 is not supported for in-memory sources (rows/ticks/DataFrame): "
                "parallel workers rebuild adapters per process and cannot see in-process "
                "data — use workers=1 or write the data to files first")

    # ---- inspectors (cheap, non-terminal) ------------------------------------------------
    def head(self, k: int = 5) -> list:
        """First k Ticks of the first source (reads only what it needs)."""
        return list(islice(iter(self.adapter.ticks(self.sources[0])), int(k)))

    def describe(self) -> dict:
        """The pipeline's resolved state — no data is touched."""
        return {"schema": self.schema.to_dict(),
                "strategy": self._raw_strategy(require_method=False),
                "output": None,
                "segmenter": dict(self._window_desc),
                "options": {**{k: _jsonable(v) for k, v in self._options.items()},
                            "mode": self._mode,
                            "overrides": sorted(self._explicit_options)},
                "sources": [str(s) for s in self.sources]}

    def explain(self, *, inspect_sources: bool = False):
        """Return a stable, structured execution plan without reading rows by default."""
        from .results import ExecutionPlan
        warnings = []
        categorical = [n for n, f in self.schema.fields.items() if f.kind == "categorical"]
        if categorical:
            warnings.append({"code": "categorical-output-policy", "message":
                             f"categorical fields {categorical} need an explicit output policy"})
        if self._options["on_error"] == "skip":
            warnings.append({"code": "source-errors-skipped", "message":
                             "source errors will be skipped and reported"})
        if inspect_sources and hasattr(self.adapter, "check_source"):
            for source in self.sources:
                self.adapter.check_source(source)
        strategy = self._raw_strategy(require_method=False)
        full_pool = bool(strategy.get("filters") or strategy.get("scorers") or
                         strategy.get("selection", {}).get("method") in
                         {"cluster-balanced", "k-center", "facility-location", "dsir",
                          "rare", "typical"})
        stages = ({"name": "input", "streaming": hasattr(self.adapter, "iter_ticks"),
                   "materializes_total_pool": False},
                  {"name": "selection", "streaming": not full_pool,
                   "materializes_total_pool": full_pool},
                  {"name": "output", "streaming": True, "materializes_total_pool": False})
        return ExecutionPlan(source={"kind": type(self.adapter).__name__,
                                     "count": len(self.sources),
                                     "sources": [str(s) for s in self.sources]},
                             schema=self.schema.to_dict(), window=dict(self._window_desc),
                             strategy=strategy, output=None,
                             options=self.describe()["options"], stages=stages,
                             warnings=tuple(warnings))

    # ---- terminals -----------------------------------------------------------------------
    def collect(self, n: Optional[int] = None):
        """Count-path selection -> CurateResult."""
        if n is None:
            n = self._n
        if n is None:
            hint = (" (.curate(hours=...) routes to .output())"
                    if self._hours is not None else "")
            raise ValueError(f"collect() needs a count: pass n= or call .curate(n=...) "
                             f"first{hint}")
        from .strategy import validate_target_size
        n = validate_target_size(n)
        self._check_exec()
        strategy = self._compile()
        from .curate import curate as _curate
        return _curate(schema=self.schema, adapter=self.adapter, sources=self.sources,
                       strategy=strategy, segmenter=self._segmenter_or_default(),
                       target_size=n, seed=self._options["seed"],
                       group_of=self._group_of, tags_of=None,
                       test_frac=self._split_test or 0.0, workers=self._options["workers"],
                       on_error=self._options["on_error"],
                       streaming=self._options["streaming"], progress=self._options["progress"],
                       transforms=self._transforms, registry=self._registry,
                       executor=self._options["executor"])

    def report(self, **kw) -> dict:
        """Feature pass only (no selection) -> dataset_report over the scored pool."""
        self._check_exec()
        from .report import dataset_report
        from .scoring import score_sources
        for nm, fn in self._callables.items():
            register_compiled(nm, fn)
        scorer_outs = {s["as"] for s in self._scorers}
        transform_outs = {o for t in self._transforms for o in t["outputs"]}
        registered = [nm for nm in self._callables
                      if nm not in scorer_outs and nm not in transform_outs]
        segments, _errors, _tracks = score_sources(
            schema=self.schema, adapter=self.adapter, sources=self.sources,
            define=dict(self._define), registered=registered,
            segmenter=self._segmenter_or_default(),
            tags_spec={"from": list(self._tags)} if self._tags else None,
            group_of=self._group_of, workers=self._options["workers"],
            on_error=self._options["on_error"], streaming=self._options["streaming"],
            progress=self._options["progress"], executor=self._options["executor"])
        if self._transforms:
            from .transforms import apply_transforms
            apply_transforms(self._transforms, segments, adapter=self.adapter,
                             registry=self._registry, progress=self._options["progress"])
        if self._scorers:
            from .scorers import apply_scorers, parse_scorers
            apply_scorers(parse_scorers([dict(s) for s in self._scorers]), segments,
                          schema=self.schema, adapter=self.adapter,
                          seed=self._options["seed"], progress=self._options["progress"],
                          registry=self._registry)
        return dataset_report(segments, **kw)

    def experiment(self, arms: dict, n: int, baseline: Optional[str] = None) -> dict:
        """Multi-arm comparison over THIS pipeline's sources. `arms` maps name ->
        Pipeline (its compiled strategy) or raw strategy dict."""
        from .strategy import validate_target_size
        n = validate_target_size(n)
        self._check_exec()
        from .experiment import Arm, run_experiment
        from .strategy import load_strategy
        built = []
        for name, arm in arms.items():
            if isinstance(arm, Pipeline):
                built.append(Arm(name, arm._compile()))
            elif isinstance(arm, dict):
                built.append(Arm(name, load_strategy(arm, self.schema)))
            else:
                raise ValueError(f"arm {name!r} must be a Pipeline or a raw strategy dict, "
                                 f"got {type(arm).__name__}")
        return run_experiment(schema=self.schema, adapter=self.adapter,
                              sources=self.sources, arms=built,
                              segmenter=self._segmenter_or_default(), target_size=n,
                              seed=self._options["seed"], baseline=baseline,
                              group_of=self._group_of, workers=self._options["workers"],
                              on_error=self._options["on_error"],
                              streaming=self._options["streaming"],
                              progress=self._options["progress"],
                              executor=self._options["executor"])

    def output(self, path, *, preset: Optional[str] = None, spec=None,
               container: Optional[str] = None, shard_length: int = 64,
               short_policy: str = "keep", modalities: Optional[dict] = None,
               include=None, exclude=None, encoder=None, span_steps: Optional[int] = None,
               span_seconds: Optional[float] = None, categorical: Optional[str] = None,
               preset_params: Optional[dict] = None) -> dict:
        """Write shards + manifests + config.json. `.curate(hours=...)` routes through the
        hours-budget `write_dataset`; `.curate(n=...)` selects then writes each chosen
        segment as a sub-track. config.json gains a 'pipeline' key: the operator chain."""
        self._check_exec()
        path = str(path)
        if spec is not None and (preset is not None or modalities is not None or include is not None
                                 or exclude is not None or container is not None
                                 or shard_length != 64 or short_policy != "keep"
                                 or categorical is not None or preset_params):
            raise ValueError("output(spec=...) is the direct escape hatch and cannot be combined "
                             "with preset/modalities/include/exclude/container/shard_length/"
                             "short_policy/categorical/preset_params")
        if preset is not None and modalities is not None:
            raise ValueError("output(preset=...) defines its modalities and cannot be combined "
                             "with modalities=; use include=/exclude= to filter the preset")
        preset_resolution = None
        from .output.spec import OutputSpec
        if spec is not None:
            if isinstance(spec, dict):
                spec = OutputSpec.from_dict(spec)
            if not isinstance(spec, OutputSpec):
                raise ValueError("spec must be an OutputSpec or output-spec dict")
        elif preset is not None:
            from .output.presets import output_preset
            params = dict(preset_params or {})
            if categorical is not None:
                params["categorical"] = categorical
            if container is not None:
                params["container"] = container
            if shard_length != 64:
                params["shard_length"] = shard_length
            if short_policy != "keep":
                params["short_policy"] = short_policy
            categorical_fields = [n for n, f in self.schema.fields.items()
                                  if f.kind == "categorical"]
            if self._mode == "strict" and categorical_fields and "categorical" not in params:
                raise ValueError("strict mode requires an explicit categorical policy for "
                                 f"fields {categorical_fields}")
            spec = output_preset(preset, schema=self.schema, **params)
            preset_resolution = {"name": preset, "version": 1, "parameters": params,
                                 "resolved": spec.to_dict()}
            if include is not None or exclude is not None:
                raw_spec = spec.to_dict()
                preset_mods = raw_spec["modalities"]
                if include is not None:
                    unknown = sorted(set(include) - set(preset_mods))
                    if unknown:
                        raise ValueError(f"output(): include names unknown preset modalities "
                                         f"{unknown} (have: {sorted(preset_mods)})")
                    preset_mods = {k: v for k, v in preset_mods.items() if k in set(include)}
                if exclude is not None:
                    unknown = sorted(set(exclude) - set(preset_mods))
                    if unknown:
                        raise ValueError(f"output(): exclude names unknown preset modalities "
                                         f"{unknown} (have: {sorted(preset_mods)})")
                    preset_mods = {k: v for k, v in preset_mods.items()
                                   if k not in set(exclude)}
                if not preset_mods:
                    raise ValueError("output(): include/exclude left no preset modalities")
                raw_spec["modalities"] = preset_mods
                spec = OutputSpec.from_dict(raw_spec)
                preset_resolution["resolved"] = spec.to_dict()
        else:
            if container is None:
                container = _SUFFIX_CONTAINER.get(os.path.splitext(path)[1].lower(), "npz")
            if categorical is None:
                categorical = "error" if self._mode == "strict" else "omit"
            if categorical not in {"omit", "error"}:
                raise ValueError("default output categorical policy must be 'omit' or 'error'; "
                                 "use a preset for encoder policies")
        if spec is None and modalities is None:
            mods: dict = {}
            for fname, fs in self.schema.fields.items():
                if fs.kind == "categorical":
                    if categorical == "error":
                        raise ValueError(f"output(): categorical field {fname!r} would be omitted; "
                                         "strict/error policy requires explicit modalities")
                    print(f"  [output] categorical field {fname!r} skipped from default "
                          f"modalities (pass modalities= to encode it)")
                    continue
                mods[fname] = {"dtype": "float32",
                               "shape": [len(fs.dims)] if fs.kind == "vector" else [],
                               "source": f"field:{fname}"}
            if not mods:
                raise ValueError("output(): no numeric schema fields for default modalities "
                                 "— pass modalities= explicitly")
        elif spec is None:
            assert modalities is not None
            mods = {k: dict(v) for k, v in modalities.items()}
        if spec is None and include is not None:
            unknown = sorted(set(include) - set(mods))
            if unknown:
                raise ValueError(f"output(): include names unknown modalities {unknown} "
                                 f"(have: {sorted(mods)})")
            mods = {k: v for k, v in mods.items() if k in set(include)}
        if spec is None and exclude is not None:
            unknown = sorted(set(exclude) - set(mods))
            if unknown:
                raise ValueError(f"output(): exclude names unknown modalities {unknown} "
                                 f"(have: {sorted(mods)})")
            mods = {k: v for k, v in mods.items() if k not in set(exclude)}
        if spec is None and not mods:
            raise ValueError("output(): include/exclude left no modalities")
        if spec is None:
            spec = OutputSpec.from_dict({"modalities": mods, "shard_length": shard_length,
                                         "short_policy": short_policy, "container": container})
        spec.validate_schema(self.schema)
        output_op = {"op": "output", "args": [path], "kwargs": {
            "preset": preset, "spec": spec.to_dict(), "categorical": categorical}}
        if isinstance(encoder, str):
            from .output.base import get_encoder
            encoder = get_encoder(encoder)
        elif isinstance(encoder, dict):
            from .output.base import get_encoder
            encoder = get_encoder(encoder["name"], **dict(encoder.get("params") or {}))
        strategy = self._compile()
        opt = self._options

        if self._hours is not None:                       # ---- hours-budget path ----
            from .output.dataset import write_dataset
            hours = float(self._hours)
            train_hours, test_hours = hours, self._test_hours
            if self._split_test is not None and not self._test_hours_explicit:
                train_hours = (1.0 - self._split_test) * hours
                test_hours = self._split_test * hours
            result = write_dataset(
                path, schema=self.schema, adapter=self.adapter, sources=self.sources,
                strategy=strategy, segmenter=self._segmenter_or_default(), spec=spec,
                encoder=encoder, train_hours=train_hours, test_hours=test_hours,
                sample_hz=opt["sample_hz"] or None, seed=opt["seed"],
                group_of=self._group_of, span_steps=span_steps, span_seconds=span_seconds,
                workers=opt["workers"], on_error=opt["on_error"], resume=opt["resume"],
                streaming=opt["streaming"], progress=opt["progress"],
                transforms=self._transforms, registry=self._registry,
                executor=opt["executor"])
            from .io import read_text
            from .manifest import write_config
            cfg = json.loads(read_text(f"{path}/config.json"))
            cfg["pipeline"] = list(self._ops)
            cfg["output_request"] = output_op
            if preset_resolution:
                cfg["output_preset"] = preset_resolution
            cfg["mode"] = {"name": self._mode, "resolved": self.describe()["options"]}
            cfg["warnings"] = [dict(w) for w in self.explain().warnings]
            write_config(f"{path}/config.json", cfg)
            return result

        # ---- count path -------------------------------------------------------------
        if opt["resume"]:
            raise ValueError("resume is supported on the hours path only "
                             "(.curate(hours=...)); the count path rewrites its "
                             "selection deterministically per run")
        if self._n is None:
            raise ValueError("output() needs .curate(n=...) or .curate(hours=...) first")
        result = self.collect()
        if not result.selected:
            raise ValueError("selection is empty — nothing to write (check "
                             "filters/predicates and the segment window)")
        written = self._write_selected(path, result, spec, encoder, opt["seed"], opt["progress"])
        from .io import read_text
        from .manifest import write_config
        cfg = json.loads(read_text(f"{path}/config.json"))
        cfg["pipeline"] = list(self._ops)
        cfg["output_request"] = output_op
        if preset_resolution:
            cfg["output_preset"] = preset_resolution
        cfg["mode"] = {"name": self._mode, "resolved": self.describe()["options"]}
        cfg["warnings"] = [dict(w) for w in self.explain().warnings]
        write_config(f"{path}/config.json", cfg)
        return written

    def _write_selected(self, out_dir: str, result, spec, encoder, seed: int, progress=None) -> dict:
        """Count-path write: each selected segment becomes a sub-track per entity, written
        with the same shard mechanics as the hours path."""
        from .io import write_text
        from .manifest import build_config, write_config
        from .output.shardwriter import write_track
        from .output.track import Track, tracks_from_ticks
        by_source: dict = {}
        for s in result.selected:
            by_source.setdefault(s.source_id, []).append(s)
        entries: list = []
        from .progress import emit
        emit(progress, "writing", "start", total=len(by_source))
        for source_i, src in enumerate(sorted(by_source, key=str), 1):
            segs = sorted(by_source[src], key=lambda s: s.start_idx)
            tracks = tracks_from_ticks(self.schema, list(self.adapter.ticks(src)), src)
            split = result.splits.get(src, "train")
            for k, seg in enumerate(segs):
                for tr in tracks:
                    recs = tr.records[seg.start_idx:seg.end_idx]
                    if not recs:
                        continue
                    sub = Track(tr.source_id, tr.entity_id, recs, dict(tr.meta))
                    sub.meta.update(seg.features)
                    sub.meta["group"] = seg.group
                    entries += write_track(out_dir, sub, spec, encoder=encoder,
                                           schema=self.schema, split=split,
                                           name_suffix=f"_seg{k:04d}",
                                           media_provider=getattr(self.adapter, "provider", None))
            emit(progress, "writing", "advance", current=source_i, total=len(by_source),
                 source=src, status="ok")
        emit(progress, "writing", "complete", current=len(by_source), total=len(by_source),
             detail=f"{len(entries)} shards")
        write_text(f"{out_dir}/dataset_manifest.json",
                   json.dumps({"artifact_format_version":
                               __import__("kuration.versioning", fromlist=["ARTIFACT_FORMAT_VERSION"]).ARTIFACT_FORMAT_VERSION,
                               "count": len(entries), "shards": entries}, indent=2))
        results = {"shards": len(entries), "selected": len(result.selected),
                   "segments_total": len(result.all_segments),
                   "sources_skipped": len(result.errors),
                   "n_train": sum(1 for v in result.splits.values() if v == "train"),
                   "n_test": sum(1 for v in result.splits.values() if v == "test")}
        cfg = build_config(self.schema, result.strategy, results=results, seed=seed)
        cfg["output"] = spec.to_dict()
        cfg["pipeline"] = list(self._ops)
        write_config(f"{out_dir}/config.json", cfg)
        from .results import DatasetWriteResult
        return DatasetWriteResult(shards=entries, selected=len(result.selected), results=results,
                                  errors=result.errors)
