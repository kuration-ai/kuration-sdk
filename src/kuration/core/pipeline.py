"""Pipeline facade — `input(...)` -> chained operators -> terminals.

`input(src)` sniffs the source (file/glob/dir/DataFrame/rows/ticks/columnar dict), infers a
`SchemaSpec` (via `kuration.core.infer`) unless one is given, and builds the right adapter — all
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

import dis
import json
import math
import os
import copy
import re
from functools import lru_cache, wraps
from itertools import islice
from typing import Any, Callable, Optional

import numpy as np

from .adapters.base import get_adapter
from .adapters.hdf5 import HDF5Adapter
from .adapters.jsonl import JsonlAdapter
from .adapters.npz import NpzAdapter
from .adapters.parquet_dataset import ParquetDatasetAdapter
from .adapters.timeseries import TimeseriesAdapter
from .adapters.zarr_store import ZarrAdapter
from .buffer import Tick
from .dsl.grammar import compile_metric
from .results import DatasetReport, ExperimentResult
from .schema import SchemaSpec
from .segment import Segmenter
from ._pipeline_support import (
    FILTER_TYPES as _FILTER_TYPES,
    GLOB_CHARS as _GLOB_CHARS,
    METHOD_ALIASES as _METHOD_ALIASES,
    OP_ALIASES as _OP_ALIASES,
    OPTION_KEYS as _OPTION_KEYS,
    SUFFIX_CONTAINER as _SUFFIX_CONTAINER,
    TABULAR_KINDS as _TABULAR_KINDS,
    JsonRowsAdapter as _JsonRowsAdapter,
    RowsAdapter as _RowsAdapter,
    TickAdapter as _TickAdapter,
    auto_qualify as _auto_qualify,
    default_field_map as _default_field_map,
    describe_src as _describe_src,
    jsonable as _jsonable,
)

__all__ = ["Pipeline", "input"]

class _Unset:
    """Sentinel distinguishing "argument not passed" from an explicit ``None``, with a stable
    repr (unlike a bare ``object()``) so it can appear in a reproducible signature string."""

    def __repr__(self) -> str:
        return "..."


_UNSET = _Unset()


def _leased_output(method):
    """Serialize publication to one destination while preserving the public signature."""
    @wraps(method)
    def locked(self, path, *args, **kwargs):
        if kwargs.get("dry_run", False):
            return method(self, path, *args, **kwargs)
        from .manifest import (
            dataset_writer_lease,
            validate_source_destination_separation,
        )
        validate_source_destination_separation(
            str(path), self.sources, adapter=self.adapter)
        with dataset_writer_lease(str(path)):
            return method(self, path, *args, **kwargs)
    return locked


def _unique_sources(sources: list[str], *, context: str) -> list[str]:
    """Reject repeated source identities before they can duplicate windows or checkpoint keys."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for source in sources:
        if source in seen and source not in duplicates:
            duplicates.append(source)
        seen.add(source)
    if duplicates:
        raise ValueError(f"{context}: duplicate source ID(s) {duplicates}; sources must be unique")
    return sources


def _adapter_chain(adapter):
    """Yield transparent adapter wrappers and their nested data adapter once each."""
    current, seen = adapter, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "adapter", None)


def _registry_inherits(registry, ancestor) -> bool:
    """Return whether a live registry overlay ultimately resolves through ``ancestor``."""
    current = registry
    while current is not None:
        if current is ancestor:
            return True
        current = current.parent
    return False


_METADATA_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _metadata_part(value: str, *, label: str) -> str:
    """Validate one stable pipeline-metadata namespace or key."""
    if not isinstance(value, str) or not _METADATA_PART.fullmatch(value):
        raise ValueError(
            f"{label} must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_', or '-'"
        )
    return value


def _friendly_operator_part(value: str) -> str:
    """Turn one public component-name part into a readable Python identifier."""
    identifier = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if identifier[0].isdigit():
        identifier = f"operator_{identifier}"
    return identifier


def _canonical_operator_part(value: str) -> str:
    """Encode a component-name part injectively for an unambiguous method fallback."""
    substitutions = {"_": "_u", "-": "_h", ".": "_d"}
    identifier = "".join(substitutions.get(character, character) for character in value)
    if identifier[0].isdigit():
        identifier = f"operator__{identifier}"
    return identifier


def _operator_aliases(name: str) -> tuple[str, ...]:
    """Return short, qualified, and collision-proof method names in preference order."""
    parts = name.split("/", 1)
    local = parts[-1]
    aliases = [_friendly_operator_part(local)]
    if len(parts) == 2:
        aliases.append("__".join(_friendly_operator_part(part) for part in parts))
    aliases.append(_canonical_operator_alias(name))
    return tuple(dict.fromkeys(aliases))


def _canonical_operator_alias(name: str) -> str:
    return "__".join(_canonical_operator_part(part) for part in name.split("/", 1))


def _alias_claim_rank(name: str, alias: str) -> int:
    """Rank how strongly ``name`` claims ``alias``: 0 spells the whole component name, 1 is the
    local shorthand of a namespaced component.

    A tighter claim wins outright, so a plain ``smooth`` operator keeps ``pipeline.smooth()``
    when a plugin also registers ``acme/smooth`` (still reachable as ``acme__smooth()``). Only
    equally strong claims are ambiguous, and every claimant of an ambiguous spelling then has a
    stronger spelling of its own to suggest.
    """
    if "/" in name and alias == _friendly_operator_part(name.split("/", 1)[1]):
        return 1
    return 0


@lru_cache(maxsize=1)
def _pipeline_attribute_names() -> frozenset[str]:
    """Public ``Pipeline`` attribute names that a registered operator must not try to reuse.

    ``__getattr__`` runs only after normal attribute lookup fails, so an operator whose method
    spelling equals a real attribute would be silently shadowed by the built-in. The reserved set
    is derived from the class itself — class attributes plus every attribute the class's own
    methods assign — so it can never drift from the real method surface. It is deliberately a
    superset: an attribute a method assigns on a collaborator object is reserved too, which only
    ever makes the guard stricter.
    """
    names = {name for name in dir(Pipeline) if not name.startswith("_")}
    for value in vars(Pipeline).values():
        for candidate in (value, getattr(value, "__func__", None), getattr(value, "fget", None),
                          getattr(value, "fset", None)):
            code = getattr(candidate, "__code__", None)
            if code is None:
                continue
            names.update(instruction.argval for instruction in dis.get_instructions(code)
                         if instruction.opname == "STORE_ATTR"
                         and not str(instruction.argval).startswith("_"))
    return frozenset(names)


def operator_name_conflict(name: str) -> Optional[str]:
    """Return a loud message when a registered operator name could never be reached as a method.

    Registration-time guard shared by ``Registry.register`` and plugin validation: any method
    spelling of the operator that collides with a real ``Pipeline`` attribute resolves to the
    built-in, so the operator would be silently shadowed instead of callable.
    """
    reserved = sorted(alias for alias in _operator_aliases(name)
                      if alias in _pipeline_attribute_names())
    if not reserved:
        return None
    local = name.split("/", 1)[-1]
    return (f"operator {name!r} method name(s) {reserved} collide with existing Pipeline "
            f"attribute(s) of the same name; Python resolves the attribute first, so the "
            f"operator would be silently shadowed — rename the component (for example "
            f"{local}-<qualifier>)")


def _clone_adapter(adapter, memo=None):
    """Copy wrapper/configuration state without copying potentially large source payloads."""
    memo = {} if memo is None else memo
    identity = id(adapter)
    if identity in memo:
        return memo[identity]
    cloned = copy.copy(adapter)
    memo[identity] = cloned
    for attr in ("fields", "globals", "videos", "load_kwargs", "dataset_of", "_container"):
        value = getattr(cloned, attr, None)
        if isinstance(value, dict):
            setattr(cloned, attr, dict(value))
    nested = getattr(adapter, "adapter", None)
    if nested is not None:
        cloned.adapter = _clone_adapter(nested, memo)
    return cloned


def _adapter_with_fields(adapter):
    return next((current for current in _adapter_chain(adapter)
                 if hasattr(current, "fields")), None)


def _normalized_source_mapping(src: dict, *, value_copy: Callable) -> dict[str, Any]:
    """String-normalize source IDs without silently merging distinct keys."""
    normalized: dict[str, Any] = {}
    origins: dict[str, object] = {}
    for source_id, value in src.items():
        key = str(source_id)
        if not key:
            raise ValueError("input(): source IDs must not normalize to an empty string")
        if key in normalized:
            previous = origins[key]
            raise ValueError(
                "input(): distinct source IDs collide after string normalization: "
                f"{previous!r} and {source_id!r} both become {key!r}")
        origins[key] = source_id
        normalized[key] = value_copy(value)
    return normalized


# ======================================================================================
# input() — the factory (intentionally shadows builtins.input in this module)
# ======================================================================================
def input(src=None, *, schema=None, adapter=None, sources=None, fields=None,
          time=None, entity=None, hz=None, kinds=None, sample=200, globals=None,
          registry=None, **adapter_kw) -> "Pipeline":
    """Open a source as a `Pipeline`. Sniffs the source kind, infers the schema (unless
    `schema=` is given) from a sample, and builds the adapter — eagerly and loudly.

    Accepted `src`: file path / glob / list of paths / parquet-parts directory; a pandas
    DataFrame (duck-typed via .to_dict); list[dict] rows; list[Tick]; dict of columns
    (columnar in-memory); {source_id: rows|ticks} (multi-source in-memory). Escape hatch:
    `input(adapter=<InputAdapter or registry name>, sources=[...], schema=...)`. Pass
    ``registry=`` to resolve a named adapter and all later extensions through an isolated
    registry snapshot.
    """
    if registry is not None:
        from .registry import Registry
        if not isinstance(registry, Registry):
            raise ValueError("input(registry=...) needs a Registry")
    op = {"op": "input",
          "args": [] if src is None else [_describe_src(src)],
          "kwargs": {k: _jsonable(v) for k, v in dict(
              schema=schema, adapter=adapter,
              sources=_describe_src(sources) if sources is not None else None,
              fields=fields, time=time,
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
            adapter = get_adapter(adapter, registry=registry, schema=schema, **kw)
        srcs = list(sources) if sources is not None else (
            list(adapter.sources()) if hasattr(adapter, "sources") else [])
        if not srcs:
            raise ValueError("input(adapter=...) requires sources=[...] "
                             "(or an adapter exposing .sources())")
        if any(not isinstance(source, str) or not source for source in srcs):
            raise ValueError("input(adapter=...) sources must be non-empty strings")
        _unique_sources(srcs, context="input(adapter=...)")
        return Pipeline(schema=schema, adapter=adapter, sources=srcs,
                        in_memory=bool(getattr(adapter, "in_memory", False)), ops=[op],
                        registry=registry)

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
                            hz=hz, kinds=kinds, sample=sample, globals=globals,
                            registry=registry, **adapter_kw)

    if adapter_kw:
        raise ValueError(f"unknown input() argument(s) {sorted(adapter_kw)} — adapter "
                         f"keywords only apply to file-backed sources")

    # -- in-memory lists ----------------------------------------------------------------
    if isinstance(src, (list, tuple)):
        if not src:
            raise ValueError("input(): empty list — nothing to curate")
        if all(isinstance(x, Tick) for x in src):
            return _input_ticks({"mem": list(src)}, op, schema=schema, time=time,
                                entity=entity, hz=hz, kinds=kinds, sample=sample,
                                registry=registry)
        if all(isinstance(x, dict) for x in src):
            return _input_rows({"mem": list(src)}, op, schema=schema, fields=fields,
                               time=time, entity=entity, hz=hz, kinds=kinds, sample=sample,
                               globals=globals, registry=registry)
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
            tick_sources = _normalized_source_mapping(src, value_copy=list)
            return _input_ticks(tick_sources, op, schema=schema,
                                time=time, entity=entity, hz=hz, kinds=kinds, sample=sample,
                                registry=registry)
        if all(_is_rows(v) for v in vals):
            row_sources = _normalized_source_mapping(src, value_copy=list)
            return _input_rows(row_sources, op, schema=schema,
                               fields=fields, time=time, entity=entity, hz=hz, kinds=kinds,
                               sample=sample, globals=globals, registry=registry)
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
                               globals=globals, registry=registry)
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
    _unique_sources(paths, context="input()")
    kinds_ = {p: sniff(p) for p in paths}
    if len(set(kinds_.values())) > 1:
        raise ValueError(f"input(): mixed source kinds {kinds_} — all sources must be the "
                         f"same kind")
    return paths, next(iter(kinds_.values()))


def _input_files(src, op, *, schema, fields, time, entity, hz, kinds, sample, globals,
                 registry=None, **adapter_kw) -> "Pipeline":
    paths, kind = _expand_paths(src)
    if kind == "video":
        if schema is not None or fields is not None or time is not None or entity is not None:
            raise ValueError("video input probes its media schema; schema/fields/time/entity "
                             "overrides are not supported (use input(adapter=...) instead)")
        from .media import VideoAdapter
        video_ad = VideoAdapter(paths, **adapter_kw)
        return Pipeline(schema=video_ad.schema, adapter=video_ad, sources=paths,
                        in_memory=False, ops=[op], registry=registry)
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
    elif kind == "zarr":
        ad = ZarrAdapter(schema, fields=fmap, globals=globals, **adapter_kw)
    elif kind == "parquet-dataset":
        ad = ParquetDatasetAdapter(schema, fields=fmap, path=paths[0], time_column=tc,
                                   entity_column=ec, globals=globals, **adapter_kw)
        paths = ad.sources()
        _unique_sources(paths, context="input()")
    else:
        raise ValueError(f"input(): unsupported sniffed source kind {kind!r}")
    return Pipeline(schema=schema, adapter=ad, sources=paths, in_memory=False, ops=[op],
                    registry=registry)


def _input_rows(data: dict, op, *, schema, fields, time, entity, hz, kinds, sample,
                globals, registry=None) -> "Pipeline":
    first = next(iter(data.values()))
    schema, fmap, tc, ec = _infer_or_explicit(
        first[:sample], schema=schema, fields=fields, time=time, entity=entity, hz=hz,
        kinds=kinds, tabular=True)
    ad = _RowsAdapter(schema, data=data, fields=fmap, time_column=tc, entity_column=ec,
                      globals=globals)
    return Pipeline(schema=schema, adapter=ad, sources=list(data), in_memory=True, ops=[op],
                    registry=registry)


def _input_ticks(data: dict, op, *, schema, time, entity, hz, kinds, sample,
                 registry=None) -> "Pipeline":
    if schema is None:
        from .infer import infer_schema
        first = next(iter(data.values()))
        schema, _hints = infer_schema(first[:sample], name="inferred", time=time,
                                      entity=entity, hz=hz, kinds=kinds)
    ad = _TickAdapter(schema, data)
    return Pipeline(schema=schema, adapter=ad, sources=list(data), in_memory=True, ops=[op],
                    registry=registry)


# ======================================================================================
# Pipeline
# ======================================================================================
class Pipeline:
    """Immutable curation pipeline. Build via `kuration.core.pipeline.input(...)`;
    every operator returns a new Pipeline."""

    def __init__(self, *, schema: SchemaSpec, adapter, sources: list, in_memory: bool,
                 ops: list, registry=None):
        self.schema = schema
        self.adapter = adapter
        self.sources = list(sources)
        self._in_memory = in_memory
        self._ops = list(ops)
        self._segmenter: Optional[Segmenter] = None
        self._window_desc = {"window": 64, "stride": 64, "unit": "steps"}
        self._define: dict = {}          # measure name -> DSL spec (insertion-ordered)
        self._callables: dict = {}       # measure name -> fn(WindowBuffer) -> float
        self._registered_metrics: list[str] = []  # registry metrics explicitly declared
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
            "resume": False, "sample_hz": None, "progress": None, "executor": "local",
            "memory_budget": None, "batch_size": 1024, "checkpoint_dir": None,
            "spill_dir": None, "approximation": "exact", "max_in_flight": 2}
        self._progress_spec: Any = None
        self._mode: Optional[str] = None
        self._explicit_options: set[str] = set()
        self._expectations: dict = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._materialized_metadata: set[tuple[str, str]] = set()
        from .registry import default_registry
        self._registry = default_registry if registry is None else registry.snapshot()

    @property
    def metadata(self) -> dict[str, dict[str, Any]]:
        """Defensive copy of namespaced metadata carried by this pipeline.

        Use :meth:`with_metadata` for caller-owned values or :meth:`enrich` to run a
        registered analyzer and attach its materialized result.
        """
        return copy.deepcopy(self._metadata)

    def _operator_method_map(self, *, discover_plugins: bool) -> dict[str, list[str]]:
        """Map Python method spellings to the registered operator names they identify.

        Claims are resolved by strength (see ``_alias_claim_rank``), so a spelling is reported as
        ambiguous only when several operators claim it equally strongly.
        """
        registry = self.__dict__.get("_registry")
        if registry is None:
            return {}
        if discover_plugins:
            from .plugins import load_plugins
            from .registry import default_registry
            if _registry_inherits(registry, default_registry):
                load_plugins()
        claims: dict[str, dict[int, list[str]]] = {}
        for operator_name in registry.names("operator"):
            for alias in _operator_aliases(operator_name):
                rank = _alias_claim_rank(operator_name, alias)
                claims.setdefault(alias, {}).setdefault(rank, []).append(operator_name)
        return {alias: by_rank[min(by_rank)] for alias, by_rank in claims.items()}

    def __getattr__(self, name: str) -> Callable[..., "Pipeline"]:
        """Resolve registered operators as immutable methods.

        The local component name is the normal spelling (``acme/signal-diversity`` becomes
        ``pipeline.signal_diversity(...)``). Namespace-qualified and collision-proof aliases
        keep every registered operator callable without a string-based dispatch escape hatch.
        """
        if name.startswith("_"):
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")
        methods = self._operator_method_map(discover_plugins=False)
        matches = methods.get(name, [])
        if not matches:
            methods = self._operator_method_map(discover_plugins=True)
            matches = methods.get(name, [])
        if not matches:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")
        if len(matches) > 1:
            # Only suggest spellings that actually resolve on the same registry view that
            # produced the ambiguity; `apply()` is the always-available fallback.
            alternatives = []
            for operator_name in matches:
                alternative = next(
                    (
                        alias for alias in _operator_aliases(operator_name)
                        if methods.get(alias) == [operator_name]
                    ),
                    None,
                )
                alternatives.append(f"{alternative}()" if alternative is not None
                                    else f"apply({operator_name!r})")
            raise AttributeError(
                f"registered operator method {name!r} is ambiguous between {matches}; "
                f"use one of {alternatives}")

        operator_name = matches[0]
        descriptor = self._registry.descriptor("operator", operator_name)

        def registered_operator(**params: Any) -> "Pipeline":
            return self._invoke_operator(operator_name, params)

        registered_operator.__name__ = name
        registered_operator.__qualname__ = f"{type(self).__name__}.{name}"
        registered_operator.__doc__ = (
            descriptor.description
            or f"Run the registered {operator_name!r} compositional operator."
        )
        return registered_operator

    def __dir__(self) -> list[str]:
        """Include currently registered operator methods in interactive discovery."""
        names = set(super().__dir__())
        methods = self._operator_method_map(discover_plugins=False)
        names.update(name for name, matches in methods.items() if len(matches) == 1)
        return sorted(names)

    # ---- immutability plumbing -------------------------------------------------------
    def _clone(self) -> "Pipeline":
        # Copy pipeline state without rerunning eager input construction.
        p = object.__new__(type(self))
        p.__dict__.update(self.__dict__)
        p.sources = list(self.sources)
        # Keep clone identities isolated as part of the immutability contract.
        p.schema = copy.deepcopy(self.schema)
        p.adapter = _clone_adapter(self.adapter)
        for configured in _adapter_chain(p.adapter):
            if hasattr(configured, "schema"):
                configured.schema = p.schema
        for attr in ("_ops", "_define", "_registered_metrics", "_predicates", "_prepass", "_axes", "_tags",
                     "_scorers", "_transforms", "_method_params", "_window_desc",
                     "_expectations", "_weights", "_target", "_tag_coverage", "_group_cap",
                     "_feature_space", "_metadata"):
            setattr(p, attr, copy.deepcopy(getattr(self, attr)))
        p._callables = dict(self._callables)
        p._options = dict(self._options)  # callbacks/providers are deliberately shared values
        p._progress_spec = self._progress_spec
        p._explicit_options = set(self._explicit_options)
        p._materialized_metadata = set(self._materialized_metadata)
        return p

    def _next(self, opname: str, *args, **kwargs) -> "Pipeline":
        p = self._clone()
        p._ops.append({"op": opname, "args": [_jsonable(a) for a in args],
                       "kwargs": {k: _jsonable(v) for k, v in kwargs.items()
                                  if v is not None}})
        return p

    def _metric_names(self) -> set:
        return (set(self._define) | set(self._callables) | set(self._registered_metrics) |
                {s["as"] for s in self._scorers} |
                {o for t in self._transforms for o in t["outputs"]})

    def _metadata_target(self, namespace: str, key: str, *, replace: bool) -> tuple[str, str]:
        namespace = _metadata_part(namespace, label="metadata namespace")
        key = _metadata_part(key, label="metadata key")
        if not isinstance(replace, bool):
            raise ValueError("metadata replace must be true or false")
        if key in self._metadata.get(namespace, {}) and not replace:
            raise ValueError(
                f"pipeline metadata {namespace!r}/{key!r} already exists; "
                "pass replace=True to replace it explicitly"
            )
        return namespace, key

    def _store_metadata(self, namespace: str, key: str, value: Any, *,
                        materialized: bool) -> None:
        self._metadata.setdefault(namespace, {})[key] = copy.deepcopy(value)
        target = (namespace, key)
        if materialized:
            self._materialized_metadata.add(target)
        else:
            self._materialized_metadata.discard(target)

    def _reject_materialized_input_change(self, operation: str) -> None:
        if not self._materialized_metadata:
            return
        attached = [
            f"{namespace}/{key}"
            for namespace, key in sorted(self._materialized_metadata)
        ]
        raise ValueError(
            f"{operation}() cannot change the input basis after materialized pipeline "
            f"metadata {attached}; configure input, schema, media, window, and registry "
            "before enrichment, or branch from the pre-enrichment pipeline and run it again"
        )

    def using(self, registry) -> "Pipeline":
        """Return a pipeline resolving extension names through an explicit overlay registry."""
        from .registry import Registry
        if not isinstance(registry, Registry):
            raise ValueError("using() needs a Registry")
        self._reject_materialized_input_change("using")
        p = self._next("using", "<registry>")
        p._registry = registry.snapshot()
        return p

    def with_metadata(self, namespace: str, key: str, value: Any, *,
                      replace: bool = False) -> "Pipeline":
        """Return a pipeline carrying caller-owned, namespaced JSON metadata.

        The input pipeline is unchanged and :attr:`metadata` returns a defensive copy.
        This method attaches descriptive state only; use :meth:`enrich` when the value
        should be produced by a registered analyzer over this pipeline's input.
        """
        namespace, key = self._metadata_target(namespace, key, replace=replace)
        from .provenance import strict_json_value

        normalized = strict_json_value(
            value, label=f"pipeline metadata {namespace!r}/{key!r}"
        )
        p = self._next(
            "with_metadata", namespace, key, _jsonable(normalized), replace=replace
        )
        p._store_metadata(namespace, key, normalized, materialized=False)
        return p

    def _invoke_operator(self, name: str, params: dict[str, Any]) -> "Pipeline":
        """Validate and expand one registered compositional operator."""
        from .plugins import load_plugins
        from .registry import validate_parameters
        from .registry import default_registry
        if (name not in self._registry.names("operator")
                and _registry_inherits(self._registry, default_registry)):
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

    def apply(self, name: str, **params) -> "Pipeline":
        """Compatibility wrapper for string-based registered-operator dispatch.

        The operator receives a clone and must return a derived ``Pipeline`` containing at
        least one additional ordinary operator. Its descriptor validates parameters before
        expansion. Prefer the direct method generated at registration time, such as
        ``pipeline.signal_diversity(...)`` for ``acme/signal-diversity``.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("apply() needs a non-empty registered operator name")
        return self._invoke_operator(name, params)

    def _with_schema(self, schema: SchemaSpec, opname: str, *args, **kwargs) -> "Pipeline":
        self._reject_materialized_input_change(opname)
        p = self._next(opname, *args, **kwargs)
        p.schema = schema
        for configured in _adapter_chain(p.adapter):
            if hasattr(configured, "schema"):
                configured.schema = schema
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
            field_adapter = _adapter_with_fields(self.adapter)
            if field_adapter is not None:
                mapped = []
                for d in dims:
                    value = field_adapter.fields[d]
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
        target_adapter = _adapter_with_fields(p.adapter)
        if field not in self.schema.fields and target_adapter is not None:
            target_adapter.fields = {
                k: v for k, v in target_adapter.fields.items() if k not in set(dims)}
            target_adapter.fields[field] = mapped
            if hasattr(target_adapter, "available_fields"):
                target_adapter.available_fields = frozenset(target_adapter.fields)
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
        self._reject_materialized_input_change("window")
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
        self._reject_materialized_input_change("join_video")
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
        recorded_mapping = {
            _describe_src(source): _describe_src(uri)
            for source, uri in mapping.items()
        }
        p = self._next("join_video", videos=recorded_mapping, column=column, on=on,
                       tolerance=tolerance, mismatch=mismatch)
        p.adapter = JoinedMediaAdapter(p.adapter, mapping, provider=provider, column=column,
                                       on=on, tolerance=tolerance, mismatch=mismatch)
        p.schema = p.adapter.schema
        return p

    def transform(self, name: str, *, using=None, inputs=None, batch_size=None,
                  device="cpu", **params) -> "Pipeline":
        """Add a registered bounded transform that produces scalar segment features."""
        from .transforms import resolve_transform
        call = resolve_transform(name, using=using, inputs=inputs, batch_size=batch_size,
                                 device=device, params=params, schema=self.schema,
                                 registry=self._registry,
                                 available_features=(set(self._define) | set(self._callables) |
                                     set(self._registered_metrics) |
                                     {o for t in self._transforms for o in t["outputs"]}))
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
        if name in self._metric_names():
            raise ValueError(f"measure {name!r} already declared — names must be unique")
        bare_measure = (spec is None and of is None and map is None and filter is None
                        and scope is None)
        if bare_measure and name not in self._registry.names("metric"):
            from .plugins import load_plugins
            from .registry import default_registry
            if _registry_inherits(self._registry, default_registry):
                load_plugins()
        p = self._next("measure", name, spec, of=of, map=map, filter=filter, scope=scope)
        if callable(spec):
            if of is not None or map is not None or filter is not None or scope is not None:
                raise ValueError(f"measure {name!r}: a callable metric takes no "
                                 f"of/map/filter/scope kwargs")
            p._registry = self._registry.child()
            p._registry.register_metric(name, spec)
            p._callables[name] = spec
            return p
        if (bare_measure
                and name in self._registry.names("metric") and name not in self.schema.fields
                and name not in self.schema.globals):
            p._registered_metrics.append(name)
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
        if bins == 1:
            import warnings
            warnings.warn(f"{operator}({metric!r}, bins=1): every segment lands in one bin; "
                          "the axis adds no selection signal", stacklevel=2)
        p = self._next(operator, metric, bins=bins)
        p._axes.append((metric, bins))
        return p

    def score(self, name: str, out: Optional[str] = None, **params) -> "Pipeline":
        """Add a whole-pool SegmentScorer; its output feature is `out` (default: name)."""
        from .plugins import load_plugins
        from .registry import default_registry
        if (name not in self._registry.names("scorer")
                and _registry_inherits(self._registry, default_registry)):
            load_plugins()
        if name not in self._registry.names("scorer"):
            raise ValueError(f"unknown scorer {name!r}; registered: "
                             f"{self._registry.names('scorer')}")
        from .scorers import validate_scorer_parameters
        validate_scorer_parameters(name, params, registry=self._registry)
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
        resume, sample_hz, and progress callback or display mode. Unknown options are loud."""
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
        resolved_progress = None
        if "progress" in kw:
            from .progress import progress_callback

            resolved_progress = progress_callback(kw["progress"])
        if "executor" in kw and kw["executor"] not in {"local", "ray"}:
            raise ValueError("executor must be 'local' or 'ray'")
        for key in ("memory_budget", "batch_size", "max_in_flight"):
            if key in kw and kw[key] is not None and (
                    not isinstance(kw[key], int) or isinstance(kw[key], bool) or kw[key] < 1):
                raise ValueError(f"{key} must be a positive integer or None")
        for key in ("checkpoint_dir", "spill_dir"):
            if key in kw and kw[key] is not None and (
                    not isinstance(kw[key], str) or not kw[key]):
                raise ValueError(f"{key} must be a non-empty path string or None")
        if "approximation" in kw and kw["approximation"] not in {"exact", "approximate"}:
            raise ValueError("approximation must be 'exact' or 'approximate'")
        p = self._next("options", mode=mode, **kw)
        if mode is not None:
            p._mode = mode
            defaults = {"on_error": "raise"} if mode == "strict" else {"on_error": "skip"}
            for key, value in defaults.items():
                if key not in self._explicit_options and key not in kw:
                    p._options[key] = value
        p._options.update(kw)
        if "progress" in kw:
            p._options["progress"] = resolved_progress
            p._progress_spec = kw["progress"]
        p._explicit_options.update(kw)
        return p

    def expect(self, **expectations) -> "Pipeline":
        """Attach explicit quality gates evaluated by ``preview()`` and ``review()``."""
        from .quality import validate_expectations
        resolved = validate_expectations(expectations)
        p = self._next("expect", **resolved)
        p._expectations.update(resolved)
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
        if normalize is not None and normalize not in {"quantile", "minmax", "zscore", "none"}:
            raise ValueError("curate normalize must be quantile, minmax, zscore, or none")
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
            raise ValueError("curate(): no selection method resolvable — add .bin(...) "
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
        """Configure selection; preferred alias for :meth:`curate`."""
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
                raise ValueError("no selection method resolvable — add .bin(...) axes, "
                                 "pass weights= to .sample(), or name a method")
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
        """Assemble and validate the strategy against this pipeline's registry snapshot."""
        from .strategy import load_strategy
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

    def _optional_extras(self) -> list[str]:
        """Best-effort install extras implied by the resolved pipeline, without importing them."""
        extras: set[str] = set()
        adapter_extra = {
            "HDF5Adapter": "array", "ParquetDatasetAdapter": "parquet",
            "HuggingFaceAdapter": "hf", "ZarrAdapter": "zarr",
            "VideoAdapter": "video", "JoinedMediaAdapter": "video",
        }.get(type(self.adapter).__name__)
        if adapter_extra:
            extras.add(adapter_extra)
        if any(str(source).lower().endswith(".parquet") for source in self.sources):
            extras.add("parquet")
        from .io import is_remote
        if any(is_remote(source) for source in self.sources):
            extras.add("cloud")
        if self._options["executor"] == "ray":
            extras.add("ray")
        if self._options["memory_budget"] is not None:
            extras.add("scale")
        if self._prepass or self._method in {
                "cluster-balanced", "k-center", "facility", "dsir", "rare", "typical"}:
            extras.add("methods")
        for call in self._transforms:
            try:
                extra = self._registry.descriptor("transform", call["implementation"]).extra
            except ValueError:
                extra = None
            if extra:
                extras.add(extra)
        return sorted(extras)

    def _estimate_sources(self, *, max_sources: int = 3,
                          max_ticks_per_source: int = 10_000) -> dict:
        """Inspect a bounded source sample and estimate pool size without retaining rows."""
        sampled = self.sources[:max_sources]
        observed_ticks = 0
        observed_candidates = 0
        observed_entities = 0
        truncated = False
        segmenter = self._segmenter_or_default()
        from itertools import islice
        for source in sampled:
            ticks = (
                self.adapter.iter_ticks(source)
                if hasattr(self.adapter, "iter_ticks")
                and not getattr(self.adapter, "in_memory", False)
                else iter(self.adapter.ticks(source))
            )
            batch = list(islice(ticks, max_ticks_per_source + 1))
            if len(batch) > max_ticks_per_source:
                truncated = True
                batch = batch[:max_ticks_per_source]
            count = len(batch)
            observed_ticks += count
            observed_candidates += max(0, 1 + (count - segmenter.window) // segmenter.stride)
            observed_entities += len({entity for tick in batch for entity in tick.entities})
        scale = len(self.sources) / len(sampled) if sampled else 0.0
        return {
            "sampled_sources": len(sampled), "total_sources": len(self.sources),
            "max_ticks_per_source": max_ticks_per_source,
            "ticks_observed": observed_ticks,
            "candidate_windows_estimate": round(observed_candidates * scale),
            "entities_per_source_estimate": (
                observed_entities / len(sampled) if sampled else 0.0),
            "lower_bound": truncated,
            "basis": "bounded source sample" if truncated or len(sampled) < len(self.sources)
                     else "all sources",
        }

    # ---- inspectors (cheap, non-terminal) ------------------------------------------------
    def head(self, k: int = 5) -> list:
        """First k Ticks of the first source (reads only what it needs)."""
        return list(islice(iter(self.adapter.ticks(self.sources[0])), int(k)))

    def describe(self) -> dict:
        """The pipeline's resolved state — no data is touched."""
        described_options = {
            k: _jsonable(self._progress_spec if k == "progress" else v)
            for k, v in self._options.items()
        }
        return {"schema": self.schema.to_dict(),
                "strategy": self._raw_strategy(require_method=False),
                "output": None,
                "segmenter": dict(self._window_desc),
                "options": {**described_options,
                            "mode": self._mode,
                            "overrides": sorted(self._explicit_options)},
                "sources": _describe_src(self.sources),
                "expectations": dict(self._expectations),
                "metadata": self.metadata}

    def preview(self, *, max_sources: int = 3, max_ticks_per_source: int = 10_000,
                examples: int = 5):
        """Read a bounded source sample and summarize fields, missingness, and examples."""
        from .review import preview_dataset
        report = preview_dataset(schema=self.schema, adapter=self.adapter, sources=self.sources,
                                 max_sources=max_sources,
                                 max_ticks_per_source=max_ticks_per_source,
                                 examples=examples)
        if self._expectations:
            from .quality import check_quality
            report["quality"] = check_quality(report, self._expectations).to_dict()
        return report

    def review(self, n: Any = _UNSET, *, path=None, examples: int = 10,
               hist_bins: int = 10, **output_options):
        """Review count selection, or write and review an hours-budget selection.

        Count review intentionally uses the reference ``collect`` path so it can expose the
        complete candidate pool. Hours review needs ``path=`` because spans are defined by the
        output declaration; it reuses the resulting writer decisions without rescoring. Pass
        ``n=None`` explicitly to review every scored and filtered segment, same as
        ``collect(n=None)``; leaving ``n`` unset uses whatever ``.sample(n=...)` configured.
        """
        report: Any
        if self._hours is not None:
            if n is not _UNSET:
                raise ValueError("hours review does not accept n")
            if path is None:
                raise ValueError("hours review needs path= plus any output options")
            result = self.output(path, **output_options)
            from .review import hours_selection_review
            report = hours_selection_review(result)
        else:
            if path is not None or output_options:
                raise ValueError("path/output options are only used by hours review")
            result = self.collect(n=n)
            from .review import selection_review
            report = selection_review(result, seed=self._options["seed"], examples=examples,
                                      hist_bins=hist_bins)
        if self._expectations:
            from .quality import check_quality
            report["quality"] = check_quality(report, self._expectations).to_dict()
        return report

    def explain(self, *, inspect_sources: bool = False):
        """Return a stable, structured execution plan without reading rows by default."""
        from .results import ExecutionPlan
        warnings = []
        categorical = [n for n, f in self.schema.fields.items() if f.kind == "categorical"]
        if categorical:
            warnings.append({"code": "categorical-output-policy", "message":
                             f"categorical fields {categorical} need an explicit output policy",
                             "hint": "Pass categorical='omit', 'integer-vocab', or 'encoder' "
                                     "to output()."})
        if self._options["on_error"] == "skip":
            # F2: worded prospectively ("configured to", not "were") -- this fires on every
            # pipeline using the default policy, including a totally clean run with nothing
            # to report, and read as "something already went wrong" at a skim.
            warnings.append({"code": "source-errors-skipped", "message":
                             "on_error='skip' (the default): a source error during this run "
                             "would be silently admitted as a data gap instead of stopping it",
                             "hint": "Use .options(mode='strict') while developing to fail fast."})
        if self._method is None and not self._axes and not self._weights:
            warnings.append({"code": "selection-unresolved", "message":
                             "no selection method or budget has been configured",
                             "hint": "Add .sample('equal', n=...) for a count or "
                                     ".sample('top-k', hours=..., weights=...) for duration."})
        if inspect_sources and not self._in_memory and hasattr(self.adapter, "check_source"):
            for source in self.sources:
                self.adapter.check_source(source)
        strategy = self._raw_strategy(require_method=False)
        from .planning import compile_physical_plan
        physical = compile_physical_plan(
            self, memory_budget=self._options["memory_budget"],
            spill_path=self._options["spill_dir"])
        warnings.extend(dict(warning) for warning in physical.warnings)
        stages = tuple(stage.to_dict() for stage in physical.stages)
        options = dict(self.describe()["options"])
        options["selection_budget"] = (
            {"kind": "count", "value": self._n} if self._n is not None else
            {"kind": "hours", "value": self._hours} if self._hours is not None else None)
        options["optional_extras"] = self._optional_extras()
        options["quality_expectations"] = dict(self._expectations)
        if inspect_sources:
            options["source_estimate"] = self._estimate_sources()
        return ExecutionPlan(source={"kind": type(self.adapter).__name__,
                                     "count": len(self.sources),
                                     "sources": _describe_src(self.sources)},
                             schema=self.schema.to_dict(), window=dict(self._window_desc),
                             strategy=strategy, output=None,
                             options=options, stages=stages,
                             warnings=tuple(warnings), physical_plan=physical.to_dict())

    # ---- terminals -----------------------------------------------------------------------
    def collect(self, n: Any = _UNSET):
        """Count-path selection -> CurateResult. Pass ``n=None`` explicitly for every scored
        and filtered segment, with no selection method/weights/bins required — the count-path
        equivalent of `.report()`'s scored pool, but returned as a `CurateResult` so
        `.selected`/`.all_segments` and `.stats()` work the same way they do for a real
        selection."""
        if n is _UNSET:
            n = self._n
            if n is None:
                hint = (" (.sample(hours=...) routes to .output())"
                        if self._hours is not None else "")
                raise ValueError(
                    f"collect() needs a count: pass n=, call .sample(n=...) first, "
                    f"or pass n=None explicitly for every scored and filtered segment{hint}")
        from .strategy import load_strategy, validate_target_size
        n = validate_target_size(n)
        self._check_exec()
        if n is None:
            raw = self._raw_strategy(require_method=False)
            # No selection is run when target_size=None (curate() bypasses it entirely), but
            # load_strategy() still defaults a missing "selection" block to "uniform-bins",
            # which then fails validation for want of binning.axes. "none" needs neither axes,
            # weights, nor predicates, and means exactly this: no selection method, return
            # every segment the universal predicate pre-filter admits.
            if "method" not in raw.get("selection", {}):
                raw.setdefault("selection", {})["method"] = "none"
            transform_outputs = {o for t in self._transforms for o in t["outputs"]}
            strategy = load_strategy(raw, self.schema, registry=self._registry,
                                     available_metrics=transform_outputs)
        else:
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

    def report(self, **kw) -> DatasetReport:
        """Feature pass only (no selection) -> dataset_report over the scored pool."""
        self._check_exec()
        from .report import dataset_report
        from .scoring import score_sources
        scorer_outs = {s["as"] for s in self._scorers}
        transform_outs = {o for t in self._transforms for o in t["outputs"]}
        registered = [nm for nm in [*self._callables, *self._registered_metrics]
                      if nm not in scorer_outs and nm not in transform_outs]
        segments, _errors, _tracks = score_sources(
            schema=self.schema, adapter=self.adapter, sources=self.sources,
            define=dict(self._define), registered=registered,
            segmenter=self._segmenter_or_default(),
            tags_spec={"from": list(self._tags)} if self._tags else None,
            group_of=self._group_of, workers=self._options["workers"],
            on_error=self._options["on_error"], streaming=self._options["streaming"],
            progress=self._options["progress"], executor=self._options["executor"],
            registry=self._registry,
            required_fields={token.split(":", 1)[1]
                             for call in self._transforms for token in call["inputs"]
                             if token.startswith("field:")})
        if self._transforms:
            from .transforms import apply_transforms
            segments = apply_transforms(
                self._transforms, segments, adapter=self.adapter,
                registry=self._registry, progress=self._options["progress"],
                on_error=self._options["on_error"], errors=_errors)
        if self._scorers:
            from .scorers import apply_scorers, parse_scorers
            apply_scorers(parse_scorers([dict(s) for s in self._scorers]), segments,
                          schema=self.schema, adapter=self.adapter,
                          seed=self._options["seed"], progress=self._options["progress"],
                          registry=self._registry, on_error=self._options["on_error"],
                          errors=_errors)
        return dataset_report(segments, **kw)

    def stats(self, **kwargs):
        """Summarize this pipeline at its current uncurated or count-selected stage.

        With no count selection this profiles the input and complete candidate pool. After
        ``sample(n=...)`` it compares candidates with the selected set. Hours admission is
        finalized by the output layer, so an hours-configured pipeline reports candidates and
        directs callers to ``output(path).stats()`` for admitted/emitted statistics.

        Thin sugar for ``kuration.stats(pipeline, **kwargs)``: every keyword that function
        accepts (``hist_bins``, ``max_sources``, ``max_ticks_per_source``, ``examples``,
        ``progress``, ...) is forwarded and validated there, so the two spellings can never
        drift.
        """
        from .stats import stats

        return stats(self, **kwargs)

    def _artifact_datasets(self) -> list[str]:
        """Written dataset directories this pipeline reads, if its adapter names any.

        The seam is one optional adapter method, ``datasets()`` (implemented by
        ``CuratedDatasetAdapter``): an adapter that reads kuration's own artifacts can say so,
        which is what lets ``analyze()``/``check()`` reach the artifact-level hooks instead of
        only the tick stream.
        """
        for current in _adapter_chain(self.adapter):
            names = getattr(current, "datasets", None)
            if callable(names):
                return [str(dataset) for dataset in names()]
        return []

    def _single_artifact(self, caller: str) -> str:
        datasets = self._artifact_datasets()
        if not datasets:
            raise ValueError(
                f"{caller} needs an artifact-backed pipeline — an adapter that names written "
                f"datasets, such as input(adapter='curated', dataset_dir=...); this pipeline "
                f"reads {type(self.adapter).__name__}")
        if len(datasets) > 1:
            raise ValueError(
                f"{caller} needs exactly one written dataset; this pipeline reads "
                f"{len(datasets)}: {datasets} — analyze them one at a time")
        return datasets[0]

    def analysis_mode(self, mode: Optional[str] = None) -> str:
        """Resolve the analyzer execution mode this pipeline runs under.

        ``None`` resolves to ``"dataset"`` when the source is a written kuration dataset and
        ``"stream"`` otherwise — exactly the rule :meth:`analyze`/:meth:`enrich`/:meth:`check`
        apply. Domain operators that accept mode-specific options call this to reject an
        inapplicable option up front instead of re-deriving the rule from pipeline internals.
        """
        if mode not in {None, "dataset", "stream"}:
            raise ValueError("analysis_mode(mode=...) must be 'dataset', 'stream', or None")
        return mode or ("dataset" if self._artifact_datasets() else "stream")

    def _drive_analyzer(self, name: str, parameters: Optional[dict] = None, *,
                        mode: Optional[str] = None, inputs=None,
                        batch_size: Optional[int] = None, analyzer=None,
                        _caller: str = "analyzer", **options) -> tuple[dict, str]:
        """Drive one analyzer once and return its result plus resolved execution mode."""
        if not isinstance(name, str) or not name:
            raise ValueError(f"{_caller}() needs a registered analyzer name")
        if mode not in {None, "dataset", "stream"}:
            raise ValueError(
                f"{_caller}(mode=...) must be 'dataset', 'stream', or None"
            )
        if inputs is not None and (not isinstance(inputs, (list, tuple)) or not inputs):
            raise ValueError(
                f"{_caller}(inputs=...) must be a non-empty sequence of tokens"
            )
        resolved = self.analysis_mode(mode)
        from .analyzers import drive_dataset_analyzer, get_analyzer, run_stream_analyzer

        if resolved == "dataset":
            dataset = self._single_artifact(f"{_caller}(mode='dataset')")
            if inputs is not None or batch_size is not None:
                raise ValueError(
                    f"{_caller}(inputs=/batch_size=) apply to the tick-stream path; "
                    "dataset analysis reads written modalities"
                )
            if analyzer is None:
                analyzer = get_analyzer(name, registry=self._registry,
                                        **dict(parameters or {}))
            elif parameters:
                raise ValueError("pass analyzer= or parameters=, not both")
            result = drive_dataset_analyzer(analyzer, name, dataset, options=options)
            return result, resolved
        self._check_exec()
        result = run_stream_analyzer(
            name, schema=self.schema, adapter=self.adapter, sources=self.sources,
            segmenter=self._segmenter_or_default(), parameters=parameters, options=options,
            inputs=inputs, batch_size=batch_size, on_error=self._options["on_error"],
            progress=self._options["progress"], registry=self._registry, analyzer=analyzer)
        return result, resolved

    def _enrichment_basis(self, name: str, *, mode: str, inputs) -> str:
        """Fingerprint the pipeline state a materialized analyzer result describes."""
        from .checkpoints import canonical_hash

        payload = {
            "analyzer": self._registry.descriptor("analyzer", name).to_dict(),
            "adapter": f"{type(self.adapter).__module__}.{type(self.adapter).__qualname__}",
            "datasets": _jsonable(self._artifact_datasets()),
            "inputs": _jsonable(inputs),
            "mode": mode,
            "schema": self.schema.to_dict(),
            "segmenter": dict(self._window_desc),
            "sources": _jsonable(self.sources),
        }
        return canonical_hash(payload)

    def enrich(self, name: str, parameters: Optional[dict] = None, *,
               namespace: Optional[str] = None, key: Optional[str] = None,
               replace: bool = False, mode: Optional[str] = None, inputs=None,
               batch_size: Optional[int] = None, analyzer=None, **options) -> "Pipeline":
        """Run a registered analyzer and attach its result to a new pipeline.

        This is the pipeline-preserving counterpart of :meth:`analyze`. Domain
        operators normally wrap it with a direct method and a stable metadata location.
        The analyzer executes exactly once; its JSON result, component descriptor,
        invocation, and input-basis fingerprint are retained under ``metadata``.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("enrich() needs a registered analyzer name")
        parts = name.split("/", 1)
        default_namespace = parts[0] if len(parts) == 2 else "analysis"
        default_key = _friendly_operator_part(parts[-1])
        namespace, key = self._metadata_target(
            default_namespace if namespace is None else namespace,
            default_key if key is None else key,
            replace=replace,
        )
        result, resolved = self._drive_analyzer(
            name, parameters, mode=mode, inputs=inputs, batch_size=batch_size,
            analyzer=analyzer, _caller="enrich", **options
        )
        from .provenance import strict_json_value

        normalized_result = strict_json_value(
            result, label=f"analyzer {name!r} result for pipeline metadata"
        )
        request = strict_json_value(
            {
                "parameters": _jsonable(dict(parameters or {})),
                "options": _jsonable(options),
                "inputs": _jsonable(inputs),
                "batch_size": batch_size,
                "owned_analyzer": analyzer is not None,
            },
            label=f"analyzer {name!r} metadata request",
        )
        attachment = {
            "component": self._registry.descriptor("analyzer", name).to_dict(),
            "mode": resolved,
            "basis_sha256": self._enrichment_basis(name, mode=resolved, inputs=inputs),
            "request": request,
            "result": normalized_result,
        }
        validated_attachment = strict_json_value(
            attachment, label=f"pipeline enrichment {namespace!r}/{key!r}"
        )
        p = self._next(
            "enrich", name, namespace=namespace, key=key, replace=replace,
            mode=mode, inputs=inputs, batch_size=batch_size,
            parameters=_jsonable(dict(parameters or {})),
            options=_jsonable(options), owned_analyzer=analyzer is not None,
        )
        p._store_metadata(namespace, key, validated_attachment, materialized=True)
        return p

    def analyze(self, name: str, parameters: Optional[dict] = None, *,
                mode: Optional[str] = None, inputs=None, batch_size: Optional[int] = None,
                analyzer=None, **options):
        """Run a registered stateful analyzer over this pipeline. Terminal.

        Two modes, resolved from the pipeline's own shape and overridable with ``mode=``:

        * ``"stream"`` (default for a raw pipeline) drives the analyzer's ``add_batch`` hook
          over this pipeline's segmented tick stream, with the window from ``.window(...)``,
          the analyzer's declared lookbehind/lookahead context, and lazily decoded media —
          one bounded pass, safe under ``.options(streaming=True)``. Segment ids are real
          ``(source_id, start_idx, end_idx)`` source coordinates.
        * ``"dataset"`` (default when the source *is* a written kuration dataset) drives the
          analyzer's ``check_dataset`` hook and returns exactly what
          ``run_dataset_analyzer()`` and ``kur-check --analyzer`` return.

        ``parameters`` are analyzer construction parameters, validated against the registered
        descriptor; ``**options`` belong to the hook being driven (``check_dataset`` options,
        or ``add_batch`` keywords). ``inputs=``/``batch_size=`` remap and size the stream
        path's declared inputs. ``analyzer=`` drives an instance you already own — its
        accumulated evidence stays yours after the call.
        """
        result, _resolved = self._drive_analyzer(
            name, parameters, mode=mode, inputs=inputs, batch_size=batch_size,
            analyzer=analyzer, _caller="analyze", **options
        )
        return result

    def check(self, policy: Optional[dict] = None, *, verify: bool = False, **expectations):
        """Evaluate quality gates over the cheapest evidence this pipeline can produce.

        The programmatic counterpart of ``kur-check --policy``: an artifact-backed pipeline is
        inspected and gated exactly as the CLI gates it, and any other pipeline is gated on a
        bounded input preview. Expectations come from ``.expect(...)``, a ``policy`` mapping
        (the ``{"expectations": {...}}`` policy-file shape is accepted), and keywords, in that
        precedence order. Nothing raises on failure — the returned ``QualityGateResult``
        reports every check; use ``check_quality()`` when a failure should raise. Selection,
        budget, and shard gates need selection evidence: run them through ``.review()`` or
        gate the written artifact.
        """
        from .quality import evaluate_quality, validate_expectations
        if not isinstance(verify, bool):
            raise ValueError("check(verify=...) must be true or false")
        merged = dict(self._expectations)
        if policy is not None:
            if not isinstance(policy, dict):
                raise ValueError("check(policy=...) must be a quality-policy dict")
            declared = policy.get("expectations", policy)
            if not isinstance(declared, dict):
                raise ValueError("quality policy expectations must be a JSON object")
            merged.update(declared)
        merged.update(expectations)
        if not merged:
            raise ValueError("check() needs quality expectations: declare .expect(...), pass "
                             "policy=, or pass expectation keywords")
        validate_expectations(merged)          # eager, loud, before touching data
        datasets = self._artifact_datasets()
        if datasets:
            from .inspection import inspect_dataset
            report: Any = inspect_dataset(
                self._single_artifact("check()"),
                verify=verify or bool(merged.get("require_verified")),
                registry=self._registry)
        else:
            if verify:
                raise ValueError("check(verify=True) applies to an artifact-backed pipeline; "
                                 "an input preview has no written shards to decode")
            from .review import preview_dataset
            report = preview_dataset(schema=self.schema, adapter=self.adapter,
                                     sources=self.sources)
        return evaluate_quality(report, merged)

    def experiment(
        self,
        arms: dict,
        n: int,
        baseline: Optional[str] = None,
    ) -> ExperimentResult:
        """Multi-arm comparison over THIS pipeline's sources. `arms` maps name ->
        Pipeline (its compiled strategy) or raw strategy dict."""
        from .strategy import require_target_size
        n = require_target_size(n)
        self._check_exec()
        from .experiment import Arm, run_experiment
        from .strategy import load_strategy
        built = []
        for name, arm in arms.items():
            if isinstance(arm, Pipeline):
                incompatible = []
                if arm.schema != self.schema:
                    incompatible.append("schema")
                if arm.sources != self.sources or arm._ops[:1] != self._ops[:1]:
                    incompatible.append("input sources/adapter")
                if arm._window_desc != self._window_desc:
                    incompatible.append("window")
                if arm._options != self._options:
                    incompatible.append("execution options")
                if arm._group_of is not self._group_of:
                    incompatible.append("grouping callback")
                if incompatible:
                    raise ValueError(
                        f"experiment arm {name!r} changes {', '.join(incompatible)}; arms "
                        "share this pipeline's input/window/execution and may vary only "
                        "features, transforms, scorers, filters, and selection")
                built.append(Arm(name, arm._compile(), tuple(arm._transforms), arm._registry))
            elif isinstance(arm, dict):
                built.append(Arm(name, load_strategy(arm, self.schema,
                                                    registry=self._registry,
                                                    available_metrics={
                                                        o for t in self._transforms
                                                        for o in t["outputs"]}),
                                 tuple(self._transforms), self._registry))
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

    @_leased_output
    def output(self, path, *, preset: Optional[str] = None, spec=None,
               container: Optional[str] = None, shard_length: int = 64,
               short_policy: str = "keep", modalities: Optional[dict] = None,
               include=None, exclude=None, encoder=None, span_steps: Optional[int] = None,
               span_seconds: Optional[float] = None, categorical: Optional[str] = None,
               preset_params: Optional[dict] = None, force: bool = False, dry_run: bool = False,
               inspect_sources: bool = False):
        """Write shards + manifests + config.json. `.sample(hours=...)` routes through the
        hours-budget `write_dataset`; `.sample(n=...)` selects then writes each chosen
        segment as a sub-track. config.json gains a 'pipeline' key: the operator chain."""
        self._check_exec()
        if (not isinstance(force, bool) or not isinstance(dry_run, bool)
                or not isinstance(inspect_sources, bool)):
            raise ValueError("force, dry_run, and inspect_sources must be true or false")
        from .io import _normalize_explicit_path
        path = _normalize_explicit_path(path, label="path")
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
                spec = OutputSpec.from_dict(spec, registry=self._registry)
            if not isinstance(spec, OutputSpec):
                raise ValueError("spec must be an OutputSpec or output-spec dict")
            spec = OutputSpec.from_dict(spec.to_dict(), registry=self._registry)
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
            spec = output_preset(preset, schema=self.schema, registry=self._registry, **params)
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
                spec = OutputSpec.from_dict(raw_spec, registry=self._registry)
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
                                         "short_policy": short_policy, "container": container},
                                        registry=self._registry)
        spec.validate_schema(self.schema)
        from .output.containers import validate_container_availability
        validate_container_availability(spec.container, registry=self._registry)
        # Count-mode windows are a fixed, statically-known length, so this is checkable up
        # front; an hours-budget span's length depends on runtime budget/group_of resolution
        # inside _write_dataset_impl, so it isn't covered here.
        shard_chop_warning = None
        if self._hours is None:
            window = self._segmenter_or_default().window
            if window % spec.shard_length != 0:
                full, remainder = divmod(window, spec.shard_length)
                if spec.short_policy == "drop":
                    tail = f"its trailing {remainder} steps are dropped (short_policy='drop')"
                elif spec.short_policy == "pad":
                    tail = (f"plus one padded to length {spec.shard_length} for the trailing "
                            f"{remainder} real steps (short_policy='pad')")
                else:
                    tail = f"plus one of length {remainder} (short_policy='keep')"
                shard_chop_warning = {
                    "code": "shard-length-chops-windows",
                    "message": (
                        f"shard_length={spec.shard_length} does not evenly divide the "
                        f"window length ({window}); each window writes {full} full shard"
                        f"{'s' if full != 1 else ''} of length {spec.shard_length}, {tail}"),
                    "hint": "Pass shard_length=<window length> (or a divisor of it) to "
                            "output() for one shard per window.",
                }
        output_op = {"op": "output", "args": [_describe_src(path)], "kwargs": {
            "preset": preset, "spec": spec.to_dict(), "categorical": categorical,
            "force": force}}
        if isinstance(encoder, str):
            from .output.base import get_encoder
            encoder = get_encoder(encoder, registry=self._registry)
        elif isinstance(encoder, dict):
            from .output.base import get_encoder
            encoder = get_encoder(encoder["name"], registry=self._registry,
                                  **dict(encoder.get("params") or {}))
        strategy = self._compile()
        opt = self._options
        if opt["approximation"] == "approximate":
            raise ValueError(f"selection method {strategy.method!r} has no declared "
                             "approximate physical implementation yet; use "
                             "approximation='exact'")
        if self._hours is None and opt["memory_budget"] is not None:
            from .output.dataset import _minimum_output_working_bytes
            output_minimum = _minimum_output_working_bytes(
                spec, self._segmenter_or_default().window)
            if opt["memory_budget"] < output_minimum:
                raise ValueError(
                    f"memory_budget={opt['memory_budget']} bytes cannot hold one count-output "
                    f"shard accumulator (estimated minimum {output_minimum} bytes); reduce "
                    "shard_length or raise memory_budget")

        if dry_run:
            if self._hours is None and opt["resume"]:
                raise ValueError("resume is supported on the hours path only "
                                 "(.sample(hours=...)); count output is deterministic")
            if self._hours is None and self._n is None:
                raise ValueError("output() needs .sample(n=...) or .sample(hours=...) first")
            if span_steps is not None and span_seconds is not None:
                raise ValueError("pass only one of span_steps or span_seconds")
            if span_steps is not None and (
                    not isinstance(span_steps, int) or isinstance(span_steps, bool)
                    or span_steps < 1):
                raise ValueError("span_steps must be a positive integer")
            if span_seconds is not None and (
                    not isinstance(span_seconds, (int, float))
                    or isinstance(span_seconds, bool) or not math.isfinite(span_seconds)
                    or span_seconds <= 0):
                raise ValueError("span_seconds must be a finite positive number")
            if self._hours is not None and (opt["sample_hz"] or self.schema.sample_hz) is None:
                raise ValueError("hours output needs a sample rate; pass hz= to input(), "
                                 ".at_rate(), or .options(sample_hz=...)")
            effective_hz = opt["sample_hz"] or self.schema.sample_hz
            span_len = span_steps if span_steps is not None else (
                round(span_seconds * effective_hz)
                if span_seconds is not None and effective_hz is not None else None)
            if span_len is not None and span_len < 1:
                raise ValueError("span_seconds is shorter than one sample at the effective "
                                 "sample_hz")
            if (span_len is not None and spec.short_policy == "drop"
                    and span_len < spec.shard_length):
                raise ValueError(
                    f"span_steps={span_len} < shard_length={spec.shard_length} with "
                    "short_policy='drop': every admitted span would chunk to zero shards "
                    "while still consuming the hours budget")
            if self._hours is None and opt["memory_budget"] is not None \
                    and not self._can_run_bounded_count(strategy):
                raise ValueError(
                    "memory_budget requires a bounded exact count plan; currently supported "
                    "only for local top-k/uniform-bins without global filters, scorers, tag "
                    "coverage, or group caps (and with PyArrow installed)")
            if self._hours is not None and opt["memory_budget"] is not None:
                bounded_hours = (strategy.method in {"top-k", "weighted", "uniform-bins"}
                                 and not strategy.filters and not strategy.scorers
                                 and opt["executor"] == "local")
                if bounded_hours:
                    try:
                        import pyarrow  # noqa: F401
                    except ImportError:
                        bounded_hours = False
                if not bounded_hours:
                    raise ValueError("memory_budget requires a bounded exact hours plan; use "
                                     "local top-k/weighted/uniform-bins without global filters "
                                     "or scorers, and install the scale extra")
            from dataclasses import replace
            plan = self.explain(inspect_sources=inspect_sources)
            plan_options = dict(plan.options)
            extras = set(plan_options.get("optional_extras") or ())
            if spec.container == "hdf5":
                extras.add("array")
            elif spec.container == "parquet":
                extras.add("parquet")
            plan_options["optional_extras"] = sorted(extras)
            estimate = plan_options.get("source_estimate")
            if estimate:
                if self._n is not None:
                    selected = min(self._n, estimate["candidate_windows_estimate"])
                    per_segment = max(1, -(-self._segmenter_or_default().window
                                           // spec.shard_length))
                    estimate["output_shards_estimate"] = round(
                        selected * estimate["entities_per_source_estimate"] * per_segment)
                else:
                    effective_hz = opt["sample_hz"] or self.schema.sample_hz
                    assert effective_hz is not None and self._hours is not None
                    hz = float(effective_hz)
                    seconds = float(self._hours + self._test_hours) * 3600.0
                    estimate["output_shards_estimate"] = max(
                        1, round(seconds * hz / spec.shard_length))
                estimate["output_estimate_assumptions"] = (
                    "before short-policy drops and per-source split constraints")
            output_plan = {**spec.to_dict(), "path": path,
                           "preset": preset_resolution, "write": False}
            warnings = [w for w in plan.warnings
                        if w.get("code") != "categorical-output-policy"]
            omitted = [name for name, field in self.schema.fields.items()
                       if field.kind == "categorical" and name not in spec.to_dict()["modalities"]]
            if omitted:
                warnings.append({"code": "categorical-fields-omitted",
                                 "message": f"categorical fields {omitted} are not in output",
                                 "hint": "Pass explicit modalities or an output preset with "
                                         "categorical='integer-vocab' or 'encoder' to retain them."})
            if shard_chop_warning:
                warnings.append(shard_chop_warning)
            return replace(plan, output=output_plan, options=plan_options,
                           warnings=tuple(warnings))

        if self._hours is not None:                       # ---- hours-budget path ----
            from .output.dataset import _write_dataset_impl
            hours = float(self._hours)
            train_hours, test_hours = hours, self._test_hours
            if self._split_test is not None and not self._test_hours_explicit:
                train_hours = (1.0 - self._split_test) * hours
                test_hours = self._split_test * hours
            config_updates = {
                "pipeline": list(self._ops),
                "output_request": output_op,
                "mode": {"name": self._mode, "resolved": self.describe()["options"]},
                "warnings": [dict(w) for w in self.explain().warnings],
            }
            if self._metadata:
                config_updates["metadata"] = _jsonable(self._metadata)
            if preset_resolution:
                config_updates["output_preset"] = preset_resolution
            return _write_dataset_impl(
                path, schema=self.schema, adapter=self.adapter, sources=self.sources,
                strategy=strategy, segmenter=self._segmenter_or_default(), spec=spec,
                encoder=encoder, train_hours=train_hours, test_hours=test_hours,
                sample_hz=opt["sample_hz"] or None, seed=opt["seed"],
                group_of=self._group_of, span_steps=span_steps, span_seconds=span_seconds,
                workers=opt["workers"], on_error=opt["on_error"], resume=opt["resume"],
                force=force, streaming=opt["streaming"], progress=opt["progress"],
                transforms=self._transforms, registry=self._registry,
                executor=opt["executor"], memory_budget=opt["memory_budget"],
                batch_size=opt["batch_size"], checkpoint_dir=opt["checkpoint_dir"],
                spill_dir=opt["spill_dir"], approximation=opt["approximation"],
                max_in_flight=opt["max_in_flight"], _config_updates=config_updates)

        # ---- count path -------------------------------------------------------------
        if opt["resume"]:
            raise ValueError("resume is supported on the hours path only "
                             "(.sample(hours=...)); the count path rewrites its "
                             "selection deterministically per run")
        if self._n is None:
            raise ValueError("output() needs .sample(n=...) or .sample(hours=...) first")
        if opt["memory_budget"] is not None and (
                getattr(self.adapter, "in_memory", False)
                or getattr(self.adapter, "supports_streaming", True) is False
                or not hasattr(self.adapter, "iter_ticks")):
            raise ValueError(
                "memory_budget on count output requires a file-backed, streaming-capable "
                f"adapter; {type(self.adapter).__name__} materializes complete sources")
        can_run_bounded = self._can_run_bounded_count(strategy)
        if opt["memory_budget"] is not None and not can_run_bounded:
            raise ValueError(
                "memory_budget requires a bounded exact count plan; currently supported only "
                "for local top-k/uniform-bins without global filters, scorers, tag coverage, "
                "or group caps (and with PyArrow installed)")
        from .manifest import (
            validate_dataset_destination,
            validate_source_destination_separation,
        )
        destination_state = validate_dataset_destination(path, force=force, resume=False)
        validate_source_destination_separation(path, self.sources, adapter=self.adapter)
        checkpoint_root = spill_root = None
        from .output.base import PassthroughEncoder
        from .provenance import (
            build_run_signature,
            require_reusable_source_identities,
            source_identities_allow_reuse,
        )
        count_signature = build_run_signature(
            schema=self.schema, strategy=strategy, spec=spec,
            segmenter=self._segmenter_or_default(), sources=self.sources,
            adapter=self.adapter, encoder=encoder or PassthroughEncoder(), seed=opt["seed"],
            sample_hz=opt["sample_hz"] or self.schema.sample_hz,
            train_hours=None, test_hours=None, span_len=self._segmenter_or_default().window,
            clip_aggregate="count-selection", group_of=self._group_of, tags_of=None,
            transforms=self._transforms, registry=self._registry)
        source_identities = count_signature["payload"]["sources"]
        if can_run_bounded and not source_identities_allow_reuse(source_identities):
            if opt["memory_budget"] is not None:
                require_reusable_source_identities(
                    source_identities, purpose="memory-bounded checkpoint reuse")
            # The exact in-memory collector has no persisted feature cache and is therefore
            # the safe automatic fallback for logical IDs without content/version evidence.
            can_run_bounded = False
        if can_run_bounded:
            from .work_paths import resolve_bounded_work_paths
            checkpoint_root, spill_root = resolve_bounded_work_paths(
                out_dir=path, sources=self.sources, adapter=self.adapter,
                checkpoint_dir=opt["checkpoint_dir"], spill_dir=opt["spill_dir"])
        result = self._bounded_count_result(
            path, strategy, checkpoint_root=checkpoint_root, spill_dir=spill_root
        ) if can_run_bounded else self.collect()
        if not result.selected:
            raise ValueError("selection is empty — nothing to write (check "
                             "filters/predicates and the segment window)")
        # Everything used to derive shard paths must be validated before force=True can clear
        # the prior generation. This also rejects a selected set that short_policy='drop' would
        # silently turn into a committed zero-shard artifact.
        from .output.shardwriter import entity_id_token
        selected_by_source: dict = {}
        for segment in result.selected:
            selected_by_source.setdefault(segment.source_id, []).append(segment)
        predicted_shards = 0
        for source, segments in selected_by_source.items():
            expected_track_rows = list(result.source_tracks.get(source, ()))
            if not expected_track_rows:
                raise ValueError(f"selected source {source!r} has no scored track metadata")
            expected_lengths = {int(length) for _, length in expected_track_rows}
            if len(expected_lengths) != 1:
                raise ValueError(
                    f"source {source!r} has inconsistent scored track lengths "
                    f"{sorted(expected_lengths)}"
                )
            expected_ticks = next(iter(expected_lengths))
            for entity_id, _length in expected_track_rows:
                entity_id_token(entity_id)
            for segment in segments:
                span_length = int(segment.end_idx) - int(segment.start_idx)
                if span_length < 1 or int(segment.end_idx) > expected_ticks:
                    raise ValueError(
                        f"selected segment {segment.start_idx}:{segment.end_idx} for "
                        f"{source!r} is outside its scored track length {expected_ticks}"
                    )
                shards_per_entity = (
                    span_length // spec.shard_length
                    if spec.short_policy == "drop"
                    else math.ceil(span_length / spec.shard_length)
                )
                predicted_shards += shards_per_entity * len(expected_track_rows)
        if predicted_shards == 0:
            raise ValueError(
                "selection would emit zero shards: every selected segment is shorter than "
                f"shard_length={spec.shard_length} with short_policy='drop'; use "
                "short_policy='keep'/'pad' or reduce shard_length"
            )
        from .manifest import (begin_dataset, clear_dataset_resume_marker,
                               clear_dataset_shards, pin_writing_dataset)
        begin_dataset(path)
        if force and destination_state != "absent":
            clear_dataset_shards(path)
        clear_dataset_resume_marker(path)
        pin_writing_dataset(path)
        written = self._write_selected(path, result, spec, encoder, opt["seed"],
                                       signature=count_signature, progress=opt["progress"])
        from .io import read_text
        from .manifest import write_config
        cfg = json.loads(read_text(f"{path}/config.json"))
        cfg["pipeline"] = list(self._ops)
        cfg["output_request"] = output_op
        if preset_resolution:
            cfg["output_preset"] = preset_resolution
        cfg["mode"] = {"name": self._mode, "resolved": self.describe()["options"]}
        cfg["warnings"] = [dict(w) for w in self.explain().warnings]
        if shard_chop_warning:
            cfg["warnings"].append(shard_chop_warning)
        if self._metadata:
            cfg["metadata"] = _jsonable(self._metadata)
        if getattr(result, "execution", None):
            cfg["execution"] = result.execution
        write_config(f"{path}/config.json", cfg)
        from .manifest import commit_dataset
        commit_dataset(path)
        return written

    def _can_run_bounded_count(self, strategy) -> bool:
        """Whether the implemented exact bounded runtime covers this logical pipeline."""
        if (strategy.method not in {"top-k", "uniform-bins"}
                or strategy.filters or strategy.scorers
                or strategy.tag_coverage or strategy.group_cap
                or self._options["executor"] != "local"):
            return False
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            return False
        return True

    def _bounded_count_result(self, out_dir: str, strategy, *,
                              checkpoint_root: str | None = None,
                              spill_dir: str | None = None):
        """Execute the currently supported exact count plan without a total Python pool."""
        from .bounded import (SizedSegmentPool, checkpoint_feature_stage,
                              resolve_bounded_memory_limits)
        from .checkpoints import (
            CheckpointIdentity,
            StageCheckpoint,
            artifact_checkpoint_summary,
        )
        from .curate import CurateResult
        from .external import external_top_k, external_uniform_bins
        from .planning import compile_physical_plan
        from .plugins import plugin_provenance
        from .provenance import component_identity, source_descriptor
        if checkpoint_root is None:
            from .work_paths import resolve_bounded_work_paths
            checkpoint_root, spill_dir = resolve_bounded_work_paths(
                out_dir=out_dir, sources=self.sources, adapter=self.adapter,
                checkpoint_dir=self._options["checkpoint_dir"],
                spill_dir=self._options["spill_dir"])
        physical = compile_physical_plan(
            self, memory_budget=self._options["memory_budget"], spill_path=spill_dir)
        artifact_physical = {
            **physical.to_dict(),
            "spill_path": None,
            "spill_configured": spill_dir is not None,
        }
        partition_policy = resolve_bounded_memory_limits(
            schema=self.schema, segmenter=self._segmenter_or_default(),
            memory_budget=self._options["memory_budget"],
            batch_size=self._options["batch_size"],
            max_in_flight=self._options["max_in_flight"], transforms=self._transforms)
        logical = {"schema": self.schema.to_dict(), "strategy": strategy.resolved,
                   "window": dict(self._window_desc), "sources": [str(s) for s in self.sources],
                   "transforms": list(self._transforms)}
        transform_outputs = {output for call in self._transforms for output in call["outputs"]}
        transform_inputs = {
            token.split(":", 1)[1]
            for call in self._transforms for token in call["inputs"]
            if token.startswith("feature:")
        }
        registered = sorted(name for name in strategy.referenced_metrics() | transform_inputs
                            if name not in strategy.define and name not in transform_outputs)
        identity = CheckpointIdentity.from_plans(
            logical, artifact_physical,
            input_fingerprints=tuple(source_descriptor(source, self.adapter)
                                     for source in self.sources),
            schema=self.schema.to_dict(), components=tuple([
                *plugin_provenance(registry=self._registry),
                *(self._registry.descriptor("metric", name).to_dict()
                  for name in registered if name in self._registry.names("metric")),
                *(self._registry.descriptor("transform", call["implementation"]).to_dict()
                  for call in self._transforms),
            ]),
            parameters={"seed": self._options["seed"],
                        "adapter": component_identity(self.adapter),
                        "partition_policy": {"version": 1, **partition_policy}})
        checkpoint = StageCheckpoint(
            f"{checkpoint_root.rstrip('/')}/{identity.fingerprint}",
            physical.stages[0].id, identity)
        if self._n is None:  # guarded by output(); keeps this private method type-safe
            raise ValueError("bounded count output requires a target size")
        target_size = self._n
        from .scoring import _cached_group_callback
        group_of = _cached_group_callback(self._group_of)
        stage = checkpoint_feature_stage(
            schema=self.schema, adapter=self.adapter, sources=self.sources,
            define=strategy.define, registered=registered,
            segmenter=self._segmenter_or_default(), checkpoint=checkpoint,
            batch_size=self._options["batch_size"],
            tags_spec=strategy.tags, predicates=strategy.predicates,
            transforms=self._transforms, registry=self._registry,
            group_of=group_of, on_error=self._options["on_error"],
            progress=self._options["progress"],
            max_in_flight=self._options["max_in_flight"],
            memory_budget=self._options["memory_budget"],
            retain_predicate_failures=bool(strategy.predicates))
        marker = checkpoint.finalize([commit.partition_id for commit in stage.commits])
        if strategy.method == "top-k":
            # Preserve the current reference behavior: `_normalize(..., "none")`
            # historically falls through to stable quantile ranks.
            normalize = "quantile" if strategy.normalize == "none" else strategy.normalize
            selected = external_top_k(
                stage.commits, weights=strategy.weights, target_size=target_size,
                normalize=normalize, spill_dir=spill_dir,
                predicates=strategy.predicates)
        elif strategy.method == "uniform-bins":
            selected = external_uniform_bins(
                stage.commits, axes=strategy.axes, target_size=target_size,
                seed=self._options["seed"], spill_dir=spill_dir,
                predicates=strategy.predicates)
        splits = {}
        if self._split_test:
            from .select import assign_splits
            source_groups = {source: group_of(source) if group_of else None
                             for source in stage.source_tracks}
            splits = assign_splits(source_groups, self._split_test, self._options["seed"])
        result = CurateResult(selected=selected, all_segments=SizedSegmentPool(
            stage.candidate_rows), schema=self.schema, strategy=strategy,
            splits=splits, errors=list(stage.errors), source_tracks=dict(stage.source_tracks),
            seed=self._options["seed"])
        result.execution = {
            "physical_plan": artifact_physical,
            "checkpoint": artifact_checkpoint_summary(marker),
                            "bounded": True, "telemetry": stage.telemetry}
        return result

    def _write_selected(self, out_dir: str, result, spec, encoder, seed: int, *, signature,
                        progress=None) -> dict:
        """Count-path write: each selected segment becomes a sub-track per entity, written
        with the same shard mechanics as the hours path."""
        from .io import write_text
        from .manifest import build_config, portable_shard_entries, write_config
        from .output.dataset import _minimum_output_working_bytes
        from .output.shardwriter import entity_id_token
        from .output.stream import stream_source_shards
        by_source: dict = {}
        for s in result.selected:
            by_source.setdefault(s.source_id, []).append(s)
        entries: list = []
        from .progress import emit
        emit(progress, "writing", "start", total=len(by_source))

        def tick_iter(source):
            if hasattr(self.adapter, "iter_ticks") and not getattr(
                    self.adapter, "in_memory", False):
                return self.adapter.iter_ticks(source)
            return iter(self.adapter.ticks(source))

        for source_i, src in enumerate(sorted(by_source, key=str), 1):
            segs = sorted(by_source[src], key=lambda s: s.start_idx)
            expected_track_rows = sorted(
                result.source_tracks.get(src, ()),
                key=lambda row: entity_id_token(row[0]),
            )
            if not expected_track_rows:
                raise ValueError(f"selected source {src!r} has no scored track metadata")
            entity_ids = [entity_id for entity_id, _ in expected_track_rows]
            expected_lengths = {int(length) for _, length in expected_track_rows}
            if len(expected_lengths) != 1:
                raise ValueError(f"source {src!r} has inconsistent scored track lengths "
                                 f"{sorted(expected_lengths)}")
            expected_ticks = next(iter(expected_lengths))
            split = result.splits.get(src, "train")
            jobs = [
                {"eid": entity_id, "span_index": segment_index,
                 "start": segment.start_idx, "end": segment.end_idx, "split": split,
                 # Content-derived, not rank-derived (F1): two curations that each pick a
                 # different span at the same selection rank must not alias one filename.
                 "name_suffix": f"_seg{segment.start_idx:08d}_{segment.end_idx:08d}",
                 "group": segment.group, "meta": dict(segment.features)}
                for segment_index, segment in enumerate(segs) for entity_id in entity_ids]
            jobs_per_pass = len(jobs) or 1
            if self._options["memory_budget"] is not None:
                span_len = max((job["end"] - job["start"] for job in jobs), default=1)
                per_job = _minimum_output_working_bytes(spec, span_len)
                jobs_per_pass = self._options["memory_budget"] // per_job
                if jobs_per_pass < 1:
                    raise ValueError(
                        f"memory_budget={self._options['memory_budget']} bytes cannot hold one "
                        f"count-output shard accumulator (estimated minimum {per_job} bytes); "
                        "reduce shard_length or raise memory_budget")
                if getattr(result, "execution", None):
                    telemetry = result.execution.setdefault("telemetry", {})
                    telemetry["output_accumulator_bytes"] = per_job
                    telemetry["output_jobs_per_pass"] = jobs_per_pass
            for start in range(0, len(jobs), jobs_per_pass):
                entries += stream_source_shards(
                    out_dir, src, tick_iter(src), jobs[start:start + jobs_per_pass],
                    schema=self.schema, spec=spec, encoder=encoder,
                    media_provider=getattr(self.adapter, "provider", None),
                    registry=self._registry, expected_ticks=expected_ticks,
                    expected_entities=entity_ids)
            emit(progress, "writing", "advance", current=source_i, total=len(by_source),
                 source=src, status="ok")
        emit(progress, "writing", "complete", current=len(by_source), total=len(by_source),
             detail=f"{len(entries)} shards")
        from .provenance import assert_sources_unchanged
        from .versioning import ARTIFACT_FORMAT_VERSION
        assert_sources_unchanged(self.sources, self.adapter, signature["payload"]["sources"])
        write_text(f"{out_dir}/dataset_manifest.json",
                   json.dumps({"artifact_format_version": ARTIFACT_FORMAT_VERSION,
                               "commit_protocol": "sidecar-v1",
                               "count": len(entries),
                               "shards": portable_shard_entries(out_dir, entries)}, indent=2))
        results = {"shards": len(entries), "selected": len(result.selected),
                   "segments_total": len(result.all_segments),
                   "sources_skipped": len(result.errors),
                   "n_train_sources": len({entry["source_id"] for entry in entries
                                           if entry["split"] == "train"}),
                   "n_test_sources": len({entry["source_id"] for entry in entries
                                          if entry["split"] == "test"})}
        cfg = build_config(self.schema, result.strategy, results=results, seed=seed)
        cfg["output"] = spec.to_dict()
        cfg["provenance"] = signature
        cfg["pipeline"] = list(self._ops)
        cfg["errors"] = list(result.errors)
        if self._metadata:
            cfg["metadata"] = _jsonable(self._metadata)
        write_config(f"{out_dir}/config.json", cfg)
        from .results import DatasetWriteResult
        return DatasetWriteResult(dataset=str(out_dir), shards=entries,
                                  selected=len(result.selected), results=results,
                                  errors=result.errors)
