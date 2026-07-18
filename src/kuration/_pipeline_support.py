"""Input adapters and validation helpers used by the fluent pipeline facade.

Kept separate so ``pipeline.py`` can focus on fluent plan state and terminal execution. These
objects remain module-level because in-memory adapters must be pickleable.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .adapters.timeseries import TimeseriesAdapter
from .schema import SchemaSpec

GLOB_CHARS = set("*?[")
FILTER_TYPES = {"anomaly", "dedup"}
OPTION_KEYS = {"workers", "on_error", "seed", "streaming", "resume", "sample_hz", "progress",
               "mode", "executor", "memory_budget", "batch_size", "checkpoint_dir",
               "spill_dir", "approximation", "max_in_flight"}
TABULAR_KINDS = {"timeseries", "jsonl", "json-rows", "parquet-dataset"}
SUFFIX_CONTAINER = {".hdf5": "hdf5", ".h5": "hdf5", ".npz": "npz",
                    ".parquet": "parquet", ".json": "json"}

OP_ALIASES = {
    "<=": "<=", "le": "<=", "lte": "<=", "lessorequal": "<=", "leq": "<=",
    ">=": ">=", "ge": ">=", "gte": ">=", "greaterorequal": ">=", "geq": ">=",
    "<": "<", "lt": "<", "less": "<", ">": ">", "gt": ">", "greater": ">",
    "==": "==", "eq": "==", "equal": "==", "=": "==",
    "!=": "!=", "ne": "!=", "neq": "!=", "notequal": "!=",
}

METHOD_ALIASES = {
    "equal": "uniform-bins", "uniform": "uniform-bins", "uniform-bins": "uniform-bins",
    "top": "top-k", "top-k": "top-k", "topk": "top-k",
    "target": "target-distribution", "target-distribution": "target-distribution",
    "cluster": "cluster-balanced", "cluster-balanced": "cluster-balanced",
    "facility": "facility-location", "facility-location": "facility-location",
    "weighted": "weighted", "k-center": "k-center", "dsir": "dsir",
    "rare": "rare", "typical": "typical", "filter": "filter",
}


def jsonable(value):
    """Return a compact JSON-safe provenance representation."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    rendered = repr(value)
    return rendered if len(rendered) <= 200 else rendered[:200] + "..."


def describe_src(src) -> Any:
    if isinstance(src, str):
        return src
    if isinstance(src, (list, tuple)) and src and all(isinstance(item, str) for item in src):
        return list(src)
    return f"<in-memory {type(src).__name__}>"


def auto_qualify(token: str, schema: SchemaSpec) -> str:
    if "." in token:
        return token
    if token in schema.fields:
        return f"{schema.entity_axis}.{token}" if schema.entity_axis else token
    if token in schema.globals:
        return token
    if token in schema.event_streams:
        raise ValueError(
            f"{token!r} is an event stream — name a field: '{token}.<field>' "
            f"(fields: {sorted(schema.event_streams[token]['fields'])})")
    raise ValueError(f"unknown metric source {token!r}; schema fields: "
                     f"{sorted(schema.fields)}, globals: {sorted(schema.globals)}")


def default_field_map(schema: SchemaSpec, *, tabular: bool) -> dict:
    out: dict = {}
    for name, field in schema.fields.items():
        out[name] = ([f"{name}.{i}" for i in range(len(field.dims))]
                     if tabular and field.kind == "vector" else name)
    return out


def expand_row(row: dict) -> dict:
    out = dict(row)
    for key, value in row.items():
        if isinstance(value, np.ndarray) and value.ndim == 1:
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                out[f"{key}.{index}"] = item
    return out


class RowsAdapter(TimeseriesAdapter):
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
        return self._rows_to_ticks([expand_row(row) for row in self._data[source_uri]])

    def source_fingerprint(self, source_uri):
        from .provenance import _state_hash
        return _state_hash(self._data[source_uri])


class TickAdapter:
    in_memory = True

    def __init__(self, schema, data: dict):
        self.schema = schema
        self._data = data

    def ticks(self, source_uri):
        if source_uri not in self._data:
            raise ValueError(f"unknown in-memory source {source_uri!r} "
                             f"(have: {sorted(self._data)})")
        return self._data[source_uri]

    def source_fingerprint(self, source_uri):
        from .provenance import _state_hash
        return _state_hash(self._data[source_uri])


class JsonRowsAdapter(TimeseriesAdapter):
    def ticks(self, source_uri):
        from .infer import load_rows
        return self._rows_to_ticks([expand_row(row) for row in load_rows(source_uri)])

    def iter_ticks(self, source_uri):
        yield from self.ticks(source_uri)
