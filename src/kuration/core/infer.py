"""Schema inference — bootstrap a `SchemaSpec` (+ adapter hints) from a data sample.

The zero-config on-ramp: `sniff` maps a path to an adapter kind, `sample_source` pulls a
small sample in the shape that kind produces, and `infer_schema` classifies it into a
schema plus the column `hints` a `TimeseriesAdapter` needs (`fields=` mapping, time/entity
columns, list-cell expansion). Inference is deliberately conservative and loud: a column
it cannot classify unambiguously raises with a pointer at `kinds=` / an explicit
`schema=` — no silent guessing.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from .buffer import Tick
from .schema import SchemaSpec

_EXT_KINDS = {
    ".csv": "timeseries", ".parquet": "timeseries", ".jsonl": "jsonl",
    ".npz": "npz", ".h5": "hdf5", ".hdf5": "hdf5", ".json": "json-rows",
    ".zarr": "zarr",
    ".mp4": "video", ".m4v": "video", ".mov": "video",
}
_TIME_CANDIDATES = ("t", "time", "timestamp", "ts", "step", "frame", "tick")
_ENTITY_CANDIDATES = ("entity", "entity_id", "id", "agent", "player", "device", "object")
_NUMERIC = (int, float, np.integer, np.floating)
_BOOL = (bool, np.bool_)


# ---- source sniffing --------------------------------------------------------
def sniff(source) -> str:
    """Adapter kind from a path, by extension (or a directory of .parquet parts)."""
    p = str(source)
    ext = os.path.splitext(p)[1].lower()
    if ext in _EXT_KINDS:
        return _EXT_KINDS[ext]
    if Path(p).is_dir():
        from .io import glob
        if glob(f"{p.rstrip('/')}/**/*.parquet"):
            return "parquet-dataset"
        raise ValueError(
            f"cannot infer an adapter for directory {p!r}: it contains no .parquet parts — "
            f"pass an explicit adapter= for other layouts")
    raise ValueError(
        f"cannot infer an adapter for {p!r}: supported extensions are .csv/.parquet "
        f"(timeseries), .jsonl, .npz, .h5/.hdf5 (hdf5), .zarr, .json (rows), or a directory of "
        f".parquet parts, or .mp4/.m4v/.mov video — pass an explicit adapter= for anything else")


# ---- plain-JSON rows --------------------------------------------------------
def load_rows(source) -> list[dict]:
    """Read a plain .json file as a list of row dicts: a JSON array of objects is returned
    as-is; a columnar dict of equal-length lists is zipped into rows. Anything else raises."""
    from .io import read_text
    doc = json.loads(read_text(source))
    bad = ("expected a JSON array of row objects or a columnar dict of equal-length lists")
    if isinstance(doc, list):
        for i, r in enumerate(doc):
            if not isinstance(r, dict):
                raise ValueError(f"{source!s}: {bad}; array element {i} is "
                                 f"{type(r).__name__}, not an object")
        return doc
    if isinstance(doc, dict):
        if not doc:
            raise ValueError(f"{source!s}: {bad}; got an empty dict")
        non_list = {k: type(v).__name__ for k, v in doc.items() if not isinstance(v, list)}
        if non_list:
            raise ValueError(f"{source!s}: {bad}; non-list column value(s): {non_list}")
        lengths = {k: len(v) for k, v in doc.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"{source!s}: {bad}; ragged column lengths: {lengths}")
        n = next(iter(lengths.values()))
        return [{k: doc[k][i] for k in doc} for i in range(n)]
    raise ValueError(f"{source!s}: {bad}; got {type(doc).__name__}")


# ---- sampling ---------------------------------------------------------------
def _coerce_cell(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _csv_rows(path, n: int) -> list[dict]:
    from .io import open_file
    rows: list[dict] = []
    with open_file(path, "rt") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= n:
                break
            rows.append({k: _coerce_cell(v) for k, v in row.items()})
    return rows


def _parquet_rows(path, n: int) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError("sampling .parquet needs the optional extra: "
                          "pip install 'kuration-sdk[parquet]'") from e
    from .io import local_copy
    with local_copy(path) as p:
        batch = next(pq.ParquetFile(p).iter_batches(batch_size=n), None)
        return batch.to_pylist()[:n] if batch is not None else []


def sample_source(source, *, n: int = 200) -> tuple[str, object]:
    """Sniff `source` and pull a small sample in the shape `infer_schema` accepts:
    (kind, rows | dict-of-arrays)."""
    kind = sniff(source)
    p = str(source)
    if kind == "timeseries":
        if p.lower().endswith(".parquet"):
            return kind, _parquet_rows(p, n)
        return kind, _csv_rows(p, n)
    if kind == "jsonl":
        from .io import open_file
        rows: list[dict] = []
        with open_file(p, "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
                if len(rows) >= n:
                    break
        return kind, rows
    if kind == "json-rows":
        return kind, load_rows(source)[:n]
    if kind == "npz":
        from .io import local_copy
        with local_copy(p) as lp:
            with np.load(lp, allow_pickle=False) as z:
                sample = {k: np.asarray(z[k][:n]) for k in z.files}
        return kind, sample
    if kind == "hdf5":
        try:
            import h5py
        except ImportError as e:
            raise ImportError("sampling HDF5 needs h5py: pip install 'kuration-sdk[array]'") from e
        from .io import local_copy
        with local_copy(p) as lp, h5py.File(lp, "r") as f:
            sample = {k: np.asarray(f[k][:n]) for k in f.keys()
                      if isinstance(f[k], h5py.Dataset)}
        return kind, sample
    if kind == "zarr":
        try:
            import zarr
        except ImportError as e:
            raise ImportError(
                "sampling Zarr needs zarr: pip install 'kuration-sdk[zarr]'"
            ) from e
        group = zarr.open(p, mode="r")
        sample = {key: np.asarray(group[key][:n]) for key in group.array_keys()}
        return kind, sample
    if kind == "video":
        raise ValueError(
            "sample_source() does not decode video for tabular schema inference; "
            "use input(video_path) or VideoAdapter to probe the media schema")
    # parquet-dataset: sample the first part file (sorted)
    from .io import glob
    parts = glob(f"{p.rstrip('/')}/**/*.parquet")
    return kind, _parquet_rows(parts[0], n)


# ---- hz ---------------------------------------------------------------------
def _hz_from_times(times) -> Optional[float]:
    vals = []
    for t in times:
        if t is None:
            continue
        if isinstance(t, _BOOL) or not isinstance(t, _NUMERIC):
            return None
        vals.append(float(t))
    uniq = np.unique(np.asarray(vals, dtype=np.float64)) if vals else np.empty(0)
    if uniq.size < 2:
        return None
    diffs = np.diff(uniq)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    return 1.0 / float(np.median(diffs))


_HZ_MISMATCH_THRESHOLD = 0.2


def _warn_if_hz_mismatches_times(hz: Optional[float], times, *, source: str) -> None:
    """A declared ``hz=`` is taken on faith -- nothing compares it to the data's own
    timestamps. 25Hz data declared as 1Hz runs clean, then throws off any hours budget by
    25x with no signal anything is wrong. Warn once the declared rate and the rate implied
    by the median timestamp delta disagree past a relative 20%."""
    if hz is None:
        return
    observed = _hz_from_times(times)
    if observed is None or observed <= 0:
        return
    relative_gap = abs(observed - hz) / max(observed, hz)
    if relative_gap <= _HZ_MISMATCH_THRESHOLD:
        return
    import warnings
    warnings.warn(
        f"{source}: declared hz={hz:g} does not match the sample's own timestamps "
        f"(median delta implies ~{observed:.3g}Hz, a {relative_gap:.0%} relative gap) — "
        "an hours budget or time-unit window would be off by roughly that factor",
        stacklevel=3)


# ---- cell classification ----------------------------------------------------
def _classify_column(col: str, vals: list) -> tuple[str, Optional[int]]:
    """(kind, vector_length) from a column's non-None cells; ambiguity raises."""
    hint = "pass kinds= or an explicit schema="
    if not vals:
        raise ValueError(f"column {col!r} is all-None in the sample — {hint}")
    kinds_seen: set[str] = set()
    veclen: Optional[int] = None
    for v in vals:
        if isinstance(v, _BOOL) or isinstance(v, str):
            kinds_seen.add("categorical")
        elif isinstance(v, _NUMERIC):
            kinds_seen.add("scalar")
        elif isinstance(v, (list, tuple)):
            if len(v) == 0:
                raise ValueError(f"column {col!r} has an empty list cell — {hint}")
            for x in v:
                if isinstance(x, _BOOL) or not isinstance(x, _NUMERIC):
                    raise ValueError(f"column {col!r}: list cell holds non-numeric element "
                                     f"{x!r} — {hint}")
            if veclen is None:
                veclen = len(v)
            elif veclen != len(v):
                raise ValueError(f"column {col!r} has inconsistent list lengths "
                                 f"({veclen} vs {len(v)}) — {hint}")
            kinds_seen.add("vector")
        else:
            raise ValueError(f"column {col!r} has unsupported cell type "
                             f"{type(v).__name__} ({v!r}) — {hint}")
    if len(kinds_seen) > 1:
        raise ValueError(f"column {col!r} mixes types ({sorted(kinds_seen)}) — {hint}")
    return kinds_seen.pop(), veclen


def _no_dots(names) -> None:
    dotted = [n for n in names if "." in n]
    if dotted:
        raise ValueError(f"field column name(s) {dotted} contain '.' — the metric DSL parses "
                         f"sources on '.'; rename the column(s)")


def _mk_hints(fields: dict, time_col, entity_col, list_columns: dict) -> dict:
    return {"fields": fields, "time_column": time_col, "entity_column": entity_col,
            "list_columns": list_columns}


# ---- inference: rows --------------------------------------------------------
def _match_column(columns: list, candidates: tuple, exclude=()) -> Optional[str]:
    lower: dict[str, str] = {}
    for c in columns:
        if c not in exclude:
            lower.setdefault(c.lower(), c)
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _infer_rows(rows: list, *, name, time, entity, hz, kinds) -> tuple[SchemaSpec, dict]:
    columns: list[str] = []
    for r in rows:
        for k in r:
            if k not in columns:
                columns.append(k)

    if time is not None:
        if time not in columns:
            raise ValueError(f"time column {time!r} not in sample columns {columns}")
        time_col: Optional[str] = time
    else:
        time_col = _match_column(columns, _TIME_CANDIDATES)
    if entity is not None:
        if entity not in columns:
            raise ValueError(f"entity column {entity!r} not in sample columns {columns}")
        entity_col: Optional[str] = entity
    else:
        entity_col = _match_column(columns, _ENTITY_CANDIDATES, exclude=(time_col,))

    field_cols = [c for c in columns if c not in (time_col, entity_col)]
    kinds = dict(kinds or {})
    for k, v in kinds.items():
        if k not in field_cols:
            raise ValueError(f"kinds= references unknown field column {k!r}; "
                             f"field columns: {field_cols}")
        if v == "vector":
            raise ValueError(f"kinds= cannot force column {k!r} to 'vector' — vectors are "
                             f"inferred from list cells; use an explicit schema=")
        if v not in ("scalar", "categorical"):
            raise ValueError(f"kinds[{k!r}] must be 'scalar' or 'categorical', got {v!r}")
    _no_dots(field_cols)

    fields_spec: dict = {}
    hints_fields: dict = {}
    list_columns: dict = {}
    for col in field_cols:
        vals = [r.get(col) for r in rows]
        vals = [v for v in vals if v is not None]
        if col in kinds:
            if any(isinstance(v, (list, tuple)) for v in vals):
                raise ValueError(f"column {col!r} holds list cells (a vector) — kinds= cannot "
                                 f"override it; use an explicit schema=")
            if not vals:
                raise ValueError(f"column {col!r} is all-None in the sample — "
                                 f"pass an explicit schema=")
            kind, veclen = kinds[col], None
        else:
            kind, veclen = _classify_column(col, vals)
        if kind == "vector":
            assert veclen is not None
            fields_spec[col] = {"kind": "vector", "dims": [f"d{i}" for i in range(veclen)]}
            hints_fields[col] = [f"{col}.{i}" for i in range(veclen)]
            list_columns[col] = veclen
        else:
            fields_spec[col] = {"kind": kind}
            hints_fields[col] = col

    if time_col is not None:
        times = [r.get(time_col) for r in rows]
        if hz is None:
            hz = _hz_from_times(times)
        else:
            _warn_if_hz_mismatches_times(hz, times, source="input()")
    schema = SchemaSpec.from_dict({
        "name": name, "entity_axis": entity_col or "entity", "sample_hz": hz,
        "fields": fields_spec})
    return schema, _mk_hints(hints_fields, time_col, entity_col, list_columns)


# ---- inference: dict-of-arrays ----------------------------------------------
def _infer_arrays(sample: dict, *, name, hz) -> tuple[SchemaSpec, dict]:
    _no_dots(sample)
    fields_spec: dict = {}
    for key, arr in sample.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            if arr.dtype.kind in "USOb":          # bool arrays classify like bool cells
                fields_spec[key] = {"kind": "categorical"}
            elif np.issubdtype(arr.dtype, np.number):
                fields_spec[key] = {"kind": "scalar"}
            else:
                raise ValueError(f"array {key!r} has unsupported dtype {arr.dtype!r} — "
                                 f"pass an explicit schema=")
        elif arr.ndim == 2:
            if not np.issubdtype(arr.dtype, np.number):
                raise ValueError(f"array {key!r} is 2-D but not numeric ({arr.dtype!r}) — "
                                 f"pass an explicit schema=")
            fields_spec[key] = {"kind": "vector",
                                "dims": [f"d{i}" for i in range(arr.shape[1])]}
        else:
            raise ValueError(f"array {key!r} has ndim {arr.ndim} (shape {arr.shape}) — "
                             f"only 1-D/2-D arrays are inferable; pass an explicit schema=")
    schema = SchemaSpec.from_dict({
        "name": name, "entity_axis": "entity", "sample_hz": hz, "fields": fields_spec})
    return schema, _mk_hints({k: k for k in sample}, None, None, {})


# ---- inference: ticks -------------------------------------------------------
def _infer_ticks(ticks: list, *, name, hz) -> tuple[SchemaSpec, dict]:
    per_field: dict[str, list] = {}
    for tk in ticks:
        for fvals in tk.entities.values():
            for fname, v in fvals.items():
                per_field.setdefault(fname, [])
                if v is not None:
                    per_field[fname].append(v)
    if not per_field:
        raise ValueError("empty sample: no entity fields in the sampled ticks")
    _no_dots(per_field)
    fields_spec: dict = {}
    for fname, vals in per_field.items():
        kind, veclen = _classify_column(fname, vals)
        if kind == "vector":
            assert veclen is not None
            fields_spec[fname] = {"kind": "vector", "dims": [f"d{i}" for i in range(veclen)]}
        else:
            fields_spec[fname] = {"kind": kind}
    if hz is None:
        hz = _hz_from_times([tk.time for tk in ticks])
    else:
        _warn_if_hz_mismatches_times(hz, [tk.time for tk in ticks], source="input()")
    schema = SchemaSpec.from_dict({
        "name": name, "entity_axis": "entity", "sample_hz": hz, "fields": fields_spec})
    return schema, _mk_hints({f: f for f in per_field}, None, None, {})


# ---- entry point --------------------------------------------------------------
def infer_schema(sample, *, name: str = "inferred", time: Optional[str] = None,
                 entity: Optional[str] = None, hz: Optional[float] = None,
                 kinds: Optional[dict] = None) -> tuple[SchemaSpec, dict]:
    """Infer `(SchemaSpec, hints)` from a sample: list[dict] rows, dict-of-arrays, or
    list[Tick]. `hints["fields"]` is exactly the `fields=` mapping `TimeseriesAdapter`
    takes; `list_columns` names row columns whose list cells need per-dim expansion."""
    if isinstance(sample, dict):
        if not sample:
            raise ValueError("empty sample: cannot infer a schema from no data")
        return _infer_arrays(sample, name=name, hz=hz)
    if isinstance(sample, list):
        if not sample:
            raise ValueError("empty sample: cannot infer a schema from no data")
        if isinstance(sample[0], Tick):
            return _infer_ticks(sample, name=name, hz=hz)
        if isinstance(sample[0], dict):
            return _infer_rows(sample, name=name, time=time, entity=entity, hz=hz, kinds=kinds)
    raise ValueError(f"cannot infer a schema from sample of type {type(sample).__name__} — "
                     f"expected list[dict] rows, dict of arrays, or list[Tick]")
