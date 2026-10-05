"""Generic tabular adapter — the "any columns of numbers" case, with zero per-dataset code.

A CSV/Parquet table (or an in-memory list of row dicts) becomes a canonical `Tick` stream:
rows sharing a `time_column` value form one timestep; an optional `entity_column` splits a
timestep into entities; declared `fields` map schema fields to column(s). This covers the
common multivariate-timeseries dataset without anyone writing an adapter class.

Bespoke formats (ROS bags, proprietary logs, the CS2 `data.json`) still implement their own
`InputAdapter`, but they only owe the same tiny contract: yield `Tick`s.
"""
from __future__ import annotations

import math
import numbers
import re
from typing import Iterable, Optional

from ..buffer import Tick
from ..schema import SchemaSpec
from .base import register_adapter


def _is_parquet_source(source_uri: str) -> bool:
    return str(source_uri).lower().endswith(".parquet")


def _num_time(v):
    # Preserve integral timestamps exactly. Converting epoch nanoseconds to float collapses
    # adjacent values above 2**53 and can merge distinct timesteps.
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, str):
        stripped = v.strip()
        if re.fullmatch(r"[+-]?\d+", stripped):
            return int(stripped)
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


class TimeseriesAdapter:
    """Configured by a column mapping over a `SchemaSpec`.

    fields:        {schema_field -> column_name}  (vector field -> [col_per_dim, ...])
    time_column:   groups rows into timesteps; None -> each row is its own timestep
    entity_column: splits a timestep into entities; None -> a single entity per timestep
    globals:       {global_name -> column_name}  (per-tick scalar, read from the tick's first row)
    Rows must be ordered by time (timesteps are formed from runs of equal time values).
    """

    @staticmethod
    def detect_source(source_uri: str) -> int:
        source = str(source_uri)
        return 80 if source.lower().endswith(".csv") or _is_parquet_source(source) else 0

    def __init__(self, schema: SchemaSpec, *, fields: dict,
                 time_column: Optional[str] = None, entity_column: Optional[str] = None,
                 globals: Optional[dict] = None):
        self.schema = schema
        self.fields = fields
        self.time_column = time_column
        self.entity_column = entity_column
        self.globals = globals or {}
        for f in fields:
            if f not in schema.fields:
                raise ValueError(f"field map references undeclared schema field {f!r}")
        for g in self.globals:
            if g not in schema.globals:
                raise ValueError(f"global map references undeclared schema global {g!r}")
        for name, column in fields.items():
            fs = schema.fields[name]
            if fs.kind == "vector":
                if (not isinstance(column, (list, tuple)) or len(column) != fs.width
                        or any(not isinstance(c, str) or not c for c in column)):
                    raise ValueError(f"vector field {name!r} needs exactly {fs.width} non-empty "
                                     "column names")
                if len(set(column)) != len(column):
                    raise ValueError(f"vector field {name!r} maps duplicate columns {column!r}")
            elif not isinstance(column, str) or not column:
                raise ValueError(f"{fs.kind} field {name!r} needs one non-empty column name")
        for name, column in self.globals.items():
            if not isinstance(column, str) or not column:
                raise ValueError(f"global {name!r} needs one non-empty column name")
        for label, column in (("time_column", time_column), ("entity_column", entity_column)):
            if column is not None and (not isinstance(column, str) or not column):
                raise ValueError(f"{label} must be a non-empty string or None")
        self.available_fields = frozenset(fields)
        self.available_globals = frozenset(self.globals)

    # ---- value extraction ---------------------------------------------------
    def _value(self, row: dict, fname: str):
        fs = self.schema.fields[fname]
        col = self.fields[fname]
        if fs.kind == "vector":
            return tuple(float(row[c]) for c in col)
        if fs.kind == "categorical":
            value = row[col]
            if value is None:
                return None
            return value.decode() if isinstance(value, bytes) else str(value)
        return float(row[col])

    def _mk_tick(self, time, rows: list[dict]) -> Tick:
        entities: dict = {}
        for row in rows:
            eid = row[self.entity_column] if self.entity_column else 0
            if eid in entities:
                raise ValueError(f"duplicate row for entity {eid!r} at time {time!r} — "
                                 f"the later row would silently overwrite the earlier one")
            entities[eid] = {f: self._value(row, f) for f in self.fields}
        glb = {}
        if rows:
            for g, c in self.globals.items():
                values = [float(row[c]) for row in rows]
                if any(not math.isfinite(value) for value in values):
                    raise ValueError(f"global {g!r} contains a non-finite value at time {time!r}")
                if any(value != values[0] for value in values[1:]):
                    raise ValueError(f"global {g!r} disagrees across entity rows at time "
                                     f"{time!r}: {values!r}")
                glb[g] = values[0]
        return Tick(time=_num_time(time), entities=entities, globals=glb)

    def _iter_rows_to_ticks(self, rows: Iterable[dict]):
        if self.time_column is None:
            for i, row in enumerate(rows):
                yield self._mk_tick(i, [row])
            return
        cur = None
        group: list[dict] = []
        warned_nonnumeric = False
        for row in rows:
            tv = row[self.time_column]
            ordered = _num_time(tv)
            if isinstance(ordered, str) and not warned_nonnumeric:
                # _num_time() falls back to the raw string when it isn't integer/float-parseable
                # (e.g. an ISO timestamp) -- rows still group/order by plain string comparison,
                # but sample_hz/seconds-unit windowing needs real epoch arithmetic and will fail
                # far from here (or silently misorder non-zero-padded strings) if that's assumed.
                import warnings
                warnings.warn(
                    f"time column {self.time_column!r} is not numeric (e.g. {tv!r}) -- rows "
                    "are grouped/ordered by plain string comparison; hz-based or seconds-unit "
                    "windowing needs a numeric (epoch) time column", stacklevel=3)
                warned_nonnumeric = True
            if isinstance(ordered, float) and not math.isfinite(ordered):
                raise ValueError(f"time column {self.time_column!r} contains non-finite {tv!r}")
            if group and ordered != cur:
                try:
                    decreasing = ordered < cur
                except TypeError as error:
                    raise ValueError(f"time column {self.time_column!r} contains values that "
                                     f"cannot be ordered consistently: {tv!r}") from error
                if decreasing:
                    raise ValueError(f"rows are not ordered by {self.time_column!r}: time "
                                     f"{tv!r} follows {cur!r}")
                yield self._mk_tick(cur, group)
                group = []
            group.append(row)
            cur = ordered
        if group:
            yield self._mk_tick(cur, group)

    def _rows_to_ticks(self, rows: list[dict]) -> list[Tick]:
        return list(self._iter_rows_to_ticks(rows))

    # ---- preflight ----------------------------------------------------------
    def _required_columns(self) -> set:
        need = set()
        if self.time_column:
            need.add(self.time_column)
        if self.entity_column:
            need.add(self.entity_column)
        for f, col in self.fields.items():
            need.update(col if isinstance(col, (list, tuple)) else [col])
        need.update(self.globals.values())
        return need

    def check_source(self, source_uri: str) -> None:
        """Preflight one source without reading data rows: sniff the CSV header (or the
        Parquet file schema) and verify every mapped column exists — so a column typo fails
        at `kur-validate --check-sources` time, not after a full feature pass."""
        from ..io import local_copy, open_file
        if _is_parquet_source(source_uri):
            try:
                import pyarrow.parquet as pq
            except ImportError as e:
                raise ImportError("checking .parquet needs the optional extra: "
                                  "pip install 'kuration-sdk[parquet]'") from e
            with local_copy(source_uri) as p:
                cols = set(pq.ParquetFile(p).schema_arrow.names)
        else:
            import csv
            with open_file(source_uri, "rt") as f:
                header = next(csv.reader(f), None)
            if header is None:
                raise ValueError(f"source {source_uri!r} is empty — no header row")
            cols = set(header)
        missing = sorted(self._required_columns() - cols)
        if missing:
            raise ValueError(f"source {source_uri!r} is missing mapped column(s) {missing}; "
                             f"available: {sorted(cols)}")

    # ---- sources ------------------------------------------------------------
    def from_rows(self, rows: Iterable[dict]) -> list[Tick]:
        return self._rows_to_ticks(list(rows))

    def ticks(self, source_uri: str) -> list[Tick]:
        return list(self.iter_ticks(source_uri))

    def iter_ticks(self, source_uri: str):
        from ..io import local_copy, open_file
        if _is_parquet_source(source_uri):
            try:
                import pyarrow.parquet as pq  # optional [parquet] extra
            except ImportError as e:
                raise ImportError("reading .parquet needs the optional extra: "
                                  "pip install 'kuration-sdk[parquet]'") from e
            with local_copy(source_uri) as p:
                def rows():
                    for batch in pq.ParquetFile(p).iter_batches():
                        yield from batch.to_pylist()
                yield from self._iter_rows_to_ticks(rows())
        else:
            import csv
            with open_file(source_uri, "rt") as f:
                yield from self._iter_rows_to_ticks(csv.DictReader(f))


register_adapter("timeseries", TimeseriesAdapter)
