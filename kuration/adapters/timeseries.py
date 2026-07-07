"""Generic tabular adapter — the "any columns of numbers" case, with zero per-dataset code.

A CSV/Parquet table (or an in-memory list of row dicts) becomes a canonical `Tick` stream:
rows sharing a `time_column` value form one timestep; an optional `entity_column` splits a
timestep into entities; declared `fields` map schema fields to column(s). This covers the
common multivariate-timeseries dataset without anyone writing an adapter class.

Bespoke formats (ROS bags, proprietary logs, the CS2 `data.json`) still implement their own
`InputAdapter`, but they only owe the same tiny contract: yield `Tick`s.
"""
from __future__ import annotations

from typing import Iterable, Optional

from ..buffer import Tick
from ..schema import SchemaSpec
from .base import register_adapter


def _num_time(v):
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

    # ---- value extraction ---------------------------------------------------
    def _value(self, row: dict, fname: str):
        fs = self.schema.fields[fname]
        col = self.fields[fname]
        if fs.kind == "vector":
            return tuple(float(row[c]) for c in col)
        if fs.kind == "categorical":
            return str(row[col])
        return float(row[col])

    def _mk_tick(self, time, rows: list[dict]) -> Tick:
        entities: dict = {}
        for row in rows:
            eid = row[self.entity_column] if self.entity_column else 0
            if eid in entities:
                raise ValueError(f"duplicate row for entity {eid!r} at time {time!r} — "
                                 f"the later row would silently overwrite the earlier one")
            entities[eid] = {f: self._value(row, f) for f in self.fields}
        glb = {g: float(rows[0][c]) for g, c in self.globals.items()} if rows else {}
        return Tick(time=_num_time(time), entities=entities, globals=glb)

    def _rows_to_ticks(self, rows: list[dict]) -> list[Tick]:
        if self.time_column is None:
            return [self._mk_tick(i, [row]) for i, row in enumerate(rows)]
        ticks: list[Tick] = []
        cur = None
        group: list[dict] = []
        seen_times: set = set()
        for row in rows:
            tv = row[self.time_column]
            if group and tv != cur:
                ticks.append(self._mk_tick(cur, group))
                seen_times.add(cur)
                group = []
                if tv in seen_times:     # e.g. rows sorted entity-major, not by time
                    raise ValueError(
                        f"rows are not ordered by {self.time_column!r}: time {tv!r} reappears "
                        f"after its timestep closed — sort rows by time first")
            group.append(row)
            cur = tv
        if group:
            ticks.append(self._mk_tick(cur, group))
        return ticks

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
        if source_uri.endswith(".parquet"):
            try:
                import pyarrow.parquet as pq
            except ImportError as e:
                raise ImportError("checking .parquet needs the optional extra: "
                                  "pip install -e '.[parquet]'") from e
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
        from ..io import local_copy, open_file
        if source_uri.endswith(".parquet"):
            try:
                import pyarrow.parquet as pq  # optional [parquet] extra
            except ImportError as e:
                raise ImportError("reading .parquet needs the optional extra: "
                                  "pip install -e '.[parquet]'") from e
            with local_copy(source_uri) as p:     # gs://, s3://, … staged to a tempfile
                rows = pq.read_table(p).to_pylist()
        else:
            import csv
            with open_file(source_uri, "rt") as f:
                rows = list(csv.DictReader(f))
        return self._rows_to_ticks(rows)


register_adapter("timeseries", TimeseriesAdapter)
