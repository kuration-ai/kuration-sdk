"""Partitioned-Parquet adapter — a directory/glob of Parquet parts as curation sources.

The single-file `TimeseriesAdapter` already reads one `.parquet`; this generalizes it to a
*dataset* of parts (e.g. Hive-style `date=*/part-*.parquet`) using the same column mapping. Two
modes via `source_per`:

  "part"     each Parquet file is its own source (the common case — one trajectory/episode per
             file; keeps the per-source train/test split leakage-free)
  "dataset"  all parts are one source — rows are concatenated (and sorted by `time_column` if
             set) before forming ticks

Reuses the `[parquet]` extra (pyarrow); no new dependency. Remote globs (`gs://…/**/*.parquet`)
work via fsspec (`[cloud]` extra + a backend).
"""
from __future__ import annotations

from typing import Optional

from ..buffer import Tick
from ..schema import SchemaSpec
from .base import register_adapter
from .timeseries import TimeseriesAdapter

_GLOB_CHARS = set("*?[")


class ParquetDatasetAdapter(TimeseriesAdapter):
    def __init__(self, schema: SchemaSpec, *, fields: dict, path: str,
                 source_per: str = "part", time_column: Optional[str] = None,
                 entity_column: Optional[str] = None, globals: Optional[dict] = None):
        super().__init__(schema, fields=fields, time_column=time_column,
                         entity_column=entity_column, globals=globals)
        if source_per not in ("part", "dataset"):
            raise ValueError(f"source_per must be 'part' or 'dataset', got {source_per!r}")
        self.path = path
        self.source_per = source_per

    def _parts(self) -> list[str]:
        from ..io import glob
        pat = self.path if any(c in self.path for c in _GLOB_CHARS) \
            else f"{self.path.rstrip('/')}/**/*.parquet"
        return glob(pat)

    def _read_rows(self, parquet_path: str) -> list[dict]:
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError("reading .parquet needs the optional extra: "
                              "pip install -e '.[parquet]'") from e
        from ..io import local_copy
        with local_copy(parquet_path) as p:
            return pq.read_table(p).to_pylist()

    def sources(self) -> list[str]:
        parts = self._parts()
        if not parts:
            raise ValueError(f"no .parquet parts found under {self.path!r}")
        return parts if self.source_per == "part" else [self.path]

    def ticks(self, source_uri: str) -> list[Tick]:
        if self.source_per == "part":
            return self._rows_to_ticks(self._read_rows(source_uri))
        rows: list[dict] = []
        for part in self._parts():
            rows.extend(self._read_rows(part))
        if self.time_column is not None:
            rows.sort(key=lambda r: r[self.time_column])     # ticks form from runs of equal time
        return self._rows_to_ticks(rows)


register_adapter("parquet-dataset", ParquetDatasetAdapter)
