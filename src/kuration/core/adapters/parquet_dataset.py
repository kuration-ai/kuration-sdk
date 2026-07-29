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
_DETECT_DIR_LIMIT = 512          # directories a detection walk may examine before giving up


class ParquetDatasetAdapter(TimeseriesAdapter):
    @staticmethod
    def detect_source(source_uri: str) -> int:
        """Recognize a *local* directory that contains Parquet parts.

        Detection is deliberately cheap: a remote prefix (`gs://`, `s3://`, …) would need a
        recursive listing per candidate source, so remote datasets are not auto-detected —
        pass ``adapter="parquet-dataset"`` (plus ``schema=``) for those. The local walk stops
        after ``_DETECT_DIR_LIMIT`` directories so a large Parquet-free tree cannot make
        detection arbitrarily slow; a deeper layout also needs the explicit adapter.
        """
        import os
        from pathlib import Path

        from ..io import is_remote

        source = str(source_uri)
        if is_remote(source) or not Path(source).is_dir():
            return 0
        for scanned, (_root, _dirs, files) in enumerate(os.walk(source), start=1):
            if any(name.endswith(".parquet") for name in files):
                return 100
            if scanned >= _DETECT_DIR_LIMIT:
                return 0
        return 0

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
                              "pip install 'kuration-sdk[parquet]'") from e
        from ..io import local_copy
        with local_copy(parquet_path) as p:
            return pq.read_table(p).to_pylist()

    def sources(self) -> list[str]:
        parts = self._parts()
        if not parts:
            raise ValueError(f"no .parquet parts found under {self.path!r}")
        return parts if self.source_per == "part" else [self.path]

    def source_fingerprint(self, source_uri: str):
        """Fingerprint resolved parts, not a literal dataset glob identifier."""
        from ..provenance import source_descriptor, source_descriptors
        if self.source_per == "part":
            return source_descriptor(source_uri, None)
        parts = self._parts()
        if not parts:
            raise ValueError(f"no .parquet parts found under {self.path!r}")
        return {"parts": source_descriptors(parts, None)}

    @property
    def sources_are_paths(self) -> bool:
        """In "dataset" mode the single source URI is the root path/glob itself — a plain
        existence stat on it is meaningless (and false for a glob); `check_source` resolves
        the parts instead."""
        return self.source_per == "part"

    @property
    def supports_streaming(self) -> bool:
        # Dataset mode globally sorts rows and can merge equal timesteps across parts. That
        # operation is inherently materializing unless the dataset declares ordering metadata.
        return self.source_per == "part"

    def check_source(self, source_uri: str) -> None:
        """Preflight: in "part" mode the source is one .parquet file (inherited footer-schema
        sniff); in "dataset" mode the source is the whole tree — resolve the parts and sniff
        each footer (schema metadata only, no row data)."""
        if self.source_per == "part":
            return super().check_source(source_uri)
        parts = self._parts()
        if not parts:
            raise ValueError(f"no .parquet parts found under {self.path!r}")
        for part in parts:
            super().check_source(part)

    def ticks(self, source_uri: str) -> list[Tick]:
        if self.source_per == "part":
            return self._rows_to_ticks(self._read_rows(source_uri))
        rows: list[dict] = []
        for part in self._parts():
            rows.extend(self._read_rows(part))
        if self.time_column is not None:
            rows.sort(key=lambda r: r[self.time_column])     # ticks form from runs of equal time
        return self._rows_to_ticks(rows)

    def iter_ticks(self, source_uri: str):
        if self.source_per != "part":
            raise ValueError("bounded-memory streaming is unavailable for source_per='dataset'")
        yield from super().iter_ticks(source_uri)


register_adapter("parquet-dataset", ParquetDatasetAdapter)
