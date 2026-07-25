"""JSON Lines input adapter — one JSON object per line is one row, same column mapping as the
timeseries adapter (rows sharing `time_column` form a timestep; `entity_column` splits it).
Stdlib only.
"""
from __future__ import annotations

import json

from ..buffer import Tick
from .base import register_adapter
from .timeseries import TimeseriesAdapter


class JsonlAdapter(TimeseriesAdapter):
    @staticmethod
    def detect_source(source_uri: str) -> int:
        return 100 if str(source_uri).lower().endswith(".jsonl") else 0

    def ticks(self, source_uri: str) -> list[Tick]:
        return list(self.iter_ticks(source_uri))

    def iter_ticks(self, source_uri: str):
        from ..io import open_file
        with open_file(source_uri, "rt") as f:
            rows = (json.loads(line) for raw in f if (line := raw.strip()))
            yield from self._iter_rows_to_ticks(rows)

    def check_source(self, source_uri: str) -> None:
        """Preflight one .jsonl by its first non-empty line only: parse it and verify every
        mapped column is a key. JSONL rows may in principle be heterogeneous, so this is a
        first-row check — but a mapping typo shows up in row one."""
        from ..io import open_file
        with open_file(source_uri, "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    break
            else:
                raise ValueError(f"source {source_uri!r} is empty — no JSON lines")
        missing = sorted(self._required_columns() - set(row))
        if missing:
            raise ValueError(f"source {source_uri!r}: first row is missing mapped column(s) "
                             f"{missing}; available: {sorted(row)}")


register_adapter("jsonl", JsonlAdapter)
