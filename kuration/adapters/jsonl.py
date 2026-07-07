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
    def ticks(self, source_uri: str) -> list[Tick]:
        from ..io import open_file
        rows = []
        with open_file(source_uri, "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return self._rows_to_ticks(rows)

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
