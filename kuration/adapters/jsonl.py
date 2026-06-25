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


register_adapter("jsonl", JsonlAdapter)
