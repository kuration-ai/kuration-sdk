"""HuggingFace `datasets` input adapter — a dataset/split as a canonical Tick stream.

Same column mapping as the timeseries adapter (rows sharing `time_column` form a timestep;
`entity_column` splits it). One split is one source by default; set `source_column` to split it
into one source per distinct value (e.g. per episode/trajectory id — keeps the per-source
train/test split leakage-free). Needs the `[hf]` extra (`datasets`).

    adapter: {type: huggingface, dataset: "org/name", split: "train",
              fields: {...}, time_column: "t", source_column: "episode_id"}
"""
from __future__ import annotations

from typing import Optional

from ..buffer import Tick
from ..schema import SchemaSpec
from .base import register_adapter
from .timeseries import TimeseriesAdapter


class HuggingFaceAdapter(TimeseriesAdapter):
    def __init__(self, schema: SchemaSpec, *, fields: dict, dataset: str, split: str = "train",
                 config: Optional[str] = None, time_column: Optional[str] = None,
                 entity_column: Optional[str] = None, globals: Optional[dict] = None,
                 source_column: Optional[str] = None, load_kwargs: Optional[dict] = None):
        super().__init__(schema, fields=fields, time_column=time_column,
                         entity_column=entity_column, globals=globals)
        self.dataset_name = dataset
        self.split = split
        self.config = config
        self.source_column = source_column
        self.load_kwargs = load_kwargs or {}
        self._rows: Optional[list[dict]] = None      # cached after first load (per process)

    def _load_rows(self) -> list[dict]:
        if self._rows is None:
            try:
                from datasets import load_dataset
            except ImportError as e:
                raise ImportError("HuggingFaceAdapter needs the datasets library: "
                                  "pip install -e '.[hf]'") from e
            ds = load_dataset(self.dataset_name, self.config, split=self.split, **self.load_kwargs)
            self._rows = list(ds)
        return self._rows

    def sources(self) -> list[str]:
        if self.source_column is None:
            return [f"{self.dataset_name}:{self.split}"]
        return sorted({str(r[self.source_column]) for r in self._load_rows()})

    def ticks(self, source_uri: str) -> list[Tick]:
        rows = self._load_rows()
        if self.source_column is not None:
            rows = [r for r in rows if str(r[self.source_column]) == str(source_uri)]
        return self._rows_to_ticks(rows)


register_adapter("huggingface", HuggingFaceAdapter)
