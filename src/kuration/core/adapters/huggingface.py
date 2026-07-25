"""HuggingFace `datasets` input adapter — a dataset/split as a canonical Tick stream.

Same column mapping as the timeseries adapter (rows sharing `time_column` form a timestep;
`entity_column` splits it). One split is one source by default; set `source_column` to split it
into one source per distinct value (e.g. per episode/trajectory id — keeps the per-source
train/test split leakage-free). Needs the `[hf]` extra (`datasets`).

    adapter: {type: huggingface, dataset: "org/name", split: "train",
              fields: {...}, time_column: "t", source_column: "episode_id"}
"""
from __future__ import annotations

from typing import Any, Optional

from ..buffer import Tick
from ..schema import SchemaSpec
from .base import register_adapter
from .timeseries import TimeseriesAdapter


class HuggingFaceAdapter(TimeseriesAdapter):
    @staticmethod
    def detect_source(source_uri: str) -> int:
        # Hub identifiers and local dataset scripts are not safely distinguishable from
        # ordinary paths. Require an explicit adapter instead of inheriting CSV detection.
        return 0

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
        self._dataset: Any = None
        self._source_indices: Optional[dict[str, list[int]]] = None
        self._source_rows: Optional[dict[str, list[dict]]] = None
        self._source_origins: dict[str, tuple[str, str]] = {}

    # HF sources are dataset/split names (or source_column values), not filesystem paths —
    # tells `kur-validate --check-sources` to skip the existence stat and rely on check_source.
    sources_are_paths = False

    @property
    def supports_streaming(self) -> bool:
        """Partitioned splits need a whole-dataset index and are therefore not bounded."""
        return self.source_column is None

    def _source_key(self, value) -> str:
        key = str(value)
        if not key:
            raise ValueError(f"source column {self.source_column!r} contains an empty source ID")
        origin = (f"{type(value).__module__}.{type(value).__qualname__}", repr(value))
        previous = self._source_origins.setdefault(key, origin)
        if previous != origin:
            raise ValueError(
                f"source column {self.source_column!r} contains distinct values that collide "
                f"after string normalization as {key!r}: {previous[1]} ({previous[0]}) and "
                f"{value!r} ({origin[0]})")
        return key

    def sources(self) -> list[str]:
        if self.source_column is None:
            return [f"{self.dataset_name}:{self.split}"]
        self._build_source_index()
        index = self._source_indices if self._source_indices is not None else self._source_rows
        return sorted(index or {})

    def _load_dataset(self):
        if self._rows is not None:
            return None
        if self._dataset is not None:
            return self._dataset
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("HuggingFaceAdapter needs the datasets library: "
                              "pip install 'kuration-sdk[hf]'") from e
        self._dataset = load_dataset(self.dataset_name, self.config, split=self.split,
                                     **self.load_kwargs)
        return self._dataset

    def _build_source_index(self) -> None:
        if self.source_column is None or self._source_indices is not None \
                or self._source_rows is not None:
            return
        if self._rows is not None:
            index: dict[str, list[int]] = {}
            for position, row in enumerate(self._rows):
                index.setdefault(self._source_key(row[self.source_column]), []).append(position)
            self._source_indices = index
            return
        dataset = self._load_dataset()
        if hasattr(dataset, "select") and hasattr(dataset, "__len__"):
            index = {}
            # Read only the source column once. This is O(rows), after which each source is an
            # Arrow index selection instead of a fresh full-split scan.
            for position, value in enumerate(dataset[self.source_column]):
                index.setdefault(self._source_key(value), []).append(position)
            self._source_indices = index
            return
        rows: dict[str, list[dict]] = {}
        for row in dataset:  # streaming datasets cannot random-select; partition in one pass
            rows.setdefault(self._source_key(row[self.source_column]), []).append(row)
        self._source_rows = rows

    def _iter_rows(self):
        if self._rows is not None:
            yield from self._rows
            return
        # A datasets.Dataset is Arrow-backed and iterable without converting the whole split
        # into a Python list. Users may also pass streaming=True through load_kwargs.
        yield from self._load_dataset()

    def check_source(self, source_uri: str) -> None:
        """Preflight the dataset's declared features against the column map without
        downloading data: the builder's metadata already names the columns. If rows were
        injected or cached in-process, check the first row instead. A dataset whose builder
        declares no features and
        has no cached rows is unverifiable here; the run-time path stays loud."""
        need = self._required_columns()
        if self.source_column:
            need = need | {self.source_column}
        if self._rows is not None:
            cols = set(self._rows[0]) if self._rows else set()
        else:
            try:
                from datasets import load_dataset_builder
            except ImportError as e:
                raise ImportError("HuggingFaceAdapter needs the datasets library: "
                                  "pip install 'kuration-sdk[hf]'") from e
            features = load_dataset_builder(self.dataset_name, self.config).info.features
            if features is None:
                return
            cols = set(features)
        missing = sorted(need - cols)
        if missing:
            raise ValueError(f"dataset {self.dataset_name!r} is missing mapped column(s) "
                             f"{missing}; available: {sorted(cols)}")

    def ticks(self, source_uri: str) -> list[Tick]:
        return list(self.iter_ticks(source_uri))

    def iter_ticks(self, source_uri: str):
        if self.source_column is None:
            rows = self._iter_rows()
        else:
            self._build_source_index()
            key = str(source_uri)
            if self._source_indices is not None:
                indices = self._source_indices.get(key)
                if indices is None:
                    raise KeyError(f"unknown Hugging Face source {source_uri!r}")
                if self._rows is not None:
                    rows = (self._rows[index] for index in indices)
                else:
                    rows = iter(self._load_dataset().select(indices))
            else:
                source_rows = self._source_rows or {}
                if key not in source_rows:
                    raise KeyError(f"unknown Hugging Face source {source_uri!r}")
                rows = iter(source_rows[key])
        yield from self._iter_rows_to_ticks(rows)


register_adapter("huggingface", HuggingFaceAdapter)
