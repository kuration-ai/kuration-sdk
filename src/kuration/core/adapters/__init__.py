"""Input adapters: raw source -> canonical Tick stream."""
from .base import InputAdapter, detect_adapter, get_adapter, register_adapter
from .curated import CuratedDatasetAdapter, schema_from_output
from .hdf5 import HDF5Adapter
from .huggingface import HuggingFaceAdapter
from .jsonl import JsonlAdapter
from .npz import NpzAdapter
from .parquet_dataset import ParquetDatasetAdapter
from .timeseries import TimeseriesAdapter
from .zarr_store import ZarrAdapter
from ..media import JoinedMediaAdapter, VideoAdapter

__all__ = ["InputAdapter", "register_adapter", "get_adapter", "detect_adapter",
           "TimeseriesAdapter",
           "HDF5Adapter", "NpzAdapter", "JsonlAdapter", "CuratedDatasetAdapter",
           "ParquetDatasetAdapter", "HuggingFaceAdapter", "ZarrAdapter",
           "schema_from_output"]
__all__ += ["VideoAdapter", "JoinedMediaAdapter"]
