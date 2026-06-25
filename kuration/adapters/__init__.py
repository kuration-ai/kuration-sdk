"""Input adapters: raw source -> canonical Tick stream."""
from .base import InputAdapter, get_adapter, register_adapter
from .curated import CuratedDatasetAdapter, schema_from_output
from .hdf5 import HDF5Adapter
from .timeseries import TimeseriesAdapter

__all__ = ["InputAdapter", "register_adapter", "get_adapter", "TimeseriesAdapter",
           "HDF5Adapter", "CuratedDatasetAdapter", "schema_from_output"]
