"""kuration — a dataset-agnostic curation SDK.

Full pipeline: schema-driven metric DSL, input adapters, segmentation, feature engine,
binning, declarative strategies (base + feature-space + filter + pre-pass filters + a
learned-scorer seam), the output layer (declared modalities, npz/json/hdf5/parquet shards,
hours-budget fill, sub-track spans), tag-coverage, experiment + report, parallel /
skip-and-log / resume / out-of-core execution, path-agnostic local + cloud I/O, schema
inference, and a fluent `input(...).operator(...)` pipeline API.
See README.md, docs/, and PLAN.md.
"""
from .adapters import (
    CuratedDatasetAdapter,
    HDF5Adapter,
    HuggingFaceAdapter,
    InputAdapter,
    JsonlAdapter,
    NpzAdapter,
    ParquetDatasetAdapter,
    TimeseriesAdapter,
    ZarrAdapter,
    get_adapter,
    register_adapter,
    schema_from_output,
)
from .binning import bin_segments, quantile_bins
from .buffer import Tick, WindowBuffer, build_window, empty_like
from .curate import CurateResult, curate
from .dsl import compile_metric, register_compiled, register_metric
from .experiment import Arm, run_experiment, union_define
from .features import FeatureEngine
from .infer import infer_schema
from .io import exists, local_copy, read_text, write_bytes, write_text
from .loaders import iter_shards, read_shard_arrays, shard_entries
from .manifest import build_config, dump_segment_manifest, write_config
from .methods import apply_filters, method_priorities
from .pipeline import Pipeline, input
from .report import dataset_report
from .scorers import SegmentScorer, apply_scorers, get_scorer, register_scorer
from .output import (
    Modality,
    OutputEncoder,
    OutputSpec,
    PassthroughEncoder,
    Track,
    fill_to_hours,
    get_encoder,
    register_encoder,
    tracks_from_ticks,
    write_dataset,
    write_track,
)
from .schema import FieldSpec, SchemaSpec
from .scoring import score_sources
from .segment import Segment, Segmenter
from .select import assign_splits
from .strategy import Strategy, load_strategy, segment_priorities, select

__all__ = [
    "SchemaSpec", "FieldSpec",
    "WindowBuffer", "Tick", "build_window", "empty_like",
    "compile_metric", "register_metric", "register_compiled",
    "InputAdapter", "TimeseriesAdapter", "HDF5Adapter", "NpzAdapter", "JsonlAdapter",
    "CuratedDatasetAdapter", "ParquetDatasetAdapter", "HuggingFaceAdapter", "ZarrAdapter",
    "schema_from_output", "register_adapter", "get_adapter",
    "Arm", "run_experiment", "union_define",
    "Segment", "Segmenter", "FeatureEngine",
    "bin_segments", "quantile_bins",
    "Strategy", "load_strategy", "select", "segment_priorities", "assign_splits",
    "apply_filters", "method_priorities", "score_sources", "dataset_report",
    "SegmentScorer", "register_scorer", "get_scorer", "apply_scorers",
    "curate", "CurateResult",
    "build_config", "dump_segment_manifest", "write_config",
    "read_text", "write_text", "write_bytes", "exists", "local_copy",
    "OutputSpec", "Modality", "OutputEncoder", "PassthroughEncoder",
    "register_encoder", "get_encoder", "Track", "tracks_from_ticks", "write_track",
    "fill_to_hours", "write_dataset",
    "iter_shards", "read_shard_arrays", "shard_entries",
    "infer_schema", "Pipeline", "input",
]
