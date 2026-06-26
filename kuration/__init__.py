"""kuration — a dataset-agnostic curation SDK.

Phase A+B+C surface: the schema-driven metric engine, input adapters, segmentation, the
feature engine, and the selection engine (binning + declarative strategies + set-based
selection with tag-coverage/group-cap + source-level splits + reproducible config). The
output layer (shards/encoders) arrives next — see `kuration-sdk-plan.md`.
"""
from .adapters import (
    CuratedDatasetAdapter,
    HDF5Adapter,
    InputAdapter,
    JsonlAdapter,
    NpzAdapter,
    TimeseriesAdapter,
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
from .io import exists, local_copy, read_text, write_bytes, write_text
from .manifest import build_config, dump_segment_manifest, write_config
from .methods import apply_filters, method_priorities
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
    "CuratedDatasetAdapter", "schema_from_output", "register_adapter", "get_adapter",
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
]
