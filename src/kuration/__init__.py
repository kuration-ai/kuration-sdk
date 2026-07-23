"""kuration — a dataset-agnostic curation SDK.

Full pipeline: schema-driven metric DSL, input adapters, segmentation, feature engine,
binning, declarative strategies (base + feature-space + filter + pre-pass filters + a
learned-scorer seam), the output layer (declared modalities, npz/json/hdf5/parquet shards,
hours-budget fill, sub-track spans), tag-coverage, experiment + report, parallel /
skip-and-log / resume / out-of-core execution, path-agnostic local + cloud I/O, schema
inference, and a fluent `input(...).operator(...)` pipeline API.
See README.md, docs/, and PLAN.md.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kuration-sdk")
except PackageNotFoundError:  # source tree imported before installation
    __version__ = "0+unknown"

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
    VideoAdapter,
    JoinedMediaAdapter,
    get_adapter,
    register_adapter,
    schema_from_output,
)
from .binning import bin_segments, quantile_bins
from .buffer import Tick, WindowBuffer, build_window, empty_like
from .columnar import IDENTITY_COLUMNS, SegmentBatch, iter_segment_batches
from .checkpoints import (CheckpointIdentity, PartitionCommit, StageCheckpoint,
                          canonical_hash)
from .bounded import FeatureStageResult, checkpoint_feature_stage
from .external import (external_quantile_edges, external_top_k, external_uniform_bins,
                       external_weighted, iter_external_span_items,
                       iter_external_hash_dedup, iter_external_top_k_priorities,
                       iter_external_uniform_priorities,
                       iter_external_weighted_priorities)
from .curate import CurateResult, curate
from .convenience import action_state_checker, slam_pipeline
from .dsl import compile_metric, register_compiled, register_metric
from .experiment import Arm, run_experiment, union_define
from .features import FeatureEngine
from .infer import infer_schema
from .io import exists, local_copy, read_text, write_bytes, write_text
from .loaders import iter_shards, read_shard_arrays, shard_entries
from .inspection import compare_datasets, inspect_dataset
from .html_report import render_html_report, write_html_report
from .manifest import build_config, dump_segment_manifest, write_config
from .methods import apply_filters, method_priorities, register_filter
from .pipeline import Pipeline, input
from .plugins import load_plugins, loaded_plugin_modules, plugin_failures, plugin_provenance
from .plugin_validation import PluginCheckResult, check_plugins
from .progress import ProgressCallback
from .report import dataset_report
from .review import hours_selection_review, preview_dataset, selection_review
from .quality import (QualityGateError, check_quality, evaluate_quality,
                      validate_expectations)
from .scorers import SegmentScorer, apply_scorers, get_scorer, register_scorer
from .output import (
    Container,
    Modality,
    OutputEncoder,
    OutputSpec,
    PassthroughEncoder,
    Track,
    fill_to_hours,
    get_encoder,
    register_encoder,
    register_container,
    output_preset,
    tracks_from_ticks,
    write_dataset,
    write_track,
)
from .schema import FieldSpec, MediaSpec, SchemaSpec
from .media import FrameRef, MediaProvider, PyAVMediaProvider, VideoMetadata
from .transforms import (BatchTransform, TransformContext, apply_transforms,
                         apply_transforms_to_windows, get_transform,
                         register_batch_transform, register_transform)
from .analyzers import Analyzer, SplitLeakageAnalyzer, get_analyzer, register_analyzer
from .execution import (ExecutionBackend, LocalExecutionBackend, RayExecutionBackend,
                        ResourceSpec, get_execution_backend)
from .planning import (OperatorResources, OperatorTraits, PhysicalOperator, PhysicalPlan,
                       PhysicalStage, compile_physical_plan)
from .scoring import score_sources
from .segment import Segment, Segmenter
from .select import assign_splits
from .strategy import Strategy, load_strategy, segment_priorities, select
from .registry import (ComponentDescriptor, Registry, default_registry, describe_component,
                       list_components, register_method, register_operator, register_preset)
from .results import (DatasetComparison, DatasetInspection, DatasetPreview, DatasetReport,
                      DatasetWriteResult, ExecutionPlan, ExperimentResult,
                      HoursSelectionReview, QualityGateResult, SelectionReview,
                      ValidationResult)
from .versioning import (ARTIFACT_FORMAT_VERSION, JOB_SCHEMA_VERSION, migrate,
                         migrate_config, migrate_job)

__all__ = [
    "__version__",
    "SchemaSpec", "FieldSpec", "MediaSpec",
    "WindowBuffer", "Tick", "build_window", "empty_like",
    "SegmentBatch", "IDENTITY_COLUMNS", "iter_segment_batches",
    "CheckpointIdentity", "PartitionCommit", "StageCheckpoint", "canonical_hash",
    "FeatureStageResult", "checkpoint_feature_stage",
    "external_quantile_edges", "external_top_k",
    "external_uniform_bins",
    "iter_external_top_k_priorities", "iter_external_span_items",
    "iter_external_uniform_priorities",
    "external_weighted", "iter_external_weighted_priorities",
    "iter_external_hash_dedup",
    "compile_metric", "register_metric", "register_compiled",
    "InputAdapter", "TimeseriesAdapter", "HDF5Adapter", "NpzAdapter", "JsonlAdapter",
    "CuratedDatasetAdapter", "ParquetDatasetAdapter", "HuggingFaceAdapter", "ZarrAdapter",
    "schema_from_output", "register_adapter", "get_adapter",
    "FrameRef", "VideoMetadata", "MediaProvider", "PyAVMediaProvider",
    "VideoAdapter", "JoinedMediaAdapter",
    "BatchTransform", "TransformContext", "register_batch_transform", "register_transform",
    "get_transform",
    "apply_transforms", "apply_transforms_to_windows",
    "Analyzer", "SplitLeakageAnalyzer", "register_analyzer", "get_analyzer",
    "ExecutionBackend", "LocalExecutionBackend", "RayExecutionBackend", "ResourceSpec",
    "get_execution_backend",
    "OperatorResources", "OperatorTraits", "PhysicalOperator", "PhysicalStage",
    "PhysicalPlan", "compile_physical_plan",
    "Arm", "run_experiment", "union_define",
    "Segment", "Segmenter", "FeatureEngine",
    "bin_segments", "quantile_bins",
    "Strategy", "load_strategy", "select", "segment_priorities", "assign_splits",
    "apply_filters", "method_priorities", "score_sources", "dataset_report",
    "register_method", "register_filter", "register_operator",
    "SegmentScorer", "register_scorer", "get_scorer", "apply_scorers",
    "curate", "CurateResult",
    "build_config", "dump_segment_manifest", "write_config",
    "read_text", "write_text", "write_bytes", "exists", "local_copy",
    "OutputSpec", "Modality", "OutputEncoder", "PassthroughEncoder",
    "register_encoder", "get_encoder", "Track", "tracks_from_ticks", "write_track",
    "fill_to_hours", "write_dataset",
    "Container", "register_container", "output_preset", "register_preset",
    "iter_shards", "read_shard_arrays", "shard_entries",
    "inspect_dataset", "DatasetInspection", "compare_datasets", "DatasetComparison",
    "preview_dataset", "DatasetPreview", "selection_review", "SelectionReview",
    "hours_selection_review", "HoursSelectionReview",
    "evaluate_quality", "check_quality", "validate_expectations",
    "QualityGateResult", "QualityGateError",
    "render_html_report", "write_html_report",
    "infer_schema", "Pipeline", "input",
    "action_state_checker", "slam_pipeline",
    "load_plugins", "loaded_plugin_modules", "plugin_failures", "plugin_provenance",
    "check_plugins",
    "PluginCheckResult", "ProgressCallback",
    "Registry", "default_registry", "ComponentDescriptor", "list_components",
    "describe_component", "ExecutionPlan", "ARTIFACT_FORMAT_VERSION",
    "DatasetWriteResult", "ExperimentResult", "DatasetReport", "ValidationResult",
    "JOB_SCHEMA_VERSION", "migrate", "migrate_job", "migrate_config",
    "pixel_shift", "action_state_consistency", "slam",
    "SLAMPipeline", "SLAMReconstruction", "input_slam",
]

# First-party domain capabilities are built into this evaluation branch. Their optional
# frameworks remain lazy, but their descriptors are available in every Registry snapshot
# without scanning the external ``kuration.plugins`` entry-point group.
from . import action_state_consistency, pixel_shift, slam
from .slam import SLAMPipeline, SLAMReconstruction, input_slam

pixel_shift.register()
action_state_consistency.register()
slam.register()
