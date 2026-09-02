"""Kuration gameplay integration for SLAM trajectories and scenes."""
from __future__ import annotations

from typing import Any

from .adapter import SLAMTrajectoryAdapter, input_slam
from .export import EXPORT_FORMAT, EXPORT_VERSION, export_slam_output
from .motion import OUTPUT_FEATURES, SLAMMotion, slam_motion_diversity
from .pointcloud import (
    POINTCLOUD_FORMAT,
    POINTCLOUD_VERSION,
    PointCloudArtifact,
    curate_point_cloud,
    load_point_cloud,
)
from .preflight import SLAMPreflightResult, preflight_slam
from .quality import QUALITY_FEATURES, SLAMQuality, slam_quality
from .runner import (
    SLAMReconstructor,
    SLAMReconstruction,
    decode_video_clips,
    prepare_slam_input,
    run_slam_reconstruction,
    slam_provenance,
)
from .schema import (
    CAMERA_FRUSTUM,
    CAMERA_INTRINSICS,
    CAMERA_POSITION,
    CAMERA_X_AXIS,
    CAMERA_Y_AXIS,
    CAMERA_Z_AXIS,
    TRACKING_CONFIDENCE,
    slam_schema,
)
from .viz import build_pointcloud_scene, load_slam_output
from .workflow import (
    SELECTION_FORMAT,
    SELECTION_VERSION,
    SLAMCurationResult,
    curate_slam_gameplay,
)


PACKAGE = "kuration-sdk"
INSTALL_HINT = 'pip install "kuration-sdk[slam]"'
ADAPTER_NAME = "aintropy/slam-trajectory"
TRANSFORM_NAME = "aintropy/slam-motion"
OPERATOR_NAME = "aintropy/slam-motion-diversity"
QUALITY_TRANSFORM_NAME = "aintropy/slam-quality"
QUALITY_OPERATOR_NAME = "aintropy/slam-quality"

EMPTY_PARAMETERS: dict[str, Any] = {
    "type": "object", "properties": {}, "additionalProperties": False
}
OPERATOR_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "axis": {"type": "string", "enum": list(OUTPUT_FEATURES)},
        "bins": {"type": "integer", "minimum": 2},
    },
    "additionalProperties": False,
}
ADAPTER_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {"manifest": {"type": "string", "minLength": 1}},
    "required": ["manifest"],
    "additionalProperties": False,
}
QUALITY_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "min_tracking_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "translation_ratio": {"type": "number", "exclusiveMinimum": 0},
        "translation_mad_multiplier": {"type": "number", "exclusiveMinimum": 0},
        "max_translation_step": {"type": "number", "exclusiveMinimum": 0},
        "max_turn_degrees": {"type": "number", "exclusiveMinimum": 0, "maximum": 180},
    },
    "additionalProperties": False,
}
QUALITY_OPERATOR_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        **QUALITY_PARAMETERS["properties"],
        "min_score": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "additionalProperties": False,
}


# Torch-backed names served lazily below; deliberately NOT in __all__ so a star import stays
# zero-Torch like everything else in this module. Attribute access still works.
_LAZY_RUNTIME_NAMES = ("CameraPose", "SLAMOperator", "SLAMOutput")


def __getattr__(name: str):
    """Lazily expose `SLAMOperator`/`CameraPose`/`SLAMOutput` so importing
    `kuration.gameplay.slam` keeps a zero-Torch import graph -- only code
    that actually touches `SLAMOperator` pulls in `torch`."""
    if name in _LAZY_RUNTIME_NAMES:
        from . import _runtime

        return getattr(_runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_RUNTIME_NAMES})


def register() -> None:
    """Idempotently register the adapter, bounded transform, and curation operator."""
    from kuration import (
        default_registry,
        register_adapter,
        register_batch_transform,
        register_operator,
    )
    from ...core._registration import needs_registration

    common: dict[str, Any] = {
        "package": PACKAGE,
        "extra": "slam",
        "install_hint": INSTALL_HINT,
        "version": "1",
        "stability": "experimental",
    }
    if needs_registration(
        default_registry, "adapter", ADAPTER_NAME, SLAMTrajectoryAdapter
    ):
        register_adapter(
            ADAPTER_NAME,
            SLAMTrajectoryAdapter,
            description="Read per-clip camera trajectories from a Kuration SLAM manifest",
            parameters=ADAPTER_PARAMETERS,
            **common,
        )
    if needs_registration(default_registry, "transform", TRANSFORM_NAME, SLAMMotion):
        register_batch_transform(
            TRANSFORM_NAME,
            SLAMMotion,
            outputs=OUTPUT_FEATURES,
            inputs=(f"field:{CAMERA_POSITION}", f"field:{CAMERA_Z_AXIS}"),
            resources={"num_gpus": 0, "preferred_batch_size": 128},
            description="Measure translation and camera-turn motion in SLAM trajectory windows",
            parameters=EMPTY_PARAMETERS,
            **common,
        )
    if needs_registration(
        default_registry,
        "transform",
        QUALITY_TRANSFORM_NAME,
        SLAMQuality,
    ):
        register_batch_transform(
            QUALITY_TRANSFORM_NAME,
            SLAMQuality,
            outputs=QUALITY_FEATURES,
            inputs=(
                f"field:{CAMERA_POSITION}",
                f"field:{CAMERA_Z_AXIS}",
                f"field:{TRACKING_CONFIDENCE}",
            ),
            resources={"num_gpus": 0, "preferred_batch_size": 128},
            description="Flag tracking gaps, pose jumps, and unstable trajectory scale",
            parameters=QUALITY_PARAMETERS,
            **common,
        )
    if needs_registration(
        default_registry,
        "operator",
        OPERATOR_NAME,
        slam_motion_diversity,
    ):
        register_operator(
            OPERATOR_NAME,
            slam_motion_diversity,
            description="Add SLAM motion features and quantile-bin a selected diversity axis",
            parameters=OPERATOR_PARAMETERS,
            **common,
        )
    if needs_registration(
        default_registry,
        "operator",
        QUALITY_OPERATOR_NAME,
        slam_quality,
    ):
        register_operator(
            QUALITY_OPERATOR_NAME,
            slam_quality,
            description="Add SLAM reconstruction-quality signals and optionally filter windows",
            parameters=QUALITY_OPERATOR_PARAMETERS,
            **common,
        )


__all__ = [
    "ADAPTER_NAME",
    "CAMERA_FRUSTUM",
    "CAMERA_INTRINSICS",
    "CAMERA_POSITION",
    "CAMERA_X_AXIS",
    "CAMERA_Y_AXIS",
    "CAMERA_Z_AXIS",
    "EXPORT_FORMAT",
    "EXPORT_VERSION",
    "INSTALL_HINT",
    "OPERATOR_NAME",
    "OUTPUT_FEATURES",
    "POINTCLOUD_FORMAT",
    "POINTCLOUD_VERSION",
    "QUALITY_FEATURES",
    "QUALITY_OPERATOR_NAME",
    "QUALITY_TRANSFORM_NAME",
    "SELECTION_FORMAT",
    "SELECTION_VERSION",
    "PointCloudArtifact",
    "SLAMCurationResult",
    "SLAMMotion",
    "SLAMPreflightResult",
    "SLAMQuality",
    "SLAMReconstruction",
    "SLAMReconstructor",
    "SLAMTrajectoryAdapter",
    "TRANSFORM_NAME",
    "TRACKING_CONFIDENCE",
    "build_pointcloud_scene",
    "curate_slam_gameplay",
    "curate_point_cloud",
    "decode_video_clips",
    "export_slam_output",
    "input_slam",
    "load_point_cloud",
    "load_slam_output",
    "prepare_slam_input",
    "preflight_slam",
    "register",
    "run_slam_reconstruction",
    "slam_motion_diversity",
    "slam_quality",
    "slam_provenance",
    "slam_schema",
]
