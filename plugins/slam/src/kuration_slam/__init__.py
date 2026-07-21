"""Kuration integration for precomputed SLAM trajectories and scenes."""
from __future__ import annotations

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
from .runner import decode_video_clips, prepare_slam_input, run_slam_pipeline
from .schema import (
    CAMERA_FRUSTUM,
    CAMERA_INTRINSICS,
    CAMERA_POSITION,
    CAMERA_X_AXIS,
    CAMERA_Y_AXIS,
    CAMERA_Z_AXIS,
    slam_schema,
)


PACKAGE = "kuration-plugin-slam"
INSTALL_HINT = "pip install kuration-plugin-slam"
ADAPTER_NAME = "aintropy/slam-trajectory"
TRANSFORM_NAME = "aintropy/slam-motion"
OPERATOR_NAME = "aintropy/slam-motion-diversity"

EMPTY_PARAMETERS = {"type": "object", "properties": {}, "additionalProperties": False}
OPERATOR_PARAMETERS = {
    "type": "object",
    "properties": {
        "axis": {"type": "string", "enum": list(OUTPUT_FEATURES)},
        "bins": {"type": "integer", "minimum": 2},
    },
    "additionalProperties": False,
}
ADAPTER_PARAMETERS = {
    "type": "object",
    "properties": {"manifest": {"type": "string", "minLength": 1}},
    "required": ["manifest"],
    "additionalProperties": False,
}


def register() -> None:
    """Idempotently register the adapter, bounded transform, and curation operator."""
    from kuration import (
        default_registry,
        register_adapter,
        register_batch_transform,
        register_operator,
    )

    common = {
        "package": PACKAGE,
        "install_hint": INSTALL_HINT,
        "version": "1",
        "stability": "experimental",
    }
    if ADAPTER_NAME not in default_registry.names("adapter"):
        register_adapter(
            ADAPTER_NAME,
            SLAMTrajectoryAdapter,
            description="Read per-clip camera trajectories from a Kuration SLAM manifest",
            parameters=ADAPTER_PARAMETERS,
            **common,
        )
    if TRANSFORM_NAME not in default_registry.names("transform"):
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
    if OPERATOR_NAME not in default_registry.names("operator"):
        register_operator(
            OPERATOR_NAME,
            slam_motion_diversity,
            description="Add SLAM motion features and quantile-bin a selected diversity axis",
            parameters=OPERATOR_PARAMETERS,
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
    "OPERATOR_NAME",
    "OUTPUT_FEATURES",
    "POINTCLOUD_FORMAT",
    "POINTCLOUD_VERSION",
    "PointCloudArtifact",
    "SLAMMotion",
    "SLAMTrajectoryAdapter",
    "TRANSFORM_NAME",
    "curate_point_cloud",
    "decode_video_clips",
    "export_slam_output",
    "input_slam",
    "load_point_cloud",
    "prepare_slam_input",
    "register",
    "run_slam_pipeline",
    "slam_motion_diversity",
    "slam_schema",
]
