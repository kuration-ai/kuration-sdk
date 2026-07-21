"""Canonical schema for time-aligned camera reconstruction data."""
from __future__ import annotations

from kuration import SchemaSpec


CAMERA_POSITION = "camera_position"
CAMERA_X_AXIS = "camera_x_axis"
CAMERA_Y_AXIS = "camera_y_axis"
CAMERA_Z_AXIS = "camera_z_axis"
CAMERA_INTRINSICS = "camera_intrinsics"
CAMERA_FRUSTUM = "camera_frustum"

FIELD_TO_ARRAY = {
    CAMERA_POSITION: "camera_position",
    CAMERA_X_AXIS: "camera_x_axis",
    CAMERA_Y_AXIS: "camera_y_axis",
    CAMERA_Z_AXIS: "camera_z_axis",
    CAMERA_INTRINSICS: "camera_intrinsics",
    CAMERA_FRUSTUM: "camera_frustum",
}


def slam_schema(*, sample_hz: float | None = None) -> SchemaSpec:
    """Return the stable schema emitted by :class:`SLAMTrajectoryAdapter`."""
    raw = {
        "name": "aintropy-slam-trajectory-v1",
        "entity_axis": "camera",
        "fields": {
            CAMERA_POSITION: {
                "kind": "vector",
                "dims": ["x", "y", "z"],
                "semantics": "position",
            },
            CAMERA_X_AXIS: {"kind": "vector", "dims": ["x", "y", "z"]},
            CAMERA_Y_AXIS: {"kind": "vector", "dims": ["x", "y", "z"]},
            CAMERA_Z_AXIS: {"kind": "vector", "dims": ["x", "y", "z"]},
            CAMERA_INTRINSICS: {"kind": "vector", "dims": ["cx", "cy", "f"]},
            CAMERA_FRUSTUM: {
                "kind": "vector",
                "dims": [f"p{point}_{axis}" for point in range(5) for axis in ("x", "y", "z")],
            },
        },
    }
    if sample_hz is not None:
        raw["sample_hz"] = sample_hz
    return SchemaSpec.from_dict(raw)
