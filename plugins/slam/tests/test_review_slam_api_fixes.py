"""SLAM-specific regressions from the comprehensive SDK review."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from kuration_slam import PointCloudArtifact, export_slam_output, load_point_cloud
from kuration_slam.export import _times_for_clip


def _pose():
    return SimpleNamespace(
        frustum_points=np.zeros((5, 3), dtype=np.float64),
        cx=1.0,
        cy=1.0,
        f=1.0,
        x_axis=np.array([1.0, 0.0, 0.0]),
        y_axis=np.array([0.0, 1.0, 0.0]),
        z_axis=np.array([0.0, 0.0, 1.0]),
    )


def test_export_requires_timestamp_coverage_for_every_clip(tmp_path):
    output = SimpleNamespace(
        trajectories={"one": np.zeros((1, 3)), "two": np.ones((1, 3))},
        poses={"one": [_pose()], "two": [_pose()]},
        points=None,
    )
    destination = tmp_path / "export"
    with pytest.raises(ValueError, match=r"cover exactly.*missing=\['two'\]"):
        export_slam_output(output, destination, timestamps={"one": [10.0]})
    assert not destination.exists()


def test_time_helper_reports_a_missing_timestamp_clip():
    with pytest.raises(ValueError, match="missing clip 'two'"):
        _times_for_clip("two", 1, timestamps={"one": [10.0]}, sample_hz=None)


@pytest.mark.parametrize(
    ("format_value", "version_value", "message"),
    [
        ("wrong-format", 1, "has format"),
        ("kuration-pointcloud-npz", 999, "unsupported version"),
        (None, None, "has format None"),
    ],
)
def test_point_cloud_loader_validates_format_and_version(
        tmp_path, format_value, version_value, message):
    path = tmp_path / "cloud.npz"
    payload = {"points": np.array([[0.0, 1.0, 2.0]])}
    if format_value is not None:
        payload["_format"] = np.asarray(format_value)
    if version_value is not None:
        payload["_version"] = np.asarray(version_value)
    np.savez(path, **payload)
    with pytest.raises(ValueError, match=message):
        load_point_cloud(path)


def test_point_cloud_current_metadata_still_round_trips(tmp_path):
    path = PointCloudArtifact(np.array([[1.0, 2.0, 3.0]])).save(tmp_path / "cloud.npz")
    assert load_point_cloud(path).points.tolist() == [[1.0, 2.0, 3.0]]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format", "wrong-format", "scene has format 'wrong-format'"),
        ("format", None, "scene has format None"),
        ("version", 999, "scene has unsupported version 999"),
        ("version", True, "scene has unsupported version True"),
    ],
)
def test_point_cloud_loader_validates_manifest_scene_metadata(
        tmp_path, field, value, message):
    manifest = export_slam_output(
        SimpleNamespace(
            trajectories={"clip": np.zeros((1, 3))},
            poses={"clip": [_pose()]},
            points=np.array([[1.0, 2.0, 3.0]]),
        ),
        tmp_path / "export",
        sample_hz=1.0,
    )
    document = json.loads(manifest.read_text())
    document["scene"][field] = value
    manifest.write_text(json.dumps(document))

    with pytest.raises(ValueError, match=message):
        load_point_cloud(manifest)
