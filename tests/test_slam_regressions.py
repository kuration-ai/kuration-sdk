"""Release and integration regressions for the native SLAM SDK surface."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import kuration.gameplay.slam.export as slam_export_module
import kuration.gameplay.slam.runner as slam_runner_module
from kuration.gameplay.slam import (
    PointCloudArtifact,
    SLAMReconstructor,
    export_slam_output,
    load_point_cloud,
)
from kuration.gameplay.slam.export import _times_for_clip


def _pose(position=(0.0, 0.0, 0.0)):
    return SimpleNamespace(
        frustum_points=np.asarray([
            position,
            (1.0, 1.0, 1.0),
            (1.0, -1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (-1.0, -1.0, 1.0),
        ]),
        cx=1.0,
        cy=1.0,
        f=1.0,
        x_axis=np.array([1.0, 0.0, 0.0]),
        y_axis=np.array([0.0, 1.0, 0.0]),
        z_axis=np.array([0.0, 0.0, 1.0]),
    )


def _output(name="clip"):
    return SimpleNamespace(
        trajectories={name: np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])},
        poses={name: [_pose(), _pose((1.0, 0.0, 0.0))]},
        points=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
        colors=np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8),
    )


def test_native_reconstructor_creates_sdk_input_with_stable_provenance(tmp_path):
    calls = []

    class FakeRuntime:
        def __init__(self, *, depth_model, output_pointcloud):
            assert depth_model is depth
            assert output_pointcloud is True

        def __call__(self, frames):
            calls.append(sorted(frames))
            return _output()

    depth = object()
    slam = SLAMReconstructor(
        depth_model=depth,
        depth_model_id="test-depth@abc123",
        runtime_factory=FakeRuntime,
        output_pointcloud=True,
    )
    frames = {"clip": np.zeros((2, 2, 2, 3), dtype=np.uint8)}
    first = slam.reconstruct(frames, tmp_path / "export", resume=True)
    second = slam.reconstruct(frames, tmp_path / "export", resume=True)

    assert calls == [["clip"]]
    assert first.manifest == second.manifest
    document = json.loads(first.manifest.read_text())
    assert document["provenance"]["models"]["depth"] == {
        "id": "test-depth@abc123",
        "source": "explicit",
    }
    assert document["provenance"]["engine"].endswith(".FakeRuntime")
    assert "0x" not in json.dumps(document["provenance"])
    assert first.to_pipeline().sources == ["clip"]

    changed = {"clip": np.ones((2, 2, 2, 3), dtype=np.uint8)}
    slam.reconstruct(changed, tmp_path / "export", resume=True)
    assert calls == [["clip"], ["clip"]]


def test_bundled_slam_operator_runs_a_complete_fake_depth_reconstruction():
    torch = pytest.importorskip("torch")
    from kuration.gameplay.slam import SLAMOperator

    class FakeDepth(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def inference(self, *, image, process_res, export_dir):
            assert process_res == 32
            assert export_dir is None
            count = len(image)
            intrinsics = np.repeat(np.eye(3)[None, ...], count, axis=0)
            intrinsics[:, 0, 0] = 2.0
            intrinsics[:, 1, 1] = 2.0
            extrinsics = np.repeat(np.eye(4)[None, ...], count, axis=0)
            return SimpleNamespace(
                depth=np.ones((count, 2, 2), dtype=np.float32),
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                conf=np.ones((count, 2, 2), dtype=np.float32),
                processed_images=np.asarray(image),
            )

    operator = SLAMOperator(
        FakeDepth(),
        output_pointcloud=False,
        devices=["cpu"],
        window_size=2,
        overlap=1,
        bridge_size=1,
        process_res=32,
    )
    output = operator({"clip": np.zeros((3, 2, 2, 3), dtype=np.uint8)})
    assert output.trajectories["clip"].shape == (3, 3)
    assert len(output.poses["clip"]) == 3
    assert output.points is None


def test_bundled_slam_operator_keeps_disconnected_clips_in_local_frames():
    torch = pytest.importorskip("torch")
    from kuration.gameplay.slam import SLAMOperator

    class FakeDepth(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def inference(self, *, image, process_res, export_dir):
            count = len(image)
            return SimpleNamespace(
                depth=np.ones((count, 2, 2), dtype=np.float32),
                intrinsics=np.repeat(np.eye(3)[None, ...], count, axis=0),
                extrinsics=np.repeat(np.eye(4)[None, ...], count, axis=0),
                conf=np.ones((count, 2, 2), dtype=np.float32),
                processed_images=np.asarray(image),
            )

    operator = SLAMOperator(
        FakeDepth(),
        output_pointcloud=False,
        devices=["cpu"],
        window_size=2,
        overlap=1,
        bridge_size=1,
        process_res=32,
    )
    frames = np.zeros((2, 2, 2, 3), dtype=np.uint8)

    output = operator({"first": frames, "second": frames.copy()})

    assert set(output.trajectories) == {"first", "second"}
    assert output.trajectories["first"].shape == (2, 3)
    assert output.trajectories["second"].shape == (2, 3)


def test_automatic_resume_requires_stable_opaque_runtime_identities(tmp_path):
    frames = {"clip": np.zeros((2, 2, 2, 3), dtype=np.uint8)}

    depth = SLAMReconstructor(depth_model=object(), runtime_factory=lambda **_: None)
    with pytest.raises(ValueError, match="depth_model_id"):
        depth.reconstruct(frames, tmp_path / "depth", resume=True)

    embedded = SLAMReconstructor(
        depth_model="da3-base",
        embeddings={"clip": np.ones((2, 4), dtype=np.float32)},
        runtime_factory=lambda **_: None,
    )
    with pytest.raises(ValueError, match="embedding_model_id"):
        embedded.reconstruct(frames, tmp_path / "embedding", resume=True)


def test_reconstruct_skips_source_hash_without_automatic_resume(tmp_path, monkeypatch):
    calls = []

    class FakeRuntime:
        def __init__(self, *, depth_model):
            assert depth_model is model

        def __call__(self, frames):
            calls.append(sorted(frames))
            return _output()

    def unexpected_hash(_source):
        raise AssertionError("source identity should not be calculated")

    model = object()
    monkeypatch.setattr(slam_runner_module, "_source_identity", unexpected_hash)
    slam = SLAMReconstructor(depth_model=model, runtime_factory=FakeRuntime)
    frames = {"clip": np.zeros((2, 2, 2, 3), dtype=np.uint8)}

    slam.reconstruct(frames, tmp_path / "ordinary")
    slam.reconstruct(frames, tmp_path / "manual-cache", resume=True, cache_key="caller-v1")

    assert calls == [["clip"], ["clip"]]


def test_invalid_export_provenance_does_not_create_partial_artifact(tmp_path):
    destination = tmp_path / "export"

    with pytest.raises(ValueError, match="provenance must be a JSON-safe mapping"):
        export_slam_output(_output(), destination, provenance={"invalid": object()})

    assert not destination.exists()


def test_failed_manifest_commit_preserves_previous_export(tmp_path, monkeypatch):
    destination = tmp_path / "export"
    manifest = export_slam_output(_output(), destination, sample_hz=1.0)
    original_manifest = manifest.read_bytes()
    original_document = json.loads(original_manifest)
    original_clip = destination / original_document["clips"]["clip"]["trajectory"]
    original_clip_bytes = original_clip.read_bytes()

    changed = _output()
    changed.trajectories["clip"] = changed.trajectories["clip"] + 10.0

    def fail_commit(_target, _value):
        raise OSError("simulated manifest publication failure")

    monkeypatch.setattr(slam_export_module, "_atomic_write_text", fail_commit)
    with pytest.raises(OSError, match="publication failure"):
        export_slam_output(changed, destination, sample_hz=1.0)

    assert manifest.read_bytes() == original_manifest
    assert original_clip.read_bytes() == original_clip_bytes
    from kuration.gameplay.slam import SLAMTrajectoryAdapter

    SLAMTrajectoryAdapter(manifest=manifest).check_source("clip")


def test_private_depth_vendor_resolves_nested_config_modules():
    pytest.importorskip("omegaconf")
    from kuration.gameplay.slam._vendor.depth_anything_3.cfg import load_config
    from kuration.gameplay.slam._vendor.depth_anything_3.registry import MODEL_REGISTRY

    config = load_config(MODEL_REGISTRY["da3nested-giant-large"])
    package = "kuration.gameplay.slam._vendor.depth_anything_3"

    assert config.__object__.path == f"{package}.model.da3"
    assert config.anyview.__object__.path == f"{package}.model.da3"
    assert config.metric.__object__.path == f"{package}.model.da3"


def test_export_requires_timestamp_coverage_before_writing(tmp_path):
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
    tmp_path, format_value, version_value, message
):
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
    tmp_path, field, value, message
):
    manifest = export_slam_output(_output(), tmp_path / "export", sample_hz=1.0)
    document = json.loads(manifest.read_text())
    document["scene"][field] = value
    manifest.write_text(json.dumps(document))

    with pytest.raises(ValueError, match=message):
        load_point_cloud(manifest)
