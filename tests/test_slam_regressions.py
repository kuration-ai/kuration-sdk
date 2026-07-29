"""Release and integration regressions for the native SLAM SDK surface."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import kuration.gameplay.slam.export as slam_export_module
import kuration.gameplay.slam.runner as slam_runner_module
import kuration.gameplay.slam.workflow as slam_workflow_module
from kuration.gameplay.slam import (
    PointCloudArtifact,
    SLAMReconstructor,
    SLAMTrajectoryAdapter,
    build_pointcloud_scene,
    curate_slam_gameplay,
    export_slam_output,
    input_slam,
    load_point_cloud,
    load_slam_output,
    prepare_slam_input,
)
from kuration.gameplay.slam.cli import main as cli_main
from kuration.gameplay.slam.demo import write_demo_reconstruction
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


def test_concurrent_exports_never_prune_another_writers_staged_payload(
    tmp_path,
    monkeypatch,
):
    """A later manifest commit must never reference a payload pruned by an earlier writer."""
    destination = tmp_path / "export"
    first = _output()
    first.trajectories["clip"] = first.trajectories["clip"] + 10.0
    second = _output()
    second.trajectories["clip"] = second.trajectories["clip"] + 20.0
    first_waiting = threading.Event()
    second_done = threading.Event()
    failures = []
    real_writer = slam_export_module._atomic_write_text

    def ordered_writer(target, value):
        if threading.current_thread().name == "first-export":
            first_waiting.set()
            if not second_done.wait(5):
                raise TimeoutError("second export did not finish")
        real_writer(target, value)

    def run(output, *, done=None):
        try:
            export_slam_output(output, destination, sample_hz=1.0)
        except Exception as error:  # pragma: no cover - surfaced by the assertion below
            failures.append(error)
        finally:
            if done is not None:
                done.set()

    monkeypatch.setattr(slam_export_module, "_atomic_write_text", ordered_writer)
    first_thread = threading.Thread(
        target=run,
        args=(first,),
        name="first-export",
    )
    second_thread = threading.Thread(
        target=run,
        args=(second,),
        kwargs={"done": second_done},
        name="second-export",
    )
    first_thread.start()
    assert first_waiting.wait(5)
    second_thread.start()
    first_thread.join(5)
    second_thread.join(5)

    assert not first_thread.is_alive() and not second_thread.is_alive()
    assert not failures
    # The deliberately delayed first writer commits last, so its manifest is authoritative.
    adapter = SLAMTrajectoryAdapter(manifest=destination / "manifest.json")
    adapter.check_source("clip")
    assert adapter.ticks("clip")[0].entities[0]["camera_position"] == (10.0, 10.0, 10.0)


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


def test_slam_manifest_readers_reject_non_object_json(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")

    calls = [
        lambda: SLAMTrajectoryAdapter(manifest=manifest),
        lambda: prepare_slam_input(manifest),
        lambda: load_point_cloud(manifest),
        lambda: load_slam_output(manifest),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="JSON object"):
            call()


def test_slam_adapter_rejects_remote_trajectory_archives(tmp_path):
    manifest = export_slam_output(_output(), tmp_path / "export")
    document = json.loads(manifest.read_text())
    document["clips"]["clip"]["trajectory"] = "gs://bucket/trajectory.npz"
    manifest.write_text(json.dumps(document))

    for call in (
        lambda: SLAMTrajectoryAdapter(manifest=manifest),
        lambda: load_slam_output(manifest),
    ):
        with pytest.raises(ValueError, match="trajectory archives must use local paths"):
            call()


# ----------------------------------------------------------------------
# 2026-07 SLAM review fixes
# ----------------------------------------------------------------------


def _fake_depth_model(torch):
    """Smallest object `SLAMOperator` accepts: a real module (so `.parameters()`/`.to()` behave)
    whose `inference` returns the identity reconstruction the runtime expects."""

    class FakeDepth(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def inference(self, *, image, process_res, export_dir):
            count = len(image)
            intrinsics = np.repeat(np.eye(3)[None, ...], count, axis=0)
            intrinsics[:, 0, 0] = 2.0
            intrinsics[:, 1, 1] = 2.0
            return SimpleNamespace(
                depth=np.ones((count, 2, 2), dtype=np.float32),
                intrinsics=intrinsics,
                extrinsics=np.repeat(np.eye(4)[None, ...], count, axis=0),
                conf=np.ones((count, 2, 2), dtype=np.float32),
                processed_images=np.asarray(image),
            )

    return FakeDepth()


def _operator(torch, model, **overrides):
    from kuration.gameplay.slam._runtime import SLAMOperator

    options = {
        "output_pointcloud": False,
        "window_size": 2,
        "overlap": 1,
        "bridge_size": 1,
        "process_res": 32,
    }
    options.update(overrides)
    return SLAMOperator(model, **options)


def test_unindexed_and_index_zero_devices_compare_as_the_same_device():
    torch = pytest.importorskip("torch")
    from kuration.gameplay.slam._runtime import SLAMOperator

    same = SLAMOperator._same_device
    # Autodetection yields the unindexed form; a parameter always reports the indexed one.
    assert same(torch.device("cuda:0"), torch.device("cuda"))
    assert same(torch.device("cuda"), torch.device("cuda:0"))
    assert same(torch.device("mps:0"), torch.device("mps"))
    assert same(torch.device("cpu"), torch.device("cpu"))
    assert not same(torch.device("cuda:1"), torch.device("cuda"))
    assert not same(torch.device("cpu"), torch.device("cuda"))
    assert not same(None, torch.device("cpu"))


def test_autodetected_single_device_reuses_the_model_instead_of_replicating_it():
    torch = pytest.importorskip("torch")
    from kuration.gameplay.slam._runtime import SLAMOperator

    device = SLAMOperator._resolve_devices(None)[0]
    model = _fake_depth_model(torch).to(device)
    operator = _operator(torch, model, devices=None)

    # A deepcopy here would put a second full copy of the model on the same accelerator.
    assert operator._get_model_for_device(operator.devices[0]) is model
    assert operator._device_models[operator.devices[0]] is model


def test_indexed_parameter_device_matches_an_unindexed_target_without_replicating():
    """The accelerator case, without needing an accelerator: a model whose parameters live on
    `cuda:0` must be reused as-is for the autodetected `cuda` target instead of being deepcopied
    onto the very same GPU."""
    torch = pytest.importorskip("torch")

    class IndexedDepth:
        def parameters(self):
            return iter([SimpleNamespace(device=torch.device("cuda:0"))])

        def to(self, device):  # pragma: no cover - must never be reached
            raise AssertionError(f"model must not be replicated onto {device}")

        def eval(self):  # pragma: no cover - must never be reached
            raise AssertionError("model must not be replicated")

    model = IndexedDepth()
    operator = _operator(torch, model, devices=["cuda"])
    assert operator.devices == [torch.device("cuda")]
    assert operator._get_model_for_device(operator.devices[0]) is model


def test_device_replica_clears_the_lazily_cached_model_device():
    torch = pytest.importorskip("torch")

    class CachingDepth:
        """Mimics the vendored DA3 model: `.device` is memoized on first use and `.to()` does
        not reset it, so a replica would keep moving inputs to the source device."""

        def __init__(self):
            self.device = torch.device("cuda:3")
            self.moved_to = None
            self.evaluated = False

        def parameters(self):
            return iter(())

        def to(self, device):
            self.moved_to = device
            return self

        def eval(self):
            self.evaluated = True
            return self

    model = CachingDepth()
    operator = _operator(torch, model, devices=["cpu"])
    replica = operator._get_model_for_device(torch.device("cpu"))

    assert replica is not model
    assert replica.moved_to == torch.device("cpu")
    assert replica.evaluated
    assert replica.device is None
    assert model.device == torch.device("cuda:3")


def test_computed_device_property_on_a_replica_is_left_alone():
    torch = pytest.importorskip("torch")

    class PropertyDepth:
        @property
        def device(self):
            return torch.device("cpu")

        def parameters(self):
            return iter(())

        def to(self, device):
            return self

        def eval(self):
            return self

    operator = _operator(torch, PropertyDepth(), devices=["cpu"])
    replica = operator._get_model_for_device(torch.device("cpu"))
    assert replica.device == torch.device("cpu")


def test_device_replicas_are_materialized_before_any_worker_thread_runs():
    torch = pytest.importorskip("torch")
    from kuration.gameplay.slam._runtime import SLAMOperator

    events: list[tuple[str, bool]] = []

    class RecordingOperator(SLAMOperator):
        def _get_model_for_device(self, device):
            events.append(("replica", threading.current_thread() is threading.main_thread()))
            return super()._get_model_for_device(device)

        def _run_chunk(self, chunk, model):
            events.append(("chunk", threading.current_thread() is threading.main_thread()))
            return super()._run_chunk(chunk, model)

    from kuration.gameplay.slam import _runtime as runtime_module

    operator = RecordingOperator(
        _fake_depth_model(torch),
        output_pointcloud=False,
        devices=["cpu"],
        window_size=2,
        overlap=1,
        bridge_size=1,
        process_res=32,
    )
    assert isinstance(operator, runtime_module.SLAMOperator)
    operator({"clip": np.zeros((3, 2, 2, 3), dtype=np.uint8)})

    kinds = [kind for kind, _ in events]
    assert kinds.count("replica") == 1
    assert "chunk" in kinds
    assert kinds.index("replica") < kinds.index("chunk")
    assert all(on_main for kind, on_main in events if kind == "replica")


def test_replica_creation_is_serialized_by_the_model_lock():
    torch = pytest.importorskip("torch")

    operator = _operator(torch, _fake_depth_model(torch), devices=["cpu"])
    entered = threading.Event()
    finished = threading.Event()

    def call():
        entered.set()
        operator._get_model_for_device(torch.device("cpu"))
        finished.set()

    thread = threading.Thread(target=call)
    with operator._model_lock:
        thread.start()
        assert entered.wait(5)
        assert not finished.wait(0.25)
    assert finished.wait(5)
    thread.join(5)
    assert not thread.is_alive()


def test_export_keeps_in_tree_videos_relative_and_bundles_external_files(
    tmp_path, monkeypatch
):
    destination = tmp_path / "export"
    inside = destination / "media" / "clip.mp4"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"")
    monkeypatch.chdir(tmp_path)

    manifest = export_slam_output(
        _output(), destination, sample_hz=1.0, videos={"clip": "export/media/clip.mp4"}
    )
    assert json.loads(manifest.read_text())["clips"]["clip"]["video"] \
        == str(Path("media") / "clip.mp4")

    outside = tmp_path / "elsewhere" / "customer-secret-match.mp4"
    outside.parent.mkdir()
    outside.write_bytes(b"external-video")
    manifest = export_slam_output(
        _output(),
        destination,
        sample_hz=1.0,
        videos={"clip": "elsewhere/customer-secret-match.mp4"},
    )
    reference = json.loads(manifest.read_text())["clips"]["clip"]["video"]
    assert Path(reference).parent == Path("media")
    assert Path(reference).name.startswith("video-")
    assert outside.name not in Path(reference).name
    assert (destination / reference).read_bytes() == b"external-video"
    renamed = tmp_path / "renamed.mp4"
    renamed.write_bytes(outside.read_bytes())
    repeated = export_slam_output(
        _output(), destination, sample_hz=1.0, videos={"clip": renamed}
    )
    assert json.loads(repeated.read_text())["clips"]["clip"]["video"] == reference

    manifest = export_slam_output(
        _output(), destination, sample_hz=1.0, videos={"clip": "gs://bucket/clip.mp4"}
    )
    assert json.loads(manifest.read_text())["clips"]["clip"]["video"] == "gs://bucket/clip.mp4"

    with pytest.raises(ValueError, match="non-empty path or URI"):
        export_slam_output(_output(), destination, sample_hz=1.0, videos={"clip": ""})
    with pytest.raises(ValueError, match="does not exist"):
        export_slam_output(
            _output(),
            destination,
            sample_hz=1.0,
            videos={"clip": tmp_path / "missing.mp4"},
        )
    with pytest.raises(ValueError, match="regular file"):
        export_slam_output(
            _output(), destination, sample_hz=1.0, videos={"clip": tmp_path}
        )
    for unsafe in (
        "https://user:password@example.test/clip.mp4",
        "gs://bucket/clip.mp4?X-Goog-Signature=secret",
        "s3://bucket/clip.mp4#token",
    ):
        with pytest.raises(ValueError, match="credential|query parameters"):
            export_slam_output(
                _output(), destination, sample_hz=1.0, videos={"clip": unsafe}
            )


def test_cwd_relative_video_reingests_from_a_different_working_directory(tmp_path, monkeypatch):
    workdir = tmp_path / "work"
    (workdir / "videos").mkdir(parents=True)
    video = workdir / "videos" / "clip.mp4"
    video.write_bytes(b"")

    monkeypatch.chdir(workdir)
    manifest = prepare_slam_input(
        _output(),
        tmp_path / "export",
        export_options={"sample_hz": 1.0, "videos": {"clip": "videos/clip.mp4"}},
        preflight=False,
    )

    class Provider:
        capabilities: dict = {}

        def __init__(self):
            self.probed: list[str] = []

        def probe(self, uri):
            assert Path(uri).is_file(), f"unresolvable video reference {uri!r}"
            self.probed.append(uri)
            return SimpleNamespace(uri=uri, height=2, width=2)

    monkeypatch.chdir(tmp_path)
    provider = Provider()
    pipeline = input_slam(manifest, join_video=True, provider=provider)

    bundled = manifest.parent / json.loads(manifest.read_text())["clips"]["clip"]["video"]
    assert bundled.parent == manifest.parent / "media"
    assert bundled.read_bytes() == video.read_bytes()
    assert provider.probed == [str(bundled)]
    assert pipeline.adapter.videos == {"clip": str(bundled)}
    assert SLAMTrajectoryAdapter(manifest=manifest).video_mapping == {
        "clip": str(bundled)
    }


def test_unrecordable_runtime_option_is_rejected_instead_of_dropped():
    with pytest.raises(ValueError, match="devices.*cannot be recorded"):
        SLAMReconstructor(depth_model="da3-base", devices={"cuda:0"})
    # Non-allowlisted options only reach provenance with a custom runtime factory; the
    # default factory rejects them by name first (see the eager-validation test below).
    with pytest.raises(ValueError, match="tuning.*cannot be recorded"):
        SLAMReconstructor(
            depth_model="da3-base", runtime_factory=lambda **_: None, tuning=object()
        )

    first = SLAMReconstructor(depth_model="da3-base", devices=["cuda:0"], window_size=8)
    second = SLAMReconstructor(depth_model="da3-base", devices=["cuda:1"], window_size=8)
    nested = SLAMReconstructor(
        depth_model="da3-base", devices=["cuda:0"], window_size=8,
        runtime_factory=lambda **_: None, extra={"a": [1, 2]},
    )

    assert first.provenance["options"]["devices"] == ["cuda:0"]
    assert nested.provenance["options"]["extra"] == {"a": [1, 2]}
    assert first.provenance != second.provenance
    assert first.provenance != nested.provenance


def test_default_runtime_option_allowlist_matches_the_real_operator_signature():
    pytest.importorskip("torch")
    import inspect

    from kuration.gameplay.slam._runtime import SLAMOperator
    from kuration.gameplay.slam.runner import DEFAULT_RUNTIME_OPTIONS

    accepted = set(inspect.signature(SLAMOperator.__init__).parameters)
    accepted -= {"self", "depth_model", "embedding_model", "embeddings"}
    assert accepted == set(DEFAULT_RUNTIME_OPTIONS)


def test_pointcloud_scene_renders_a_cloud_without_colors():
    trimesh = pytest.importorskip("trimesh")

    output = SimpleNamespace(
        points=np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]),
        colors=None,
        trajectories={},
        poses={},
    )
    scene = build_pointcloud_scene(output)
    clouds = [
        geometry for geometry in scene.geometry.values()
        if isinstance(geometry, trimesh.PointCloud)
    ]
    assert len(clouds) == 1
    assert clouds[0].vertices.shape == (2, 3)
    assert len(clouds[0].colors) == 2


def test_selection_json_is_written_through_the_atomic_writer(tmp_path, monkeypatch):
    calls = []
    real_writer = slam_workflow_module._atomic_write_text

    def recording(target, value):
        calls.append(Path(target))
        real_writer(target, value)

    monkeypatch.setattr(slam_workflow_module, "_atomic_write_text", recording)
    manifest = write_demo_reconstruction(tmp_path / "demo", frames_per_clip=24)
    destination = tmp_path / "nested" / "selection.json"
    result = curate_slam_gameplay(
        manifest, destination, window=8, stride=4, n=2, min_quality=None
    )

    assert calls == [destination]
    assert json.loads(result.selection.read_text())["format"] == "kuration-slam-selection"
    assert not list(destination.parent.glob("*.tmp"))


def test_slam_curation_rejects_empty_or_duplicate_feature_spaces(tmp_path):
    manifest = write_demo_reconstruction(tmp_path / "demo", frames_per_clip=24)

    with pytest.raises(ValueError, match="non-empty"):
        curate_slam_gameplay(manifest, tmp_path / "empty.json", feature_space=[])
    with pytest.raises(ValueError, match="duplicates"):
        curate_slam_gameplay(
            manifest,
            tmp_path / "duplicate.json",
            feature_space=["slam_path_length", "slam_path_length"],
        )


def test_slam_cli_no_clobber_force_and_strict_json(tmp_path, capsys):
    destination = tmp_path / "demo"

    assert cli_main(["demo", "--out", str(destination), "--quiet"]) == 0
    first = capsys.readouterr()
    result = json.loads(
        first.out,
        parse_constant=lambda value: pytest.fail(f"non-standard JSON token {value}"),
    )
    assert result["selected_count"] > 0

    marker = destination / "caller-owned.txt"
    marker.write_text("keep")
    with pytest.raises(SystemExit) as blocked:
        cli_main(["demo", "--out", str(destination), "--quiet"])
    assert blocked.value.code == 1
    failure = capsys.readouterr()
    assert failure.out == ""
    assert failure.err.count("\n") == 1
    assert failure.err.startswith("error: FileExistsError:")
    assert "--force" in failure.err
    assert "Traceback" not in failure.err

    assert cli_main(
        ["demo", "--out", str(destination), "--quiet", "--force"]
    ) == 0
    json.loads(capsys.readouterr().out)
    assert marker.read_text() == "keep"


def test_slam_cli_resume_reuses_output_but_default_refuses_it(tmp_path, capsys):
    manifest = write_demo_reconstruction(tmp_path / "source", frames_per_clip=24)
    destination = tmp_path / "run"
    arguments = [
        "run",
        "--manifest",
        str(manifest),
        "--out",
        str(destination),
        "--window",
        "8",
        "--stride",
        "4",
        "--count",
        "2",
        "--min-quality",
        "0",
        "--quiet",
    ]

    assert cli_main(arguments) == 0
    capsys.readouterr()
    with pytest.raises(SystemExit):
        cli_main(arguments)
    assert "--force" in capsys.readouterr().err

    assert cli_main([*arguments, "--resume"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["selected_count"] == 2


def test_slam_cli_rejects_input_overlap_before_work(tmp_path, capsys):
    destination = tmp_path / "work"
    destination.mkdir()
    video = destination / "input.mp4"
    video.write_bytes(b"do-not-touch")

    with pytest.raises(SystemExit) as blocked:
        cli_main([
            "run",
            "--video",
            f"clip={video}",
            "--out",
            str(destination),
            "--force",
            "--quiet",
        ])

    assert blocked.value.code == 1
    failure = capsys.readouterr()
    assert "overlaps input path" in failure.err
    assert "Traceback" not in failure.err
    assert video.read_bytes() == b"do-not-touch"
    assert not (destination / "slam-export").exists()


def test_slam_cli_failure_is_one_line_unless_debug_requested(tmp_path, capsys):
    arguments = [
        "run",
        "--manifest",
        str(tmp_path / "missing.json"),
        "--out",
        str(tmp_path / "out"),
        "--runtime-factory",
        "invalid-factory",
        "--quiet",
    ]
    with pytest.raises(SystemExit):
        cli_main(arguments)
    failure = capsys.readouterr()
    assert failure.err.count("\n") == 1
    assert failure.err.startswith("error: ValueError:")
    assert "Traceback" not in failure.err

    with pytest.raises(ValueError, match="module:attribute"):
        cli_main([*arguments, "--debug"])


def test_vendored_depth_export_helpers_are_an_explicit_stub():
    from kuration.gameplay.slam._vendor.depth_anything_3.utils.export import export

    with pytest.raises(NotImplementedError, match="not vendored into kuration-sdk"):
        export(None, "glb", "/tmp/does-not-matter")


def test_importing_the_slam_runtime_does_not_suppress_openmp_duplicate_loading():
    pytest.importorskip("torch")
    script = (
        "import os, sys;"
        "os.environ.pop('KMP_DUPLICATE_LIB_OK', None);"
        "import kuration.gameplay.slam._runtime;"
        "sys.stdout.write(str('KMP_DUPLICATE_LIB_OK' in os.environ))"
    )
    environment = dict(os.environ)
    environment.pop("KMP_DUPLICATE_LIB_OK", None)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=300, env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"


def test_slam_artifact_paths_reject_remote_uris(tmp_path):
    frames = {"clip": np.zeros((2, 2, 2, 3), dtype=np.uint8)}
    calls = [
        lambda: SLAMTrajectoryAdapter(manifest="gs://bucket/run/manifest.json"),
        lambda: input_slam("gs://bucket/run/manifest.json"),
        lambda: export_slam_output(_output(), "s3://bucket/run"),
        lambda: prepare_slam_input("gs://bucket/run/manifest.json"),
        lambda: prepare_slam_input(frames, "az://container/run"),
        lambda: load_point_cloud("gs://bucket/run/scene_pointcloud.npz"),
        lambda: load_slam_output("gs://bucket/run/manifest.json"),
        lambda: curate_slam_gameplay("gs://bucket/run/manifest.json", tmp_path / "sel.json"),
        lambda: curate_slam_gameplay(tmp_path / "manifest.json", "gs://bucket/sel.json"),
        lambda: PointCloudArtifact(np.zeros((1, 3))).save("gs://bucket/cloud.npz"),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="local-only"):
            call()


def test_slam_artifact_paths_reject_empty_strings_before_writing():
    with pytest.raises(ValueError, match="non-empty local path"):
        export_slam_output(_output(), "")
