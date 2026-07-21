from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kuration import default_registry, describe_component
from kuration_slam import (
    ADAPTER_NAME,
    OPERATOR_NAME,
    OUTPUT_FEATURES,
    PointCloudArtifact,
    TRANSFORM_NAME,
    SLAMMotion,
    SLAMTrajectoryAdapter,
    curate_point_cloud,
    decode_video_clips,
    export_slam_output,
    input_slam,
    load_point_cloud,
    prepare_slam_input,
    register,
    run_slam_pipeline,
    slam_schema,
)


def _pose(position, forward=(0.0, 0.0, 1.0)):
    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(forward, dtype=np.float64)
    frustum = np.stack(
        [
            position,
            position + [-1.0, -1.0, 1.0],
            position + [1.0, -1.0, 1.0],
            position + [-1.0, 1.0, 1.0],
            position + [1.0, 1.0, 1.0],
        ]
    )
    return SimpleNamespace(
        frustum_points=frustum,
        cx=320.0,
        cy=240.0,
        f=500.0,
        x_axis=np.array([1.0, 0.0, 0.0]),
        y_axis=np.array([0.0, 1.0, 0.0]),
        z_axis=forward,
    )


def _output():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    forwards = [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    return SimpleNamespace(
        trajectories={"clip one": positions},
        poses={"clip one": [_pose(p, f) for p, f in zip(positions, forwards, strict=True)]},
        points=np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]),
        colors=np.array([[0, 0, 255], [0, 255, 0]], dtype=np.uint8),
    )


def test_export_and_adapter_round_trip(tmp_path):
    video = tmp_path / "clip one.mp4"
    manifest_path = export_slam_output(
        _output(),
        tmp_path / "export",
        sample_hz=2.0,
        videos={"clip one": video},
        point_attributes={"confidence": np.array([0.25, 0.9])},
    )
    document = json.loads(manifest_path.read_text())
    assert document["format"] == "kuration-slam-export"
    assert document["timebase"] == "timestamp"
    assert document["scene"]["pointcloud"] == "scene_pointcloud.npz"
    assert document["scene"]["count"] == 2
    assert document["scene"]["attributes"] == ["confidence"]
    trajectory_name = document["clips"]["clip one"]["trajectory"]
    assert "clip one" not in Path(trajectory_name).name

    adapter = SLAMTrajectoryAdapter(manifest=manifest_path)
    assert adapter.sources() == ["clip one"]
    assert adapter.schema.sample_hz == 2.0
    assert adapter.video_mapping == {"clip one": str(video)}
    ticks = adapter.ticks("clip one")
    assert [tick.time for tick in ticks] == [0.0, 0.5, 1.0]
    assert ticks[2].entities[0]["camera_position"] == (1.0, 1.0, 0.0)
    assert len(ticks[0].entities[0]["camera_frustum"]) == 15
    adapter.check_source("clip one")

    with np.load(manifest_path.parent / document["scene"]["pointcloud"]) as cloud:
        assert cloud["points"].shape == (2, 3)
        assert cloud["colors"].dtype == np.uint8
        assert cloud["confidence"].tolist() == [0.25, 0.9]


def test_export_rejects_mismatched_pose_count(tmp_path):
    output = _output()
    output.poses["clip one"] = output.poses["clip one"][:-1]
    with pytest.raises(ValueError, match="2 poses; trajectory has 3"):
        export_slam_output(output, tmp_path)


def test_export_rejects_nonfinite_trajectory(tmp_path):
    output = _output()
    output.trajectories["clip one"][1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        export_slam_output(output, tmp_path)


def test_motion_transform_computes_translation_and_turn_features():
    positions = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
    forwards = np.array([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    result = SLAMMotion().transform_batch(
        {"field:camera_position": positions, "field:camera_z_axis": forwards}
    )
    assert set(result) == set(OUTPUT_FEATURES)
    assert result["slam_path_length"] == pytest.approx([2.0])
    assert result["slam_displacement"] == pytest.approx([np.sqrt(2.0)])
    assert result["slam_motion_efficiency"] == pytest.approx([np.sqrt(2.0) / 2.0])
    assert result["slam_mean_step"] == pytest.approx([1.0])
    assert result["slam_max_step"] == pytest.approx([1.0])
    assert result["slam_turn_degrees"] == pytest.approx([90.0])
    assert result["slam_mean_turn_degrees"] == pytest.approx([45.0])
    assert result["slam_max_turn_degrees"] == pytest.approx([90.0])
    assert result["slam_spatial_extent"] == pytest.approx([np.sqrt(2.0)])


def test_motion_transform_rejects_zero_forward_axis():
    with pytest.raises(ValueError, match="zero-length"):
        SLAMMotion().transform_batch(
            {
                "field:camera_position": np.zeros((1, 2, 3)),
                "field:camera_z_axis": np.zeros((1, 2, 3)),
            }
        )


def test_registered_operator_runs_end_to_end(tmp_path):
    manifest = export_slam_output(_output(), tmp_path / "export", sample_hz=2.0)
    register()
    pipeline = (
        input_slam(manifest)
        .window(3)
        .apply(OPERATOR_NAME, axis="slam_path_length", bins=3)
        .curate("top-k", n=1, weights={"slam_path_length": 1.0})
    )
    result = pipeline.collect()
    assert result.selected[0].features["slam_path_length"] == pytest.approx(2.0)
    assert result.selected[0].features["slam_turn_degrees"] == pytest.approx(90.0)


def test_registration_exposes_complete_descriptors():
    register()
    snapshot = default_registry.snapshot()
    adapter = describe_component("adapter", ADAPTER_NAME, registry=snapshot)
    transform = describe_component("transform", TRANSFORM_NAME, registry=snapshot)
    operator = describe_component("operator", OPERATOR_NAME, registry=snapshot)
    assert adapter["parameters"]["required"] == ["manifest"]
    assert transform["package"] == "kuration-plugin-slam"
    assert transform["description"]
    assert operator["parameters"]["properties"]["bins"]["minimum"] == 2


def test_schema_has_fixed_pose_field_widths():
    schema = slam_schema(sample_hz=7.5)
    assert schema.fields["camera_position"].width == 3
    assert schema.fields["camera_intrinsics"].width == 3
    assert schema.fields["camera_frustum"].width == 15
    assert schema.sample_hz == 7.5


def test_prepare_reuses_existing_manifest_without_running_slam(tmp_path):
    manifest = export_slam_output(_output(), tmp_path / "existing")

    def must_not_run(**kwargs):
        raise AssertionError("SLAM must not be constructed for manifest input")

    prepared = prepare_slam_input(
        manifest,
        runner_options={"depth_model": object(), "pipeline_factory": must_not_run},
    )
    assert prepared == manifest


def test_prepare_exports_completed_output_without_running_slam(tmp_path):
    prepared = prepare_slam_input(
        _output(),
        tmp_path / "prepared",
        export_options={"sample_hz": 3.0},
    )
    document = json.loads(prepared.read_text())
    assert document["sample_hz"] == 3.0
    assert set(document["clips"]) == {"clip one"}


def test_prepare_raw_frames_runs_injected_pipeline_once(tmp_path):
    calls = {"constructed": 0, "run": 0}

    class FakePipeline:
        def __init__(self, *, depth_model, output_pointcloud):
            assert depth_model == "depth"
            assert output_pointcloud is True
            calls["constructed"] += 1

        def __call__(self, data):
            assert set(data) == {"clip one"}
            calls["run"] += 1
            return _output()

    frames = {"clip one": np.zeros((3, 4, 5, 3), dtype=np.uint8)}
    prepared = prepare_slam_input(
        frames,
        tmp_path / "prepared",
        runner_options={
            "depth_model": "depth",
            "pipeline_factory": FakePipeline,
            "output_pointcloud": True,
        },
        export_options={"sample_hz": 2.0},
    )
    assert prepared.is_file()
    assert calls == {"constructed": 1, "run": 1}


def test_prepare_can_forbid_slam_execution(tmp_path):
    frames = {"clip": np.zeros((1, 2, 2, 3), dtype=np.uint8)}
    with pytest.raises(ValueError, match="run_if_needed=False"):
        prepare_slam_input(frames, tmp_path / "out", run_if_needed=False)


def test_runner_rejects_conflicting_embedding_sources():
    frames = {"clip": np.zeros((1, 2, 2, 3), dtype=np.uint8)}
    with pytest.raises(ValueError, match="not both"):
        run_slam_pipeline(
            frames,
            depth_model=object(),
            embedding_model=object(),
            embeddings={"clip": np.zeros((1, 2))},
            pipeline_factory=lambda **kwargs: None,
        )


def test_point_cloud_confidence_curation_and_output(tmp_path):
    cloud = PointCloudArtifact(
        points=np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        ),
        colors=np.arange(12, dtype=np.uint8).reshape(4, 3),
        attributes={"confidence": np.array([0.1, 0.8, 0.6, 0.9])},
    )
    raw = cloud.save(tmp_path / "raw.npz")
    destination = tmp_path / "curated.npz"
    curated = curate_point_cloud(
        raw,
        method="confidence",
        min_confidence=0.5,
        max_points=2,
        output=destination,
    )
    assert curated.points.tolist() == [[2.0, 0.0, 0.0], [0.1, 0.1, 0.1]]
    assert curated.attributes["confidence"].tolist() == [0.9, 0.8]
    restored = load_point_cloud(destination)
    assert restored.points.tolist() == curated.points.tolist()
    assert restored.colors.tolist() == curated.colors.tolist()


def test_point_cloud_voxel_and_random_curation_are_deterministic():
    points = np.array(
        [[0.0, 0.0, 0.0], [0.2, 0.2, 0.2], [1.1, 0.0, 0.0], [2.1, 0.0, 0.0]]
    )
    cloud = PointCloudArtifact(points)
    voxel = curate_point_cloud(cloud, method="voxel", voxel_size=1.0)
    assert voxel.points.tolist() == [
        [0.0, 0.0, 0.0],
        [1.1, 0.0, 0.0],
        [2.1, 0.0, 0.0],
    ]
    first = curate_point_cloud(cloud, method="random", max_points=2, seed=7)
    second = curate_point_cloud(cloud, method="random", max_points=2, seed=7)
    assert first.points.tolist() == second.points.tolist()


def test_point_cloud_loads_directly_from_slam_manifest(tmp_path):
    manifest = export_slam_output(_output(), tmp_path / "export")
    cloud = load_point_cloud(manifest)
    assert len(cloud) == 2
    assert cloud.points[1].tolist() == [1.0, 2.0, 3.0]


def test_video_decode_preserves_timestamps_skip_and_notebook_bgr_order():
    class Provider:
        def iter_refs(self, uri, *, source_id):
            assert uri == "clip.mp4"
            assert source_id == "clip"
            return [
                SimpleNamespace(frame_index=index, timestamp=index / 10)
                for index in range(4)
            ]

        def decode_batch(self, refs):
            return np.stack([
                np.full((2, 3, 3), [ref.frame_index, 10, 20], dtype=np.uint8)
                for ref in refs
            ])

    frames, timestamps = decode_video_clips(
        {"clip": "clip.mp4"}, frame_skip=2, batch_size=1, provider=Provider()
    )
    assert timestamps["clip"].tolist() == [0.0, 0.2]
    assert frames["clip"].shape == (2, 2, 3, 3)
    assert frames["clip"][1, 0, 0].tolist() == [20, 10, 2]


def test_prepare_video_mapping_decodes_runs_and_exports_alignment(tmp_path):
    class Provider:
        def iter_refs(self, uri, *, source_id):
            return [SimpleNamespace(frame_index=i, timestamp=i * 0.25) for i in range(3)]

        def decode_batch(self, refs):
            return np.zeros((len(refs), 2, 2, 3), dtype=np.uint8)

    class FakePipeline:
        def __init__(self, *, depth_model):
            assert depth_model == "depth"

        def __call__(self, frames):
            assert frames["clip one"].shape == (3, 2, 2, 3)
            return _output()

    manifest = prepare_slam_input(
        {"clip one": "gameplay.mp4"},
        tmp_path / "prepared-video",
        runner_options={"depth_model": "depth", "pipeline_factory": FakePipeline},
        decoder_options={"provider": Provider()},
    )
    document = json.loads(manifest.read_text())
    assert document["timebase"] == "timestamp"
    assert document["clips"]["clip one"]["video"] == "gameplay.mp4"
    adapter = SLAMTrajectoryAdapter(manifest=manifest)
    assert [tick.time for tick in adapter.ticks("clip one")] == [0.0, 0.25, 0.5]
