from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kuration import Pipeline, TransformContext, default_registry, describe_component
from kuration import gameplay
from kuration.gameplay.slam import (
    ADAPTER_NAME,
    OPERATOR_NAME,
    OUTPUT_FEATURES,
    QUALITY_FEATURES,
    QUALITY_TRANSFORM_NAME,
    PointCloudArtifact,
    SLAMQuality,
    TRANSFORM_NAME,
    SLAMMotion,
    SLAMTrajectoryAdapter,
    build_pointcloud_scene,
    curate_point_cloud,
    curate_slam_gameplay,
    decode_video_clips,
    export_slam_output,
    input_slam,
    load_point_cloud,
    prepare_slam_input,
    preflight_slam,
    register,
    run_slam_reconstruction,
    slam_provenance,
    slam_schema,
)
from kuration.gameplay.slam.cli import main as cli_main
from kuration.gameplay.slam.demo import write_demo_reconstruction


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
    video.write_bytes(b"portable-video")
    manifest_path = export_slam_output(
        _output(),
        tmp_path / "export",
        sample_hz=2.0,
        videos={"clip one": video},
        point_attributes={"confidence": np.array([0.25, 0.9])},
        tracking_confidence={"clip one": np.array([1.0, 0.4, 0.9])},
    )
    document = json.loads(manifest_path.read_text())
    assert document["format"] == "kuration-slam-export"
    assert document["timebase"] == "timestamp"
    assert document["scene"]["pointcloud"].startswith("scene_pointcloud-")
    assert document["scene"]["pointcloud"].endswith(".npz")
    assert document["scene"]["count"] == 2
    assert document["scene"]["attributes"] == ["confidence"]
    trajectory_name = document["clips"]["clip one"]["trajectory"]
    assert "clip one" not in Path(trajectory_name).name

    adapter = SLAMTrajectoryAdapter(manifest=manifest_path)
    assert adapter.sources() == ["clip one"]
    assert adapter.schema.sample_hz == 2.0
    bundled_video = Path(adapter.video_mapping["clip one"])
    assert bundled_video.parent == manifest_path.parent / "media"
    assert bundled_video.name.startswith("video-")
    assert bundled_video.read_bytes() == b"portable-video"
    ticks = adapter.ticks("clip one")
    assert [tick.time for tick in ticks] == [0.0, 0.5, 1.0]
    assert ticks[2].entities[0]["camera_position"] == (1.0, 1.0, 0.0)
    assert ticks[1].entities[0]["tracking_confidence"] == pytest.approx(0.4)
    assert len(ticks[0].entities[0]["camera_frustum"]) == 15
    adapter.check_source("clip one")

    with np.load(manifest_path.parent / document["scene"]["pointcloud"]) as cloud:
        assert cloud["points"].shape == (2, 3)
        assert cloud["colors"].dtype == np.uint8
        assert cloud["confidence"].tolist() == [0.25, 0.9]


def test_build_pointcloud_scene_covers_exported_geometry(monkeypatch):
    trimesh = pytest.importorskip("trimesh")
    monkeypatch.setattr(
        trimesh,
        "load_path",
        lambda *_args, **_kwargs: pytest.fail(
            "ordered line segments should not require trimesh's graph backends"
        ),
    )

    scene = build_pointcloud_scene(_output(), max_display_points=1)

    assert isinstance(scene, trimesh.Scene)
    clouds = [
        geometry
        for geometry in scene.geometry.values()
        if isinstance(geometry, trimesh.PointCloud)
    ]
    assert len(clouds) == 1
    assert len(clouds[0].vertices) == 1
    assert sum(isinstance(geometry, trimesh.path.Path3D)
               for geometry in scene.geometry.values()) == 4
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            build_pointcloud_scene(_output(), max_display_points=invalid)


def test_top_level_slam_reconstruct_returns_the_normal_pipeline(tmp_path):
    manifest = export_slam_output(_output(), tmp_path / "export", sample_hz=2.0)

    pipeline = gameplay.slam_reconstruct(manifest)
    assert type(pipeline) is Pipeline

    pipeline = (
        pipeline
        .window(2, stride=1)
        .slam_motion_diversity(axis="slam_turn_degrees", bins=2)
        .slam_quality(min_score=0.0)
    )
    result = pipeline.collect(n=10)

    assert not result.errors
    assert len(result.all_segments) == 2
    assert all("slam_path_length" in segment.features for segment in result.all_segments)
    assert all("slam_quality_score" in segment.features for segment in result.all_segments)


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


def test_export_rejects_invalid_tracking_confidence(tmp_path):
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        export_slam_output(
            _output(),
            tmp_path,
            tracking_confidence={"clip one": np.array([1.0, -0.1, 0.9])},
        )


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


def test_slam_transforms_honor_input_remapping_and_ragged_valid_steps():
    """Built-in transforms follow the same role-remapping and padding contract as plugins."""
    context = TransformContext(
        device="cpu",
        schema=None,
        transform="SLAM regression",
        inputs={
            "field:camera_position": "field:position",
            "field:camera_z_axis": "field:forward",
            "field:tracking_confidence": "field:confidence",
        },
        valid_steps=np.asarray([3, 1]),
    )
    batch = {
        "field:position": np.asarray([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[4.0, 5.0, 6.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
        ]),
        "field:forward": np.asarray([
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
        ]),
        "field:confidence": np.asarray([
            [1.0, 0.8, 0.6],
            [0.75, np.nan, np.nan],
        ]),
    }

    motion = SLAMMotion().transform_batch(batch, context)
    quality = SLAMQuality().transform_batch(batch, context)

    assert motion["slam_path_length"] == pytest.approx([2.0, 0.0])
    assert motion["slam_spatial_extent"] == pytest.approx([2.0, 0.0])
    assert quality["slam_tracking_confidence_mean"] == pytest.approx([0.8, 0.75])
    assert quality["slam_translation_jump_fraction"] == pytest.approx([0.0, 0.0])
    assert np.isfinite(
        np.concatenate([*motion.values(), *quality.values()])
    ).all()


def test_quality_transform_flags_confidence_gaps_pose_jumps_and_scale_drift():
    positions = np.array([[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [22.0, 0.0, 0.0],
    ]])
    forwards = np.array([[
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]])
    confidence = np.array([[1.0, 0.2, 1.0, 1.0]])
    result = SLAMQuality().transform_batch({
        "field:camera_position": positions,
        "field:camera_z_axis": forwards,
        "field:tracking_confidence": confidence,
    })
    assert set(result) == set(QUALITY_FEATURES)
    assert result["slam_tracking_gap_fraction"] == pytest.approx([0.25])
    assert result["slam_translation_jump_fraction"] == pytest.approx([1 / 3])
    assert result["slam_rotation_jump_fraction"] == pytest.approx([1 / 3])
    assert result["slam_step_scale_drift"][0] > 1
    assert 0 < result["slam_quality_score"][0] < 1


def test_registered_operator_runs_end_to_end(tmp_path):
    manifest = export_slam_output(_output(), tmp_path / "export", sample_hz=2.0)
    register()
    pipeline = (
        input_slam(manifest)
        .window(3)
        .slam_motion_diversity(axis="slam_path_length", bins=3)
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
    quality = describe_component("transform", QUALITY_TRANSFORM_NAME, registry=snapshot)
    assert adapter["parameters"]["required"] == ["manifest"]
    assert adapter["extra"] == "slam"
    assert adapter["package"] == "kuration-sdk"
    assert adapter["install_hint"] == 'pip install "kuration-sdk[slam]"'
    assert transform["package"] == "kuration-sdk"
    assert transform["description"]
    assert operator["parameters"]["properties"]["bins"]["minimum"] == 2
    assert "slam_quality_score" in QUALITY_FEATURES
    assert quality["parameters"]["properties"]["max_turn_degrees"]["maximum"] == 180
    assert "aintropy/slam-reconstruct" not in snapshot.names("operator")


def test_schema_has_fixed_pose_field_widths():
    schema = slam_schema(sample_hz=7.5)
    assert schema.fields["camera_position"].width == 3
    assert schema.fields["camera_intrinsics"].width == 3
    assert schema.fields["camera_frustum"].width == 15
    assert schema.fields["tracking_confidence"].kind == "scalar"
    assert schema.sample_hz == 7.5


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_slam_provenance_rejects_nonfinite_runtime_options(bad):
    with pytest.raises(ValueError, match="JSON-safe and finite"):
        slam_provenance({"depth_model": "da3-base", "threshold": bad})


def test_prepare_reuses_existing_manifest_without_running_slam(tmp_path):
    manifest = export_slam_output(_output(), tmp_path / "existing")

    def must_not_run(**kwargs):
        raise AssertionError("SLAM must not be constructed for manifest input")

    prepared = prepare_slam_input(
        manifest,
        runner_options={"depth_model": object(), "runtime_factory": must_not_run},
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


def test_prepare_raw_frames_runs_injected_runtime_once(tmp_path):
    calls = {"constructed": 0, "run": 0}

    class FakeRuntime:
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
            "runtime_factory": FakeRuntime,
            "output_pointcloud": True,
        },
        export_options={"sample_hz": 2.0},
    )
    assert prepared.is_file()
    assert calls == {"constructed": 1, "run": 1}


def test_prepare_resume_reuses_matching_cache_and_reports_progress(tmp_path):
    calls = {"run": 0}

    class FakeRuntime:
        def __init__(self, *, depth_model):
            assert depth_model == "depth"

        def __call__(self, frames):
            calls["run"] += 1
            return _output()

    frames = {"clip one": np.zeros((3, 2, 2, 3), dtype=np.uint8)}
    events = []
    options = {"depth_model": "depth", "runtime_factory": FakeRuntime}
    first = prepare_slam_input(
        frames,
        tmp_path / "cached",
        runner_options=options,
        resume=True,
        cache_key="fixture-v1",
        progress=events.append,
    )
    second = prepare_slam_input(
        frames,
        tmp_path / "cached",
        runner_options=options,
        resume=True,
        cache_key="fixture-v1",
        progress=events.append,
    )
    assert first == second
    assert calls["run"] == 1
    assert any(event["phase"] == "slam-cache" and "reused" in event["detail"]
               for event in events)


def test_prepare_resume_rejects_cache_marker_for_changed_manifest(tmp_path):
    calls = {"run": 0}

    class FakeRuntime:
        def __init__(self, *, depth_model):
            assert depth_model == "depth"

        def __call__(self, frames):
            calls["run"] += 1
            return _output()

    frames = {"clip one": np.zeros((3, 2, 2, 3), dtype=np.uint8)}
    options = {"depth_model": "depth", "runtime_factory": FakeRuntime}
    manifest = prepare_slam_input(
        frames,
        tmp_path / "cached",
        runner_options=options,
        resume=True,
        cache_key="fixture-v1",
    )
    manifest.write_text(manifest.read_text() + "\n")

    prepare_slam_input(
        frames,
        tmp_path / "cached",
        runner_options=options,
        resume=True,
        cache_key="fixture-v1",
    )

    assert calls["run"] == 2


def test_preflight_is_actionable_for_manifest_and_missing_raw_runtime(tmp_path):
    manifest = export_slam_output(_output(), tmp_path / "existing")
    ready = preflight_slam(manifest)
    assert ready.ready
    assert ready.will_run_slam is False
    assert "reconstruction: skipped" in ready.render()

    frames = {"clip": np.zeros((1, 2, 2, 3), dtype=np.uint8)}
    missing = preflight_slam(frames, tmp_path / "raw")
    assert not missing.ready
    failures = {check["name"] for check in missing.checks if check["status"] == "fail"}
    assert "depth-model" in failures
    runtime = next(check for check in missing.checks if check["name"] == "slam-runtime")
    assert runtime["status"] in {"pass", "fail"}  # depends on whether [slam] is installed

    unavailable = preflight_slam(
        frames,
        tmp_path / "unavailable-runtime",
        runner_options={"runtime_factory": "missing_slam_runtime:SLAMOperator"},
    )
    assert any(
        check["name"] == "slam-runtime" and check["status"] == "fail"
        for check in unavailable.checks
    )


def test_preflight_rejects_manifest_with_missing_trajectory_archive(tmp_path):
    manifest = export_slam_output(_output(), tmp_path / "broken")
    document = json.loads(manifest.read_text())
    (manifest.parent / document["clips"]["clip one"]["trajectory"]).unlink()
    report = preflight_slam(manifest)
    assert not report.ready
    assert any(check["name"] == "manifest" and check["status"] == "fail"
               for check in report.checks)


def test_prepare_can_forbid_slam_execution(tmp_path):
    frames = {"clip": np.zeros((1, 2, 2, 3), dtype=np.uint8)}
    with pytest.raises(ValueError, match="run_if_needed=False"):
        prepare_slam_input(frames, tmp_path / "out", run_if_needed=False)


def test_runner_rejects_conflicting_embedding_sources():
    frames = {"clip": np.zeros((1, 2, 2, 3), dtype=np.uint8)}
    with pytest.raises(ValueError, match="not both"):
        run_slam_reconstruction(
            frames,
            depth_model=object(),
            embedding_model=object(),
            embeddings={"clip": np.zeros((1, 2))},
            runtime_factory=lambda **kwargs: None,
        )


def test_unknown_default_runtime_options_are_rejected_before_any_cached_path(tmp_path):
    with pytest.raises(TypeError, match="tolerence"):
        gameplay.SLAMReconstructor(object(), tolerence=0.1)
    frames = {"clip": np.zeros((1, 2, 2, 3), dtype=np.uint8)}
    with pytest.raises(TypeError, match="window_sizes"):
        run_slam_reconstruction(frames, depth_model=object(), window_sizes=8)
    # Reusing an existing manifest must not silently ignore the same mistake.
    manifest = export_slam_output(_output(), tmp_path / "existing")
    with pytest.raises(TypeError, match="tolerence"):
        prepare_slam_input(
            manifest, runner_options={"depth_model": object(), "tolerence": 0.1}
        )
    # A custom runtime factory owns its own signature, so its options pass through.
    runner = gameplay.SLAMReconstructor(
        object(), runtime_factory=lambda **_: None, custom_option=1
    )
    assert runner.runner_options["custom_option"] == 1


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
        {"clip": "clip.mp4"}, frame_skip=2, batch_size=1, color_space="BGR",
        provider=Provider()
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

    class FakeRuntime:
        def __init__(self, *, depth_model):
            assert depth_model == "depth"

        def __call__(self, frames):
            assert frames["clip one"].shape == (3, 2, 2, 3)
            return _output()

    video = tmp_path / "gameplay.mp4"
    video.write_bytes(b"gameplay")
    manifest = prepare_slam_input(
        {"clip one": str(video)},
        tmp_path / "prepared-video",
        runner_options={"depth_model": "depth", "runtime_factory": FakeRuntime},
        decoder_options={"provider": Provider()},
    )
    document = json.loads(manifest.read_text())
    assert document["timebase"] == "timestamp"
    reference = document["clips"]["clip one"]["video"]
    assert Path(reference).parent == Path("media")
    assert video.name not in reference
    assert (manifest.parent / reference).read_bytes() == b"gameplay"
    adapter = SLAMTrajectoryAdapter(manifest=manifest)
    assert [tick.time for tick in adapter.ticks("clip one")] == [0.0, 0.25, 0.5]


def test_offline_golden_path_writes_quality_selection_and_curated_cloud(tmp_path):
    manifest = write_demo_reconstruction(tmp_path / "demo", frames_per_clip=48)
    result = curate_slam_gameplay(
        manifest,
        tmp_path / "selection.json",
        window=12,
        stride=6,
        n=6,
    )
    document = json.loads(result.selection.read_text())
    assert document["format"] == "kuration-slam-selection"
    assert str(manifest.resolve()) not in document["source_manifest"]
    assert not Path(document["source_manifest"]).is_absolute()
    assert len(document["segments"]) == result.selected_count == 6
    assert document["selection"]["candidates"] == result.candidate_count
    assert all("slam_quality_score" in row["features"] for row in document["segments"])

    assert cli_main(["demo", "--out", str(tmp_path / "cli-demo"), "--quiet"]) == 0
    assert (tmp_path / "cli-demo" / "selection.json").is_file()
    assert (tmp_path / "cli-demo" / "scene_pointcloud.npz").is_file()
