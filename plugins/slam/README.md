# Kuration SLAM plugin

Curate gameplay video using camera trajectories reconstructed by a SLAM system. The first
integration targets the `SLAMPipeline` output demonstrated in `example.ipynb`, while keeping the
data contract independent of a particular depth or place-recognition model.

The plugin deliberately separates two execution scopes:

1. **Optional global preparation** runs through a direct plugin API, not a bounded transform.
   `SLAMPipeline` may compare several clips, run large GPU models, find shared places, and stitch
   one scene. It runs only for raw frames or video paths.
2. **Curation** runs inside Kuration over portable, time-aligned camera trajectories. It can
   select diverse or high-value gameplay windows and write the matching video and pose arrays.

An existing manifest or completed `SLAMPipeline` output bypasses reconstruction. This avoids
pretending that a cross-clip, stateful reconstruction is a bounded per-window transform.
Importing or installing the base plugin does not install Depth Anything, SALAD, FAISS, PyTorch,
or the user's `SLAMPipeline` implementation.

## Install

```bash
pip install kuration-plugin-slam

# Also decode MP4/MOV inputs through Kuration's PyAV provider
pip install "kuration-plugin-slam[video]"
```

For repository development:

```bash
python scripts/check_plugin.py plugins/slam
```

## 1. Normalize any supported input

`prepare_slam_input()` is the optional execution boundary. It always returns a portable manifest,
but what it does depends on the input:

```python
from kuration_slam import prepare_slam_input

# Existing manifest or export directory: validate and reuse; SLAM does not run.
manifest = prepare_slam_input("slam-export/manifest.json")

# Completed notebook output: export it; SLAM does not run again.
manifest = prepare_slam_input(
    output,
    "slam-export",
    export_options={"sample_hz": 7.5},
)

# Raw frame mapping: construct and run SLAMPipeline once, then export.
manifest = prepare_slam_input(
    data,  # {clip_name: (T,H,W,3) frames}, matching the notebook
    "slam-export",
    runner_options={
        "depth_model": depth_model,
        "embedding_model": embedding_model,
        "output_pointcloud": True,
        "window_size": 32,
        "overlap": 4,
        "bridge_size": 16,
        "process_res": 504,
        "conf_thresh": 30.0,
        "num_max_points": 1_000_000,
    },
    export_options={"sample_hz": 7.5},
)
```

With no `pipeline_factory`, the raw-input path lazily imports
`slam_pipeline:SLAMPipeline`, matching the notebook. A different implementation can be injected
as a callable or import string:

```python
runner_options={
    "depth_model": depth_model,
    "pipeline_factory": "my_slam.integration:SLAMPipeline",
}
```

Video paths are accepted directly when the `[video]` extra is installed. Timestamps and source
video paths are automatically retained for later alignment:

```python
manifest = prepare_slam_input(
    {"clip_001": "/data/clip_001.mp4", "clip_002": "/data/clip_002.mp4"},
    "slam-export",
    runner_options={"depth_model": depth_model, "embedding_model": embedding_model},
    decoder_options={"frame_skip": 4, "color_space": "BGR", "batch_size": 64},
)
```

Set `run_if_needed=False` when a workflow must reject raw input rather than unexpectedly running
an expensive reconstruction.

## 2. Export a completed reconstruction directly

After the notebook produces `output`, export it without converting any `CameraPose` objects by
hand:

```python
from kuration_slam import export_slam_output

manifest = export_slam_output(
    output,
    "slam-export",
    sample_hz=7.5,  # processed-frame rate, e.g. 30 FPS input with skip=4
    videos={name: f"/data/clips/{name}.mp4" for name in output.trajectories},
)
```

For variable-rate or irregularly sampled input, pass the exact processed-frame timestamps instead:

```python
manifest = export_slam_output(
    output,
    "slam-export",
    timestamps={"clip_001": clip_001_timestamps, "clip_002": clip_002_timestamps},
    videos={"clip_001": "/data/clip_001.mp4", "clip_002": "/data/clip_002.mp4"},
)
```

The export contains:

```text
slam-export/
├── manifest.json
├── clips/
│   └── <safe clip id>.npz
└── scene_pointcloud.npz       # present when output.points is not None
```

Each clip archive has aligned arrays for:

- `camera_position`: trajectory center `(T, 3)`
- `camera_x_axis`, `camera_y_axis`, `camera_z_axis`: orientation basis `(T, 3)`
- `camera_intrinsics`: `(cx, cy, f)` `(T, 3)`
- `camera_frustum`: five flattened 3D frustum points `(T, 15)`
- `time`: frame timestamps or processed-frame indices `(T,)`

The scene point cloud is a first-class plugin artifact referenced by the manifest. It remains
separate from Kuration's temporal tracks because points have no timestep or clip ownership in the
notebook output.

## 3. Curate trajectory windows

```python
from kuration_slam import input_slam

pipeline = (
    input_slam(manifest)
    .window(32, stride=16)
    .apply("aintropy/slam-motion-diversity", axis="slam_path_length", bins=8)
    .curate("uniform-bins", n=2_000)
)

result = pipeline.collect()
```

The operator adds these finite scalar features per window:

- `slam_path_length`
- `slam_displacement`
- `slam_motion_efficiency`
- `slam_mean_step` and `slam_max_step`
- `slam_turn_degrees`, `slam_mean_turn_degrees`, and `slam_max_turn_degrees`
- `slam_spatial_extent`

Use the transform directly when the strategy should combine several signals:

```python
pipeline = (
    input_slam(manifest)
    .window(32, stride=16)
    .transform("aintropy/slam-motion")
    .curate(
        "k-center",
        n=2_000,
        feature_space=["slam_path_length", "slam_turn_degrees", "slam_spatial_extent"],
    )
)
```

## 4. Join the original video

When every manifest clip has a `video` entry, the helper can attach lazy video frames:

```python
pipeline = input_slam(manifest, join_video=True, tolerance=0.07)
```

Timestamp alignment is recommended for the notebook because `read_video_frames(..., skip=4)`
subsamples the original MP4. `sample_hz` generates regular timestamps; explicit `timestamps=`
preserves irregular sampling. Frame-index alignment is used only when the export has neither.
It is valid only if the reconstructed trajectory contains one row for every frame in the joined
video.

After selection, ordinary Kuration output can preserve both frames and aligned fields:

```python
pipeline.output(
    "curated-gameplay",
    shard_length=32,
    modalities={
        "frame": {"dtype": "uint8", "shape": [1080, 1920, 3], "source": "media:frame"},
        "camera_position": {
            "dtype": "float32", "shape": [3], "source": "field:camera_position"
        },
        "camera_z_axis": {
            "dtype": "float32", "shape": [3], "source": "field:camera_z_axis"
        },
    },
)
```

## 5. Curate and write the scene point cloud

Load a point cloud from either `manifest.json`, the NPZ path, or an existing
`PointCloudArtifact`. Colors and any row-aligned numeric attributes remain attached through every
selection:

```python
from kuration_slam import curate_point_cloud, load_point_cloud

cloud = load_point_cloud(manifest)

# Geometry-aware reduction: retain one representative per occupied voxel.
curated = curate_point_cloud(
    cloud,
    method="voxel",
    voxel_size=0.05,
    max_points=250_000,
    output="curated-gameplay/scene_pointcloud.npz",
)
```

The available deterministic methods are:

- `voxel`: spatial coverage using one point per voxel; optional `max_points` cap
- `random`: seeded baseline requiring `max_points`
- `confidence`: highest-confidence points requiring a scalar `confidence` attribute

Point confidence or other row-aligned numeric values can be attached while exporting:

```python
manifest = export_slam_output(
    output,
    "slam-export",
    point_attributes={"confidence": point_confidence, "source_frame": source_frame},
)

curated = curate_point_cloud(
    manifest,
    method="confidence",
    min_confidence=0.7,
    max_points=250_000,
    output="curated-gameplay/high-confidence-scene.npz",
)
```

NPZ output is versioned, portable, and preserves points, optional colors, and every attribute.
Selecting scene points corresponding specifically to selected video windows requires the upstream
SLAM implementation to provide a row-aligned clip/frame ownership attribute; the notebook's
current merged `output.points` does not contain that provenance.

## Leakage boundary

Visual-place recognition can connect several clips to the same physical scene. Do not run one
joint reconstruction across the entire dataset and then randomly split those related clips into
train and test. Either:

- assign scene-connected clips to train/test before reconstruction and run SLAM separately, or
- treat every connected scene as an atomic source in the surrounding data preparation.

Kuration's ordinary split guarantee is source-level; a shared scene is a wider leakage boundary
than an individual clip.

## Registered components

- adapter: `aintropy/slam-trajectory`
- transform: `aintropy/slam-motion`
- operator: `aintropy/slam-motion-diversity`

All registrations are dependency-light and discoverable through `kur-list` / `kur-describe`.
