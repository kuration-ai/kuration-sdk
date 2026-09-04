# SLAM reconstruction and gameplay curation

`kuration.gameplay.slam` turns one or more gameplay videos into a portable 3D reconstruction:
camera trajectories, camera poses, and an optional scene point cloud. Clips connected by
place-recognition matches share an aligned coordinate frame; unmatched clips remain valid,
independent local reconstructions. The trajectory manifest is then an ordinary Kuration input, so
windows can be scored for motion, reconstruction quality, and diversity with the normal Kuration SDK.

The public facade handles both execution paths:

1. Live reconstruction is an optional, global operation. It may compare clips,
   download a depth model, use a GPU, and stitch a shared scene.
2. Precomputed SLAM output is lightweight. Opening, validating, transforming, and curating a
   manifest needs only the base SDK and NumPy; it does not run or import the model runtime.

## Install

Install the reconstruction runtime only when raw frames or videos must run through SLAM:

```bash
pip install "kuration-sdk[slam]"
```

Use `[gameplay]` to install SLAM together with Pixel Shift, action-state, and point-cloud
visualization. Point-cloud notebook visualization additionally
needs `.[slam-viz]`. Consumers that
receive `manifest.json` can use `input_slam()`, the trajectory transforms, and point-cloud
curation with the base SDK.

## Reconstruct and curate through the gameplay domain

```python
import kuration.gameplay as gameplay

pipeline = gameplay.slam_reconstruct(
    {
        "match-001": "/data/match-001.mp4",
        "match-002": "/data/match-002.mp4",
    },
    "slam-export",
    depth_model="da3-base",
    depth_model_id="depth-anything/da3-base@MODEL_REVISION",
    output_pointcloud=True,
    window_size=48,
    overlap=4,
    decoder_options={"frame_skip": 4, "color_space": "RGB"},
    resume=True,
    join_video=True,
)

result = (
    pipeline
    .window(32, stride=16)
    .slam_motion_diversity(axis="slam_path_length", bins=8)
    .slam_quality(min_score=0.6)
    .sample("uniform-bins", n=2_000)
    .collect()
)
```

`slam_reconstruct()` returns the immutable core `Pipeline`. Its gameplay methods resolve through the
same registered operators used by declarative jobs, so provenance retains stable component names.
Raw input requires the `slam` extra and an output directory. Passing an existing manifest does
not load the model runtime.

## Advanced reconstruction control

The names describe different layers:

- `gameplay.slam_reconstruct(...)` is a convenience function returning the ordinary `Pipeline`.
- `gameplay.SLAMReconstructor` is the separate global reconstruction coordinator.
  Reconstruction compares whole clips and writes a reusable artifact, so it does not satisfy the
  immutable, bounded curation-operator contract.

`gameplay.SLAMReconstructor` exposes the intermediate reconstruction when an application needs to
inspect or reuse it before constructing a curation pipeline:

```python
import kuration.gameplay as gameplay

slam = gameplay.SLAMReconstructor(
    depth_model="da3-base",
    depth_model_id="depth-anything/da3-base@MODEL_REVISION",
)
reconstruction = slam.reconstruct(videos, "slam-export", resume=True)
pipeline = reconstruction.to_pipeline(join_video=True)
```

When a model is supplied as an object, set `depth_model_id` and, when applicable,
`embedding_model_id` to a stable model name plus revision. Those IDs, the engine, and JSON-safe
runtime options are recorded in the manifest. `resume=True` derives its cache identity from that
provenance plus the input clips/frame content and decoder/export settings; a caller can override
it with `cache_key=`.

`SLAMReconstructor(data)` or `SLAMReconstructor.run(data)` preserves the notebook-style in-memory
path and returns `SLAMOutput`. `SLAMReconstructor.reconstruct(...)` is the artifact-producing SDK boundary and
returns `SLAMReconstruction`, whose `manifest` field is the path to the portable export.

Custom depth or place-recognition runtimes can be injected without changing the artifact format:

```python
slam = SLAMReconstructor(
    depth_model=my_depth_model,
    depth_model_id="my-depth-model@2026-07-15",
    embedding_model=my_place_model,
    embedding_model_id="my-place-model@v2",
    runtime_factory="my_game.slam:SLAMOperator",
)
```

The factory must construct a callable accepting `{clip_name: (T, H, W, 3) frames}` and return an
object with `trajectories` and `poses` mappings plus optional `points` and `colors`.

## Curate precomputed output without running SLAM

```python
import kuration.gameplay as gameplay

pipeline = (
    gameplay.slam_reconstruct("slam-export/manifest.json")
    .window(32, stride=16)
    .slam_motion_diversity(axis="slam_path_length", bins=8)
    .slam_quality(min_score=0.6)
)
```

`gameplay.input_slam()` is the direct manifest-only equivalent. `prepare_slam_input()` is the
lower-level normalization API. It reuses an existing manifest,
exports a completed `SLAMOutput`, or runs raw inputs only when necessary. Set
`run_if_needed=False` when an application must reject raw input rather than trigger an expensive
run.

```python
from kuration.gameplay.slam import prepare_slam_input

manifest = prepare_slam_input("slam-export/manifest.json")  # no model runtime
manifest = prepare_slam_input(output, "slam-export")         # export only
```

Local video references outside the export directory are copied atomically into `media/` under a
content-derived filename. References already inside the export remain relative, so moving the
directory preserves video joins without publishing the original host path or filename. A clean
remote URI such as `gs://bucket/clip.mp4` remains an external dependency and is not copied;
URIs containing embedded credentials, query/path parameters, or fragments are rejected. Configure
remote credentials through the media provider instead of recording them in an artifact.

## Preflight and progress

`preflight_slam()` checks input shape, manifest contents, decoder availability, output
permissions, model/factory configuration, and optional CUDA requirements without decoding video
or constructing models.

The preparation APIs emit structured `slam-preflight`, `slam-decode`,
`slam-reconstruction`, `slam-export`, `slam-cache`, and `slam-prepare` progress events.
Pass `progress="auto"` through `gameplay.slam_reconstruct(...)` to get compact updating rows in
JupyterLab and terminal progress in scripts; direct lower-level APIs continue to accept the
structured callback returned by `progress_callback("auto")`.

## Registered trajectory components

| component | kind | purpose |
|---|---|---|
| `aintropy/slam-trajectory` | adapter | Read per-clip camera trajectories from a manifest |
| `aintropy/slam-motion` | transform | Add translation and camera-turn features |
| `aintropy/slam-motion-diversity` | operator | Add motion features and quantile-bin an axis |
| `aintropy/slam-quality` | transform / operator | Score tracking gaps, jumps, and scale instability; optionally filter windows |

Live reconstruction is intentionally not a registered compositional operator: it is a global,
artifact-producing `SLAMReconstructor` operation. The registered components remain bounded and work
with direct methods, planning, provenance, and curation.

## Point clouds

```python
from kuration.gameplay.slam import curate_point_cloud

curated = curate_point_cloud(
    "slam-export/manifest.json",
    method="voxel",
    voxel_size=0.05,
    max_points=250_000,
    output="curated-scene.npz",
)
```

`voxel` preserves spatial coverage, `random` is a seeded baseline, and `confidence` keeps the
highest-confidence rows when a scalar confidence attribute was exported.

## Notebook and leakage boundary

The checked-in [SLAM notebook](https://github.com/aintropy-solutions/kuration-sdk/blob/main/examples/notebooks/slam_kurious.ipynb) is output-free and uses
parameterized paths; it is a walkthrough, not a bundled gameplay dataset. The separate
[gameplay stats notebook](https://github.com/aintropy-solutions/kuration-sdk/blob/main/examples/notebooks/gameplay_stats.ipynb)
is model-free and demonstrates lifecycle stats, notebook progress, domain plots, and optional
video comparison.

Place recognition can connect multiple clips to one physical scene. Assign scene-connected clips
to a split before reconstruction, or treat the connected scene as one atomic group. Do not
reconstruct a shared scene and then randomly split its clips across train and test.

## Command-line interface

```bash
kur-slam preflight \
  --video match-001=/data/match-001.mp4 \
  --out slam-work \
  --depth-model da3-base \
  --depth-model-id depth-anything/da3-base@MODEL_REVISION \
  --frame-skip 4 \
  --require-gpu

kur-slam run \
  --video match-001=/data/match-001.mp4 \
  --out slam-work \
  --depth-model da3-base \
  --depth-model-id depth-anything/da3-base@MODEL_REVISION \
  --frame-skip 4 \
  --window 32 --stride 16 --count 2000 --min-quality 0.6
```

For custom runtimes, use `--runtime-factory module:callable` instead of `--depth-model`. The
factory returns `run_slam_reconstruction()` options and should honor `preflight=True` without allocating
models.

For a model-free end-to-end smoke test:

```bash
kur-slam demo --out slam-demo
```

`kur-slam run` and `kur-slam demo` refuse a non-empty output by default. Use `--resume` on `run`
to reuse a matching reconstruction, or `--force` to update only the files owned by the command;
`--force` does not delete unrelated files. Output roots that overlap an input manifest or local
video are rejected before decoding or reconstruction starts. Add `--debug` to any subcommand when
a full traceback is needed; ordinary failures are a single `error:` line.
