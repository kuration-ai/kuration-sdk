# SLAM trajectory curation

`kuration.slam` turns video into a portable SLAM export (camera trajectory + scene point
cloud), then lets you curate it like any other Kuration pipeline: window it, score it, and
select a diverse, high-quality subset.

## Install

```bash
pip install "kuration-sdk[slam]"
```

## Quick example

Build a `SLAMOperator` once, reconstruct a folder of clips with `.apply(...)`, then curate
the result. This is the same flow walked through end to end in `video_to_pointcloud.ipynb`:

```python
from pathlib import Path

from kuration import input
from kuration.slam import SLAMOperator, curate_slam_gameplay

pipeline = SLAMOperator(
    depth_model="da3-base",
    output_pointcloud=True,
    window_size=32,
    overlap=4,
    bridge_size=16,
    process_res=504,
    conf_thresh=20.0,
    num_max_points=1_000_000,
)

videos = sorted(str(p) for p in Path("clips").glob("*.mp4"))
reconstructed = input(videos).apply(
    "aintropy/slam-reconstruct", pipeline=pipeline, out_dir="slam-export"
)

manifest_path = Path("slam-export") / "manifest.json"
selection = curate_slam_gameplay(
    manifest_path, "selection.json", window=16, stride=8, n=8, min_quality=None,
)
```

`aintropy/slam-reconstruct` decodes every clip, runs the `SLAMOperator`, and exports
`manifest.json`, one trajectory NPZ per clip, and `scene_pointcloud.npz` into `out_dir`.
Clip names come from each video's filename stem, so sources must have unique stems. No
place-recognition model was passed to `SLAMOperator` above, so each clip is just cut into
equal-size windows with no cross-clip stitching — pass `embedding_model=...` to stitch
clips that revisit the same place.

See `video_to_pointcloud.ipynb` for the full walkthrough, including a 3D point-cloud
viewer over the reconstructed scene.

## Stitching clips with a place-recognition model

Pass `embedding_model=` to detect when two clips (or two moments in the same clip)
revisit the same physical place, so `SLAMOperator` anchors chunk boundaries there and
stitches them into one continuous trajectory instead of reconstructing each clip in
isolation.

The only requirement is an object exposing `.encode(frames)` (or `__call__(frames)`)
that takes a list of `(H, W, 3)` uint8 frames and returns an `(N, D)` array of global
descriptors. `example.ipynb` wraps SALAD, a visual place-recognition model, like this:

```python
import cv2
import numpy as np
import torch
import torchvision.transforms as T


class SaladEmbeddingModel(torch.nn.Module):
    def __init__(self, device, image_size=(224, 224)):
        super().__init__()
        self.backbone = salad_helper.get_backbone(
            'dinov2_vitb14', {'num_trainable_blocks': 4, 'return_token': True, 'norm_layer': True}
        )
        self.aggregator = salad_helper.get_aggregator(
            'SALAD', {'num_channels': 768, 'num_clusters': 64, 'cluster_dim': 128, 'token_dim': 256}
        )
        self.load_state_dict(torch.load(SALAD_CKPT_PATH, map_location='cpu', weights_only=True))
        self.device = device
        self.to(device).eval()
        self.requires_grad_(False)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(image_size, T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        return self.aggregator(self.backbone(x))

    @torch.no_grad()
    def encode(self, frames, batch_size=64):
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        embs = []
        for start in range(0, len(rgb_frames), batch_size):
            batch = torch.stack(
                [self.transform(f) for f in rgb_frames[start:start + batch_size]]
            ).to(self.device)
            embs.append(self(batch).cpu().float().numpy())
        return np.concatenate(embs, axis=0)


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
embedding_model = SaladEmbeddingModel(device)
```

`salad_helper` and `SALAD_CKPT_PATH` come from the SALAD repo, cloned and wired up
alongside Depth Anything 3 in `example.ipynb`'s `## 1. Setup` and `## 2. Load models`
cells — pull those cells if you want this exact model; any other VPR model with the
same `.encode(frames)` contract works too. That wrapper's `cv2.cvtColor(..., BGR2RGB)`
step assumes BGR input frames, matching `example.ipynb`'s raw `cv2`-based loader — drop
that conversion when frames come from `decode_video_clips`/`aintropy/slam-reconstruct`,
which decode as RGB by default.

Wire it into `SLAMOperator` the same way as `depth_model`:

```python
pipeline = SLAMOperator(
    depth_model="da3-base",
    embedding_model=embedding_model,
    output_pointcloud=True,
    window_size=32,
    overlap=4,
    bridge_size=16,
    process_res=504,
)
```

Already have embeddings computed elsewhere? Pass them directly instead of a model —
either a list aligned with your clip order, or a `{clip_name: (N_i, D)}` mapping:

```python
pipeline = SLAMOperator(
    depth_model="da3-base",
    embeddings={name: embedding_model.encode(list(frames)) for name, frames in data.items()},
    output_pointcloud=True,
)
```

`find_intersections()` also treats a clip crossing its own earlier path (at least
`window_size` frames apart) as a loop closure, so a single long clip benefits from
`embedding_model=` too, not just multi-clip datasets.

## No GPU? Try the demo

```bash
kur-slam demo --out slam-demo
```

Runs a deterministic, model-free version of the same pipeline (export, quality scoring,
window selection, point-cloud reduction) so you can exercise the whole flow without a
depth model or real video.

## Already have a manifest or a completed reconstruction?

Skip `.apply("aintropy/slam-reconstruct", ...)` and start directly from either:

```python
from kuration.slam import export_slam_output, input_slam

# From a manifest already on disk
pipeline = input_slam("slam-export/manifest.json")

# From a completed SLAMOperator output (has .trajectories/.poses/.points/.colors)
manifest = export_slam_output(output, "slam-export", sample_hz=7.5)
```

## Registered components

| Name | Kind | What it does |
| --- | --- | --- |
| `aintropy/slam-trajectory` | adapter | Reads a SLAM manifest's per-clip camera trajectories into a Kuration pipeline |
| `aintropy/slam-reconstruct` | operator | Runs a live `SLAMOperator` over a video pipeline and swaps in its exported trajectory |
| `aintropy/slam-motion` | transform | Adds translation/turn motion features to trajectory windows |
| `aintropy/slam-motion-diversity` | operator | Adds motion features, then quantile-bins one axis for diverse selection |
| `aintropy/slam-quality` | transform / operator | Flags tracking gaps, pose jumps, and unstable scale; the operator form can also filter out low-quality windows |

All five are dependency-light and discoverable via `kur-list` / `kur-describe`.

## Curating trajectory windows manually

`curate_slam_gameplay` is the one-call reference workflow above. To combine motion and
quality signals yourself:

```python
pipeline = (
    input_slam("slam-export/manifest.json")
    .window(32, stride=16)
    .apply("aintropy/slam-motion-diversity", axis="slam_path_length", bins=8)
    .apply("aintropy/slam-quality", min_score=0.6)
    .curate("uniform-bins", n=2_000)
)
result = pipeline.collect()
```

## Joining the original video back in

```python
pipeline = input_slam("slam-export/manifest.json", join_video=True, tolerance=0.07)
```

Or pass `join_video=True` directly to `.apply("aintropy/slam-reconstruct", ...)`.

## Point clouds

```python
from kuration.slam import curate_point_cloud

curated = curate_point_cloud(
    "slam-export/manifest.json",
    method="voxel",
    voxel_size=0.05,
    max_points=250_000,
    output="curated-scene.npz",
)
```

`voxel` keeps one representative point per occupied voxel for spatial coverage, `random`
is a seeded baseline that needs `max_points`, and `confidence` keeps the
highest-confidence points (needs a `confidence` point attribute set at export time via
`export_slam_output(..., point_attributes={"confidence": ...})`).

## Leakage boundary

Visual-place recognition can connect several clips to the same physical scene. Don't
reconstruct everything together and then randomly split those connected clips into
train/test — assign scene-connected clips to a split *before* reconstruction, or treat
each connected scene as one atomic source in the surrounding data preparation.
