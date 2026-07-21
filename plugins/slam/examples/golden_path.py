"""Offline SLAM plugin smoke test; no video decoder, GPU, or model download required."""
from pathlib import Path

from kuration_slam import curate_point_cloud, curate_slam_gameplay
from kuration_slam.demo import write_demo_reconstruction


workspace = Path("slam-demo")
manifest = write_demo_reconstruction(workspace / "slam-export")
selection = curate_slam_gameplay(
    manifest,
    workspace / "selection.json",
    window=16,
    stride=8,
    n=12,
    min_quality=0.5,
)
curate_point_cloud(
    manifest,
    method="voxel",
    voxel_size=0.05,
    max_points=500,
    output=workspace / "scene_pointcloud.npz",
)
print(selection.to_dict())

