"""Local modification: Depth Anything 3's GLB/PLY export helpers are NOT vendored.

The upstream module pulls in trimesh/plyfile and writes viewer-oriented scene files, which
duplicate what `kuration.gameplay.slam.export` / `.pointcloud` already produce in a validated,
reproducible form. `api.py::_export_results` imports this module lazily, so keeping an explicit
stub here turns `inference(export_dir=...)` into a clear error instead of a ModuleNotFoundError
that contradicts the comment at the import site.
"""
from __future__ import annotations

from typing import Any


def export(prediction: Any, export_format: str, export_dir: str, **kwargs: Any) -> None:
    raise NotImplementedError(
        "Depth Anything 3's export helpers are not vendored into kuration-sdk: call "
        "inference(..., export_dir=None) and use the kuration export path "
        "(kuration.gameplay.slam.export_slam_output / curate_point_cloud) to write artifacts "
        f"(requested export_format={export_format!r}, export_dir={export_dir!r})"
    )
