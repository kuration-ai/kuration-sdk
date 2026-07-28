"""Local-only path handling for SLAM artifacts."""
from __future__ import annotations

from pathlib import Path


def local_path(value: str | Path, *, name: str) -> Path:
    """Return ``value`` as a local :class:`Path`, rejecting remote URIs loudly.

    Every SLAM artifact (manifest, per-clip NPZ, point cloud, export tree) is read and written
    through ``pathlib`` and NumPy file handles, so a ``gs://`` / ``s3://`` / ``az://`` URI would
    be silently mangled into ``gs:/bucket/...`` and surface as a confusing missing-file error.
    Remote artifact stores are out of scope; video *references* recorded in a manifest may stay
    remote because they are opened through a media provider, not through this helper.
    """
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{name} must be a non-empty local path")
    text = str(value)
    if not text:
        raise ValueError(f"{name} must be a non-empty local path")
    if "://" in text:
        raise ValueError(
            f"{name} must be a local path: SLAM artifacts are local-only, got {text!r} -- "
            "download the reconstruction to a local directory first"
        )
    return Path(value)
