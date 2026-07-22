"""High-level, reproducible SLAM trajectory curation workflow."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from .adapter import input_slam


SELECTION_FORMAT = "kuration-slam-selection"
SELECTION_VERSION = 1


@dataclass(frozen=True)
class SLAMCurationResult:
    manifest: Path
    selection: Path
    selected_count: int
    candidate_count: int
    errors: tuple[dict, ...]

    def to_dict(self) -> dict:
        return {
            "manifest": str(self.manifest),
            "selection": str(self.selection),
            "selected_count": self.selected_count,
            "candidate_count": self.candidate_count,
            "errors": [dict(error) for error in self.errors],
        }


def _selection_row(segment) -> dict:
    return {
        "source": segment.source_id,
        "start_index": segment.start_idx,
        "end_index": segment.end_idx,
        "time_range": [float(value) for value in segment.time_range],
        "features": {name: float(value) for name, value in sorted(segment.features.items())},
        "bin": list(segment.bin),
    }


def curate_slam_gameplay(
    manifest: str | Path,
    output: str | Path,
    *,
    window: int = 32,
    stride: int = 16,
    n: int = 2_000,
    min_quality: float | None = 0.5,
    quality_options: Mapping[str, object] | None = None,
    feature_space: list[str] | None = None,
    selection_method: str = "uniform-bins",
    bins: int = 4,
    seed: int = 0,
    progress=None,
) -> SLAMCurationResult:
    """Select diverse, reconstruction-quality gameplay windows and write a JSON manifest."""
    from . import QUALITY_TRANSFORM_NAME, TRANSFORM_NAME, register

    register()
    path = Path(manifest)
    destination = Path(output)
    features = feature_space or [
        "slam_path_length",
        "slam_turn_degrees",
        "slam_spatial_extent",
        "slam_quality_score",
    ]
    pipeline = (
        input_slam(path)
        .window(window, stride=stride)
        .transform(TRANSFORM_NAME)
        .transform(QUALITY_TRANSFORM_NAME, **dict(quality_options or {}))
    )
    if min_quality is not None:
        pipeline = pipeline.filter("slam_quality_score", ">=", min_quality)
    if selection_method == "uniform-bins":
        for feature in features:
            pipeline = pipeline.bin(feature, bins=bins)
        pipeline = pipeline.curate("uniform-bins", n=n)
    elif selection_method == "k-center":
        pipeline = pipeline.curate("k-center", n=n, feature_space=list(features))
    else:
        raise ValueError("selection_method must be 'uniform-bins' or 'k-center'")
    pipeline = pipeline.options(seed=seed, on_error="raise", progress=progress)
    result = pipeline.collect()
    candidate_count = sum(
        min_quality is None or segment.features["slam_quality_score"] >= min_quality
        for segment in result.all_segments
    )
    document = {
        "format": SELECTION_FORMAT,
        "version": SELECTION_VERSION,
        "source_manifest": str(path.resolve()),
        "window": window,
        "stride": stride,
        "selection": {
            "method": selection_method,
            "requested": n,
            "selected": len(result.selected),
            "candidates": candidate_count,
            "feature_space": list(features),
            "bins": bins if selection_method == "uniform-bins" else None,
            "minimum_quality": min_quality,
            "seed": seed,
        },
        "segments": [_selection_row(segment) for segment in result.selected],
        "errors": list(result.errors),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return SLAMCurationResult(
        path,
        destination,
        len(result.selected),
        candidate_count,
        tuple(result.errors),
    )
