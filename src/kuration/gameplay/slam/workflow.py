"""High-level, reproducible SLAM trajectory curation workflow."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from ...core.provenance import publishable_source_id
from ._paths import local_path
from .adapter import input_slam
from .export import _atomic_write_text


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
    feature_space: Sequence[str] | None = None,
    selection_method: str = "uniform-bins",
    bins: int = 4,
    seed: int = 0,
    progress=None,
) -> SLAMCurationResult:
    """Select diverse, reconstruction-quality gameplay windows and write a JSON manifest."""
    from . import QUALITY_TRANSFORM_NAME, TRANSFORM_NAME, register
    from .quality import _fraction

    register()
    path = local_path(manifest, name="SLAM manifest")
    destination = local_path(output, name="SLAM selection output")
    if quality_options is not None and not isinstance(quality_options, Mapping):
        raise ValueError("quality_options must be a mapping or None")
    if feature_space is None:
        features = [
            "slam_path_length",
            "slam_turn_degrees",
            "slam_spatial_extent",
            "slam_quality_score",
        ]
    else:
        if (
            isinstance(feature_space, (str, bytes))
            or not isinstance(feature_space, Sequence)
            or not feature_space
            or any(not isinstance(feature, str) or not feature for feature in feature_space)
        ):
            raise ValueError("feature_space must be a non-empty sequence of feature names")
        if len(feature_space) != len(set(feature_space)):
            raise ValueError("feature_space must not contain duplicates")
        features = list(feature_space)
    if min_quality is not None:
        min_quality = _fraction(min_quality, "min_quality")
    if selection_method not in {"uniform-bins", "k-center"}:
        raise ValueError("selection_method must be 'uniform-bins' or 'k-center'")
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
    pipeline = pipeline.options(seed=seed, on_error="raise", progress=progress)
    result = pipeline.collect()
    candidate_count = sum(
        min_quality is None or segment.features["slam_quality_score"] >= min_quality
        for segment in result.all_segments
    )
    document = {
        "format": SELECTION_FORMAT,
        "version": SELECTION_VERSION,
        "source_manifest": publishable_source_id(path.resolve()),
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
    # Same atomic writer the export path uses: a fixed-name `.tmp` sibling collides between two
    # concurrent runs writing the same selection.
    _atomic_write_text(destination, json.dumps(document, indent=2, sort_keys=True) + "\n")
    return SLAMCurationResult(
        path,
        destination,
        len(result.selected),
        candidate_count,
        tuple(result.errors),
    )
