"""DiverseSubset: round-robin across k-means clusters in the feature space.

Given a set of pre-measured features on the pipeline, DiverseSubset optionally
drops near-duplicates in the joint feature space, then samples equally from
k-means clusters over those features. Useful when a dataset has heavy
near-duplication and you want to compress to a diverse subset without losing
coverage of the feature space.

Compiles to `[.filter("dedup", ...)] -> .sample("cluster-balanced", ...)` on the
Kuration SDK pipeline. See `describe()` for the exact chain a given instance
produces.

Requires the `[methods]` extra (scikit-learn).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class DiverseSubset:
    """Cluster-balanced selection over a set of pre-measured features.

    The user is expected to have measured `features` on the pipeline via
    `.measure(...)` before applying this operation.

    Args:
        features: names of already-measured columns to cluster on.
        count: total windows to sample. Minimum 1.
        n_clusters: k-means cluster count over the joint feature space.
            Defaults to 32. Increase for large / diverse datasets.
        dedup_threshold: cosine-similarity threshold for near-duplicate
            filtering, in [-1, 1]. If None (default), dedup is skipped.

    Raises:
        ValueError: on empty features, count < 1, n_clusters < 2, or
            dedup_threshold out of range.
    """

    features: tuple[str, ...]
    count: int = 100
    n_clusters: int = 32
    dedup_threshold: float | None = None

    EXTRAS: ClassVar[tuple[str, ...]] = ("methods",)

    def __post_init__(self) -> None:
        # tolerate list/sequence at construction, store as tuple for hashability
        if not isinstance(self.features, tuple):
            object.__setattr__(self, "features", tuple(self.features))
        if not self.features:
            raise ValueError(
                "DiverseSubset: 'features' must be a non-empty sequence of "
                "measured column names to cluster on"
            )
        if any(not isinstance(f, str) or not f for f in self.features):
            raise ValueError(
                "DiverseSubset: every entry in 'features' must be a "
                "non-empty string"
            )
        if self.count < 1:
            raise ValueError(
                f"DiverseSubset: 'count' must be >= 1 (got {self.count})."
            )
        if self.n_clusters < 2:
            raise ValueError(
                f"DiverseSubset: 'n_clusters' must be >= 2 "
                f"(got {self.n_clusters})."
            )
        if self.dedup_threshold is not None and not (-1.0 <= self.dedup_threshold <= 1.0):
            raise ValueError(
                f"DiverseSubset: 'dedup_threshold' must be in [-1, 1] "
                f"(got {self.dedup_threshold})."
            )

    def apply(self, pipeline):
        p = pipeline
        if self.dedup_threshold is not None:
            p = p.filter(
                "dedup",
                feature_space=list(self.features),
                threshold=self.dedup_threshold,
            )
        return p.sample(
            "cluster-balanced",
            n=self.count,
            feature_space=list(self.features),
            clusters=self.n_clusters,  # SDK method_params key is 'clusters'
        )

    def describe(self) -> str:
        """One-line description of the SDK chain this operation compiles to."""
        chain = []
        if self.dedup_threshold is not None:
            chain.append(
                f".filter('dedup', feature_space={list(self.features)}, "
                f"threshold={self.dedup_threshold})"
            )
        chain.append(
            f".sample('cluster-balanced', n={self.count}, "
            f"feature_space={list(self.features)}, clusters={self.n_clusters})"
        )
        return f"DiverseSubset: {' -> '.join(chain)}"
