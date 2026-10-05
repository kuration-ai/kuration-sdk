"""Pre-built dataset-selection recipes.

Each recipe is a small, self-contained composition of Kuration SDK primitives
(`.measure`, `.filter`, `.bin`, `.sample`, ...) exposed as a single configurable unit,
plus the operator method each recipe compiles to (see the module ``register()``).

Add your own by writing a class with `.apply(pipeline) -> pipeline` and registering it
with :func:`kuration.selection.plan.register_operation_type` for config-dict use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class BalancedSelector:
    """Balance selected windows across one measured feature.

    Windows get measured against a feature, quantile-binned across the observed range,
    then sampled equally from each bin. Useful when you want a training set to cover the
    whole spread of a signal instead of matching its natural distribution.

    Compiles to `.measure(...) -> .bin(...) -> .sample("equal", ...)` on the Kuration
    SDK pipeline. See `describe()` for the exact chain a given instance produces.

    Args:
        feature: column name the measured value gets stored under (e.g. 'activity').
        of: source field to measure (e.g. 'signal').
        kind: metric kind supported by the SDK grammar ('std', 'mean', 'max',
            'entropy', ...).
        bins: quantile bin count across the feature range. Minimum 2.
        count: total windows to sample, split equally across bins. Minimum 1.

    Raises:
        ValueError: if any argument is out of range or empty.
    """

    feature: str
    of: str
    kind: str = "std"
    bins: int = 8
    count: int = 100

    EXTRAS: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        if not self.feature:
            raise ValueError(
                "BalancedSelector: 'feature' must be a non-empty string "
                "(the column name to store the measured value under)"
            )
        if not self.of:
            raise ValueError(
                "BalancedSelector: 'of' must be a non-empty string "
                "(the source field to measure)"
            )
        if self.bins < 2:
            raise ValueError(
                f"BalancedSelector: 'bins' must be >= 2 (got {self.bins}). "
                "Bin coverage needs at least two buckets."
            )
        if self.count < 1:
            raise ValueError(
                f"BalancedSelector: 'count' must be >= 1 (got {self.count})."
            )

    def apply(self, pipeline):
        return (
            pipeline
            .measure(self.feature, self.kind, of=self.of)
            .bin(self.feature, bins=self.bins)
            .sample("equal", n=self.count)
        )

    def describe(self) -> str:
        """One-line description of the SDK chain this operation compiles to."""
        return (
            f"BalancedSelector: "
            f".measure('{self.feature}', '{self.kind}', of='{self.of}') "
            f"-> .bin('{self.feature}', bins={self.bins}) "
            f"-> .sample('equal', n={self.count})"
        )


@dataclass(frozen=True)
class RareEventCoverage:
    """Weighted sample toward windows containing a rare event.

    Measures the per-window rate of a target event field, then weighted-samples so
    windows containing the rare event get admitted proportional to how much of the
    window they occupy. Useful when a rare failure lives inside a much larger stream
    of healthy telemetry and ordinary distribution matching would drop it.

    Compiles to `.measure(..., "rate", of=...) -> .sample("weighted", weights={...})`
    on the Kuration SDK pipeline. See `describe()` for the exact chain a given
    instance produces.

    `.sample("weighted", ...)` has no bounded/streaming execution path yet (unlike
    `top-k`/`uniform-bins`): it materializes the whole scored candidate pool in memory before
    sampling, not just the `count` selected. Fine for a candidate pool that fits in memory —
    the common case this SDK targets — but there's currently no `memory_budget=`/`streaming=True`
    equivalent for it. For a run that genuinely needs a bounded weighted path, there isn't one
    yet; `BalancedSelector` (`.sample("equal", ...)`, bounded) is the nearest sibling recipe if
    rarity-weighting specifically isn't required, or narrow the candidate pool with an earlier
    `.filter()` first.

    Args:
        of: source field whose rate to measure. Expected to be a boolean or
            0/1 column (e.g. 'fault' or 'alarm').
        feature: column name to store the measured rate under. Defaults to
            'event_rate'.
        count: total windows to sample. Minimum 1.
        weight: importance of the rate feature in the sampling weight.
            Defaults to 1.0; increase for tighter focus on rare-event windows.

    Raises:
        ValueError: if any argument is out of range or empty.
    """

    of: str
    feature: str = "event_rate"
    count: int = 100
    weight: float = 1.0

    EXTRAS: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        if not self.of:
            raise ValueError(
                "RareEventCoverage: 'of' must be a non-empty string "
                "(the source field whose rate to measure)"
            )
        if not self.feature:
            raise ValueError(
                "RareEventCoverage: 'feature' must be a non-empty string"
            )
        if self.count < 1:
            raise ValueError(
                f"RareEventCoverage: 'count' must be >= 1 (got {self.count})."
            )
        if self.weight <= 0:
            raise ValueError(
                f"RareEventCoverage: 'weight' must be > 0 (got {self.weight})."
            )

    def apply(self, pipeline):
        return (
            pipeline
            .measure(self.feature, "rate", of=self.of)
            .sample("weighted", n=self.count, weights={self.feature: self.weight})
        )

    def describe(self) -> str:
        """One-line description of the SDK chain this operation compiles to."""
        return (
            f"RareEventCoverage: "
            f".measure('{self.feature}', 'rate', of='{self.of}') "
            f"-> .sample('weighted', n={self.count}, "
            f"weights={{'{self.feature}': {self.weight}}})"
        )


@dataclass(frozen=True)
class DiverseSubset:
    """Cluster-balanced selection over a set of pre-measured features.

    Given a set of pre-measured features on the pipeline, DiverseSubset optionally
    drops near-duplicates in the joint feature space, then samples equally from
    k-means clusters over those features. Useful when a dataset has heavy
    near-duplication and you want to compress to a diverse subset without losing
    coverage of the feature space.

    Compiles to `[.filter("dedup", ...)] -> .sample("cluster-balanced", ...)` on the
    Kuration SDK pipeline. See `describe()` for the exact chain a given instance
    produces.

    Requires the `[methods]` extra (scikit-learn). The user is expected to have
    measured `features` on the pipeline via `.measure(...)` before applying this
    operation.

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
        # tolerate list/sequence at construction (e.g. a JSON config dict's "features": [...]),
        # store as tuple for hashability; mypy sees only the declared tuple[str, ...] type
        if not isinstance(self.features, tuple):
            object.__setattr__(self, "features", tuple(self.features))  # type: ignore[unreachable]
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


__all__ = ["BalancedSelector", "DiverseSubset", "RareEventCoverage"]
