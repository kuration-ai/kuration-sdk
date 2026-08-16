"""BalancedSelector — cover the whole range of a signal, not just its middle.

Windows get measured against a feature, quantile-binned across the observed range,
then sampled equally from each bin. Useful when you want a training set to cover the
whole spread of a signal instead of matching its natural distribution.

Compiles to `.measure(...) -> .bin(...) -> .sample("equal", ...)` on the Kuration
SDK pipeline. See `describe()` for the exact chain a given instance produces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class BalancedSelector:
    """Balance selected windows across one measured feature.

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
