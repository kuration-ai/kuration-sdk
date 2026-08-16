"""RareEventCoverage — sample so rare events don't get dropped.

Measures the per-window rate of a target event field, then weighted-samples so
windows containing the rare event get admitted proportional to how much of the
window they occupy. Useful when a rare failure lives inside a much larger stream
of healthy telemetry and ordinary distribution matching would drop it.

Compiles to `.measure(..., "rate", of=...) -> .sample("weighted", weights={...})`
on the Kuration SDK pipeline. See `describe()` for the exact chain a given
instance produces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class RareEventCoverage:
    """Weighted sample toward windows containing a rare event.

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
