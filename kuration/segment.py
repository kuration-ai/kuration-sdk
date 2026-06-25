"""Cut a canonical `Tick` stream into fixed-width overlapping windows.

Windows are expressed in steps, or in seconds when the schema declares `sample_hz` (the SDK
converts). `Segment` carries generic per-window state — `features` / `bin` / `tags` — keyed
by name, so adding a metric never touches this dataclass (the malleability the registry buys).

In-memory for now (operates on a materialized tick list); the interface takes the window
count up front so a streaming segmenter can later yield windows from a lazy tick iterator
without changing callers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .buffer import Tick


@dataclass
class Segment:
    source_id: str
    start_idx: int               # index into the source's tick list
    end_idx: int                 # exclusive
    time_range: tuple            # (start_time, end_time) from the ticks
    features: dict[str, float] = field(default_factory=dict)
    bin: tuple = ()
    tags: frozenset = frozenset()       # categorical labels for tag-coverage (was CS2 "objects")
    group: Any = None                   # grouping key for the cap (was CS2 "map_name")


class Segmenter:
    def __init__(self, window: float, stride: float, *, unit: str = "steps",
                 sample_hz: Optional[float] = None):
        if unit not in ("steps", "seconds"):
            raise ValueError(f"unit must be 'steps' or 'seconds', got {unit!r}")
        if unit == "seconds":
            if not sample_hz:
                raise ValueError("unit='seconds' needs the schema's sample_hz")
            window = round(window * sample_hz)
            stride = round(stride * sample_hz)
        self.window = int(window)
        self.stride = int(stride)
        if self.window < 1 or self.stride < 1:
            raise ValueError(f"window/stride must be >=1 (got {self.window}/{self.stride})")

    def segments(self, source_id: str, ticks: list[Tick]) -> list[Segment]:
        n = len(ticks)
        out: list[Segment] = []
        for start in range(0, max(1, n - self.window + 1), self.stride):
            end = start + self.window
            if end > n:
                break
            out.append(Segment(source_id=source_id, start_idx=start, end_idx=end,
                               time_range=(ticks[start].time, ticks[end - 1].time)))
        return out
