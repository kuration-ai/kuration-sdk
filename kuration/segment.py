"""Cut a canonical `Tick` stream into fixed-width overlapping windows.

Windows are expressed in steps, or in seconds when the schema declares `sample_hz` (the SDK
converts). `Segment` carries generic per-window state — `features` / `bin` / `tags` — keyed
by name, so adding a metric never touches this dataclass (the malleability the registry buys).

`segments()` is the in-memory path (random-access over a materialized list); `stream()` is the
out-of-core path — a sliding `deque` over a lazy tick iterator that never holds more than one
window, so a single source larger than RAM is fine. Both emit identical windows.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Iterator, Optional

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
            if (not isinstance(sample_hz, (int, float)) or isinstance(sample_hz, bool)
                    or not math.isfinite(sample_hz) or sample_hz <= 0):
                raise ValueError("unit='seconds' needs the schema's sample_hz")
            for name, value in (("window", window), ("stride", stride)):
                if (not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(value) or value <= 0):
                    raise ValueError(f"{name} must be a finite positive number")
            window = round(window * sample_hz)
            stride = round(stride * sample_hz)
        elif (not isinstance(window, int) or isinstance(window, bool)
              or not isinstance(stride, int) or isinstance(stride, bool)):
            raise ValueError("step-unit window/stride must be positive integers")
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

    def stream(self, source_id: str, ticks: Iterable[Tick]) -> Iterator[tuple[Segment, list[Tick]]]:
        """Yield (Segment, window_ticks) from a lazy tick iterator, holding at most one window
        (a `deque` of `window` ticks). Emits exactly the windows `segments()` would — a window
        [start, start+W) once tick start+W-1 has arrived, for start on the stride grid."""
        buf: deque = deque(maxlen=self.window)
        idx = 0
        for tk in ticks:
            buf.append(tk)
            idx += 1
            if len(buf) == self.window and (idx - self.window) % self.stride == 0:
                start = idx - self.window
                win = list(buf)
                yield (Segment(source_id=source_id, start_idx=start, end_idx=idx,
                               time_range=(win[0].time, win[-1].time)), win)
