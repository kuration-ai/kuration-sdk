"""Reusable compositional and transform-backed operators, using only the NumPy core.

Run from the repository root:

    python examples/05_custom_operators.py
"""
from __future__ import annotations

import numpy as np

from kuration import (describe_component, input, register_batch_transform,
                       register_operator)


PARAMETERS = {
    "type": "object",
    "properties": {
        "field": {"type": "string"},
        "bins": {"type": "integer", "minimum": 2},
    },
    "required": ["field"],
    "additionalProperties": False,
}


@register_operator(
    "example/signal-diversity",
    description="Balance windows by signal variability",
    parameters=PARAMETERS,
    version="1",
)
def signal_diversity(pipeline, *, field, bins=4):
    return (pipeline
            .measure("signal_variability", "std", of=field)
            .bin("signal_variability", bins=bins))


@register_batch_transform(
    "example/window-range",
    outputs=("window_range",),
    inputs=("field:signal",),
    resources={"preferred_batch_size": 8},
)
class WindowRange:
    def transform_batch(self, batch, ctx):
        values = np.asarray(batch["field:signal"], dtype=np.float64)
        axes = tuple(range(1, values.ndim))
        return {"window_range": np.ptp(values, axis=axes)}


@register_operator(
    "example/range-diversity",
    description="Compute a custom window range and balance by it",
    parameters=PARAMETERS,
    version="1",
)
def range_diversity(pipeline, *, field, bins=4):
    return (pipeline
            .transform("example/window-range", inputs=[f"field:{field}"])
            .bin("window_range", bins=bins))


sessions = {
    f"session-{source}": [
        {"t": float(step), "signal": float(source * step + (step % (source + 2)))}
        for step in range(24)
    ]
    for source in range(1, 4)
}

base = input(sessions, time="t", hz=10).window(8, stride=4)

variability = (base
               .apply("example/signal-diversity", field="signal", bins=3)
               .sample("equal", n=5)
               .options(mode="strict", seed=4)
               .collect())
assert len(variability.selected) == 5
assert all("signal_variability" in segment.features for segment in variability.selected)

window_range = (base
                .apply("example/range-diversity", field="signal", bins=3)
                .sample("equal", n=5)
                .options(mode="strict", seed=4)
                .collect())
assert len(window_range.selected) == 5
assert all(segment.features["window_range"] >= 0 for segment in window_range.selected)

descriptor = describe_component("operator", "example/range-diversity")
assert descriptor["version"] == "1"
assert descriptor["parameters"] == PARAMETERS

print("compositional operator selected:", len(variability.selected), "windows")
print("transform-backed operator selected:", len(window_range.selected), "windows")
print("discoverable component:", descriptor["kind"], descriptor["name"],
      "version", descriptor["version"])
print("all assertions passed")
