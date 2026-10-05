"""Package your own curation step as a reusable operator.

A registered operator becomes an ordinary method on every `Pipeline`, so a recipe your team
repeats stops being copied around. Two ways to build one:

  - **compositional** — a function that returns more pipeline. Nothing new is computed; it
    names a combination of existing operators.
  - **transform-backed** — a batched callable that computes a genuinely new feature, then an
    operator that bins it.

Both are discoverable through the registry, and both need only the NumPy core.

    python examples/05_custom_operators.py
"""
from __future__ import annotations

import numpy as np

from kuration import (describe_component, input, register_batch_transform,
                      register_operator)

# Parameters are declared as JSON Schema so the registry can validate calls and `kur describe`
# can print the contract without importing anything.
PARAMETERS = {
    "type": "object",
    "properties": {"field": {"type": "string"},
                   "bins": {"type": "integer", "minimum": 2}},
    "required": ["field"],
    "additionalProperties": False,
}


@register_operator("example/signal-diversity", parameters=PARAMETERS, version="1",
                   description="Balance windows by signal variability")
def signal_diversity(pipeline, *, field, bins=4):
    """Compositional: just a measure plus a bin, under one reusable name."""
    return (pipeline
            .measure("signal_variability", "std", of=field)
            .bin("signal_variability", bins=bins))


@register_batch_transform("example/window-range", outputs=("window_range",),
                          inputs=("field:signal",),
                          resources={"preferred_batch_size": 8})
class WindowRange:
    """Transform-backed: computes peak-to-peak per window, in bounded batches."""

    def transform_batch(self, batch, ctx):
        values = np.asarray(batch["field:signal"], dtype=np.float64)
        axes = tuple(range(1, values.ndim))
        return {"window_range": np.ptp(values, axis=axes)}


@register_operator("example/range-diversity", parameters=PARAMETERS, version="1",
                   description="Compute a custom window range and balance by it")
def range_diversity(pipeline, *, field, bins=4):
    return (pipeline
            .transform("example/window-range", inputs=[f"field:{field}"])
            .bin("window_range", bins=bins))


sessions = {
    f"session-{source}": [{"t": float(step), "signal": float(source * step +
                                                             (step % (source + 2)))}
                          for step in range(24)]
    for source in range(1, 4)
}
base = input(sessions, time="t", hz=1).window(8, stride=4)

# Registering generated two spellings: a short method, and a namespace-qualified one for when
# two packages would otherwise collide. They compile to the same strategy.
assert (base.signal_diversity(field="signal", bins=3).describe()["strategy"]
        == base.example__signal_diversity(field="signal", bins=3).describe()["strategy"])

for label, pipeline in (
    ("compositional", base.signal_diversity(field="signal", bins=3)),
    ("transform-backed", base.range_diversity(field="signal", bins=3)),
):
    result = (pipeline.sample("equal", n=5)
              .options(mode="strict", seed=4)
              .collect())
    feature = sorted(set(result.selected[0].features) - {"_index"})
    print(f"{label:16} selected {len(result.selected)} windows | features {feature}")

descriptor = describe_component("operator", "example/range-diversity")
print(f"\nregistry: {descriptor['kind']} {descriptor['name']} v{descriptor['version']} "
      f"— {descriptor['description']}")
print("both spellings available: .range_diversity(...) and .example__range_diversity(...)")
