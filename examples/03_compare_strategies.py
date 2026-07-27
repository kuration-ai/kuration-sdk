"""Compare curation strategies over the same data — which one shapes the set how.

Three arms disagree about what a good training set is. The experiment harness scores every
window **once**, selects per arm from that shared pool, and reports how far each arm's
selected distribution moved from a baseline (Jensen-Shannon divergence, per metric).

    python examples/03_compare_strategies.py
"""
import math

from kuration import SchemaSpec, Tick, input

# `semantics: position` is what lets `map="speed"` turn an (x, y) field into a speed —
# inference cannot guess that a vector is a position, so this schema is written out.
schema = SchemaSpec.from_dict({
    "name": "sig", "entity_axis": "e", "sample_hz": 1.0,
    "fields": {"pos": {"kind": "vector", "dims": ["x", "y"], "semantics": "position"},
               "amp": {"kind": "scalar"}},
})


def source(level: int, n: int = 160) -> list[Tick]:
    """One recording whose motion and amplitude both scale with `level`."""
    return [Tick(time=t, entities={0: {"pos": (level * math.sin(t / 8),
                                               level * math.cos(t / 11)),
                                       "amp": level + 0.3 * math.sin(t / 5)}})
            for t in range(n)]


# A dict of {source_id: ticks} is valid input on its own — no adapter class needed.
recordings = {f"s{level}": source(level) for level in range(1, 11)}

base = (
    input(recordings, schema=schema)
    .window(8, stride=8, unit="seconds")
    .measure("speed", "mean", of="pos", map="speed")
    .measure("amp", "mean", of="amp")
    .options(mode="strict", seed=0)
)

# Each arm is the same pipeline with a different ending. `base` is immutable, so branching
# off it three times is safe.
arms = {
    "baseline": base.bin("speed", bins=6).sample("equal", n=20),   # spread across speeds
    "fast": base.sample("top-k", n=20, weights={"speed": 1.0}),    # fastest windows only
    "high-amp": base.sample("top-k", n=20, weights={"amp": 1.0}),  # largest amplitude only
}
comparison = base.experiment(arms, n=20, baseline="baseline")

print(f"scored {comparison['n_total_segments']} windows once, "
      f"then selected per arm (baseline: {comparison['baseline']})\n")
for name, arm in comparison["arms"].items():
    metrics = arm["metrics"]
    print(f"{name:<10} n={arm['n_selected']:3d}  "
          f"speed={metrics['speed']['mean']:6.3f}  amp={metrics['amp']['mean']:6.3f}")
    for metric in ("speed", "amp"):
        divergence = metrics[metric].get("js_vs_baseline")
        if divergence is not None:
            print(f"{'':<10} {metric:>5} diverges from baseline by {divergence:.3f} JS")

print("\nA top-k arm concentrates on its own metric; the binned baseline stays spread out.")
