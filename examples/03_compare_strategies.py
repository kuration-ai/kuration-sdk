"""Compare curation strategies over the same data (which one shapes the set how).

Scores once over the union of all arms' metrics, selects per arm, and reports how each arm's
selected distribution diverges from a random baseline (Jensen-Shannon, per metric).

    python examples/03_compare_strategies.py
"""
import math

from kuration import SchemaSpec, Tick, input

schema = SchemaSpec.from_dict({
    "name": "sig", "entity_axis": "e", "sample_hz": 1.0,
    "fields": {"pos": {"kind": "vector", "dims": ["x", "y"], "semantics": "position"},
               "amp": {"kind": "scalar"}},
})


def source(level, n=160):
    ticks = []
    for t in range(n):
        ticks.append(Tick(time=t, entities={0: {
            "pos": (level * math.sin(t / 8), level * math.cos(t / 11)),
            "amp": level + 0.3 * math.sin(t / 5)}}))
    return ticks


class MemAdapter:
    schema = schema
    def __init__(self, d): self.data = d
    def ticks(self, src): return self.data[src]


data = {f"s{L}": source(L) for L in range(1, 11)}   # activity levels 1..10
base = (
    input(adapter=MemAdapter(data), sources=list(data), schema=schema)
    .window(8, stride=8, unit="seconds")
    .measure("speed", "mean", of="pos", map="speed")
    .measure("amp", "mean", of="amp")
    .options(mode="strict", seed=0)
)
arms = {
    "baseline": base.bin("speed", bins=6).sample("equal", n=20),
    "fast": base.sample("top-k", n=20, weights={"speed": 1.0}),
    "high-amp": base.sample("top-k", n=20, weights={"amp": 1.0}),
}
out = base.experiment(arms, n=20, baseline="baseline")

print(f"baseline = {out['baseline']}  ({out['n_total_segments']} segments scored once)\n")
for name, a in out["arms"].items():
    tag = "(baseline)" if name == out["baseline"] else ""
    print(f"  {name:<10} n={a['n_selected']:3d}  "
          f"speed.mean={a['metrics']['speed']['mean']:.3f}  amp.mean={a['metrics']['amp']['mean']:.3f}  {tag}")
    for m in ("speed", "amp"):
        js = a["metrics"][m].get("js_vs_baseline")
        if js is not None:
            print(f"             {m} JS-vs-baseline = {js:.3f}")
