"""Compare curation strategies over the same data (which one shapes the set how).

Scores once over the union of all arms' metrics, selects per arm, and reports how each arm's
selected distribution diverges from a random baseline (Jensen-Shannon, per metric).

    python examples/03_compare_strategies.py
"""
import math

from kuration import SchemaSpec, Segmenter, Tick, Arm, load_strategy, run_experiment

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
seg = Segmenter(8, 8, unit="seconds", sample_hz=1)

define = {"speed": {"source": "e.pos", "map": "speed", "reduce": "mean"},
          "amp":   {"source": "e.amp", "reduce": "mean"}}
arms = [
    Arm("baseline", load_strategy({"name": "b", "define": define,
        "binning": {"axes": [{"metric": "speed", "bins": 6}]},
        "selection": {"method": "uniform-bins"}}, schema)),
    Arm("fast", load_strategy({"name": "f", "define": define,
        "selection": {"method": "top-k", "weights": {"speed": 1.0}}}, schema)),
    Arm("high-amp", load_strategy({"name": "h", "define": define,
        "selection": {"method": "top-k", "weights": {"amp": 1.0}}}, schema)),
]

out = run_experiment(schema=schema, adapter=MemAdapter(data), sources=list(data), arms=arms,
                     segmenter=seg, target_size=20, baseline="baseline", seed=0)

print(f"baseline = {out['baseline']}  ({out['n_total_segments']} segments scored once)\n")
for name, a in out["arms"].items():
    tag = "(baseline)" if name == out["baseline"] else ""
    print(f"  {name:<10} n={a['n_selected']:3d}  "
          f"speed.mean={a['metrics']['speed']['mean']:.3f}  amp.mean={a['metrics']['amp']['mean']:.3f}  {tag}")
    for m in ("speed", "amp"):
        js = a["metrics"][m].get("js_vs_baseline")
        if js is not None:
            print(f"             {m} JS-vs-baseline = {js:.3f}")
