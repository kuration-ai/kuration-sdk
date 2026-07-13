"""Robotics trajectories: compose a learned, action-aware score with geometric coverage.

This is the robotics-credibility example. The schema declares angular joints and a positional
end-effector (so `speed` / `angle_delta` are type-safe), and the strategy combines:
  - a geometric coverage feature (end-effector speed), and
  - a learned, whole-pool, action-aware score (`action_residual`: obs->action predictive
    residual) — the class the closed grammar can't express — weighted together,
plus coverage of every motion phase. Quality and diversity as composable axes, not either/or.

    python examples/02_robotics_action_aware.py     # needs the [methods] extra (scikit-learn)
"""
import math
import random
import tempfile

from kuration import SchemaSpec, Segmenter, Tick, load_strategy, OutputSpec, write_dataset

if __import__("importlib").util.find_spec("sklearn") is None:
    raise SystemExit("this example needs scikit-learn: pip install -e '.[methods]'")

rng = random.Random(0)

schema = SchemaSpec.from_dict({
    "name": "arm", "entity_axis": "arm", "sample_hz": 50.0,
    "fields": {
        "joint":  {"kind": "vector", "dims": ["j1", "j2", "j3"], "semantics": "angular"},
        "ee":     {"kind": "vector", "dims": ["x", "y", "z"], "semantics": "position", "max_rate": 5},
        "action": {"kind": "vector", "dims": ["da", "db", "dc"]},
        "phase":  {"kind": "categorical"},
    },
})


def trajectory(kind, n=300):
    """A synthetic arm trajectory; 'precise' episodes have small, obs-correlated actions,
    'erratic' ones have larger, less predictable actions (higher action-residual)."""
    ticks, j = [], [0.0, 0.0, 0.0]
    for t in range(n):
        phase = "reach" if t < n // 3 else "grasp" if t < 2 * n // 3 else "retract"
        scale = 0.02 if kind == "precise" else 0.15
        act = [scale * math.sin(t / 7 + i) + (rng.random() - 0.5) * scale for i in range(3)]
        j = [j[i] + act[i] for i in range(3)]
        ee = (math.sin(j[0]), math.cos(j[1]), 0.1 * j[2])
        ticks.append(Tick(time=t, entities={0: {"joint": tuple(j), "ee": ee,
                                                 "action": tuple(act), "phase": phase}}))
    return ticks


class MemAdapter:
    schema = schema
    def __init__(self, data): self.data = data
    def ticks(self, src): return self.data[src]


data = {f"ep{i}": trajectory("precise" if i % 2 else "erratic") for i in range(12)}

# Compose: weight a learned action-aware score together with a geometric coverage feature.
strategy = load_strategy({
    "name": "quality+coverage",
    "define": {"ee_speed": {"source": "arm.ee", "map": "speed", "reduce": "mean"}},
    "scorers": [{"name": "action_residual", "as": "informativeness",
                 "obs": "ee", "action": "action", "k": 8}],
    "tags": {"from": ["phase"]},
    "selection": {"method": "weighted",
                  "weights": {"informativeness": 1.0, "ee_speed": 0.3},
                  "tag_coverage": {"min_tag_frac": 0.05}},
}, schema)

spec = OutputSpec.from_dict({
    "shard_length": 64, "container": "npz",
    "modalities": {"joint":  {"dtype": "float32", "shape": [3], "source": "field:joint"},
                   "action": {"dtype": "float32", "shape": [3], "source": "field:action"}},
})

out = tempfile.mkdtemp(prefix="kuration_ex02_")
res = write_dataset(out, schema=schema, adapter=MemAdapter(data), sources=list(data),
                    strategy=strategy, segmenter=Segmenter(1.0, 0.5, unit="seconds", sample_hz=50),
                    spec=spec, train_hours=0.05, test_hours=0.0, seed=0)
print(f"wrote {res['results']['shards']} shards from {res['results']['tracks_admitted']} "
      f"admitted spans")
print("  (learned action-residual score composed with geometric ee-speed + phase coverage)")
print(f"output dir: {out}")
