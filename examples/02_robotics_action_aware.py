"""Robotics: compose a learned action-aware score with geometric coverage.

Quality and diversity are separate axes, not a choice between them. This weights two
different kinds of signal in one selection:

  - `ee_speed` — geometry, computed per window by the closed metric grammar.
  - `informativeness` — a learned, whole-pool score (`action_residual`: how poorly an
    observation predicts the action taken) reached through the scorer seam, because the
    grammar deliberately cannot express it.

...plus coverage of every motion phase, so no phase is dropped for being uncommon.

    python examples/02_robotics_action_aware.py     # needs the [methods] extra
"""
import importlib.util
import math
import random
import tempfile

from kuration import SchemaSpec, Tick, input

if importlib.util.find_spec("sklearn") is None:
    raise SystemExit("this example needs scikit-learn: pip install \"kuration-sdk[methods]\"")

rng = random.Random(0)

# Semantics are the point of writing this schema out: `angular` joints wrap at 360 degrees
# and `position` supports the speed map. Inference sees only "a vector of three floats".
schema = SchemaSpec.from_dict({
    "name": "arm", "entity_axis": "arm", "sample_hz": 50.0,
    "fields": {
        "joint": {"kind": "vector", "dims": ["j1", "j2", "j3"], "semantics": "angular"},
        "ee": {"kind": "vector", "dims": ["x", "y", "z"], "semantics": "position",
               "max_rate": 5},
        "action": {"kind": "vector", "dims": ["da", "db", "dc"]},
        "phase": {"kind": "categorical"},
    },
})


def trajectory(kind: str, n: int = 300) -> list[Tick]:
    """One episode. 'precise' actions track the observation closely, so they are easy to
    predict; 'erratic' ones are noisier and score a higher action residual."""
    ticks, joints = [], [0.0, 0.0, 0.0]
    for t in range(n):
        phase = "reach" if t < n // 3 else "grasp" if t < 2 * n // 3 else "retract"
        scale = 0.02 if kind == "precise" else 0.15
        action = [scale * math.sin(t / 7 + i) + (rng.random() - 0.5) * scale
                  for i in range(3)]
        joints = [joints[i] + action[i] for i in range(3)]
        end_effector = (math.sin(joints[0]), math.cos(joints[1]), 0.1 * joints[2])
        ticks.append(Tick(time=t, entities={0: {
            "joint": tuple(joints), "ee": end_effector,
            "action": tuple(action), "phase": phase}}))
    return ticks


# Half the episodes are precise, half erratic; a dict of ticks is input on its own.
episodes = {f"ep{i}": trajectory("precise" if i % 2 else "erratic") for i in range(12)}

out_dir = tempfile.mkdtemp(prefix="kuration_ex02_")
result = (
    input(episodes, schema=schema)
    .window(1.0, stride=0.5, unit="seconds")
    .measure("ee_speed", "mean", of="ee", map="speed")
    # The scorer runs over the whole pool once and publishes a feature like any other.
    .score("action_residual", out="informativeness", obs="ee", action="action", k=8)
    .tag("phase", min_frac=0.05)
    # Both features are weighted in one selection: informative windows preferred, with
    # motion variety still pulling against picking only the noisiest ones.
    .sample("weighted", hours=0.05, weights={"informativeness": 1.0, "ee_speed": 0.3})
    .options(mode="strict", seed=0)
    .output(out_dir, modalities={
        "joint": {"dtype": "float32", "shape": [3], "source": "field:joint"},
        "action": {"dtype": "float32", "shape": [3], "source": "field:action"},
    }, shard_length=64)
)

summary = result["results"]
print(f"wrote {summary['shards']} shards from {summary['tracks_admitted']} admitted spans")
print("selection weighted a learned action-residual score against geometric ee-speed,")
print("with every motion phase guaranteed a share.")
print(f"output dir: {out_dir}")
