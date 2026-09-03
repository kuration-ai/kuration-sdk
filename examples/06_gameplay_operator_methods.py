"""Domain operators arrive as ordinary Pipeline methods when you import the domain.

`kuration.gameplay` registers its components on import, so SLAM trajectory operators become
methods on the plain core `Pipeline` — no subclass, no separate builder. This runs on
synthetic precomputed camera trajectories, so it needs only the NumPy core: no video decoder,
model download, OpenCV, or GPU.

    python examples/06_gameplay_operator_methods.py
"""
from __future__ import annotations

import math

import kuration.gameplay as gameplay  # importing the domain is what registers the operators
from kuration import Pipeline, SchemaSpec, Tick, describe_component, input

SCHEMA = SchemaSpec.from_dict({
    "name": "synthetic-camera-trajectories",
    "entity_axis": "camera",
    "sample_hz": 10.0,
    "fields": {
        "camera_position": {"kind": "vector", "dims": ["x", "y", "z"],
                            "semantics": "position"},
        "camera_z_axis": {"kind": "vector", "dims": ["x", "y", "z"]},
        "tracking_confidence": {"kind": "scalar"},
    },
})


def trajectory(*, speed: float, turn_rate: float, confidence: float) -> list[Tick]:
    """One camera path. Steps 14-15 dip in confidence, so quality filtering has something
    to reject."""
    ticks = []
    for step in range(32):
        angle = turn_rate * step
        ticks.append(Tick(time=step / 10.0, entities={0: {
            "camera_position": (speed * step, math.sin(angle), 0.05 * math.cos(angle)),
            "camera_z_axis": (math.cos(angle), math.sin(angle), 0.0),
            "tracking_confidence": confidence if step not in {14, 15} else confidence / 2,
        }}))
    return ticks


sources = {
    "slow-pan": trajectory(speed=0.03, turn_rate=0.03, confidence=0.98),
    "fast-turn": trajectory(speed=0.12, turn_rate=0.12, confidence=0.92),
    "wide-arc": trajectory(speed=0.07, turn_rate=0.07, confidence=0.95),
}
base = input(sources, schema=SCHEMA).window(8, stride=4)

# `.slam_motion_diversity` and `.slam_quality` are not core methods — importing the domain
# added them to this ordinary Pipeline.
pipeline = (base
            .slam_motion_diversity(axis="slam_turn_degrees", bins=4)
            .slam_quality(min_score=0.5)
            .sample("equal", n=6))

# The namespace-qualified spelling is always available, for when ownership should be explicit
# or two packages register the same short name.
qualified = (base
             .aintropy__slam_motion_diversity(axis="slam_turn_degrees", bins=4)
             .aintropy__slam_quality(min_score=0.5)
             .sample("equal", n=6))
assert pipeline.describe()["strategy"] == qualified.describe()["strategy"]
assert type(pipeline) is Pipeline, "still the core Pipeline, not a domain subclass"

result = pipeline.collect()
features = sorted(set(result.selected[0].features) - {"_index"})

print(f"imported {gameplay.__name__} -> operators became {type(pipeline).__name__} methods")
for name in ("aintropy/slam-motion-diversity", "aintropy/slam-quality"):
    descriptor = describe_component("operator", name)
    print(f"  {descriptor['name']:<34} {descriptor.get('description', '')}")
print(f"selected {len(result.selected)} windows")
print(f"the two operators contributed {len(features)} SLAM features, including: "
      f"{', '.join(features[:4])}, ...")
