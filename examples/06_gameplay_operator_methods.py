"""Gameplay operators as direct methods on the ordinary Kuration Pipeline.

This example uses synthetic precomputed camera trajectories, so it needs only the NumPy core:
no video decoder, model download, OpenCV, or GPU.

Run from the repository root:

    python examples/06_gameplay_operator_methods.py
"""
from __future__ import annotations

import math

import kuration.gameplay as gameplay  # importing the domain registers its maintained components
from kuration import Pipeline, SchemaSpec, Tick, describe_component, input


SCHEMA = SchemaSpec.from_dict({
    "name": "synthetic-camera-trajectories",
    "entity_axis": "camera",
    "sample_hz": 10.0,
    "fields": {
        "camera_position": {
            "kind": "vector",
            "dims": ["x", "y", "z"],
            "semantics": "position",
        },
        "camera_z_axis": {"kind": "vector", "dims": ["x", "y", "z"]},
        "tracking_confidence": {"kind": "scalar"},
    },
})


def trajectory(*, speed: float, turn_rate: float, confidence: float) -> list[Tick]:
    ticks = []
    for step in range(32):
        angle = turn_rate * step
        ticks.append(Tick(
            time=step / 10.0,
            entities={0: {
                "camera_position": (
                    speed * step,
                    math.sin(angle),
                    0.05 * math.cos(angle),
                ),
                "camera_z_axis": (math.cos(angle), math.sin(angle), 0.0),
                "tracking_confidence": confidence if step not in {14, 15} else confidence / 2,
            }},
        ))
    return ticks


sources = {
    "slow-pan": trajectory(speed=0.03, turn_rate=0.03, confidence=0.98),
    "fast-turn": trajectory(speed=0.12, turn_rate=0.12, confidence=0.92),
    "wide-arc": trajectory(speed=0.07, turn_rate=0.07, confidence=0.95),
}

base = input(sources, schema=SCHEMA).window(8, stride=4)
assert type(base) is Pipeline

# Preferred syntax: importing the domain made the registered operators ordinary fluent methods.
pipeline = (
    base
    .slam_motion_diversity(axis="slam_turn_degrees", bins=4)
    .slam_quality(min_score=0.5)
    .sample("equal", n=6)
)

# Namespace-qualified sugar is always available when a library wants ownership to be explicit
# or two packages register the same short method name.
qualified = (
    base
    .aintropy__slam_motion_diversity(axis="slam_turn_degrees", bins=4)
    .aintropy__slam_quality(min_score=0.5)
    .sample("equal", n=6)
)
assert pipeline.describe()["strategy"] == qualified.describe()["strategy"]
assert type(pipeline) is Pipeline

result = pipeline.collect()
assert len(result.selected) == 6
assert all("slam_turn_degrees" in segment.features for segment in result.selected)
assert all("slam_quality_score" in segment.features for segment in result.selected)

motion = describe_component("operator", "aintropy/slam-motion-diversity")
quality = describe_component("operator", "aintropy/slam-quality")

print("pipeline type:", type(pipeline).__name__)
print("imported domain:", gameplay.__name__)
print("direct methods:", "slam_motion_diversity()", "slam_quality()")
print("registered operators:", motion["name"], quality["name"])
print("selected windows:", len(result.selected))
print("all assertions passed")
