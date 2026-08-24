"""Diversity sampling: cluster in feature space, sample equally.

Cluster-balanced selection over pre-measured features, with optional
dedup for near-duplicate windows. Uses `DiverseSubset` from kuration_ai,
which compiles to `.filter("dedup", ...) -> .sample("cluster-balanced", ...)`
on the Kuration SDK pipeline.

The blog target is multi-axis motion diversity: the Kuration Introduction
blog describes `slam_motion_diversity` as "binning windows by how the
camera moved, using path length, turning, and climb." That's three axes;
DiverseSubset generalizes to any feature space and samples equally from
k-means clusters over it.

Runnable with:

    python examples/diversity-sampling/diversity_sampling.py
"""

from kuration_ai import input, DiverseSubset

# 4 clips, 80 rows each. `turn_rate` and `climb_rate` are correlated
# within-clip and near-duplicate across clips, so ordinary sampling
# would return redundant windows.
sessions = {
    f"clip-{i}": [
        {"t": float(step),
         "turn_rate": float(i * step % 7),
         "climb_rate": float((i + step) % 10)}
        for step in range(80)
    ]
    for i in range(4)
}

pipeline = (
    input(sessions, time="t", hz=10.0)
    .window(8, stride=4)
    .measure("motion_std", "std", of="turn_rate")
    .measure("motion_mean", "mean", of="turn_rate")
    .measure("climb_std", "std", of="climb_rate")
)

op = DiverseSubset(
    features=["motion_std", "motion_mean", "climb_std"],
    count=15,
    n_clusters=5,
    dedup_threshold=0.98,
)
result = op.apply(pipeline).options(mode="strict", seed=1).collect()

print(f"selected: {len(result.selected)} windows")
print(f"features on first window: {sorted(set(result.selected[0].features) - {'_index'})}")
print()
print("compiled chain:")
print(f"  {op.describe()}")
