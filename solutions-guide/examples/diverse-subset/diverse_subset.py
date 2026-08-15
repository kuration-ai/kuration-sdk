"""Diverse subset — compress a redundant dataset by clustering in feature space.

Cluster-balanced selection over pre-measured features, with optional dedup for
near-duplicate windows. Uses `DiverseSubset` from kuration_ai, which compiles
to `.filter("dedup", ...) -> .sample("cluster-balanced", ...)` on the Kuration
SDK pipeline.



Runnable with:

    python examples/diverse-subset/diverse_subset.py
"""

from kuration_ai import input, DiverseSubset

# 4 sessions, 80 rows each. `signal` and `temp` are correlated within-session
# and near-duplicate across sessions, so ordinary sampling would return
# redundant windows.
sessions = {
    f"session-{i}": [
        {"t": float(step),
         "signal": float(i * step % 7),
         "temp": float(20 + (i + step) % 10)}
        for step in range(80)
    ]
    for i in range(4)
}

# Build the windowed pipeline + measure the features to cluster on
pipeline = (
    input(sessions, time="t", hz=10.0)
    .window(8, stride=4)
    .measure("signal_std", "std", of="signal")
    .measure("signal_mean", "mean", of="signal")
    .measure("temp_std", "std", of="temp")
)

# Cluster-balanced selection with a light dedup pre-pass
op = DiverseSubset(
    features=["signal_std", "signal_mean", "temp_std"],
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
