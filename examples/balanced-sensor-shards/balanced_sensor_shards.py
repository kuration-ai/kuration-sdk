"""Balanced sensor shards: cover the whole range of a signal.

Turn synthetic device-log data into windows selected across the full range of
signal variability, not just the natural distribution peak. Uses
`BalancedSelector` from kuration_ai, which compiles to
`.measure() -> .bin() -> .sample("equal", ...)`.

Runnable with:

    python examples/balanced-sensor-shards/balanced_sensor_shards.py
"""

from kuration_ai import input, BalancedSelector

# Synthetic sensor logs: 4 devices, 100 rows each. `signal` cycles at different
# rates per device so activity (std of the signal per window) varies across
# the corpus.
sessions = {
    f"device-{i}": [
        {"t": float(step),
         "signal": float(i * step % 7),
         "temp": float(20 + step % 10)}
        for step in range(100)
    ]
    for i in range(4)
}

# Build the windowed pipeline
pipeline = input(sessions, time="t", hz=10.0).window(8, stride=4)

# One operation, one line
op = BalancedSelector(feature="activity", of="signal", kind="std", bins=4, count=20)
result = op.apply(pipeline).options(mode="strict", seed=1).collect()

print(f"selected: {len(result.selected)} windows")
print(f"features on first window: {sorted(set(result.selected[0].features) - {'_index'})}")
print()
print("compiled chain:")
print(f"  {op.describe()}")
