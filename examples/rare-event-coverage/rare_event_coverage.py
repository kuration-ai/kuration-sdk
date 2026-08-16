"""Rare-event coverage — don't drop rare failures when sampling.

Weighted-sample a telemetry stream toward windows that contain rare fault
events. Uses `RareEventCoverage` from kuration_ai, which compiles to
`.measure("rate", of=...) -> .sample("weighted", weights={...})` on the
Kuration SDK pipeline.

Runnable with:

    python examples/rare-event-coverage/rare_event_coverage.py
"""

from kuration_ai import input, RareEventCoverage

# 3 devices, 200 rows each. `fault=1` occurs sparsely (roughly 1 in every 30
# rows, different offset per device) so ordinary distribution matching would
# drop most of the fault windows.
sessions = {
    f"device-{i}": [
        {"t": float(step),
         "signal": float(i * step % 7),
         "fault": 1 if step % 30 == i else 0}
        for step in range(200)
    ]
    for i in range(3)
}

# Build the windowed pipeline
pipeline = input(sessions, time="t", hz=10.0).window(8, stride=4)

# One operation, one line
op = RareEventCoverage(of="fault", count=15, weight=2.0)
result = op.apply(pipeline).options(mode="strict", seed=1).collect()

print(f"selected: {len(result.selected)} windows")
print(f"features on first window: {sorted(set(result.selected[0].features) - {'_index'})}")
print()
print("compiled chain:")
print(f"  {op.describe()}")
