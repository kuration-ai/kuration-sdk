"""Rare-event coverage: oversample rare events buried in ordinary footage.

Weighted-sample a windowed corpus toward windows containing a rare event.
Uses `RareEventCoverage` from kuration_sdk_wrapper, which compiles to
`.measure("rate", of=...) -> .sample("weighted", weights={...})` on the
Kuration SDK pipeline.

The blog target for this operator is collision oversampling: player-wall
and player-object contact events are rare in real gameplay, so evenly
sampled footage barely trains on them. The dataset below is a small
synthetic stand-in with the shape a real corpus would have after running
the SDK's `collision` operator (per-tick 0/1 flag over a set of clips);
`.measure(..., "rate", of=...)` is what aggregates it to per-window.

Runnable with:

    python examples/rare-event-coverage/rare_event_coverage.py
"""

from kuration_sdk_wrapper import input, RareEventCoverage

# 3 clips, 200 rows each. `collision=1` occurs sparsely (roughly 1 in 30
# rows, different offset per clip) so ordinary distribution matching
# would leave most collision windows out of the sample.
sessions = {
    f"clip-{i}": [
        {"t": float(step),
         "signal": float(i * step % 7),
         "collision": 1 if step % 30 == i else 0}
        for step in range(200)
    ]
    for i in range(3)
}

pipeline = input(sessions, time="t", hz=10.0).window(8, stride=4)

op = RareEventCoverage(of="collision", count=15, weight=2.0)
result = op.apply(pipeline).options(mode="strict", seed=1).collect()

print(f"selected: {len(result.selected)} windows")
print(f"features on first window: {sorted(set(result.selected[0].features) - {'_index'})}")
print()
print("compiled chain:")
print(f"  {op.describe()}")
