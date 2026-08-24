"""Motion-diversity coverage: cover a motion axis evenly.

Measure a per-window motion metric (e.g. std of a turn-rate signal), bin
the values evenly across the observed range, then sample the same count
from every bin. Uses `BalancedSelector` from kuration_ai, which compiles
to `.measure(...) -> .bin(...) -> .sample("equal", ...)` on the Kuration
SDK pipeline.

The blog target is motion-diversity binning. The Kuration Introduction
blog describes `slam_motion_diversity` as binning windows by how the
camera moved; the SDK ships motion axes like `slam_path_length`
(default) and `slam_turn_degrees`. The dataset below is a synthetic
stand-in with the shape a real corpus would have after running
`slam_pipeline` (per-tick turn-rate column over a set of clips).

Runnable with:

    python examples/motion-coverage/motion_coverage.py
"""

from kuration_ai import input, BalancedSelector

# 4 clips, 100 rows each. `turn_rate` cycles at different rates per clip
# so the per-window std varies across the corpus and there's something
# to bin over.
sessions = {
    f"clip-{i}": [
        {"t": float(step),
         "turn_rate": float(i * step % 7)}
        for step in range(100)
    ]
    for i in range(4)
}

pipeline = input(sessions, time="t", hz=10.0).window(8, stride=4)

op = BalancedSelector(feature="motion", of="turn_rate", kind="std", bins=4, count=20)
result = op.apply(pipeline).options(mode="strict", seed=1).collect()

print(f"selected: {len(result.selected)} windows")
print(f"features on first window: {sorted(set(result.selected[0].features) - {'_index'})}")
print()
print("compiled chain:")
print(f"  {op.describe()}")
