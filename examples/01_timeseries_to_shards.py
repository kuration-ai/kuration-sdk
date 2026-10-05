"""CSV files on disk to training shards — the five-minute start.

The common starting point: a folder of logs and no schema written yet. `input()` sniffs the
files, infers the schema from a sample, and every operator validates itself against it.

    python examples/01_timeseries_to_shards.py

For the same path against a real published dataset, see public_data/ett.py.
"""
import csv
import math
import random
import tempfile
from pathlib import Path

from kuration import input, iter_shards

rng = random.Random(0)
work = Path(tempfile.mkdtemp(prefix="kuration_ex01_"))

# 1. Synthesize eight device logs, one CSV per device. "fault" is deliberately rare, so
#    ordinary distribution matching would drop it — that is what tag coverage is for.
MODES = ["idle", "active", "fault"]
for device in range(8):
    path = work / f"device_{device}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["t", "device", "x", "y", "temp", "mode"])
        writer.writeheader()
        mode = MODES[device % len(MODES)] if device < 6 else "fault"
        x = y = 0.0
        for t in range(200):
            x += math.sin(t / 10) * (device + 1) * 0.1
            y += math.cos(t / 13) * 0.1
            writer.writerow({"t": t, "device": f"d{device}",
                             "x": round(x, 3), "y": round(y, 3),
                             "temp": round(20 + 5 * math.sin(t / 20) + rng.random(), 2),
                             "mode": mode})

# 2. One chain, start to finish. No SchemaSpec, no adapter, no strategy dict: the time and
#    entity columns are recognized by name, and the sample rate comes from `t` itself.
out_dir = work / "out"
pipeline = (
    input(f"{work}/*.csv")
    # The schema knows the rate, so windows can be stated in seconds.
    .window(8, stride=4, unit="seconds")
    .measure("drift", "std", of="x")          # how much the device moved
    .measure("temp", "mean", of="temp")       # how hot it ran
    .bin("drift", bins=5)                     # spread the picks over both axes instead
    .bin("temp", bins=5)                      # of over-sampling the average device
    .tag("mode", min_frac=0.05)               # keep every operating mode, incl. "fault"
    .split(test=0.25)                         # whole devices held out, never half a window
    # Budget in recorded time, not window count: 0.25h train + 0.1h test out of the ~0.44h
    # the eight devices hold between them.
    .sample("equal", hours=0.25, test_hours=0.1)
    .options(mode="strict", seed=0)
)

# `mode` selected windows but is not a training tensor. Strict mode refuses to guess what
# should happen to a categorical field, so the decision is stated here.
result = pipeline.output(out_dir, include=["x", "y", "temp"], categorical="omit",
                         shard_length=32)
summary = result["results"]
print(f"inferred fields: {', '.join(pipeline.describe()['schema']['fields'])}")
print(f"wrote {summary['shards']} shards "
      f"({summary['n_train_sources']} train / {summary['n_test_sources']} test devices)")

# 3. Read one back as ordinary numpy arrays.
arrays = next(iter(iter_shards(out_dir)))
print("one shard:", {name: tuple(array.shape) for name, array in arrays.items()})
print(f"output dir: {out_dir}")
