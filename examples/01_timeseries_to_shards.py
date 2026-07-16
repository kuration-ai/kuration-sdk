"""Generic multivariate time-series → curated training shards (the 5-minute start).

Synthesizes a few CSV "device logs", then curates them for diversity across movement + a
temperature axis, with coverage of every operating mode, and writes train/test shards.

    python examples/01_timeseries_to_shards.py
"""
import csv
import math
import random
import tempfile
from pathlib import Path

from kuration import SchemaSpec, TimeseriesAdapter, OutputSpec, input

rng = random.Random(0)
work = Path(tempfile.mkdtemp(prefix="kuration_ex01_"))

# 1. Declare the dataset's vocabulary (data, not code).
schema = SchemaSpec.from_dict({
    "name": "devices", "entity_axis": "device", "sample_hz": 1.0,
    "fields": {
        "pos":  {"kind": "vector", "dims": ["x", "y"], "semantics": "position", "max_rate": 100},
        "temp": {"kind": "scalar"},
        "mode": {"kind": "categorical"},
    },
})

# 2. Synthesize 8 device logs as CSV (one entity per file).
MODES = ["idle", "active", "fault"]
sources = []
for d in range(8):
    p = work / f"device_{d}.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "device", "x", "y", "temp", "mode"])
        w.writeheader()
        x = y = 0.0
        mode = MODES[d % len(MODES)] if d < 6 else "fault"   # "fault" deliberately rare
        for t in range(200):
            x += math.sin(t / 10) * (d + 1) * 0.1
            y += math.cos(t / 13) * 0.1
            w.writerow({"t": t, "device": f"d{d}", "x": round(x, 3), "y": round(y, 3),
                        "temp": round(20 + 5 * math.sin(t / 20) + rng.random(), 2), "mode": mode})
    sources.append(str(p))

adapter = TimeseriesAdapter(schema, fields={"pos": ["x", "y"], "temp": "temp", "mode": "mode"},
                            time_column="t", entity_column="device")

# 3. Output: each timestep is a 2-vector position + a scalar temperature.
spec = OutputSpec.from_dict({
    "shard_length": 32, "container": "npz",
    "modalities": {"pos":  {"dtype": "float32", "shape": [2], "source": "field:pos"},
                   "temp": {"dtype": "float32", "shape": [],  "source": "field:temp"}},
})

# 4. Build and execute the fluent pipeline.
pipeline = (
    input(adapter=adapter, sources=sources, schema=schema)
    .window(8, stride=4, unit="seconds")
    .measure("speed", "mean", of="pos", map="speed")
    .measure("temp", "mean", of="temp")
    .bin("speed", bins=5)
    .bin("temp", bins=5)
    .tag("mode", min_frac=0.05)
    .sample("equal", hours=0.04, test_hours=0.02)
    .options(mode="strict", seed=0)
)
res = pipeline.output(str(work / "out"), spec=spec)
r = res["results"]
print(f"wrote {r['shards']} shards  ({r['n_train']} train / {r['n_test']} test sources)")
print(f"output dir: {work / 'out'}")
print(f"  e.g. {res['shards'][0]['path'] if res['shards'] else '(none)'}")
