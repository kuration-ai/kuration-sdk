"""Fluent operator pipeline: input() -> chained operators -> terminals (kuration/pipeline.py).

Synthesizes plain .json sources (array-of-row-objects — a numeric "t" time column, an
"agent" entity column, a scalar "hp", and a categorical "action"), then runs the
motivating one-chain shape end to end:

    input(...).measure("hp").filter("hp", "<=", 50).measure("action_entropy", "entropy", of="action")
              .organize("action_entropy", bins=4).curate("equal", n=...).output(out_dir, container=...)

No schema, no adapter, no strategy dict — `input()` infers the schema from a sample and
every operator validates itself loudly against it.

    python examples/04_operator_pipeline.py
"""
import json
import random
import tempfile
from pathlib import Path

from kuration import input, iter_shards

rng = random.Random(0)
work = Path(tempfile.mkdtemp(prefix="kuration_ex04_"))

# 1. Synthesize plain-JSON device logs: source d{i} has hp centered at 20*i (+ ripple), so
#    filter("hp", "<=", 50) keeps d0/d1/d2 and drops d3/d4. "action" is drawn per-step with a
#    per-source skew (source i favors "attack" more as i grows) -> action_entropy varies both
#    across sources and across windows within a source, giving `organize` real bins to spread.
ACTIONS = ["idle", "move", "attack"]
N_SOURCES, N_STEPS = 5, 128
paths = []
for i in range(N_SOURCES):
    weights = [1.0, 1.0, 1.0 + 2.0 * i]
    rows = [{"t": float(t), "agent": f"a{i}", "hp": 20.0 * i + float(t % 7),
             "action": rng.choices(ACTIONS, weights=weights)[0]}
            for t in range(N_STEPS)]
    p = work / f"device_{i}.json"
    p.write_text(json.dumps(rows))
    paths.append(str(p))

# 2. The motivating chain: measure -> predicate filter -> a second measure (entropy of a
#    categorical) -> a bin axis over it -> count-based curation -> write shards.
out_count = work / "out_count"
pipe = (
    input(paths, time="t")
    .window(16)
    .measure("hp")
    .filter("hp", "<=", 50)
    .measure("action_entropy", "entropy", of="action")
    .organize("action_entropy", bins=4)
    .curate("equal", n=12)
    .options(on_error="raise")
)

print(pipe.describe()["schema"]["fields"].keys())
print("head (first 3 ticks of source 0):",
      [(tk.time, tk.entities) for tk in pipe.head(3)])

try:
    result = pipe.output(str(out_count), container="hdf5")
    container = "hdf5"
except ImportError as e:
    print(f"  [hdf5 unavailable ({e}); falling back to parquet]")
    result = pipe.output(str(out_count), container="parquet")
    container = "parquet"

assert result["selected"] == 12
assert result["shards"]
print(f"count path: wrote {len(result['shards'])} shards ({container}) selected="
      f"{result['selected']} to {out_count}")

# 3. Hours-budget variant of the same chain, writing npz.
out_hours = work / "out_hours"
pipe_hours = (
    input(paths, time="t")
    .window(16)
    .measure("hp")
    .filter("hp", "<=", 50)
    .measure("action_entropy", "entropy", of="action")
    .organize("action_entropy", bins=4)
    .curate("equal", hours=0.05)
    .options(on_error="raise")
)
res_hours = pipe_hours.output(str(out_hours), container="npz")
assert res_hours["results"]["shards"] > 0
print(f"hours path: wrote {res_hours['results']['shards']} shards (npz) to {out_hours}")

# 4. .describe() — the resolved, un-run state of the pipeline.
desc = pipe.describe()
print("describe(): method =", desc["strategy"]["selection"]["method"],
      "| axes =", desc["strategy"]["binning"]["axes"],
      "| predicates =", desc["strategy"]["selection"]["predicates"])

# 5. Read shards back via kuration.iter_shards and check shapes.
shards = list(iter_shards(str(out_count), split="train"))
assert len(shards) == len(result["shards"])
for arrays in shards:
    assert arrays["hp"].shape == (16,)          # window-sized sub-track, scalar modality
print(f"read back {len(shards)} train shards; hp shape = {shards[0]['hp'].shape}")

# 6. Peek at config.json's embedded 'pipeline' provenance — the operator chain, as data.
cfg = json.loads((out_count / "config.json").read_text())
ops = [o["op"] for o in cfg["pipeline"]]
print("config.json 'pipeline' provenance:", ops)
assert ops == ["input", "window", "measure", "filter", "measure", "organize", "curate",
               "options"]

print("\nall assertions passed.")
