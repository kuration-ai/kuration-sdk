"""The pipeline path in depth: one chain, both budgets, and the provenance it leaves behind.

This is the chain documented in docs/pipeline.md, run end to end:

    input(...).window(16).measure("hp").filter("hp", "<=", 50)
              .measure("action_entropy", "entropy", of="action")
              .bin("action_entropy", bins=4).sample("equal", n=12).output(...)

It shows three things the shorter examples do not: selecting by **count** versus by **hours**,
`.describe()` as the resolved-but-unrun plan, and the operator chain recorded inside
`config.json` so a run can be reconstructed from its artifact.

    python examples/04_operator_pipeline.py
"""
import json
import random
import tempfile
from pathlib import Path

from kuration import input, iter_shards

rng = random.Random(0)
work = Path(tempfile.mkdtemp(prefix="kuration_ex04_"))

# Five device logs as plain JSON rows. Source i has hp centred on 20*i, so the predicate below
# keeps d0-d2 and drops d3/d4; "action" is skewed more towards "attack" as i grows, giving
# action_entropy a real range to bin across.
ACTIONS = ["idle", "move", "attack"]
sources = []
for i in range(5):
    weights = [1.0, 1.0, 1.0 + 2.0 * i]
    rows = [{"t": float(t), "agent": f"a{i}", "hp": 20.0 * i + float(t % 7),
             "action": rng.choices(ACTIONS, weights=weights)[0]}
            for t in range(128)]
    path = work / f"device_{i}.json"
    path.write_text(json.dumps(rows))
    sources.append(str(path))


def chain(pipeline):
    """The shared middle of both runs: measure, drop, measure again, bin."""
    return (pipeline
            .window(16)
            .measure("hp")                                    # defaults to mean of "hp"
            .filter("hp", "<=", 50)                           # a universal pre-filter
            .measure("action_entropy", "entropy", of="action")
            .bin("action_entropy", bins=4))


# ---- count path: "give me exactly 12 windows" --------------------------------
count_pipe = chain(input(sources, time="t")).sample("equal", n=12).options(on_error="raise")

plan = count_pipe.describe()
print("resolved plan (nothing has run yet):")
print(f"  method  = {plan['strategy']['selection']['method']}")
print(f"  axes    = {plan['strategy']['binning']['axes']}")
print(f"  filters = {plan['strategy']['selection']['predicates']}")

out_count = work / "out_count"
try:
    result = count_pipe.output(out_count, container="hdf5")
    container = "hdf5"
except ImportError:
    result = count_pipe.output(out_count, container="parquet")   # [array] extra absent
    container = "parquet"
print(f"\ncount path: selected {result['selected']} windows -> "
      f"{len(result['shards'])} {container} shards")

# Each selected window is written as its own sub-track, so shards are window-sized.
shards = list(iter_shards(out_count, split="train"))
print(f"read back {len(shards)} shards; hp shape = {shards[0]['hp'].shape}")

# ---- hours path: "give me 0.05 hours of data" --------------------------------
hours_pipe = chain(input(sources, time="t")).sample("equal", hours=0.05).options(
    on_error="raise")
hours_result = hours_pipe.output(work / "out_hours", container="npz")
print(f"hours path: wrote {hours_result['results']['shards']} npz shards "
      f"for a {0.05:.2f}h budget")

# ---- provenance --------------------------------------------------------------
config = json.loads((out_count / "config.json").read_text())
print("\nconfig.json records the chain as data:",
      " -> ".join(step["op"] for step in config["pipeline"]))
print(f"output dir: {work}")
