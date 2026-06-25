# Output layer — selected curation → training shards

The output layer mirrors the input side: a declarative `OutputSpec` (what each timestep holds)
+ a pluggable `OutputEncoder` (where domain logic lives) + a generic `ShardWriter` (chunking,
short-policy, containers, manifests). The SDK ships the *mechanics*; the *semantics* (decoding
a video frame, binning an action) are application encoder plugins — never core.

## The output unit: a Track

A **track** is one entity's timestep sequence — the output unit (generalizing a per-POV
episode/clip). `tracks_from_ticks(schema, ticks, source_id)` produces one track per entity,
each record being that entity's fields at a timestep merged with the tick's globals. A
**shard** is a fixed-length chunk of a track.

## OutputSpec

```python
from kuration import OutputSpec
spec = OutputSpec.from_dict({
    "shard_length": 64,             # timesteps per shard
    "short_policy": "keep",         # keep | pad | drop  (tail handling)
    "container": "npz",             # npz | json
    "modalities": {
        "pos":    {"dtype": "float32", "shape": [3], "source": "field:pos"},
        "battery":{"dtype": "float32", "shape": [],  "source": "field:battery"},
        "frame":  {"dtype": "uint8",   "shape": [150,280,3], "source": "encoder:frame"},
        "label":  {"dtype": "int64",   "shape": [],  "source": "segment:label"}
    }
})
```

**Modality `source` routing:**

| kind | pulls from |
|---|---|
| `field:<name>` | the track entity's schema field at that timestep |
| `segment:<attr>` | a per-track scalar in `track.meta` (e.g. a precomputed numeric label) |
| `encoder:<hook>` | an `OutputEncoder` — domain code |

**`short_policy`:** `keep` → last shard is its (shorter) real tail; `pad` → last shard padded
to `shard_length` with zero rows (real length recorded); `drop` → short tail discarded.

**Containers:** `npz` (numpy-native, tensor default) and `json` (human-readable) are
dependency-free; `hdf5` needs the `[array]` extra and `parquet` the `[parquet]` extra. All four
have identical semantics — arrays written as-shaped `(L, *modality_shape)` plus the real
(pre-pad) `_n_steps` in-band — and all four re-ingest through `CuratedDatasetAdapter`. In
`parquet`, scalar modalities are columns and tensor modalities are flattened list-columns
(reshaped on read via the manifest's per-modality `shape`).

## OutputEncoder (the domain seam)

```python
import numpy as np

class FrameEncoder:
    def timestep(self, record, ctx):
        # ctx = {"schema": ..., "track": {...meta, source_id, entity_id}}
        return {"frame": decode_my_frame(record)}   # -> {modality_name: array}
```

Core ships `PassthroughEncoder` (no `encoder:` modalities). Register custom ones for worker
reconstruction with `register_encoder(name, factory)` / `get_encoder(name, **kw)`.

## Writing

**One track** (low-level):

```python
from kuration import write_track
entries = write_track("out/", track, spec, encoder=FrameEncoder(), schema=schema, split="train")
```

**Whole dataset** (the usual path) — score, fill an hour budget, write:

```python
from kuration import write_dataset, Segmenter
result = write_dataset(
    "out/", schema=schema, adapter=adapter, sources=sources,
    strategy=strategy, segmenter=Segmenter(4, 2, unit="seconds", sample_hz=20),
    spec=spec, encoder=None,            # PassthroughEncoder if omitted
    train_hours=10, test_hours=1,
    sample_hz=20,                       # defaults to schema.sample_hz
    seed=0, group_of=None, clip_aggregate="max",
)
```

### The hours-as-budget fill

`write_dataset` ranks each track by its source's aggregated segment priority
(`clip_aggregate`: `max` for inclusion strategies, `mean`/`min` for avoidance), then fills the
train and test **hour budgets** in descending priority. Properties:

- **source-level, leakage-free split** — a whole source goes to exactly one of train/test;
- **decoupled budgets** — sources are pre-partitioned into split pools so a spread-out priority
  can't starve the test budget;
- **group cap in hours** — a group can't exceed `max_group_overshoot × (budget / n_groups)`;
- budget honored to within one track.

So the emitted N hours is the strategy's *best* N hours, not a random draw.

## Outputs on disk

```
out/
  train/<source>_<entity>_shard0000.npz, ...
  test/<source>_<entity>_shard0000.npz, ...
  dataset_manifest.json     # {count, shards:[{path, source_id, entity_id, split, n_steps, padded, shape}]}
  config.json               # resolved schema + strategy + output spec + results + seed
```

`config.json` is the reproducibility record — schema, strategy, and output spec together
regenerate this dataset from the same sources.
