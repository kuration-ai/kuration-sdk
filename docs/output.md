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

**Sub-track spans.** By default a whole track is admitted (or not) as one unit. Pass
`write_dataset(..., span_steps=N)` (or `span_seconds=`) to subdivide each track into
independently-budgeted spans, so the fill keeps the *best parts* of a long source rather than
all-or-nothing. Each span's priority and tags aggregate only the segments overlapping it, and
admitted ids become `(source, entity, span_index)`. Unset → one whole-track span.
(`span_steps < shard_length` with `short_policy: "drop"` is rejected at preflight — every span
would chunk to zero shards while still consuming the hours budget.)

## OutputSpec

```python
from kuration import OutputSpec
spec = OutputSpec.from_dict({
    "shard_length": 64,             # timesteps per shard
    "short_policy": "keep",         # keep | pad | drop  (tail handling)
    "container": "npz",             # npz | json | hdf5 | parquet
    "presence_mask": True,           # emit boolean _present per timestep (default)
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

**Missing entities:** every shard includes `_present: bool[L]` by default. It is false when the
track entity is absent at a real timestep, so zero-filled modality values remain distinguishable
from genuine zeros. Set `presence_mask: false` only when a legacy consumer cannot accept the
auxiliary array.

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
- **optional group cap in hours** — when configured, a group can't exceed
  `max_group_overshoot × (budget / n_groups)`;
- **tag reserve** — with `selection.tag_coverage`, each tag's highest-priority track is admitted
  first (bounded by `min_tag_frac` of the budget) so a rare label survives the cut;
- budget honored to within one track.

`results.budget` and the manifest's `budget` block report requested, candidate, admitted, and
emitted seconds per split, plus underfill, dropped-tail emission loss, non-finite exclusions,
and group-cap exclusions.

With the `filter` method, spans whose segments all fail the predicates get `-inf` priority and
are **never admitted**, even when the budget has room — the fill emits only predicate-passing
data (an under-filled budget is reported, not silently topped up with rejects).

**`workers` / `on_error` / `resume`**: `write_dataset(..., workers=8)` parallelizes the feature
pass; `on_error="skip"` (default) skips+logs an unreadable source (but a *fully empty* segment
pool raises — see the troubleshooting recipe); `resume=True` skips re-encoding shards that
already exist. Resume is safe by construction: shards are written atomically (temp + rename,
locally and remotely — an interrupted run never leaves a partial shard at its final path), and
the resolved output spec is pinned in a `.kuration_output_spec.json` sidecar — resuming the
same `out_dir` with a **different** spec raises rather than silently reusing stale shards under
a fresh manifest.

So the emitted N hours is the strategy's *best* N hours, not a random draw.

## Outputs on disk

```
out/
  train/<source>_<entity>[_spanNNN]_<hash12>_shard0000.npz, ...
  test/<source>_<entity>[_spanNNN]_<hash12>_shard0000.npz, ...
  dataset_manifest.json         # shards plus requested/admitted/emitted budget accounting
  config.json                   # resolved job, results, timestamps, and full provenance
  .kuration_output_spec.json    # complete run fingerprint checked by resume
```

The readable filename prefix is portable and length-bounded; the 12-char hash disambiguates
across distinct `(source, entity)` pairs (path-like source ids can otherwise alias after
flattening). Manifest entry order is deterministic (sorted by source/entity/span), so
`ShardDataset[i]` is stable across runs and machines.

`config.json` is the reproducibility record. Its provenance block fingerprints the effective
job, SDK version, adapter/encoder/callback code and state, and source metadata. Resume refuses
to reuse shards when any fingerprinted input changes.

## Reading a dataset back

The write side produces shards + a manifest; the read side turns them back into arrays. This is
**numpy-only** (`kuration.loaders`) so a JAX/TF/plain-numpy consumer needs no extra dependency;
the PyTorch wrapper (`kuration.torch`, the `[torch]` extra) is a thin layer over it.

```python
from kuration import iter_shards

for arr in iter_shards("out/", split="train"):
    # declared modalities plus boolean _present, trimmed to the shard's real n_steps
    obs, act, present = arr["obs"], arr["act"], arr["_present"]
```

`iter_shards(dataset, *, split=None, modalities=None, trim=True, container=None, with_meta=False)`
accepts a **dataset directory**, a path to its `dataset_manifest.json`, or an already-loaded
manifest dict (pass `container=` for a bare dict — a directory reads it from `config.json`). It
handles all four containers, stages remote shards (`gs://`/`s3://`/…) automatically, **trims pad
rows** to each shard's real `n_steps`, **reshapes** tensor modalities back to their declared
rank (Parquet stores them flat), and **restores the declared dtype** (the json/parquet
round-trips widen to float64/int64 on disk; the loaders cast back, so a `uint8` frame comes
back `uint8` from every container). A dataset directory that has been **moved or copied**
still loads: when the manifest's recorded shard paths no longer exist, they are rebased onto
the directory you passed. `with_meta=True` yields `(arrays, entry)` so you also get the
shard's `source_id`/`entity_id`/`split`/`padded`. `shard_entries(dataset, split=...)` and
`read_shard_arrays(entry, container)` are the building blocks if you want manifest entries or a
single shard directly.

**PyTorch.**

```python
from kuration.torch import ShardDataset            # pip install -e '.[torch]'
from torch.utils.data import DataLoader

ds = ShardDataset("out/", split="train")           # one shard = one item, modalities -> tensors
loader = DataLoader(ds, batch_size=8)
```

`ShardDataset` is map-style (`__len__`/`__getitem__`) and works with `DataLoader` directly; pass
`transform=fn` for per-item post-processing and `include_meta=True` to attach an `_meta` dict.
With `short_policy="keep"` shards have **variable length** — supply a `collate_fn`, or write with
`short_policy="pad"` (or an evenly-dividing `shard_length`) for default batching.

**Remote output.** `out_dir` may be a cloud URL — `write_dataset("gs://bucket/run1", …)` writes
shards, manifest, and config straight to the bucket (needs the `[cloud]` extra + a backend like
`gcsfs`/`s3fs`/`adlfs`). The dataset is then re-ingestable with
`CuratedDatasetAdapter("gs://bucket/run1")`. Local paths need nothing.
