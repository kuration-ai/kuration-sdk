# Output layer — selected curation → training shards

An output modality may use `source: media:<column>`. The writer decodes lazy references in one
bounded batch and writes a `(T, ...)` tensor. Encoders may implement optional
`encode_batch(records, ctx)`, which is preferred over the compatible per-record `timestep` hook.

The output layer mirrors the input side: a declarative `OutputSpec` (what each timestep holds)
+ a pluggable `OutputEncoder` (where domain logic lives) + a generic `ShardWriter` (chunking,
short-policy, containers, manifests). Provider-backed `media:` columns are decoded generically;
application-derived tensors such as action bins remain encoder plugins.

## The output unit: a Track

A **track** is one entity's timestep sequence — the output unit (generalizing a per-POV
episode/clip). `tracks_from_ticks(schema, ticks, source_id)` produces one track per entity,
each record being that entity's fields at a timestep merged with the tick's globals. A
**shard** is a fixed-length chunk of a track.

**Sub-track spans.** By default a whole track is admitted (or not) as one unit. Pass
`write_dataset(..., span_steps=N)` (or `span_seconds=`) to subdivide each track into
independently-budgeted spans, so the fill keeps the *best parts* of a long source rather than
all-or-nothing. Each span's priority and tags aggregate only the segments overlapping it, and
the same `clip_aggregate` (`max`/`mean`/`min`) aggregates every segment feature into the span's
`track.meta`; this is what `segment:<feature>` modalities read. Admitted ids become
`(source, entity, span_index)`. Unset → one whole-track span.
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
        "frame":  {"dtype": "uint8",   "shape": [150,280,3], "source": "media:frame"},
        "label":  {"dtype": "int64",   "shape": [],  "source": "segment:label"}
    }
})
```

**Modality `source` routing:**

| kind | pulls from |
|---|---|
| `field:<name>` | the track entity's schema field at that timestep |
| `segment:<attr>` | a selected segment feature (count path) or span-aggregated feature (hours path); low-level `write_track` reads the same key from `track.meta` |
| `media:<name>` | the schema's lazy media column, decoded by its provider in one bounded shard batch |
| `encoder:<hook>` | an `OutputEncoder` — domain code |

**`short_policy`:** `keep` → last shard is its (shorter) real tail; `pad` → last shard padded
to `shard_length` with zero rows (real length recorded); `drop` → short tail discarded.

**Missing entities:** every shard includes `_present: bool[L]` by default. It is false when the
track entity is absent at a real timestep, so zero-filled modality values remain distinguishable
from genuine zeros. Set `presence_mask: false` only when a legacy consumer cannot accept the
auxiliary array.

**Missing modality values are errors:** for a present entity, every declared `field:` and
`encoder:` source must return a value, and every `segment:` attribute must exist. A missing hook,
field, or segment feature raises with source/entity/row context instead of silently writing zero.

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

    def encode_batch(self, records, ctx):
        # Optional replacement for timestep(); arrays lead with len(records).
        return {"frame": decode_many(records)}
```

When present, `encode_batch` is preferred; `timestep` remains compatible. Core ships
`PassthroughEncoder` (no `encoder:` modalities). Register custom ones for worker
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

`result["results"]["budget"]` and the manifest's `budget` block report requested, candidate, admitted, and
emitted seconds per split, plus underfill, dropped-tail emission loss, non-finite exclusions,
and group-cap exclusions.

With the `filter` method, spans whose segments all fail the predicates get `-inf` priority and
are **never admitted**, even when the budget has room — the fill emits only predicate-passing
data (an under-filled budget is reported, not silently topped up with rejects).
The same hard boundary applies to count selection: tag coverage can only repair coverage from
the predicate-passing pool. Fractional count floors round up, so a positive configured minimum
never becomes zero through rounding.

**`executor` / `workers` / `on_error` / `resume` / `force`**:
`write_dataset(..., workers=8)` parallelizes the feature
pass; `on_error="skip"` (default) skips+logs a source that fails feature reading, transform
execution, or scorer buffer rereading (but a *fully empty* segment pool raises — see the
troubleshooting recipe); `resume=True` reuses an existing shard only when
its recorded byte size and SHA-256 still match. Missing, legacy, or corrupt shards are rebuilt.
An existing destination is protected by default: use `resume=True` for the same pinned hours run
or `force=True` to replace it. These intents are mutually exclusive. Validation and selection
finish before publication begins, so a pre-publication failure leaves the published
manifest, config, shards, and commit byte-identical and readable. Bounded checkpoint caches are
stored in run-fingerprint namespaces, so a failed replacement cannot overwrite the prior run's
reusable cache.
Resume is safe by construction: local shards use same-directory temp files plus replace; remote
shards upload a complete attempt object before asking the configured fsspec backend to move it
into place. Native atomic rename and conditional-create guarantees vary by backend; the
checksummed commit protocol keeps readers fail-closed, but does not turn every object store into
a filesystem with atomic rename. The resolved output spec is
pinned in `.kuration_output_spec.json`, and that marker's checksum is itself pinned by
`dataset.commit.json`, so a missing, legacy-unpinned, or changed marker cannot authorize reuse.
Before any shard can be replaced, `dataset.commit.json` moves to `state=writing`; readers reject
that dataset until the manifest and config are durable and the final checksummed
`state=committed` record is published. Interrupted runs retain checksum-safe resume metadata.
An exclusive destination lease spans preflight through the final commit, so two SDK writers
cannot interleave one artifact. Local leases release automatically on process exit. A remote
lease is a persistent exclusive-create object; after a crashed writer, remove
`.kuration-writer.lock` manually only after confirming that no writer still owns the destination.
Persisted reuse also requires a strong source identity: an adapter-provided fingerprint, local
content hash, or immutable remote version/ETag. Logical IDs and metadata-only remote objects can
run without reuse, but cannot authorize an old shard or checkpoint; a hard memory budget that
requires checkpoints fails with instructions to provide a fingerprint.

Pass `executor="ray"` for the optional ordered Ray source backend (`[ray]`); the default remains
`"local"`. Canonical segment sorting keeps selection independent of task completion order.

So the emitted N hours is the strategy's *best* N hours, not a random draw.

## Outputs on disk

```
out/
  train/<source>_<entity>[_spanNNN]_<hash12>_shard0000.npz, ...
  test/<source>_<entity>[_spanNNN]_<hash12>_shard0000.npz, ...
  dataset_manifest.json         # shards plus requested/admitted/emitted budget accounting
  config.json                   # resolved job, results, timestamps, and full provenance
  dataset.commit.json           # final state + metadata SHA-256 publication record
  .kuration_output_spec.json    # hours-only, commit-pinned run fingerprint checked by resume
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
**numpy-only** (`kuration.core.loaders`) so a JAX/TF/plain-numpy consumer needs no extra dependency;
the PyTorch wrapper (`kuration.core.torch`, the `[torch]` extra) is a thin layer over it.

```python
from kuration import iter_shards

for arr in iter_shards("out/", split="train"):
    # declared modalities plus boolean _present, trimmed to the shard's real n_steps
    obs, act, present = arr["obs"], arr["act"], arr["_present"]
```

`iter_shards(dataset, *, split=None, modalities=None, trim=True, container=None,
with_meta=False, registry=None, verify=True)`
accepts a **dataset directory** or a path to its `dataset_manifest.json`. Legacy absolute-path
manifest dictionaries remain usable, but current portable manifests must be loaded by path so
their relative shard paths and commit sidecar can be resolved and verified. It
rejects duplicate object keys, non-finite JSON numbers, malformed embedded schemas, symlinked
metadata, and special files before decoding shards. It
handles all four containers, stages supported object-store shards (`gs://`, `s3://`, and
Azure URLs) automatically, **trims pad
rows** to each shard's real `n_steps`, **reshapes** tensor modalities back to their declared
rank (Parquet stores them flat), and **restores the declared dtype** (the json/parquet
round-trips widen to float64/int64 on disk; the loaders cast back, so a `uint8` frame comes
back `uint8` from every container). Recorded shard SHA-256 and size metadata are verified before
each shard is returned; `verify=False` is an explicit opt-out for trusted artifacts where that
extra integrity read is too expensive. A dataset directory that has been **moved or copied**
still loads: current entries are always relative to—and every legacy entry is independently
rebased onto—the directory you passed, even when the original dataset still exists.
`with_meta=True` yields `(arrays, entry)` so you also get the
shard's `source_id`/`entity_id`/`split`/`padded`. Manifest `entity_id` is an opaque,
type-tagged string so distinct IDs such as integer `1` and string `"1"` cannot alias a shard.
Writable entity IDs must have a canonical value representation: `None`, booleans, integers,
finite floats, strings, bytes, tuples, frozensets, and NumPy scalars composed from those types
are supported. Extension objects are rejected with `ValueError` instead of risking a shard-name
collision.

`shard_entries(dataset, split=...)` and
`read_shard_arrays(entry, container, verify=True)` are the building blocks if you want manifest
entries or a single shard directly.

Before handing an artifact to training, summarize it without the original job or input corpus:

```python
from kuration import inspect_dataset

report = inspect_dataset("out/")                 # manifest/config metadata only
verified = inspect_dataset("out/", verify=True) # decode every shard
assert verified.verification["valid"]
```

The report includes split sizes and durations, modalities, padding, budget information, skipped
source errors, provenance, and execution mode. The equivalent CLI is `kur-inspect out/`; use
`kur-inspect out/ --verify` in handoff or CI, `--json` for automation, or `--html report.html`
for a dependency-free shareable report.

Compare two artifacts with `compare_datasets(left, right, verify=False)` or `kur-diff LEFT RIGHT`.
The result reports shard/split/modality changes and recursively identifies differences in schema,
strategy, output, pipeline, results, and warnings. `kur-diff --html diff.html` writes the
same structured comparison as a self-contained static document.

For hours output, `hours_selection_review(result)` consumes the returned `DatasetWriteResult` and
explains item admission, rejection reasons, priorities, split utilization, underfill, and bounded
span examples without rescoring. The pipeline shortcut is
`pipeline.review(path="out/", **output_options)` after `.sample(hours=...)`.

`kur-check DATASET --policy quality.json` evaluates artifact gates such as required modalities,
minimum sources/shards, full verification, budget utilization, and skipped-source limits. It exits
2 when a gate fails, making artifact approval usable directly from CI.

**PyTorch.**

```python
from kuration.core.torch import ShardDataset            # pip install "kuration-sdk[torch]"
from torch.utils.data import DataLoader

ds = ShardDataset("out/", split="train")           # one shard = one item, modalities -> tensors
loader = DataLoader(ds, batch_size=8)
```

`ShardDataset` is map-style (`__len__`/`__getitem__`) and works with `DataLoader` directly; pass
`transform=fn` for per-item post-processing and `include_meta=True` to attach an `_meta` dict.
With `short_policy="keep"` shards have **variable length** — supply a `collate_fn`, or write with
`short_policy="pad"` (or an evenly-dividing `shard_length`) for default batching.

**Remote output.** `out_dir` may be a cloud URL — `write_dataset("gs://bucket/run1", …)` writes
shards, manifest, and config straight to the bucket (use `[s3]`, `[gcs]`, or `[azure]`; managed
environments may use `[cloud]` plus their separately supplied backend). The dataset is then
re-ingestable with `CuratedDatasetAdapter("gs://bucket/run1")`. Mutating hierarchical fsspec
backends such as FTP/SFTP is rejected because symlink-safe replacement and deletion cannot be
guaranteed. Local paths need nothing.
