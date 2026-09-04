# Usage cookbook

Task-oriented recipes spanning what the SDK can do. Each is a minimal, runnable snippet; see the
per-layer references ([schema](schema.md), [dsl](dsl.md), [adapters](adapters.md),
[strategies](strategies.md), [output](output.md)) for full options, [pipeline.md](pipeline.md)
for the `input(...).operator(...)` API, [api.md](api.md) for every public signature,
[jobs.md](jobs.md) for the CLI job-spec keys, and [overview](overview.md) for the explicit
end-to-end example.

Choose the shortest entry point:

| need | use |
|---|---|
| First dataset or ordinary files/rows | [Quick start](quickstart.md), then the [pipeline guide](pipeline.md#worked-example) |
| Exact schema, adapter, or strategy control | Recipes 1–14 below |
| Reproducible configuration without Python code | [JSON jobs](jobs.md) |
| Validate a written dataset | [`inspect_dataset` / `kur-inspect`](output.md#reading-a-dataset-back) |

The explicit recipes assume this schema and adapter:

```python
from kuration import SchemaSpec, TimeseriesAdapter, Segmenter, load_strategy

schema = SchemaSpec.from_dict({
    "name": "demo", "entity_axis": "agent", "sample_hz": 20.0,
    "fields": {
        "pos":    {"kind": "vector", "dims": ["x", "y", "z"], "semantics": "position", "max_rate": 50},
        "action": {"kind": "vector", "dims": ["dx", "dy", "dz"]},
        "reward": {"kind": "scalar"},
        "phase":  {"kind": "categorical"},
    },
})
adapter = TimeseriesAdapter(schema, fields={"pos": ["x", "y", "z"], "action": ["dx", "dy", "dz"],
                                            "reward": "r", "phase": "phase"},
                            time_column="t", entity_column="agent_id")
seg = Segmenter(4, 2, unit="seconds", sample_hz=20)
```

---

## 1 · Reading data (any format)

```python
from kuration import (TimeseriesAdapter, HDF5Adapter, NpzAdapter, JsonlAdapter,
                      CuratedDatasetAdapter)

TimeseriesAdapter(schema, fields={...}, time_column="t", entity_column="id")  # CSV / Parquet / rows
HDF5Adapter(schema, fields={"pos": "xyz", "reward": "r"})                      # .h5  ([array] extra)
NpzAdapter(schema, fields={"pos": "xyz", "reward": "r"})                        # .npz (no deps)
JsonlAdapter(schema, fields={...}, time_column="t")                            # .jsonl
CuratedDatasetAdapter("out/run1")                                              # re-read your own output
```
Sources may be local paths or cloud URLs (`gs://…`, `s3://…`) — see recipe 13. Bespoke formats:
recipe 14. Tabular sources must be time-ordered; unordered or duplicate `(time, entity)` rows
raise (see recipe 15).

---

## 2 · Curate a target *count* of the best segments

```python
from kuration import curate

strategy = load_strategy({
    "name": "lively",
    "define": {"speed": {"source": "agent.pos", "map": "speed", "reduce": "mean"}},
    "selection": {"method": "top-k", "weights": {"speed": 1.0}},
}, schema)

res = curate(schema=schema, adapter=adapter, sources=["a.csv", "b.csv"],
             strategy=strategy, segmenter=seg, target_size=5000, seed=0)
res.selected        # the chosen segments
res.all_segments    # the full scored pool
```

## 3 · Build a train/test dataset to an *hour budget* (writes shards)

```python
from kuration import OutputSpec, write_dataset

spec = OutputSpec.from_dict({
    "shard_length": 64, "container": "npz",
    "modalities": {"pos":    {"dtype": "float32", "shape": [3], "source": "field:pos"},
                   "action": {"dtype": "float32", "shape": [3], "source": "field:action"}},
})
write_dataset("out/run1", schema=schema, adapter=adapter, sources=[...],
              strategy=strategy, segmenter=seg, spec=spec,
              train_hours=10, test_hours=1, seed=0)
# -> out/run1/{train,test}/<src>_<entity>_<hash12>_shardNNNN.npz + dataset_manifest.json + config.json
```
The emitted N hours is the strategy's *best* N hours (not a random draw). `config.json`
regenerates the run.

---

## 4 · Selection patterns

```python
# diversity over quantile bins
{"binning": {"axes": [{"metric": "speed", "bins": 10}]}, "selection": {"method": "uniform-bins"}}

# weighted score (sampled w/o replacement) or deterministic top-k
{"selection": {"method": "weighted", "weights": {"speed": 1.0, "turn": 0.5}}}
{"selection": {"method": "top-k",   "weights": {"speed": 1.0}}}

# match a target distribution over bins
{"binning": {"axes": [{"metric": "speed", "bins": 8}]},
 "selection": {"method": "target-distribution", "target": "flat"}}

# joint-space coverage (needs [methods]): cluster-balanced / k-center / facility-location
{"feature_space": ["speed", "turn", "reward"], "selection": {"method": "k-center", "clusters": 512}}

# importance resampling toward an anchor subset (DSIR)
{"feature_space": ["speed", "reward"],
 "selection": {"method": "dsir", "target": {"anchor_metric": "reward", "anchor_quantile": 0.8}}}
```

## 5 · Keep only clips matching DSL criteria (`filter`)

```python
load_strategy({
    "name": "fast-and-smooth",
    "define": {"speed": {"source": "agent.pos", "map": "speed", "reduce": "mean"},
               "jerk":  {"source": "agent.pos", "map": "speed", "reduce": "jerk"}},
    "selection": {"method": "filter", "predicates": [
        {"metric": "speed", "op": ">=", "value": 10},
        {"metric": "jerk",  "op": "<=", "value": 2}]},
}, schema)
```

## 6 · Prune the pool first (outliers / near-duplicates)

```python
{"feature_space": ["speed", "turn", "reward"],
 "filters": [{"type": "anomaly", "contamination": 0.02},
             {"type": "dedup", "threshold": 0.98, "per_group": true}],
 "selection": {"method": "k-center", "clusters": 256}}
```

## 7 · Guarantee coverage of rare labels (declarative tags)

```python
{"define": {"speed": {"source": "agent.pos", "map": "speed", "reduce": "mean"}},
 "tags": {"from": ["phase"]},                       # categorical fields and/or "stream.field"
 "selection": {"method": "top-k", "weights": {"speed": 1.0},
               "tag_coverage": {"min_tag_frac": 0.02}}}   # floor per tag, both count & hours paths
```

---

## 8 · Compose & re-use datasets

```python
from kuration import CuratedDatasetAdapter, assign_splits

# mix two curated datasets and split 25/75, stratified + leakage-free
adapter = CuratedDatasetAdapter(["out/ds_a", "out/ds_b"])
sources = adapter.sources()
split = assign_splits({s: adapter.dataset_of[s] for s in sources}, test_frac=0.25, seed=0)

# or look *inside* them for matching clips, then re-emit
res = curate(schema=adapter.schema, adapter=adapter, sources=sources,
             strategy=load_strategy({"name": "hot",
                 "define": {"speed": {"source": "e.pos.pos0", "reduce": "mean"}},  # modalities -> fields
                 "selection": {"method": "filter",
                               "predicates": [{"metric": "speed", "op": ">", "value": 5}]}},
                 adapter.schema),
             segmenter=Segmenter(64, 64), target_size=10_000)
```

## 9 · Compare strategies (which curation shapes the data how)

```python
from kuration import Arm, run_experiment

out = run_experiment(schema=schema, adapter=adapter, sources=[...], segmenter=seg, target_size=5000,
    arms=[Arm("baseline", load_strategy("…baseline…", schema)),
          Arm("coverage", load_strategy("…k-center…", schema)),
          Arm("fast",     load_strategy("…top-k speed…", schema))],
    baseline="baseline", seed=0)
out["arms"]["fast"]["metrics"]["speed"]["js_vs_baseline"]   # per-metric divergence from baseline
```

## 10 · Profile a dataset (no selection)

```python
from kuration import dataset_report
rep = dataset_report(res.all_segments)
rep["metrics"]["speed"]   # mean/std/p50/p95/min/max + histogram
rep["tags"], rep["groups"]
```

---

## 11 · Scale & robustness

```python
write_dataset("out/run1", ..., workers=8,        # parallel feature pass (deterministic)
              executor="local",                  # or "ray" with the optional [ray] extra
              on_error="skip",                    # log + skip a bad source (default); "raise" to fail fast
              resume=True,                         # verify and reuse intact shards
              force=False,                         # true replaces; mutually exclusive with resume
              streaming=True)                      # bounded memory with a streaming-capable adapter
```
`streaming=True` reads each source as a lazy tick stream on both the feature pass and the shard
write (also available on `curate(...)` / `run_experiment(...)`). CSV/JSONL/single-file Parquet,
HDF5, Zarr, and single-source HuggingFace inputs support it. HuggingFace adapters configured with
`source_column`, NPZ, and partitioned-Parquet dataset mode reject the option because discovering
their partitions cannot guarantee bounded memory.

`resume=True` is safe against crashes, corruption, and drift: local shards use atomic replacement;
remote shards upload a complete attempt before publishing and never pre-delete the prior object.
Existing shards are reused only when their recorded
size and SHA-256 match, and a complete run fingerprint is pinned in a sidecar whose own
integrity is covered by the final commit. Legacy or tampered unpinned markers are never trusted.
Reuse additionally requires an adapter fingerprint, local content identity, or immutable remote
object version; mutable logical IDs and metadata-only remote listings may be processed but never
authorize persisted results. If a hard `memory_budget` requires checkpointing, provide
`adapter.source_fingerprint(source)` or use versioned/content-addressed inputs.
Changes to sources, schema, strategy, segmentation, budgets, adapter/encoder code or state,
callbacks (including captured closure/default values), seed, or output spec reject resume. To
change a run, use a fresh `out_dir` or explicit `force=True`. One writer lease is held for the
destination until commit, preventing concurrent publishers from interleaving.

Custom registry extensions can be loaded from a job with `"plugins": ["my_package.plugin"]`,
or from Python with `load_plugins(...)`. Installed packages may expose a `kuration.plugins`
entry point. Set `"include_plugin_entry_points": false` when a reproducible job should resolve
only built-ins and its explicit plugin modules—even if ambient plugins were discovered earlier
in the process. For programmatic progress, pass `progress=events.append` (or any callback) to
`curate`, `run_experiment`, or `write_dataset`; pipelines use
`.options(progress=callback)`. For end-user display, use
`.options(progress="auto")`: it updates compact rows in JupyterLab and renders throttled terminal
output in scripts. `"notebook"`, `"terminal"`, and `"off"` make the selection explicit. Run CLIs
render the same events on interactive stderr and accept `--quiet`.

## 12 · Sub-track spans (keep the best *parts* of long sources)

```python
write_dataset("out/run1", ..., span_seconds=4.0)   # or span_steps=N; admits spans, not whole tracks
# admitted ids become (source, entity, span_index)
```

## 13 · Cloud I/O (GCS / S3 / Azure)

Any source, shard, or manifest path may be a cloud URL. Install `[s3]`, `[gcs]`, or `[azure]` to
get fsspec plus the matching backend. The lower-level `[cloud]` extra installs only fsspec for
managed environments that provide `s3fs`, `gcsfs`, or `adlfs` separately. Local needs nothing.

```python
write_dataset("gs://bucket/run1", schema=schema, adapter=adapter,
              sources=["gs://bucket/raw/a.parquet", "gs://bucket/raw/b.parquet"],
              strategy=strategy, segmenter=seg, spec=spec, train_hours=10)
CuratedDatasetAdapter("gs://bucket/run1")   # re-ingest straight from the bucket
```

---

## 14 · Extend at the seams

```python
from kuration import (register_adapter, register_metric, register_encoder, register_scorer,
                      register_batch_transform)

# (a) a new raw format -> Tick stream
class RosbagAdapter:
    def __init__(self, schema): self.schema = schema
    def ticks(self, uri):
        from kuration import Tick
        for step in read_rosbag(uri):
            yield Tick(time=step.t, entities={step.id: {"pos": step.xyz}})
register_adapter("rosbag", RosbagAdapter)

# (b) a metric the closed grammar can't express  (per-segment)
@register_metric("zmax")
def zmax(w): return float(w.numeric_field("pos")[..., 2].max())   # w: WindowBuffer

# (c) a derived output modality (e.g. decode a frame)  -> OutputEncoder
class FrameEncoder:
    def timestep(self, record, ctx):
        import numpy as np
        return {"frame": np.asarray(decode(record), dtype=np.uint8)}
register_encoder("frame", FrameEncoder)
# then in OutputSpec: "frame": {"dtype": "uint8", "shape": [H,W,3], "source": "encoder:frame"}

# (d) a learned / whole-pool / action-aware score  -> SegmentScorer (composes with coverage)
class MyScorer:
    needs_buffers = True            # ask for each segment's obs+action WindowBuffer
    def score(self, segments, ctx):
        import numpy as np
        return np.array([my_quality(w) for w in ctx["buffers"]])   # one value per segment
register_scorer("quality", MyScorer)
# then in the strategy: "scorers": [{"name": "quality", "as": "quality"}],
#                       "selection": {"weights": {"quality": 1.0}}

# (e) bounded device/media inference -> BatchTransform
@register_batch_transform("visual-quality", outputs=("visual_quality",),
                          inputs=("media:frame",),
                          media_requirements={"kind": "video-frame", "layout": "HWC"})
class VisualQuality:
    def __init__(self, device="cpu"):
        self.model = load_model(device)
    def transform_batch(self, batch, ctx):
        return {"visual_quality": self.model(batch["media:frame"])}
# pipeline use: pipeline.transform("visual-quality", batch_size=16, device="cuda")
```

For raw synchronized video, use `input("clips/*.mp4")` or attach it to ordinary telemetry with
`.join_video(mapping_or_callable, on="timestamp", tolerance=...)`. Output selected RGB tensors
with a modality whose source is `media:frame`; no custom encoder is required.

---

## 15 · Curate a CSV/JSON in one chain (pipeline, no explicit schema)

For the common case, skip `SchemaSpec`/`TimeseriesAdapter`/`load_strategy` entirely — `input()`
infers the schema and every operator validates against it. Full reference: [pipeline.md](pipeline.md).

```python
from kuration import input, iter_shards

(input("logs/*.csv", time="t", entity="agent")     # or a .json file, a list of paths, rows, ...
    .window(16)
    .measure("hp")
    .filter("hp", "<=", 50)                          # universal predicate — any selection method
    .measure("action_entropy", "entropy", of="action")
    .bin("action_entropy", bins=4)
    .sample("equal", n=5000)                         # alias for uniform-bins
    .options(on_error="raise")
    .output("out/run1", container="hdf5"))            # container inferred from the path otherwise

for arrays in iter_shards("out/run1", split="train"):
    arrays["hp"]        # (window, ) float32
```

`.sample(hours=...)` instead of `n=` routes `.output(...)` through the hour-budget fill; add
`.split(test=0.2)` for a leakage-free train/test split; `.describe()` shows the resolved
schema/strategy/segmenter/options before anything runs. See `examples/04_operator_pipeline.py`
for this exact chain end to end (count path + hours path + config.json's `"pipeline"`
provenance).

---

## 16 · Generate stats before selection, after selection, and from output

`stats()` is a terminal over the current immutable pipeline stage. Branch a pipeline to compare
the uncurated corpus with a count-selected set, then inspect the actual emitted artifact:

```python
from kuration import input, stats

source = (
    input("logs/*.parquet", time="t", entity="agent")
    .window(64)
    .measure("speed", "mean", of="speed")
)

uncurated = source.stats()
selected = source.sample("top-k", n=5000, weights={"speed": 1.0})
curated = selected.stats()

written = selected.output("out/curated")
emitted = written.stats()

print(uncurated.render())
curated.plot("speed")                # pip install "kuration-sdk[stats]"
curated.available()                  # plots, evidence modes, comparisons, media
curated.examples(metric="speed", compare="selected-vs-rejected")
emitted.save("out/curated-stats.html")

# Inspect an existing artifact without its original input pipeline:
reopened = stats("out/curated")
```

The input section is a bounded field preview; candidate/selection metrics use the complete
scored segment pool. Output statistics scan the emitted, padding-trimmed modalities with exact
count/mean/std/min/max aggregates. Histograms and percentiles retain at most
`max_values=100000` evenly sampled values per modality so tensor outputs stay bounded.
`content=False` limits an artifact report to manifest/config metadata, and `verify=True` checks
recorded shard integrity while scanning.

Count-selection results also support `pipeline.collect().stats()` without rescoring. Hours
admission is decided over output tracks, so `.sample(hours=...).stats()` is explicitly marked
as planned; call `.output(path).stats()` for admitted and emitted statistics.

Importing a maintained domain unlocks relevant sections and plots without changing the call:

```python
import kuration.gameplay as gameplay

report = gameplay.slam_reconstruct("slam-export/manifest.json").window(32).slam_quality().stats()
report.plot()                         # gameplay is the automatic default
evidence = report.examples(
    metric="slam_quality_score",
    compare="selected-vs-rejected",
    n=6,
    frames=3,
)

comparison = gameplay.compare_videos("raw.mp4", "curated.mp4",
                                     labels=("Before", "After"))
comparison.save("video-review.html", embed=False)
```

The rich video result displays synchronized players in Jupyter. Small local videos embed
automatically; larger files remain referenced to avoid unexpectedly inflating notebook output.
Use `comparison.plot(frames=4)` for a matched contact sheet (`[stats]`) and
`comparison.probe()` for codec, rate, duration, and frame metadata (`[video]`).
Live pipeline evidence reads only the requested segment ranges and decodes at most `frames`
evenly spaced images per example. `selected`, `rejected`, `strongest`, and `weakest` modes reuse
the scored count-selection pool; serialized reports retain bounded copies, while live reports
can rank any available metric exactly. Whole videos and large frame tensors are never embedded
in the JSON report.
The runnable
[gameplay stats notebook](https://github.com/aintropy-solutions/kuration-sdk/blob/main/examples/notebooks/gameplay_stats.ipynb)
uses synthetic trajectories and needs no model or GPU.

---

## 17 · Review input, selection, and artifact changes

```python
from kuration import compare_datasets, render_html_report, write_html_report

preview = pipeline.preview()       # bounded raw-input quality and examples
review = pipeline.review()         # full candidate-vs-selected count review
print(preview.render())
print(review.render())

comparison = compare_datasets("out/baseline", "out/candidate", verify=True)
write_html_report("review.html", preview, review, comparison,
                  title="Curation review")
```

The review includes selection configuration, metric shifts, bin/tag/group/source coverage,
split balance, and stable selected examples with priorities. For artifact-only automation, see
[Command-line interface](#19-command-line-interface).

For an hours budget, `review(path=...)` writes the dataset and reuses the writer's admission
decisions—there is no second scoring pass. It reports requested/candidate/admitted/emitted
seconds, utilization and underfill by split, decision counts, priority summaries, and bounded
admitted/rejected span examples.

Turn those reports into quality gates with the closed `.expect(...)` grammar:

```python
checked = (
    pipeline
    .expect(
        max_missing={"action": 0.01},
        min_sources=20,
        min_tag_coverage={"rare_event": 50},
        max_group_fraction=0.25,
        min_budget_utilization={"train": 0.98, "test": 0.95},
        max_skipped_sources=0,
    )
)
checked.preview()  # evaluates input-applicable gates
checked.review()   # evaluates count-selection gates; raises QualityGateError on failure
```

| expectation | applies to | contract |
|---|---|---|
| `max_missing={field: fraction}` | preview | sampled missing fraction at or below each maximum |
| `min_sources=n` | preview, selection, hours, artifact | source count at or above `n` |
| `min_tag_coverage={tag: n}` | count review | selected occurrence floor per tag |
| `max_group_fraction=f` | count review | largest selected group share at or below `f` |
| `min_budget_utilization=f` or `{split: f}` | hours review, artifact | emitted/requested duration floor |
| `max_budget_underfill_seconds=n` or `{split: n}` | hours review, artifact | emitted underfill ceiling |
| `max_skipped_sources=n` | all reports | skipped source ceiling |
| `require_modalities=[...]` | preview, artifact | every named field/modality exists |
| `require_verified=true` | artifact | every shard was decoded successfully |
| `min_shards=n` | artifact | emitted shard floor |

A combined policy is stage-aware: `preview()` evaluates input checks and `review()`
evaluates selection or hours checks. A typo or invalid threshold is always an immediate error.

Written artifacts use a JSON policy and a CI-friendly exit code (`0` pass, `2` gate failure):

```json
{
  "require_verified": true,
  "require_modalities": ["observation", "action", "_present"],
  "min_sources": 20,
  "min_shards": 100,
  "min_budget_utilization": 0.98,
  "max_skipped_sources": 0
}
```

Run this from the command line — see [Command-line interface](#19-command-line-interface) — or
drive it from Python.

---

## 18 · Troubleshooting — common loud failures

The SDK fails loudly by design: a raised error at load or preflight beats a silently wrong
dataset. The ones you're most likely to meet, and what they mean:

| error (grep-able fragment) | cause | fix |
|---|---|---|
| `unknown metric spec key(s)` / `unknown selection key(s)` | a typo'd key in a `define` metric or the `selection` block (e.g. `"fitler"`, `"tag_covrage"`) — it would have silently dropped a constraint | fix the spelling; valid metric keys are `source/reduce/map/filter/scope` |
| `references undefined metrics [...]` | axes/weights/target name a metric that isn't in `define`, registered, or a scorer output | define it, or check the name against `define` |
| `rows are not ordered by '<col>'` | source rows sorted entity-major (all of entity A, then entity B) — grouping by runs of equal time would double-count duration | sort rows by the time column first |
| `duplicate row for entity ... at time ...` | two rows for the same `(time, entity)` — the later one would silently win | dedupe upstream, or (parquet `"dataset"` mode) set `entity_column` so parts share ticks legitimately |
| `filter '...': field ... is a vector` | comparison filters take scalar fields only | filter on a scalar field, or `define` a scalar metric from the vector first |
| `segment pool is EMPTY` | every source was skipped (see the per-source stderr lines), or every source is shorter than the segment window | fix the first per-source error, or shrink `segmenter` window; rerun with `on_error="raise"` to see the full traceback |
| `missing mapped column(s)/key(s) [...]` (or run-time `KeyError('<col>')` per skipped source) | the adapter's field/time/entity/global map names a column/key the source doesn't have | fix the name; `kur-validate --check-sources` catches this at preflight for every built-in adapter |
| `resume=True but ... different output spec/run configuration` | the pinned run, source metadata, or component fingerprint changed | use a fresh `out_dir` for a changed run (or restore the original inputs and job) |
| `span_steps=... < shard_length=... with short_policy='drop'` | every admitted span would chunk to zero shards while consuming the hours budget | raise `span_steps`, lower `shard_length`, or use `short_policy: "keep"/"pad"` |
| `binning axis '...': N NaN feature value(s)` | a registered metric or scorer produced NaN on some window; NaN would poison the quantile edges | make the metric total (return 0.0 for degenerate windows), or drop those segments with a pre-pass filter |
| `dataset ... lacks modalities [...]` | mixing curated datasets with different modalities under a schema derived from the first | pass an explicit `schema=` covering only the shared modalities |
| `needs fsspec` / `needs h5py` / `needs pyarrow` / `needs zarr` | an optional capability was reached without its extra installed | install the named extra, e.g. `pip install "kuration-sdk[s3]"` |
| `write_dataset needs sample_hz ... for the hours budget` | pipeline `.sample(hours=...).output(...)` on a source whose schema has no `sample_hz` (inference found no numeric time column, or duplicate/no-gap timestamps) | pass `hz=` to `input()`, or `.options(sample_hz=...)` before `.output(...)` |
| `workers>1 is not supported for in-memory sources` | `.options(workers=2)` on a `Pipeline` built from in-memory rows/ticks/DataFrame — parallel workers rebuild adapters by name per process and can't see in-process data | keep `workers=1` for in-memory input, or write the data to files first and `input()` the paths |

Two habits keep runs debuggable: **`kur-validate job.json` before every CLI run** (all of the
load-time errors above surface in milliseconds, before any data is read), and
**`on_error="raise"` in tests/CI** so a skipped source can't quietly shrink the pool.

---

## 19 · Command-line interface

Everything above is expressed in Python, but the same operations are reachable from the shell —
useful for reproducible, no-code configuration and for CI. This section collects every `kur-*`
command in one place.

### Run it all from one JSON job

A single `job.json` document captures schema, adapter, sources, segmenter, strategy, and output —
enough to drive curation, experiments, reporting, or import without writing Python:

`job.json`:
```json
{
  "schema": {"name": "demo", "entity_axis": "agent", "sample_hz": 20,
             "fields": {"pos": {"kind": "vector", "dims": ["x","y","z"], "semantics": "position"}}},
  "adapter": {"type": "timeseries", "fields": {"pos": ["x","y","z"]}, "time_column": "t",
              "entity_column": "agent_id"},
  "sources": ["a.csv", "b.csv"],
  "segmenter": {"window": 4, "stride": 2, "unit": "seconds"},
  "strategy": {"define": {"speed": {"source": "agent.pos", "map": "speed", "reduce": "mean"}},
               "selection": {"method": "top-k", "weights": {"speed": 1.0}}},
  "output": {"shard_length": 64, "container": "npz",
             "modalities": {"pos": {"dtype": "float32", "shape": [3], "source": "field:pos"}}},
  "train_hours": 10, "test_hours": 1, "workers": 8, "out": "out/run1"
}
```
```bash
kur-curate     job.json                  # sources -> hours-budget shards
kur-experiment job.json --out cmp.json   # add "arms": [...] + "target_size" to compare strategies
kur-report     job.json --out rep.json   # profile metric distributions (no selection)
kur-import     SOURCE --out out/raw       # detected-adapter conversion, with no selection
```

### Artifact-only automation

Diff two written datasets or inspect one, without re-running the pipeline that produced them:

```bash
kur-diff out/baseline out/candidate --verify --json
kur-diff out/baseline out/candidate --html review.html
kur-inspect out/candidate --html inspection.html
```

### Written-artifact quality gates

Evaluate a written artifact's quality gate against a JSON policy (see recipe 17 for the policy
document itself) with a CI-friendly exit code (`0` pass, `2` gate failure):

```bash
kur-check out/candidate --policy quality.json --json
kur-check out/candidate --policy quality.json --html quality.html
```
