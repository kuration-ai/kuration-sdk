# Usage cookbook

Task-oriented recipes spanning what the SDK can do. Each is a minimal, runnable snippet; see the
per-layer references ([schema](schema.md), [dsl](dsl.md), [adapters](adapters.md),
[strategies](strategies.md), [output](output.md)) for full options, and [overview](overview.md)
for the one-shot end-to-end example.

Throughout, assume a schema and an adapter:

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
Sources may be local paths or cloud URLs (`gs://…`, `s3://…`) — see recipe 14. Bespoke formats:
recipe 12.

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
# -> out/run1/{train,test}/<src>_<entity>_shardNNN.npz + dataset_manifest.json + config.json
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
                 "define": {"speed": {"source": "e.pos0", "reduce": "mean"}},  # modalities -> fields
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
              on_error="skip",                    # log + skip a bad source (default); "raise" to fail fast
              resume=True)                         # skip re-encoding shards that already exist
```

## 12 · Sub-track spans (keep the best *parts* of long sources)

```python
write_dataset("out/run1", ..., span_seconds=4.0)   # or span_steps=N; admits spans, not whole tracks
# admitted ids become (source, entity, span_index)
```

## 13 · Cloud I/O (GCS / S3 / Azure)

Any source, shard, or manifest path may be a cloud URL — needs the `[cloud]` extra + a backend
(`gcsfs` / `s3fs` / `adlfs`). Local needs nothing.

```python
write_dataset("gs://bucket/run1", schema=schema, adapter=adapter,
              sources=["gs://bucket/raw/a.parquet", "gs://bucket/raw/b.parquet"],
              strategy=strategy, segmenter=seg, spec=spec, train_hours=10)
CuratedDatasetAdapter("gs://bucket/run1")   # re-ingest straight from the bucket
```

---

## 14 · Extend at the seams

```python
from kuration import register_adapter, register_metric, register_encoder, register_scorer

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
```

---

## 15 · Run it all from one JSON job (CLIs)

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
```
