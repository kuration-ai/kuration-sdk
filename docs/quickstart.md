# Quick start — curate and inspect a dataset

In this first run, kuration will examine six small sensor sessions, score 42 candidate windows,
select 12 windows that cover different activity and fault patterns, write training-ready NumPy
shards, and read one shard back. The example is deterministic, needs no download, and uses only
the SDK's NumPy core.

Expected time: under ten minutes including installation, and normally under one minute once the
package is installed.

## 1. Install

Install the NumPy-only core in a Python 3.11 or newer environment:

```bash
pip install kuration-sdk
```

The quick start needs no optional extras.

## 2. Run the pipeline

```bash
curl -O https://raw.githubusercontent.com/aintropy-solutions/kuration-sdk/main/examples/quickstart.py
python quickstart.py --out quickstart-out
```

You should see output shaped like this:

```text
preflight:
source: RowsAdapter (6 source(s))
schema: inferred
window: 12 steps (stride 6)
selection: uniform-bins (count=12)
execution: local; errors=raise; total-pool materialization=none
optional extras: none (NumPy core)
output: npz; shard_length=12; modalities=['fault', 'signal', 'temperature']
estimate: about 42 candidate windows (bounded source sample)
estimate: about 12 output shards; before short-policy drops and per-source split constraints
input: 6 sessions, 288 rows
curation: 42 candidate windows -> 12 selected
output: 12 shards in quickstart-out
artifact: 12 shards; splits=['test', 'train']; metadata valid
first shard: split=test, modalities=['_present', 'fault', 'signal', 'temperature'], signal_shape=(12,)
reproducible config: quickstart-out/config.json
```

The exact first-shard split is deterministic for this example. The paths will reflect the value
passed to `--out`.

## 3. What the pipeline did

The runnable source is `examples/quickstart.py`. Its central pipeline
is:

```python
pipeline = (
    input(sessions, time="t", entity="device", hz=1)
    .window(12, stride=6)
    .measure("activity", "std", of="signal")
    .measure("fault_rate", "rate", of="fault")
    .bin("activity", bins=3)
    .bin("fault_rate", bins=2)
    .split(test=1 / 3)
    .sample("equal", n=12)
    .options(mode="strict", seed=7)
)

print(pipeline.output("quickstart-out", container="npz", shard_length=12,
                      dry_run=True, inspect_sources=True).render())
result = pipeline.output("quickstart-out", container="npz", shard_length=12)
```

Reading from top to bottom:

1. `input(...)` infers a schema from ordinary row dictionaries. The hints identify the time and
   entity columns and establish a one-hertz sampling rate.
2. `window(...)` turns each session into overlapping 12-step candidate windows.
3. `measure(...)` computes signal variability and the proportion of non-zero fault values in
   each window.
4. `bin(...)` places windows into activity/fault bins.
5. `split(...)` assigns whole source sessions to train or test, avoiding leakage between windows
   from the same session.
6. `sample("equal", n=12)` spreads the selection across occupied bins instead of simply taking
   the first twelve windows.
7. Strict mode makes source errors and ambiguous output choices fail immediately. The seed makes
   selection and splitting repeatable.
8. The output dry run resolves the real output choice and optional extras without writing. With
   `inspect_sources=True`, it samples sources to estimate candidate windows and shard count.
9. `output(...)` writes NPZ shards plus manifests and the fully resolved configuration.

The script constructs its small sessions in memory so the first run has no network or data-file
prerequisite. `input(...)` also accepts CSV, JSONL, Parquet, NPZ, HDF5, Zarr, Hugging Face, and
custom adapters.

## 4. Inspect the result

The output directory contains:

```text
quickstart-out/
├── config.json
├── dataset_manifest.json
├── train/
│   └── *.npz
└── test/
    └── *.npz
```

- `config.json` records the inferred schema, resolved strategy, output declaration, seed, and
  operator chain.
- `dataset_manifest.json` indexes every shard and records its source, entity, split, size, shape,
  and dtype.
- `train/` and `test/` contain the arrays consumed by training code.

Read shards with the NumPy-only API:

```python
from kuration import iter_shards

for arrays in iter_shards("quickstart-out", split="train"):
    print(arrays["signal"].shape)  # (12,)
    print(arrays["_present"].all())
    break
```

The `_present` mask distinguishes real steps from padding. This example has fixed twelve-step
windows and therefore needs no padding, but consumers can use the same field for variable-length
data.

## 5. Apply it to your data

For a CSV with `t`, `device`, `signal`, `temperature`, and `fault` columns, replace the in-memory
sessions with a path or glob:

```python
pipeline = input("logs/*.csv", time="t", entity="device", hz=1)
```

Keep the remaining operators, then adjust the fields, window size, metrics, and target count to
match the curation question.

## 6. Choose count or hours

| need | use | result |
|---|---|---|
| A fixed number of representative windows, evaluation samples, or a small experiment | `.sample(..., n=5000)` | Up to `n` selected segments for ranking methods; `filter` intentionally returns every match; each selected entity-window becomes output shards |
| A training dataset with a storage/time budget | `.sample(..., hours=100, test_hours=5)` | Highest-priority output spans fill separate train/test duration budgets; hours output supports safe resume |

Start with count selection while designing metrics because it is quick to inspect. Switch to
hours when emitted training volume—not the number of candidate windows—is the real constraint.
Keep `.options(mode="strict", seed=...)` during development, and run `explain()` again after
changing the selection method or execution options.

Use `pipeline.preview()` to inspect bounded raw-input quality before scoring and
`pipeline.review()` to compare the complete candidate pool with a count selection. Both return
JSON-safe typed results with concise `.render()` output; combine them into a shareable page with
`write_html_report("review.html", preview, review)`.

## Command-line equivalent

Every step above has a `kur-*` CLI counterpart for scripting and CI, once you have a job spec
(see the [job and CLI reference](jobs.md)). For a quick summary of the artifact this quick start
just produced, then a full integrity check:

```bash
kur-inspect quickstart-out
kur-inspect quickstart-out --verify
```

Next:

- Learn every operator in the [pipeline guide](pipeline.md).
- Package a repeatable selection decision in [building operators](operators.md).
- Find task-oriented variations in the [usage cookbook](usage.md).
- Use explicit schemas and adapters in the [end-to-end classic API example](overview.md).
- Look up exact signatures in the [Python API reference](api.md).
