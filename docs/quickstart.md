# Quick start — curate and inspect a dataset

In this first run, kuration will examine six small sensor sessions, score 42 candidate windows,
select 12 windows that cover different activity and fault patterns, write training-ready NumPy
shards, and read one shard back. The example is deterministic, needs no download, and uses only
the SDK's NumPy core.

Expected time: under ten minutes including installation, and normally under one minute once the
repository is available locally.

## 1. Install

Kuration currently installs from its repository. From the repository root, create or activate a
Python 3.11 or newer environment, then run:

```bash
python -m pip install -e .
```

The quick start needs no optional extras.

## 2. Run the pipeline

```bash
python examples/quickstart.py --out quickstart-out
```

You should see output shaped like this:

```text
input: 6 sessions, 288 rows
curation: 42 candidate windows -> 12 selected
output: 12 shards in quickstart-out
first shard: split=test, modalities=['_present', 'fault', 'signal', 'temperature'], signal_shape=(12,)
reproducible config: quickstart-out/config.json
```

The exact first-shard split is deterministic for this example. The paths will reflect the value
passed to `--out`.

## 3. What the pipeline did

The runnable source is [`examples/quickstart.py`](../examples/quickstart.py). Its central pipeline
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
8. `output(...)` writes NPZ shards plus manifests and the fully resolved configuration.

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
  fluent operator chain.
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

Next:

- Learn every fluent operator in the [pipeline guide](pipeline.md).
- Find task-oriented variations in the [usage cookbook](usage.md).
- Use explicit schemas and adapters in the [end-to-end classic API example](overview.md).
- Look up exact signatures in the [Python API reference](api.md).
