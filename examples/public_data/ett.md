# Worked example — Electricity Transformer (ETT), pipeline API only

This example curates real hourly transformer telemetry into day-length training windows using
nothing but `input(...)` and chained operators: no `SchemaSpec`, no adapter class, no strategy
dictionary. It is the counterpart to the adapter-based public-data examples — those exist
because their formats need custom parsing, while this one shows how far the plain path goes
when the data is already a table.

## The data

[`zhouhaoyi/ETDataset`](https://github.com/zhouhaoyi/ETDataset) publishes the Electricity
Transformer Dataset used by the Informer paper. This example reads `ETT-small/ETTh1.csv`:
about 17,420 hourly readings (two years) of six power-load columns plus `OT`, the oil
temperature.

The repository's `LICENSE` file is Creative Commons **Attribution-NoDerivatives 4.0
International (CC BY-ND 4.0)**. Downloading it to curate locally is ordinary use, but the
NoDerivatives term constrains publishing modified copies — so this example downloads at run
time and never vendors, reshapes, or redistributes the file. Confirm the current terms on the
repository before sharing any derived artifact.

Cite the dataset's paper if you publish results:

> Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series
> Forecasting*, AAAI 2021.

## Run it

```bash
python examples/public_data/ett.py --download --out ett-out   # ~2.5 MB
python examples/public_data/ett.py data/ETTh1.csv --out ett-out
python examples/public_data/ett.py --demo --out ett-out       # offline, synthetic
```

`--demo` generates a tiny file with the upstream column layout and no upstream content, so the
structure of the walkthrough runs with no network. Downloading is not part of CI.

## What the pipeline says

```python
pipeline = (
    input("data/ETTh1.csv", time="date", hz=1 / 3600)
    .window(24)                          # one day; stride defaults to the window
    .measure("load", "mean", of="HUFL")
    .measure("temp_swing", "std", of="OT")
    .bin("load", bins=6)
    .bin("temp_swing", bins=4)
    .sample("equal", n=60)
)
```

Two things are worth pointing at:

- **`time=` and `hz=` are the only hints inference needs.** `date` is not one of the column
  names `sniff` recognizes automatically (`t`, `time`, `timestamp`, `ts`, `step`, `frame`,
  `tick`), and its values are datetime strings rather than numbers, so the sample rate cannot
  be derived from them. Naming both keeps everything else inferred.
- **`hz` is per second.** Hourly readings are `1/3600`, which makes a 24-step window exactly
  one day. Any hours budget is then denominated in hours of *recorded* time — with hourly
  data, a meaningful budget is hundreds of hours, not single digits.

On the real file this selects 60 days out of 725 candidates, spread across 24 occupied bins
rather than concentrated in the dense middle of the load distribution.

## The same walkthrough from the shell

Every step above has a command-line equivalent, and no Python is involved:

```bash
curl -L -o ETTh1.csv \
  https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv

kur infer ETTh1.csv --time date --hz 0.000277778 --out schema.json
kur init  ETTh1.csv --out job.json --window 24 --dataset-out curated \
          --time date --hz 0.000277778 --train-hours 720
kur preview  job.json
kur validate job.json
kur curate   job.json --review
kur stats    curated --html stats.html
kur export   curated --out arrays/ --format parquet
```

`kur init` writes a complete job and resolves it before saving, so the file it produces is one
`kur validate` accepts. It picks the first scalar column as a binning axis because it cannot
know which metric matters — edit `strategy` before trusting a run.

Note the budget: `--train-hours 720` is 30 days of recorded time. The scaffold sets
`span_steps` to the window length so admission happens one window at a time; the run then
reports something like `31/726 items admitted` at 103% utilization, overshooting only by the
granularity of a single window.

## Where it goes next

- Swap `.sample("equal", n=60)` for `.sample("equal", hours=720)` to budget by recorded time
  instead of window count.
- Add `.filter("load", ">=", 5.0)` to drop idle days before binning.
- Add `.split(test=0.2)` once you have more than one recording; with a single source there is
  nothing to hold out.
- `docs/pipeline.md` documents every operator; `docs/jobs.md` documents the CLI equivalents.
