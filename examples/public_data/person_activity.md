# Worked example — wearable person activity, pipeline API only

This example curates real wearable-sensor recordings using only `input(...)` and chained
operators. Where [`ett.md`](ett.md) is the smallest possible pipeline walkthrough,
this one has the shape that actually exercises the SDK: many sources, several entities per
source, and a categorical label whose rare values are the interesting ones.

## The data

[UCI ML Repository #196 — *Localization Data for Person
Activity*](https://archive.ics.uci.edu/dataset/196/localization+data+for+person+activity).
Five people each wore four tags (left ankle, right ankle, belt, chest) and performed the same
scenario five times, giving 25 recordings and about 164,860 rows. Each row is one tag reading:
an `x`/`y`/`z` position and one of eleven activity labels, from `walking` and `lying` down to
`falling` and `on all fours`.

Creators: Vedrana Vidulin, Mitja Luštrek, Boštjan Kaluža, Rok Piltaver, Jana Krivec. UCI
presents its datasets under Creative Commons terms; check the dataset page for the licence and
citation that apply to your use before redistributing anything derived from it.

Combined tag rate is about 37 Hz (roughly 9.25 Hz per tag), which is what `hz=37.0` records.

## Run it

```bash
python examples/public_data/person_activity.py --download --out activity-out
python examples/public_data/person_activity.py 'data/person_activity/*.csv' --out activity-out
python examples/public_data/person_activity.py --demo --out activity-out
```

`--download` fetches the archive (~2 MB) and performs two preparation steps: the upstream file
is a single headerless table holding all 25 recordings, so it is given a header row and split
into one CSV per recording; and its `timestamp` column — a raw .NET tick count (100ns units),
not seconds — is rewritten to real, recording-relative seconds, so it agrees with the `hz=37.0`
declared below instead of being off by a factor of 1e7 (the SDK's own declared-hz-vs-timestamp
check would otherwise flag this). Both are ordinary data preparation rather than pipeline logic,
and together they turn the corpus into real multi-source, seconds-denominated input. To do it by
hand:

```bash
unzip localization+data+for+person+activity.zip           # -> ConfLongDemo_JSI.txt
python - <<'PY'
import csv, collections, pathlib
cols = ["sequence","tag_id","timestamp","datetime","x","y","z","activity"]
rows = collections.defaultdict(list)
for row in csv.reader(open("ConfLongDemo_JSI.txt")):
    if len(row) == 8: rows[row[0]].append(row)
out = pathlib.Path("data/person_activity"); out.mkdir(parents=True, exist_ok=True)
for name, group in rows.items():
    group.sort(key=lambda r: int(r[2]))
    origin = int(group[0][2])
    for row in group:
        row[2] = f"{(int(row[2]) - origin) / 1e7:.7f}"    # ticks (100ns) -> relative seconds
    with (out / f"{name}.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols); w.writerows(group)
PY
```

## What the pipeline says

```python
from pathlib import Path


def person_of(source: str) -> str:
    return Path(source).stem[0]


pipeline = (
    input("data/person_activity/*.csv", time="timestamp", entity="tag_id", hz=37.0)
    .window(2, unit="seconds")
    .measure("motion", "std", of="z")
    .measure("sway", "std", of="x")
    .tag("activity", min_frac=0.02)
    .bin("motion", bins=6)
    .group(person_of)
    .split(test=0.2)
    .sample("equal", n=200)
)
```

Four things this shape demonstrates that a single-column table cannot:

- **A glob is multi-source input.** `input("*.csv")` sniffs one adapter over all 25 files;
  each recording is a source, which is the unit splitting and error skipping work on.
- **`entity=` sets the entity axis.** The four tags become the entity dimension, so
  `measure(..., of="z")` reduces over a `(timesteps, tags)` window rather than a flat column.
  Metric sources are namespaced by that axis (`tag_id.z`).
- **`window(2, unit="seconds")` is real time.** Because the schema carries a sample rate, the
  window can be stated in seconds instead of a row count.
- **`tag(..., min_frac=)` protects rare labels.** This is the headline: quantile binning
  matches the *distribution*, which is exactly what discards rare events. On the real corpus
  the example prints

  ```
  activities kept: 9/11 without a floor, 11/11 with min_frac=0.02
    the floor is what guarantees a share for: sitting down, standing up from sitting
  ```

  Note what this does and does not claim. `falling` survives either way here — there are
  enough windows of it. The floor is what keeps the two briefest transitions from being
  dropped entirely.

### How the split actually works

`split(test=0.2)` is decided **per source, never per window**: a whole recording lands in one
split, so two windows from the same take can never straddle the boundary.

`group(...)` is a **stratification** key, not a holdout key. The test share is spread
proportionally across groups, so on this corpus each of the five people contributes exactly one
of their five recordings to the test split, and every person appears in both splits:

```
test:  5 recording(s) [A03, B01, C02, D03, E02] | people ['A','B','C','D','E']
train: 20 recording(s) [A01, A02, A04, A05, B02, ...] | people ['A','B','C','D','E']
```

That is the right default for "hold out comparable data from every subject". If you instead
need a strict subject-level holdout — no person in both splits — make each person a single
source, because leakage-freedom is guaranteed at the source level.

## The same walkthrough from the shell

```bash
kur infer   data/person_activity/A01.csv --time timestamp --entity tag_id --hz 37
kur init    data/person_activity/*.csv --out job.json \
            --window 74 --time timestamp --entity tag_id --hz 37 \
            --dataset-out curated --train-hours 0.5
kur preview job.json
kur curate  job.json --review
kur stats   curated --html stats.html
```

The scaffold gets you a valid job, not the job above: `kur init` chooses a single binning axis
and no tag coverage, because it cannot know that `activity` is the column whose rare values
matter. Add the `tags` block and `selection.tag_coverage` to `job.json` — or express it in
Python with `.tag("activity", min_frac=0.02)`.

## Where it goes next

- `.filter("motion", ">=", 0.05)` drops windows where the subject barely moved.
- `.measure("activity_entropy", "entropy", of="activity")` turns the label column into a
  selection axis of its own.
- `.sample("equal", hours=1.0)` budgets by recorded time instead of window count.
- `docs/strategies.md` covers coverage, group caps, and the other selection knobs.
