"""Curate real wearable-sensor recordings with nothing but the pipeline API.

Five people wore four tags (ankles, belt, chest) and repeated the same scenario five times.
That shape exercises the parts of the SDK a single-entity table cannot:

    input("*.csv", entity="tag_id")   # 25 recordings, 4 entities each
      .tag("activity", min_frac=...)  # reserve room for rare labels (falling)
      .group(person).split(test=0.2)  # whole recordings held out, stratified by person
      .sample("equal", n=...)

Three ways to run it:

    python examples/public_data/person_activity.py --download --out activity-out
    python examples/public_data/person_activity.py 'data/person_activity/*.csv' --out activity-out
    python examples/public_data/person_activity.py --demo --out activity-out

Provenance, licensing, and the equivalent CLI session: see person_activity.md.
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import random
import tempfile
import zipfile
from pathlib import Path

from kuration import input, iter_shards

SOURCE_URL = ("https://archive.ics.uci.edu/static/public/196/"
              "localization+data+for+person+activity.zip")
COLUMNS = ["sequence", "tag_id", "timestamp", "datetime", "x", "y", "z", "activity"]
TICK_HZ = 37.0        # four tags reporting at ~9.25 Hz each
TICKS_PER_SECOND = 1e7  # upstream `timestamp` is a .NET tick count (100ns units)
ACTIVITIES = ["walking", "sitting", "lying", "standing up from lying", "falling"]


def person_of(source: str) -> str:
    """Group key: the leading letter of a recording name (A01 -> person A)."""
    return Path(source).stem[0]


# ---- getting the data (not the SDK) -----------------------------------------
def download(dest_dir: Path) -> Path:
    """Fetch the upstream archive and split it into one CSV per recording.

    The upstream file is one headerless table holding all 25 recordings. Splitting it by
    recording is ordinary data preparation, not pipeline logic: it gives `input()` real
    multi-source input so whole people can be held out of the test split.
    """
    import ssl
    from urllib.error import URLError
    from urllib.request import urlopen

    existing = sorted(dest_dir.glob("*.csv"))
    if existing:
        print(f"using {len(existing)} cached recordings in {dest_dir}")
        return dest_dir
    context = None
    try:                       # some Python installs ship no usable root certificates
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    print(f"downloading {SOURCE_URL}")
    try:
        with urlopen(SOURCE_URL, timeout=300, context=context) as response:
            payload = response.read()
    except URLError as error:
        raise SystemExit(
            f"could not download {SOURCE_URL}: {error}\n"
            f"Download it manually, unzip ConfLongDemo_JSI.txt, and see "
            f"person_activity.md for the two-line split step.") from error

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".txt"))
        raw = archive.read(name).decode("utf-8", "replace")

    recordings: dict[str, list[list[str]]] = {}
    for row in csv.reader(io.StringIO(raw)):
        if len(row) == len(COLUMNS):
            recordings.setdefault(row[0], []).append(row)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for sequence, rows in recordings.items():
        rows.sort(key=lambda r: int(r[2]))               # tags interleave; order by time
        # Upstream `timestamp` is a .NET tick count (100ns units, TICKS_PER_SECOND below) --
        # not seconds. Declaring hz=TICK_HZ (readings/second) against a column in 100ns units
        # is a ~1e7x unit mismatch the SDK's own declared-hz-vs-timestamp check correctly
        # flags. Rewrite it to real, recording-relative seconds here, as ordinary data prep,
        # so the time column and hz agree like any other seconds-denominated time column.
        origin = int(rows[0][2])
        for row in rows:
            row[2] = f"{(int(row[2]) - origin) / TICKS_PER_SECOND:.7f}"
        with (dest_dir / f"{sequence}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
    print(f"wrote {len(recordings)} recordings to {dest_dir}")
    return dest_dir


def write_demo(dest_dir: Path, *, recordings: int = 6, seconds: int = 40) -> Path:
    """Write tiny recordings with the upstream column layout so the demo runs offline.

    Synthetic: four tags orbiting a walking body, with occasional falls. It copies no
    upstream data.
    """
    rng = random.Random(0)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tags = ["010-000-024-033", "020-000-033-111", "020-000-032-221", "010-000-030-096"]
    for index in range(recordings):
        person = "ABCDE"[index % 5]
        name = f"{person}{index // 5 + 1:02d}"
        with (dest_dir / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            tick = 0
            for second in range(seconds):
                activity = "falling" if second % 17 == 16 else ACTIVITIES[second % 4]
                height = 0.2 if activity in ("lying", "falling") else 1.0
                for step in range(9):
                    phase = 2 * math.pi * step / 9
                    for offset, tag in enumerate(tags):
                        tick += 1
                        writer.writerow([
                            # Recording-relative seconds, same 270296-tick (~0.027s) real
                            # per-row cadence as the upstream file -- not raw ticks, so
                            # hz=TICK_HZ agrees with the time column (see download()).
                            name, tag, f"{tick * 270296 / TICKS_PER_SECOND:.7f}",
                            f"27.05.2009 14:{second // 60:02d}:{second % 60:02d}:000",
                            round(3 + math.cos(phase + offset) + rng.gauss(0, 0.05), 4),
                            round(2 + math.sin(phase + offset) + rng.gauss(0, 0.05), 4),
                            round(height + 0.1 * offset + rng.gauss(0, 0.03), 4),
                            activity,
                        ])
    print(f"wrote {recordings} synthetic recordings to {dest_dir}")
    return dest_dir


# ---- the walkthrough --------------------------------------------------------
def build(sources: str, *, n: int):
    """Return the curation pipeline. This is the whole SDK-facing surface."""
    return (
        input(sources, time="timestamp", entity="tag_id", hz=TICK_HZ)
        # Window in real time rather than row counts — the schema knows the sample rate.
        .window(2, unit="seconds")
        # Vertical movement separates falling and standing from lying still; lateral spread
        # separates walking from sitting. Both reduce one entity column over the window.
        .measure("motion", "std", of="z")
        .measure("sway", "std", of="x")
        # Rare labels would vanish under pure distribution matching: reserve at least 2% of
        # the selection for every activity that appears, including 'falling'.
        .tag("activity", min_frac=0.02)
        .bin("motion", bins=6)
        # Splits are decided per source, never per window: a whole recording lands in one
        # split, so two windows of the same take cannot straddle the boundary. `group` is a
        # *stratification* key, not a holdout key — the test share is spread proportionally
        # across people, so a person with several recordings appears in both splits. For a
        # strict person-level holdout, make each person a single source.
        .group(person_of)
        .split(test=0.2)
        .sample("equal", n=n)
    )


def coverage_effect(sources: str, *, n: int) -> None:
    """Show what `min_frac` buys by running the same chain with and without the floor."""
    from collections import Counter

    def activities(min_frac):
        result = (input(sources, time="timestamp", entity="tag_id", hz=TICK_HZ)
                  .window(2, unit="seconds")
                  .measure("motion", "std", of="z")
                  .tag("activity", min_frac=min_frac)
                  .bin("motion", bins=6)
                  .sample("equal", n=n)).collect()
        selected = Counter(tag for s in result.selected for tag in (s.tags or []))
        pool = {tag for s in result.all_segments for tag in (s.tags or [])}
        return selected, pool

    without, pool = activities(None)
    with_floor, _ = activities(0.02)
    missing = sorted(tag.removeprefix("activity:") for tag in pool - set(without))
    print(f"\nactivities kept: {len(without)}/{len(pool)} without a floor, "
          f"{len(with_floor)}/{len(pool)} with min_frac=0.02")
    if missing:
        print(f"  the floor is what guarantees a share for: {', '.join(missing)}")


def run(sources: str, out_dir: str, *, n: int = 200) -> dict:
    pipeline = build(sources, n=n)

    schema = pipeline.describe()["schema"]
    print(f"\n{len(pipeline.sources)} recordings | entity axis {schema['entity_axis']!r} | "
          f"fields: {', '.join(schema['fields'])}")
    print(pipeline.preview(max_ticks_per_source=4000).render())

    print()
    print(pipeline.review().render())
    coverage_effect(sources, n=n)

    # Positions only: the categorical columns describe the window, they are not tensors.
    result = pipeline.output(out_dir, include=["x", "y", "z"], shard_length=74)
    print(f"\nwrote {len(result['shards'])} shards to {out_dir}")

    # Show the split the way it is actually decided: per recording, stratified by person.
    per_split: dict[str, set] = {}
    for entry in result["shards"]:
        per_split.setdefault(entry["split"], set()).add(entry["source_id"])
    for split in sorted(per_split):
        recordings = sorted(per_split[split])
        people = sorted({person_of(name) for name in recordings})
        print(f"{split}: {len(recordings)} recording(s) {recordings} | people {people}")
    assert not (per_split.get("train", set()) & per_split.get("test", set())), \
        "a recording must never appear in both splits"

    arrays = next(iter(iter_shards(out_dir)))
    print("one shard:", {name: tuple(array.shape) for name, array in arrays.items()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="?",
                        help="Glob of prepared recordings, e.g. 'data/person_activity/*.csv'.")
    parser.add_argument("--download", action="store_true",
                        help="Fetch the real archive (~2 MB) and split it per recording.")
    parser.add_argument("--demo", action="store_true",
                        help="Generate tiny synthetic recordings and run offline.")
    parser.add_argument("--out", default="activity-out", help="Output dataset directory.")
    parser.add_argument("--n", type=int, default=200, help="Windows to select.")
    args = parser.parse_args()

    if sum(bool(x) for x in (args.sources, args.download, args.demo)) != 1:
        parser.error("pass exactly one of: a source glob, --download, or --demo")
    if args.demo:
        # Demo data must live outside --out: output() rejects any pipeline whose
        # output directory overlaps one of its input sources.
        with tempfile.TemporaryDirectory(prefix="kuration_person_activity_demo_") as directory:
            sources = f"{write_demo(Path(directory))}/*.csv"
            run(sources, args.out, n=args.n)
        return
    if args.download:
        sources = f"{download(Path('data/person_activity'))}/*.csv"
    else:
        sources = args.sources

    run(sources, args.out, n=args.n)


if __name__ == "__main__":
    main()
