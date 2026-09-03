"""Curate real electricity-transformer load curves with nothing but the pipeline API.

One real CSV, one chain of operators, no schema, no adapter, no strategy dict:

    input(csv, time="date", hz=1/3600)     # infer the schema from a sample
      .window(24)                          # one day of hourly readings
      .measure(...).bin(...).sample(...)   # what matters, how to spread, how many
      .output(out_dir)                     # shards + manifests + config.json

Three ways to run it:

    python examples/public_data/ett.py --download --out ett-out   # fetch the real data
    python examples/public_data/ett.py data/ETTh1.csv --out ett-out
    python examples/public_data/ett.py --demo --out ett-out       # offline, no network

Provenance, licensing, and the equivalent CLI session: see ett.md.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import tempfile
from pathlib import Path

from kuration import input, iter_shards

SOURCE_URL = ("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/"
              "ETT-small/ETTh1.csv")
HOURLY = 1 / 3600.0          # readings are one hour apart, so hz is per-second


# ---- getting the data (not the SDK) -----------------------------------------
def fetch(url: str, dest: Path) -> Path:
    """Download `url` to `dest` once, or reuse the cached copy."""
    import ssl
    from urllib.error import URLError
    from urllib.request import urlopen

    if dest.exists():
        print(f"using cached {dest}")
        return dest
    context = None
    try:                       # some Python installs ship no usable root certificates
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    try:
        with urlopen(url, timeout=180, context=context) as response:
            dest.write_bytes(response.read())
    except URLError as error:
        raise SystemExit(
            f"could not download {url}: {error}\n"
            f"Fetch it manually and pass the path instead:\n"
            f"  curl -L -o {dest} {url}\n"
            f"  python {Path(__file__).name} {dest} --out <out-dir>") from error
    print(f"saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def download(dest: Path) -> Path:
    """Fetch the upstream hourly recording once and cache it at `dest`."""
    return fetch(SOURCE_URL, dest)


def write_demo(dest: Path, *, days: int = 40) -> Path:
    """Write a tiny file with the upstream column layout so the demo runs offline.

    Synthetic: a daily load cycle whose amplitude and baseline vary day to day, so windows
    differ from each other the way real days do and the bin axes have something to spread
    across. It copies no upstream data.
    """
    rng = random.Random(0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"])
        for day in range(days):
            busy = rng.uniform(0.4, 1.8)             # how hard this day ran
            base = rng.uniform(4.0, 9.0)
            for hour in range(24):
                daily = math.sin(2 * math.pi * hour / 24)
                jitter = rng.gauss(0.0, 0.2)
                writer.writerow([
                    f"2016-07-{1 + day:02d} {hour:02d}:00:00",
                    round(base + 3 * busy * daily + jitter, 3),
                    round(2 + busy * daily + jitter, 3),
                    round(2 + 2 * busy * daily, 3),
                    round(0.5 + busy * daily, 3),
                    round(4 + daily, 3), round(1.4 + 0.3 * daily, 3),
                    round(20 + 10 * busy + 4 * daily + jitter, 3),      # oil temperature
                ])
    return dest


# ---- the walkthrough --------------------------------------------------------
def build(source: str, *, n: int):
    """Return the curation pipeline. This is the whole SDK-facing surface."""
    return (
        input(source, time="date", hz=HOURLY)
        # A window is one day; the default stride equals the window, so days do not overlap.
        .window(24)
        # Two axes worth spreading across: how loaded the transformer was, and how much
        # its oil temperature moved. `of=` names a column; the reduce collapses the window.
        .measure("load", "mean", of="HUFL")
        .measure("temp_swing", "std", of="OT")
        # Quantile bins over each axis; selection then spreads picks across occupied cells
        # instead of over-sampling the dense middle of the distribution.
        .bin("load", bins=6)
        .bin("temp_swing", bins=4)
        .sample("equal", n=n)
    )


def run(source: str, out_dir: str, *, n: int = 60) -> dict:
    pipeline = build(source, n=n)

    # Look before running: what did inference decide, and what is actually in the file?
    schema = pipeline.describe()["schema"]
    print(f"\ninferred {len(schema['fields'])} fields at {schema['sample_hz']:.6g} Hz: "
          f"{', '.join(schema['fields'])}")
    print(pipeline.preview(max_ticks_per_source=2000).render())

    # Candidate-vs-selected, before committing to an output directory.
    print()
    print(pipeline.review().render())

    result = pipeline.output(out_dir, shard_length=24)
    print(f"\nwrote {len(result['shards'])} shards to {out_dir}")

    # Read one back as ordinary numpy.
    arrays = next(iter(iter_shards(out_dir)))
    print("one shard:", {name: tuple(array.shape) for name, array in arrays.items()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", nargs="?", help="Path to an ETT hourly CSV.")
    parser.add_argument("--download", action="store_true",
                        help="Fetch the real recording (~2.5 MB) into data/ett/.")
    parser.add_argument("--demo", action="store_true",
                        help="Generate a tiny synthetic file and run offline.")
    parser.add_argument("--out", default="ett-out", help="Output dataset directory.")
    parser.add_argument("--n", type=int, default=60, help="Windows to select.")
    args = parser.parse_args()

    if sum(bool(x) for x in (args.source, args.download, args.demo)) != 1:
        parser.error("pass exactly one of: a source path, --download, or --demo")
    if args.demo:
        # Demo data must live outside --out: output() rejects any pipeline whose
        # output directory overlaps one of its input sources.
        with tempfile.TemporaryDirectory(prefix="kuration_ett_demo_") as directory:
            source = str(write_demo(Path(directory) / "ETTh1-demo.csv"))
            run(source, args.out, n=args.n)
        return
    if args.download:
        source = str(download(Path("data/ett/ETTh1.csv")))
    else:
        source = args.source

    run(source, args.out, n=args.n)


if __name__ == "__main__":
    main()
