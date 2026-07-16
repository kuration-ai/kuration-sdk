"""The kuration ten-minute quick start.

Build a small, deterministic set of sensor sessions in memory, select twelve diverse windows,
write them as NumPy shards, and read one shard back. No download or optional dependency is
required.

    python examples/quickstart.py --out quickstart-out
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

# Keep the checked-in example runnable directly without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kuration import input, inspect_dataset, iter_shards


def example_sessions() -> dict[str, list[dict]]:
    """Return six sessions with deliberately different activity and fault patterns."""
    sessions = {}
    for session_index in range(6):
        amplitude = 0.25 + 0.35 * session_index
        rows = []
        for step in range(48):
            signal = 10.0 + amplitude * math.sin(step / 2.5)
            fault = int(session_index >= 3 and step % (13 - session_index) == 0)
            rows.append({
                "t": float(step),
                "device": f"device-{session_index}",
                "signal": round(signal, 5),
                "temperature": round(20.0 + session_index + math.cos(step / 8.0), 5),
                "fault": fault,
            })
        sessions[f"session-{session_index}"] = rows
    return sessions


def run(out_dir: str | Path) -> dict:
    """Run the documented quick-start pipeline and return its write result."""
    sessions = example_sessions()
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

    print("preflight:")
    plan = pipeline.output(out_dir, container="npz", shard_length=12, dry_run=True,
                           inspect_sources=True)
    print(plan.render())
    result = pipeline.output(out_dir, container="npz", shard_length=12)
    arrays, metadata = next(iter_shards(out_dir, with_meta=True))
    inspection = inspect_dataset(out_dir)

    print(f"input: {len(sessions)} sessions, "
          f"{sum(len(rows) for rows in sessions.values())} rows")
    print(f"curation: {result['results']['segments_total']} candidate windows -> "
          f"{result['selected']} selected")
    print(f"output: {len(result['shards'])} shards in {out_dir}")
    print(f"artifact: {inspection.shards} shards; "
          f"splits={sorted(inspection.splits)}; metadata valid")
    print(f"first shard: split={metadata['split']}, "
          f"modalities={sorted(arrays)}, signal_shape={arrays['signal'].shape}")
    print(f"reproducible config: {Path(out_dir) / 'config.json'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="quickstart-out",
                        help="output dataset directory (default: quickstart-out)")
    args = parser.parse_args()
    run(args.out)


if __name__ == "__main__":
    main()
