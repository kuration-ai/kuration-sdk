"""kur-experiment — compare several strategies over the same sources, from a JSON spec.

    kur-experiment job.json [--out comparison.json]

Adds `arms: [{name, strategy}]` and `baseline` to the shared job keys; `target_size` instead
of an hour budget. Prints per-arm JS-vs-baseline on each metric; optionally writes the full
comparison JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..experiment import Arm, run_experiment
from ..strategy import load_strategy
from ._common import build_adapter, build_schema, build_segmenter, resolve_sources


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-arm curation comparison from a JSON job spec.")
    ap.add_argument("job", help="Path to the job JSON.")
    ap.add_argument("--out", default=None, help="Write the full comparison JSON here.")
    args = ap.parse_args()
    job = json.loads(Path(args.job).read_text())

    schema = build_schema(job)
    adapter = build_adapter(job, schema)
    schema = schema or adapter.schema
    segmenter = build_segmenter(job, schema)
    sources = resolve_sources(job, adapter)
    arms = [Arm(a["name"], load_strategy(a["strategy"], schema)) for a in job["arms"]]

    out = run_experiment(schema=schema, adapter=adapter, sources=sources, arms=arms,
                         segmenter=segmenter, target_size=job["target_size"],
                         baseline=job.get("baseline"), seed=job.get("seed", 0))

    base = out["baseline"]
    print(f"baseline={base}  ({out['n_total_segments']} segments)\n")
    for name, a in out["arms"].items():
        if name == base:
            continue
        top = sorted(((m, d["js_vs_baseline"]) for m, d in a["metrics"].items()
                      if "js_vs_baseline" in d), key=lambda x: -x[1])[:3]
        shown = "  ".join(f"{m}={js:.3f}" for m, js in top)
        print(f"  {name:<16} n={a['n_selected']:5d}  top JS-vs-{base}: {shown}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
