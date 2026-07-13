from __future__ import annotations

import argparse
import json

from ._common import build_adapter, build_schema, build_segmenter, load_job_plugins, read_job


def main() -> None:
    ap = argparse.ArgumentParser(description="Explain a kuration job without reading rows.")
    ap.add_argument("job")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    job = read_job(args.job)
    load_job_plugins(job)
    schema = build_schema(job)
    adapter = build_adapter(job, schema)
    schema = schema or adapter.schema
    segmenter = build_segmenter(job, schema)
    from ..strategy import load_strategy
    strategy = load_strategy(job["strategy"], schema)
    source = {"kind": type(adapter).__name__,
              "count": len(job.get("sources") or []) or "adapter-resolved"}
    output = job.get("output") or {}
    plan: dict = {"source": source,
            "schema": schema.to_dict(),
            "window": {"window_steps": segmenter.window,
                       "stride_steps": segmenter.stride},
            "strategy": strategy.resolved, "output": job.get("output"),
            "mode": job.get("mode"), "warnings": []}
    print(json.dumps(plan, indent=2) if args.json else
          f"{source['kind']} → {strategy.method} → "
          f"{output.get('container', 'no output')}")
