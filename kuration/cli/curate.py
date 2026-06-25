"""kur-curate — run a curation job (sources → hours-budget shards) from a JSON spec.

    kur-curate job.json

job keys: schema, adapter{type,...}, sources?(else adapter.sources()), segmenter{window,stride,unit?},
strategy(inline/path/name), output(OutputSpec), train_hours, test_hours?, out, seed?.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..output import OutputSpec, write_dataset
from ..strategy import load_strategy
from ._common import build_adapter, build_schema, build_segmenter, resolve_sources


def main() -> None:
    ap = argparse.ArgumentParser(description="Curate a dataset from a JSON job spec.")
    ap.add_argument("job", help="Path to the job JSON.")
    args = ap.parse_args()
    job = json.loads(Path(args.job).read_text())

    schema = build_schema(job)
    adapter = build_adapter(job, schema)
    schema = schema or adapter.schema           # curated adapter derives its own
    strategy = load_strategy(job["strategy"], schema)
    segmenter = build_segmenter(job, schema)
    sources = resolve_sources(job, adapter)
    spec = OutputSpec.from_dict(job["output"])

    res = write_dataset(job["out"], schema=schema, adapter=adapter, sources=sources,
                        strategy=strategy, segmenter=segmenter, spec=spec,
                        train_hours=job["train_hours"], test_hours=job.get("test_hours", 0.0),
                        seed=job.get("seed", 0))
    r = res["results"]
    print(f"wrote {r['shards']} shards from {r['tracks_admitted']}/{r['tracks_total']} tracks "
          f"({r['n_train']} train / {r['n_test']} test sources) → {job['out']}")


if __name__ == "__main__":
    main()
