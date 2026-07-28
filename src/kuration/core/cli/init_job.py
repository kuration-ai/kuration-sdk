"""kur-init — scaffold a valid curation job from a data sample.

    kur-init data.csv --out job.json

`kur-validate --print-schema` documents what a job may contain; this writes one that is
already correct for a concrete corpus. Inference supplies the schema and adapter mapping, the
first scalar field becomes a binning axis, and an output preset derives modalities from the
schema. The emitted document is resolved through the ordinary job resolver before it is
written, so a scaffold that lands on disk is a job `kur-validate` accepts.
"""
from __future__ import annotations

import argparse

from ._common import (cli_run, guard_file_output, json_document,
                      parse_mapping_option)
from .infer_schema_cli import infer_source


def _axis_field(schema: dict) -> tuple[str, str] | None:
    """Return `(field, reduce)` for the first field a default binning axis can measure."""
    for name, spec in (schema.get("fields") or {}).items():
        if spec.get("kind") == "scalar":
            return name, "mean"
    return None


def build_job(sources, *, out: str, window: int, stride: int, train_hours: float,
              test_hours: float, preset: str, seed: int, **infer_kwargs) -> dict:
    """Return a resolvable job document inferred from the first source."""
    if not sources:
        raise ValueError("kur-init needs at least one source")
    inferred = infer_source(sources[0], **infer_kwargs)
    schema, adapter = inferred["schema"], inferred["adapter"]
    entity_axis = schema.get("entity_axis")

    axis = _axis_field(schema)
    if axis is not None:
        field, reduce = axis
        # Metric sources are namespaced by the entity axis the schema declares.
        source = f"{entity_axis}.{field}" if entity_axis else field
        metric = f"{field}_{reduce}"
        strategy = {"name": "scaffold",
                    "define": {metric: {"source": source, "reduce": reduce}},
                    "binning": {"axes": [{"metric": metric, "bins": 4}]},
                    "selection": {"method": "uniform-bins"}}
    else:
        # No scalar field to bin on; top-k needs weights, so fall back to an explicit
        # filter-free weighted pass only when a metric exists. Otherwise say so loudly.
        raise ValueError(
            "no scalar field was inferred, so no default binning axis could be chosen — "
            "write the strategy's define/binning blocks by hand, or pass --kinds to "
            "classify a column as scalar")

    preset_params: dict = {"name": preset}
    if any(spec.get("kind") == "categorical"
           for spec in (schema.get("fields") or {}).values()):
        # Presets refuse to guess a categorical policy; 'omit' keeps the scaffold runnable.
        preset_params["categorical"] = "omit"

    from ..versioning import JOB_SCHEMA_VERSION
    job = {"job_schema_version": JOB_SCHEMA_VERSION,
           "schema": schema,
           "adapter": adapter,
           "sources": [str(source) for source in sources],
           "segmenter": {"window": window, "stride": stride},
           "strategy": strategy,
           "output_preset": preset_params,
           "train_hours": train_hours,
           "test_hours": test_hours,
           # Without a span, the hours budget admits whole tracks: one long single-entity
           # recording is one indivisible item, so the budget cannot bite and the run emits
           # the entire source. Admitting a window at a time is what a budget implies.
           "span_steps": window,
           "out": out,
           "seed": seed}
    return job


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scaffold a validated kuration job spec from a data sample.")
    ap.add_argument("sources", nargs="+", help="Data files to curate.")
    ap.add_argument("--out", required=True, metavar="PATH", help="Job spec to write.")
    ap.add_argument("--dataset-out", default="curated", metavar="DIR",
                    help="Dataset directory the job writes to (default: curated).")
    ap.add_argument("--window", type=int, default=64, help="Window length in steps.")
    ap.add_argument("--stride", type=int, help="Window stride (default: the window length).")
    ap.add_argument("--train-hours", type=float, default=1.0, help="Train budget in hours.")
    ap.add_argument("--test-hours", type=float, default=0.0, help="Test budget in hours.")
    ap.add_argument("--preset", default="numpy-timeseries", help="Registered output preset.")
    ap.add_argument("--seed", type=int, default=0, help="Selection seed.")
    ap.add_argument("--name", default="inferred", help="Schema name.")
    ap.add_argument("--time", help="Time-column override.")
    ap.add_argument("--entity", help="Entity-column override.")
    ap.add_argument("--hz", type=float, help="Sample-rate override.")
    ap.add_argument("--kinds", metavar="JSON|PATH",
                    help="Per-column kind overrides, as inline JSON or path to JSON file.")
    ap.add_argument("--sample", type=int, default=200, help="Rows to sample for inference.")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing job spec.")
    ap.add_argument("--json", action="store_true", help="Print the job document to stdout.")
    ap.add_argument("--debug", action="store_true", help="Show full tracebacks on failure.")
    args = ap.parse_args()

    def body():
        from ..io import write_text
        guard_file_output(args.out, force=args.force,
                          protected_inputs=(*args.sources, args.dataset_out),
                          label="job output")
        from ..manifest import validate_source_destination_separation
        validate_source_destination_separation(args.dataset_out, args.sources)
        if args.window < 1:
            raise ValueError("--window must be a positive integer")
        if args.sample < 1:
            raise ValueError("--sample must be a positive integer")
        stride = args.stride if args.stride is not None else args.window
        if stride < 1:
            raise ValueError("--stride must be a positive integer")
        kinds = parse_mapping_option(args.kinds, label="--kinds")
        job = build_job(args.sources, out=args.dataset_out, window=args.window,
                        stride=stride, train_hours=args.train_hours,
                        test_hours=args.test_hours, preset=args.preset, seed=args.seed,
                        name=args.name, time=args.time, entity=args.entity, hz=args.hz,
                        kinds=kinds, sample=args.sample)
        # A scaffold that cannot be resolved is worse than none: fail before writing.
        from .job import resolve_job
        resolve_job(job, enumerate_sources=False)
        encoded = json_document(job, label="inferred job")
        write_text(args.out, encoded)
        if args.json:
            print(encoded.rstrip())
        else:
            axis = job["strategy"]["binning"]["axes"][0]
            print(f"wrote job: {args.out}\n"
                  f"schema: {job['schema']['name']} "
                  f"({len(job['schema'].get('fields') or {})} field(s))\n"
                  f"adapter: {job['adapter']['type']}\n"
                  f"window: {job['segmenter']['window']} steps "
                  f"(stride {job['segmenter']['stride']})\n"
                  f"selection: uniform-bins on {axis['metric']} ({axis['bins']} bins)\n"
                  f"output: preset {job['output_preset']['name']} -> {job['out']}\n"
                  f"next: kur-explain {args.out} && kur-curate {args.out}")

    cli_run(body, debug=args.debug)


if __name__ == "__main__":
    main()
