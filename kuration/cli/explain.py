from __future__ import annotations

import argparse
import json
import sys

from ._common import build_adapter, build_schema, build_segmenter, load_job_plugins, read_job


def explain_job(job: dict) -> dict:
    """Return an actionable, JSON-safe preflight summary without reading source rows."""
    from .validate import job_kind, validate_job
    summary = validate_job(job)
    load_job_plugins(job)
    schema = build_schema(job)
    adapter = build_adapter(job, schema)
    schema = schema or adapter.schema
    segmenter = build_segmenter(job, schema)
    from ..strategy import load_strategy
    kind = job_kind(job)
    if kind == "experiment":
        strategies = [load_strategy(arm["strategy"], schema) for arm in job["arms"]]
        method = ", ".join(f"{arm['name']}={strategy.method}"
                           for arm, strategy in zip(job["arms"], strategies))
        has_method_extras = any(strategy.filters or strategy.method in {
            "cluster-balanced", "k-center", "facility", "dsir", "rare", "typical"}
                                for strategy in strategies)
    else:
        strategy = load_strategy(job["strategy"], schema)
        method = strategy.method
        has_method_extras = bool(strategy.filters or strategy.method in {
            "cluster-balanced", "k-center", "facility", "dsir", "rare", "typical"})
    source = {"kind": type(adapter).__name__,
              "count": len(job.get("sources") or []) or "adapter-resolved"}
    output = job.get("output") or {}
    if kind == "curate":
        budget = {"kind": "hours", "train": job["train_hours"],
                  "test": job.get("test_hours", 0.0)}
    elif kind == "experiment":
        budget = {"kind": "count", "value": job["target_size"]}
    else:
        budget = None

    extras: set[str] = set()
    adapter_type = (job.get("adapter") or {}).get("type")
    adapter_extras = {"hdf5": "array", "parquet-dataset": "parquet",
                      "huggingface": "hf", "zarr": "zarr", "video": "video"}
    if adapter_type in adapter_extras:
        extras.add(adapter_extras[adapter_type])
    container_extras = {"hdf5": "array", "parquet": "parquet"}
    if output.get("container") in container_extras:
        extras.add(container_extras[output["container"]])
    if job.get("executor") == "ray":
        extras.add("ray")
    if has_method_extras:
        extras.add("methods")
    from ..io import is_remote
    if any(is_remote(source) for source in job.get("sources") or ()):
        extras.add("cloud")

    warnings = []
    if job.get("on_error", "skip") == "skip":
        warnings.append({"code": "source-errors-skipped",
                         "message": "source errors will be skipped and reported",
                         "hint": "Set mode='strict' or on_error='raise' while developing."})
    return {"kind": kind, "source": source, "schema": schema.to_dict(),
            "window": {"window_steps": segmenter.window,
                       "stride_steps": segmenter.stride},
            "selection": {"method": method, "budget": budget},
            "output": job.get("output") or job.get("output_preset"),
            "execution": {"executor": job.get("executor", "local"),
                          "workers": job.get("workers", 1),
                          "streaming": job.get("streaming", False),
                          "resume": job.get("resume", False),
                          "on_error": job.get("on_error", "skip")},
            "optional_extras": sorted(extras), "warnings": warnings,
            "validation": dict(summary)}


def render_plan(plan: dict) -> str:
    """Render the job explanation as a scannable terminal preflight."""
    selection, execution = plan["selection"], plan["execution"]
    budget = selection["budget"]
    if budget and budget["kind"] == "hours":
        budget_text = f"train={budget['train']}h, test={budget['test']}h"
    elif budget:
        budget_text = f"count={budget['value']}"
    else:
        budget_text = "none"
    output = plan.get("output") or {}
    container = output.get("container", output.get("name", "none")) \
        if isinstance(output, dict) else str(output)
    extras_text = ", ".join(plan["optional_extras"]) \
        if plan["optional_extras"] else "none (NumPy core)"
    lines = [f"job: {plan['kind']}",
             f"source: {plan['source']['kind']} ({plan['source']['count']} source(s))",
             f"schema: {plan['schema']['name']}",
             f"window: {plan['window']['window_steps']} steps "
             f"(stride {plan['window']['stride_steps']})",
             f"selection: {selection['method']} ({budget_text})",
             f"output: {container}",
             f"execution: {execution['executor']}; workers={execution['workers']}; "
             f"streaming={execution['streaming']}; resume={execution['resume']}; "
             f"errors={execution['on_error']}",
             f"optional extras: {extras_text}"]
    for warning in plan["warnings"]:
        lines.append(f"warning [{warning['code']}]: {warning['message']}")
        lines.append(f"  fix: {warning['hint']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Explain a kuration job without reading rows.")
    ap.add_argument("job")
    ap.add_argument("--json", action="store_true", help="Emit the structured plan as JSON.")
    ap.add_argument("--debug", action="store_true", help="Show full tracebacks on failure.")
    args = ap.parse_args()
    try:
        plan = explain_job(read_job(args.job))
    except Exception as error:
        if args.debug:
            raise
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(plan, indent=2) if args.json else render_plan(plan))


if __name__ == "__main__":
    main()
