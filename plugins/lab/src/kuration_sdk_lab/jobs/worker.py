"""Run one kuration job in an isolated subprocess: ``python -m kuration_sdk_lab.jobs.worker <run_dir>``.

The run directory must already contain ``job.json`` (a normalized job document) and
``run.json`` (submit metadata). The worker owns ``status.json``/``events.jsonl``/``result.json``
from here on; see ``protocol.py`` for the contract.

Isolation is the point: the SDK mutates process-global registries and plugin state, and its
only cancellation lever is raising from the synchronous progress callback — so SIGTERM sets a
flag that the callback turns into an exception on the next event, and the SDK's normal error
path unwinds (bounded-path checkpoints flush). Results with live objects (``DatasetWriteResult``)
are reduced to JSON (``hours_selection_review``) *before* exit — nothing survives the process.
"""
from __future__ import annotations

import os
import signal
import sys
import traceback
from pathlib import Path

from . import protocol


class WorkerCancelled(Exception):
    """Raised from the progress callback after SIGTERM; unwinds the SDK run cleanly."""


_cancel_requested = False


def _request_cancel(signum, frame):  # noqa: ARG001 - signal handler signature
    global _cancel_requested
    _cancel_requested = True


def _progress_callback(writer: protocol.EventWriter):
    def _cb(event: dict) -> None:
        if _cancel_requested:
            raise WorkerCancelled()
        writer.emit(dict(event))

    return _cb


def _run_curate(job: dict, resolved, progress) -> dict:
    """Mirror kur-curate's parameter pass-through (kuration.core.cli.curate._run)."""
    from kuration.core.output import write_dataset
    from kuration.core.review import hours_selection_review

    res = write_dataset(job["out"], schema=resolved.schema, adapter=resolved.adapter,
                        sources=resolved.sources, strategy=resolved.strategy,
                        segmenter=resolved.segmenter, spec=resolved.output,
                        encoder=resolved.encoder,
                        train_hours=job["train_hours"], test_hours=job.get("test_hours", 0.0),
                        seed=job.get("seed", 0), workers=job.get("workers", 1),
                        on_error=resolved.on_error, resume=job.get("resume", False),
                        force=job.get("force", False),
                        streaming=job.get("streaming", False),
                        span_steps=job.get("span_steps"), span_seconds=job.get("span_seconds"),
                        clip_aggregate=job.get("clip_aggregate", "max"),
                        progress=progress,
                        executor=job.get("executor", "local"),
                        memory_budget=job.get("memory_budget"),
                        batch_size=job.get("batch_size", 1024),
                        checkpoint_dir=job.get("checkpoint_dir"),
                        spill_dir=job.get("spill_dir"),
                        approximation=job.get("approximation", "exact"),
                        max_in_flight=job.get("max_in_flight", 2), registry=resolved.registry)
    review = hours_selection_review(res)
    return {
        "kind": "curate",
        "out": job["out"],
        "summary": res["results"],
        "review": dict(review),
        "errors": list(res["errors"])[:200],
    }


def _run_experiment(job: dict, resolved, progress) -> dict:
    from kuration.core.experiment import Arm, run_experiment

    arms = [Arm(name, strategy) for name, strategy in resolved.arms]
    out = run_experiment(schema=resolved.schema, adapter=resolved.adapter,
                         sources=resolved.sources, arms=arms,
                         segmenter=resolved.segmenter, target_size=job["target_size"],
                         baseline=job.get("baseline"), seed=job.get("seed", 0),
                         workers=job.get("workers", 1), on_error=resolved.on_error,
                         streaming=job.get("streaming", False), progress=progress,
                         executor=job.get("executor", "local"))
    return {"kind": "experiment", "comparison": dict(out)}


def _run_report(job: dict, resolved, progress) -> dict:
    """Mirror kur-report: score everything, apply scorers, report distributions."""
    from kuration.core.report import dataset_report
    from kuration.core.scoring import score_sources

    strategy = resolved.strategy
    # Scorer outputs are produced by the scorer stage, not the feature pass — exclude them
    # (see CLAUDE.md "Things that bit us").
    registered = sorted(
        r for r in strategy.referenced_metrics()
        if r not in strategy.define and r not in strategy.scorer_outputs()
    )
    segs, errors, _ = score_sources(
        schema=resolved.schema, adapter=resolved.adapter, sources=resolved.sources,
        define=strategy.define, registered=registered, segmenter=resolved.segmenter,
        tags_spec=strategy.tags, workers=job.get("workers", 1), on_error=resolved.on_error,
        streaming=job.get("streaming", False), progress=progress,
        executor=job.get("executor", "local"), registry=resolved.registry)
    if strategy.scorers:
        from kuration.core.scorers import apply_scorers

        apply_scorers(strategy.scorers, segs, schema=resolved.schema, adapter=resolved.adapter,
                      seed=job.get("seed", 0), progress=progress, registry=resolved.registry)
    rep = dataset_report(segs)
    return {"kind": "report", "report": dict(rep), "errors": list(errors)[:200]}


_RUNNERS = {"curate": _run_curate, "experiment": _run_experiment, "report": _run_report}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m kuration_sdk_lab.jobs.worker <run_dir>", file=sys.stderr)
        return 2
    run_dir = Path(argv[0])
    job = protocol.read_json(run_dir / protocol.JOB_FILE)
    if job is None:
        print(f"no readable {protocol.JOB_FILE} in {run_dir}", file=sys.stderr)
        return 2
    run_meta = protocol.read_json(run_dir / protocol.RUN_FILE) or {}
    run_id = run_meta.get("run_id") or run_dir.name

    signal.signal(signal.SIGTERM, _request_cancel)
    writer = protocol.EventWriter(run_dir, run_id)
    protocol.write_status(run_dir, protocol.STATE_RUNNING, pid=os.getpid())
    progress = _progress_callback(writer)
    try:
        from kuration.core.cli.job import resolve_job

        resolved = resolve_job(job)
        runner = _RUNNERS.get(resolved.kind)
        if runner is None:
            raise ValueError(f"unsupported job kind for the lab worker: {resolved.kind!r}")
        writer.lifecycle("start", detail=resolved.kind)
        result = runner(job, resolved, progress)
        protocol.write_json_atomic(run_dir / protocol.RESULT_FILE, result)
        writer.lifecycle("complete", detail=resolved.kind)
        protocol.write_status(run_dir, protocol.STATE_SUCCEEDED, exit={"ok": True})
        return 0
    except WorkerCancelled:
        writer.lifecycle("cancelled")
        protocol.write_status(run_dir, protocol.STATE_CANCELLED,
                              exit={"ok": False, "cancelled": True})
        return 0
    except BaseException as error:  # noqa: B036 - terminal status must always be written
        from kuration.core.cli._common import error_hint

        traceback.print_exc(file=sys.stderr)  # full detail lands in worker.log
        message = " ".join(str(error).split()) or "<no message>"
        writer.lifecycle("error", detail=f"{type(error).__name__}: {message}"[:500])
        protocol.write_status(run_dir, protocol.STATE_FAILED,
                              exit={"ok": False, "error_type": type(error).__name__,
                                    "message": message[:2000], "hint": error_hint(error)})
        return 1
    finally:
        writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
