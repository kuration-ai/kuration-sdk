# JupyterLab extension

`kuration-sdk-lab` puts the whole curation lifecycle inside JupyterLab: author a job document,
run it with live progress, and inspect the written artifact — all against the same job spec
and validators the [CLI](jobs.md) uses. It ships as a separate distribution built from
`plugins/lab/` in this repository.

```bash
pip install kuration-sdk-lab
```

The install carries both halves: a prebuilt JupyterLab (≥ 4.1) frontend and a
`jupyter-server` extension (auto-enabled). No node or build step is needed at install time.

## What you get

- **Kuration sidebar** (left panel): every run with a live status badge; entry points for
  creating a job and inspecting a dataset.
- **Job authoring**: the Launcher's *New Kuration Job* creates a `.kur.json` document —
  edited in the stock JSON editor — with a companion action panel offering **Validate**,
  **Explain**, **Preview**, and **Run**. Validation is server-side `resolve_job`, so
  messages match `kur-validate` exactly. Relative paths inside the document resolve
  against the document's own directory.
- **Run monitor**: per-phase progress bars fed by the SDK's structured progress events over
  a WebSocket (with replay after reconnects and a polling fallback), an error table, the
  raw event log, and the persisted hours-selection review once the run succeeds.
- **Dataset inspector**: overview (`inspect_dataset`), stats with capability-gated plots and
  the SDK's rendered HTML report, bounded examples (inline media frames included), a shard
  browser with npz download, and a dataset diff.

## How runs execute

Each run is an isolated worker subprocess (`python -m kuration_sdk_lab.jobs.worker`) driven by
the same job document the CLI takes. Run state lives on disk under
`<jupyter data dir>/kuration-sdk-lab/runs/<run_id>/` — progress events, worker log, and the
JSON result survive a server restart, and interrupted curate runs surface
`resume_available` when `write_dataset(resume=True)` applies. Cancellation sends SIGTERM,
which the worker turns into a clean stop at the next progress event.

Server configuration (all optional):

```python
# jupyter_server_config.py
c.KurationLabApp.runs_dir = "/data/kuration-runs"
c.KurationLabApp.max_concurrent_runs = 2
c.KurationLabApp.allow_absolute_paths = True
c.KurationLabApp.cancel_grace_seconds = 15.0
```

## REST API

The UI is a thin client over a token-authenticated HTTP API under `/kuration/v1/`
(`GET /version`, `POST /jobs/validate`, `POST /preview`, `POST /runs`,
`WS /ws/runs/{id}`, `POST /stats`, `POST /shards`, …), so the same surface is scriptable
from anywhere that can reach the Jupyter server. The job JSON Schema ships at
`GET /kuration/v1/jobs/schema` for editor integration.

## Scope notes

The extension speaks the declarative job spec, so it runs exactly what
[`kur-curate`/`kur-experiment`/`kur-report`](jobs.md) run: hours-budget curation,
experiments, and reports. Count-based selection and fluent-pipeline-only features
(`enrich`, `analyze`, `transform`, `expect`) remain Python-API territory.

Development workflow and package layout: see `plugins/lab/README.md` in the repository.
