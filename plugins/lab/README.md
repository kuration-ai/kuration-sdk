# kuration-sdk-lab

JupyterLab extension for the [kuration SDK](../../README.md): author, run, and inspect
curation jobs without leaving JupyterLab.

- **Author** — create/edit `.kur.json` job documents in the JSON editor with a companion
  action panel: Validate, Explain, Preview, Run (all server-side, using the SDK's own
  validators).
- **Run + monitor** — jobs execute in isolated worker subprocesses; per-phase progress
  streams live over a WebSocket, with cancel, resume detection, and durable run history that
  survives server restarts.
- **Inspect** — open any written dataset: overview, stats (+plots and the SDK's HTML
  report), bounded examples (with inline media frames), a shard browser with npz download,
  and dataset diff.

## Install

```bash
pip install kuration-sdk-lab
```

This installs the prebuilt JupyterLab extension and enables the `kuration_sdk_lab` server
extension automatically. Requires JupyterLab ≥ 4.1 running against `jupyter-server` ≥ 2.14.

## Server configuration

```python
# jupyter_server_config.py
c.KurationLabApp.runs_dir = "/data/kuration-runs"   # default: <jupyter data dir>/kuration-sdk-lab/runs
c.KurationLabApp.max_concurrent_runs = 2
c.KurationLabApp.allow_absolute_paths = True         # False locks all paths under the server root
c.KurationLabApp.cancel_grace_seconds = 15.0
```

## REST API

Everything the UI does is a documented HTTP call under `/kuration/v1/` (token-authenticated
like every Jupyter API). Highlights: `POST /preview`, `POST /runs`, `GET /runs/{id}/events`,
`WS /ws/runs/{id}`, `POST /stats`, `POST /shards`. See `src/kuration_sdk_lab/handlers/__init__.py`
for the full route table.

## Development

```bash
cd plugins/lab
pip install -e ".[test]"
jupyter labextension develop --overwrite .
jupyter server extension enable kuration_sdk_lab
jlpm watch          # terminal 1: rebuild on TS changes
jupyter lab         # terminal 2: refresh the browser after rebuilds
```

Run the tests (no browser or node needed):

```bash
python -m pytest tests -q
```

The Python package lives in `src/kuration_sdk_lab/`; TypeScript sources in `ts/`; the built
federated extension lands in `src/kuration_sdk_lab/labextension/` (gitignored, shipped in the
wheel).
