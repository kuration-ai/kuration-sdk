"""REST + WebSocket handlers against a live jupyter_server (pytest-jupyter fixtures)."""
from __future__ import annotations

import asyncio
import io
import json
import time

import pytest
from conftest import make_csv, make_job
from tornado.httpclient import HTTPClientError


async def _json(jp_fetch, *parts, method="GET", body=None, params=None):
    response = await jp_fetch(*parts, method=method, params=params,
                              body=None if body is None else json.dumps(body),
                              allow_nonstandard_methods=body is not None and method != "POST")
    return json.loads(response.body)


# ---- version / auth ---------------------------------------------------------

async def test_version_handshake(jp_fetch):
    payload = await _json(jp_fetch, "kuration", "v1", "version")
    assert payload["api"] == 1
    assert payload["lab"] == "0.1.0"
    assert payload["job_schema_version"]


async def test_unauthenticated_request_is_rejected(jp_serverapp, http_server_client,
                                                   jp_base_url):
    from jupyter_server.utils import url_path_join

    response = await http_server_client.fetch(
        url_path_join(jp_base_url, "kuration", "v1", "version"), raise_error=False)
    assert response.code == 403


# ---- authoring --------------------------------------------------------------

async def test_infer_and_build_and_validate(jp_fetch, jp_root_dir):
    make_csv(jp_root_dir, "sample.csv")
    inferred = await _json(jp_fetch, "kuration", "v1", "infer",
                           method="POST", body={"source": "sample.csv"})
    assert inferred["adapter_kind"] == "timeseries"
    assert "schema" in inferred and "adapter" in inferred

    built = await _json(jp_fetch, "kuration", "v1", "jobs", "build", method="POST",
                        body={"sources": ["sample.csv"], "out": "curated",
                              "window": 4, "stride": 4, "train_hours": 0.01})
    job = built["job"]
    assert job["train_hours"] == 0.01

    validated = await _json(jp_fetch, "kuration", "v1", "jobs", "validate",
                            method="POST", body={"job": job})
    assert validated.get("kind") == "curate"

    explained = await _json(jp_fetch, "kuration", "v1", "jobs", "explain",
                            method="POST", body={"job": job})
    assert explained.get("kind") == "curate"


async def test_preview_returns_dataset_preview(jp_fetch, tmp_path):
    job = make_job(tmp_path)
    payload = await _json(jp_fetch, "kuration", "v1", "preview",
                          method="POST", body={"job": job})
    assert payload["records"]["ticks"] > 0
    assert payload["sources"]["total"] == 1


async def test_job_dir_anchors_relative_job_paths(jp_fetch, jp_root_dir):
    (jp_root_dir / "proj").mkdir()
    source = make_csv(jp_root_dir / "proj", "rel.csv")
    job = make_job(jp_root_dir / "proj", sources=[str(source)])
    job["sources"] = ["rel.csv"]             # as a .kur.json in proj/ would spell it
    payload = await _json(jp_fetch, "kuration", "v1", "preview", method="POST",
                          body={"job": job, "job_dir": "proj"})
    assert payload["sources"]["total"] == 1


async def test_job_schema_served(jp_fetch):
    schema = await _json(jp_fetch, "kuration", "v1", "jobs", "schema")
    assert schema.get("$schema", "").startswith("http")
    assert "train_hours" in schema.get("properties", {})


# ---- error envelope ---------------------------------------------------------

async def test_error_envelope_carries_type_and_hint(jp_fetch):
    with pytest.raises(HTTPClientError) as excinfo:
        await jp_fetch("kuration", "v1", "inspect", method="POST",
                       body=json.dumps({"dataset": "does-not-exist"}))
    body = json.loads(excinfo.value.response.body)
    assert set(body["error"]) == {"type", "message", "hint"}
    assert body["error"]["type"]


# ---- artifact inspection ----------------------------------------------------

async def test_inspect_stats_shards_arrays(jp_fetch, written_dataset):
    inspection = await _json(jp_fetch, "kuration", "v1", "inspect",
                             method="POST", body={"dataset": written_dataset})
    assert inspection["shards"] >= 1
    assert inspection["container"]

    payload = await _json(jp_fetch, "kuration", "v1", "stats",
                          method="POST", body={"subject": written_dataset})
    assert payload["report"]["stage"] == "output"
    assert "plots" in payload["capabilities"]

    shards = await _json(jp_fetch, "kuration", "v1", "shards",
                         method="POST", body={"dataset": written_dataset})
    assert shards["entries"] and shards["container"]

    entry = shards["entries"][0]
    response = await jp_fetch("kuration", "v1", "shards", "arrays", method="POST",
                              body=json.dumps({"dataset": written_dataset, "entry": entry}))
    assert response.headers["Content-Type"] == "application/x-npz"
    import numpy as np

    arrays = np.load(io.BytesIO(response.body))
    assert arrays.files
    first = arrays[arrays.files[0]]
    assert first.shape[0] == entry["n_steps"]  # pad trimmed


async def test_stats_html_and_plot(jp_fetch, written_dataset):
    response = await jp_fetch("kuration", "v1", "stats", "html", method="POST",
                              body=json.dumps({"subject": written_dataset}))
    assert response.headers["Content-Type"].startswith("text/html")
    assert b"<" in response.body

    response = await jp_fetch("kuration", "v1", "stats", "plot", method="POST",
                              body=json.dumps({"subject": written_dataset}))
    assert response.headers["Content-Type"] == "image/png"
    assert response.body.startswith(b"\x89PNG")


# ---- runs -------------------------------------------------------------------

async def _submit_and_wait(jp_fetch, job, timeout=180.0):
    info = await _json(jp_fetch, "kuration", "v1", "runs", method="POST", body={"job": job})
    run_id = info["run_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = await _json(jp_fetch, "kuration", "v1", "runs", run_id)
        if info["state"] in {"succeeded", "failed", "cancelled", "orphaned"}:
            return info
        await asyncio.sleep(0.2)
    raise AssertionError(f"run {run_id} did not finish: {info}")


async def test_run_lifecycle_end_to_end(jp_fetch, tmp_path):
    job = make_job(tmp_path)
    info = await _submit_and_wait(jp_fetch, job)
    assert info["state"] == "succeeded", info

    result = await _json(jp_fetch, "kuration", "v1", "runs", info["run_id"], "result")
    assert result["kind"] == "curate"
    assert "review" in result

    events = await _json(jp_fetch, "kuration", "v1", "runs", info["run_id"], "events")
    phases = [(e["phase"], e["event"]) for e in events["events"]]
    assert ("lifecycle", "start") in phases and ("lifecycle", "complete") in phases
    assert events["state"] == "succeeded"

    listing = await _json(jp_fetch, "kuration", "v1", "runs")
    assert any(r["run_id"] == info["run_id"] for r in listing["runs"])


async def test_run_events_pagination(jp_fetch, tmp_path):
    job = make_job(tmp_path)
    info = await _submit_and_wait(jp_fetch, job)
    first = await _json(jp_fetch, "kuration", "v1", "runs", info["run_id"], "events",
                        params={"limit": "2"})
    assert len(first["events"]) == 2
    rest = await _json(jp_fetch, "kuration", "v1", "runs", info["run_id"], "events",
                       params={"since": str(first["next"])})
    seqs = [e["seq"] for e in first["events"] + rest["events"]]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


async def test_run_result_missing_is_404(jp_fetch):
    with pytest.raises(HTTPClientError) as excinfo:
        await jp_fetch("kuration", "v1", "runs", "nope", "result")
    assert excinfo.value.code == 404


async def test_websocket_replays_and_signals_terminal_state(jp_fetch, jp_ws_fetch, tmp_path):
    job = make_job(tmp_path)
    info = await _submit_and_wait(jp_fetch, job)
    socket = await jp_ws_fetch("kuration", "v1", "ws", "runs", info["run_id"])
    seen = []
    while True:
        message = await socket.read_message()
        if message is None:
            break
        seen.append(json.loads(message))
        if seen[-1].get("event") == "state":
            break
    socket.close()
    assert seen, "no events replayed"
    assert seen[0]["seq"] == 1
    assert seen[-1]["event"] == "state" and seen[-1]["state"] == "succeeded"


async def test_cancel_running_run_via_delete(jp_fetch, tmp_path):
    # Many small sources -> per-source progress, so the cancel lands mid-run.
    sources = [str(make_csv(tmp_path, f"s{i:03d}.csv", steps=400)) for i in range(60)]
    job = make_job(tmp_path, sources=sources, train_hours=0.5)
    info = await _json(jp_fetch, "kuration", "v1", "runs", method="POST", body={"job": job})
    run_id = info["run_id"]

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        events = await _json(jp_fetch, "kuration", "v1", "runs", run_id, "events")
        started_scoring = any(e["phase"] == "scoring" and e["event"] == "advance"
                              for e in events["events"])
        if started_scoring or events["state"] in {"succeeded", "failed", "cancelled"}:
            break
        await asyncio.sleep(0.02)
    assert started_scoring, f"run never started scoring: {events['state']}"

    await _json(jp_fetch, "kuration", "v1", "runs", run_id, method="DELETE", body=None)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        info = await _json(jp_fetch, "kuration", "v1", "runs", run_id)
        if info["state"] in {"succeeded", "failed", "cancelled", "orphaned"}:
            break
        await asyncio.sleep(0.1)
    assert info["state"] == "cancelled", info


async def test_purge_removes_terminal_run(jp_fetch, tmp_path):
    job = make_job(tmp_path)
    info = await _submit_and_wait(jp_fetch, job)
    await _json(jp_fetch, "kuration", "v1", "runs", info["run_id"],
                method="DELETE", params={"purge": "true"}, body=None)
    with pytest.raises(HTTPClientError) as excinfo:
        await jp_fetch("kuration", "v1", "runs", info["run_id"])
    assert excinfo.value.code == 404
