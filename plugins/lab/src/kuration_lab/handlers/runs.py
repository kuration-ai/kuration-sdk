"""Run lifecycle endpoints: submit, list, status, cancel/purge, events, result."""
from __future__ import annotations

from tornado import web

from ..jobs import protocol
from ..paths import normalize_job_paths
from .base import KurationAPIHandler, api_errors


class RunsHandler(KurationAPIHandler):
    @web.authenticated
    @api_errors
    async def get(self):
        await self.job_manager.ensure_started()
        self.finish_json({"runs": self.job_manager.registry.list()})

    @web.authenticated
    @api_errors
    async def post(self):
        body = self.body_json()
        job = body.get("job")
        if not isinstance(job, dict):
            raise web.HTTPError(400, reason="request needs a JSON-object 'job' field")
        if isinstance(body.get("job_dir"), str):
            job = normalize_job_paths(job, self.resolve_request_path(body["job_dir"]))
        if body.get("resume"):
            job = {**job, "resume": True}
        if body.get("force"):
            job = {**job, "force": True}

        def _kind():
            from kuration.core.cli.job import resolve_job

            return resolve_job(job).kind  # full validation before anything hits the disk

        kind = await self.run_sync(_kind)
        user = self.current_user
        submitter = getattr(user, "username", None) if user else None
        info = await self.job_manager.submit(job, kind=kind, submitter=submitter)
        self.finish_json(info, status=202)


class RunHandler(KurationAPIHandler):
    @web.authenticated
    @api_errors
    async def get(self, run_id: str):
        info = self.job_manager.registry.info(run_id)
        if info is None:
            raise web.HTTPError(404, reason=f"unknown run: {run_id}")
        self.finish_json(info)

    @web.authenticated
    @api_errors
    async def delete(self, run_id: str):
        if self.get_query_argument("purge", default="false").lower() in {"1", "true"}:
            self.job_manager.registry.purge(run_id)
            self.finish_json({"run_id": run_id, "purged": True})
            return
        info = await self.job_manager.cancel(run_id)
        self.finish_json(info)


class RunEventsHandler(KurationAPIHandler):
    """Polling fallback for the WebSocket stream: events with ``seq > since``."""

    @web.authenticated
    @api_errors
    async def get(self, run_id: str):
        run_dir = self.job_manager.registry.run_dir(run_id)
        if not run_dir.is_dir():
            raise web.HTTPError(404, reason=f"unknown run: {run_id}")
        since = int(self.get_query_argument("since", default="0"))
        limit = int(self.get_query_argument("limit", default="1000"))
        events, next_seq = protocol.read_events(run_dir, since=since, limit=limit)
        status = protocol.read_status(run_dir)
        self.finish_json({"events": events, "next": next_seq, "state": status.get("state")})


class RunResultHandler(KurationAPIHandler):
    """The worker-persisted JSON result (curate: summary + hours-selection review)."""

    @web.authenticated
    @api_errors
    async def get(self, run_id: str):
        result = self.job_manager.registry.result(run_id)
        if result is None:
            info = self.job_manager.registry.info(run_id)
            if info is None:
                raise web.HTTPError(404, reason=f"unknown run: {run_id}")
            raise web.HTTPError(404, reason=f"run {run_id} has no result (state: {info['state']})")
        self.finish_json(result)
