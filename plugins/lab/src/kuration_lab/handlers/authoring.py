"""Authoring endpoints: schema inference, job build/validate/migrate/explain, preview.

Every endpoint accepts an optional ``job_dir`` (server-side directory of the ``.kur.json``
being edited); relative paths inside the job document are anchored there before the SDK
sees them (see ``paths.normalize_job_paths``).
"""
from __future__ import annotations

from tornado import web

from ..paths import normalize_job_paths
from .base import KurationAPIHandler, api_errors


class InferHandler(KurationAPIHandler):
    @web.authenticated
    @api_errors
    async def post(self):
        from kuration.core.cli.infer_schema_cli import infer_source

        body = self.body_json()
        source = self.body_path(body, "source")
        result = await self.run_sync(
            infer_source, source,
            name=body.get("name", "inferred"), time=body.get("time"),
            entity=body.get("entity"), hz=body.get("hz"),
            kinds=body.get("kinds"), sample=body.get("sample", 200))
        self.finish_json(result)


class JobBuildHandler(KurationAPIHandler):
    @web.authenticated
    @api_errors
    async def post(self):
        from kuration.core.cli.init_job import build_job

        body = self.body_json()
        sources = body.get("sources")
        if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
            raise web.HTTPError(400, reason="request needs a list-of-strings 'sources' field")
        sources = [self.resolve_request_path(s) for s in sources]
        out = self.body_path(body, "out")
        job = await self.run_sync(
            build_job, sources, out=out,
            window=body.get("window", 64), stride=body.get("stride", 32),
            train_hours=body.get("train_hours", 1.0), test_hours=body.get("test_hours", 0.0),
            preset=body.get("preset", "numpy-timeseries"), seed=body.get("seed", 0),
            **{k: body[k] for k in ("name", "time", "entity", "hz", "kinds", "sample")
               if k in body})
        self.finish_json({"job": job})


class _JobBodyHandler(KurationAPIHandler):
    def job_from_body(self) -> dict:
        body = self.body_json()
        job = body.get("job")
        if not isinstance(job, dict):
            raise web.HTTPError(400, reason="request needs a JSON-object 'job' field")
        if isinstance(body.get("job_dir"), str):
            job = normalize_job_paths(job, self.resolve_request_path(body["job_dir"]))
        return job


class JobValidateHandler(_JobBodyHandler):
    @web.authenticated
    @api_errors
    async def post(self):
        from kuration.core.cli.validate import validate_job

        body = self.body_json()
        job = self.job_from_body()
        result = await self.run_sync(validate_job, job,
                                     check_sources=bool(body.get("check_sources", False)))
        self.finish_json(result)


class JobMigrateHandler(_JobBodyHandler):
    @web.authenticated
    @api_errors
    async def post(self):
        from kuration.core.versioning import migrate_job

        job = self.job_from_body()
        migrated = await self.run_sync(migrate_job, job)
        self.finish_json({"job": migrated})


class JobExplainHandler(_JobBodyHandler):
    @web.authenticated
    @api_errors
    async def post(self):
        from kuration.core.cli.explain import explain_job

        job = self.job_from_body()
        result = await self.run_sync(explain_job, job)
        self.finish_json(result)


class PreviewHandler(_JobBodyHandler):
    @web.authenticated
    @api_errors
    async def post(self):
        body = self.body_json()
        job = self.job_from_body()

        def _preview():
            from kuration.core.cli.job import resolve_job
            from kuration.core.review import preview_dataset

            resolved = resolve_job(job)
            if resolved.sources is None:
                raise ValueError("job resolves without sources; nothing to preview")
            return preview_dataset(
                schema=resolved.schema, adapter=resolved.adapter, sources=resolved.sources,
                max_sources=body.get("max_sources", 3),
                max_ticks_per_source=body.get("max_ticks_per_source", 10_000),
                examples=body.get("examples", 5))

        preview = await self.run_sync(_preview)
        self.finish_json(dict(preview))


class JobSchemaHandler(KurationAPIHandler):
    """Serve the SDK's shipped job JSON Schema for editors and frontend validation."""

    @web.authenticated
    @api_errors
    async def get(self):
        import json
        from importlib.resources import files

        raw = files("kuration.core").joinpath("schemas/job.schema.json").read_text("utf-8")
        self.finish_json(json.loads(raw))
