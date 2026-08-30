"""Route table for the kuration server extension (all under ``/kuration/v1/``)."""
from __future__ import annotations

from tornado import web

from .authoring import (InferHandler, JobBuildHandler, JobExplainHandler, JobMigrateHandler,
                        JobSchemaHandler, JobValidateHandler, PreviewHandler)
from .base import KurationAPIHandler, api_errors
from .inspect import (CompareHandler, InspectHandler, StatsExamplesHandler, StatsHandler,
                      StatsHtmlHandler, StatsPlotHandler)
from .media import FramesHandler
from .runs import RunEventsHandler, RunHandler, RunResultHandler, RunsHandler
from .runs_ws import RunEventsWebSocket
from .shards import ShardArraysHandler, ShardsHandler

_RUN_ID = r"([A-Za-z0-9_-]+)"


class VersionHandler(KurationAPIHandler):
    @web.authenticated
    @api_errors
    async def get(self):
        import kuration

        from .. import API_VERSION, __version__
        from kuration.core.versioning import JOB_SCHEMA_VERSION

        self.finish_json({
            "api": API_VERSION,
            "lab": __version__,
            "sdk": getattr(kuration, "__version__", "unknown"),
            "job_schema_version": JOB_SCHEMA_VERSION,
        })


handlers = [
    (r"/kuration/v1/version", VersionHandler),
    (r"/kuration/v1/infer", InferHandler),
    (r"/kuration/v1/jobs/build", JobBuildHandler),
    (r"/kuration/v1/jobs/validate", JobValidateHandler),
    (r"/kuration/v1/jobs/migrate", JobMigrateHandler),
    (r"/kuration/v1/jobs/explain", JobExplainHandler),
    (r"/kuration/v1/jobs/schema", JobSchemaHandler),
    (r"/kuration/v1/preview", PreviewHandler),
    (r"/kuration/v1/runs", RunsHandler),
    (rf"/kuration/v1/runs/{_RUN_ID}", RunHandler),
    (rf"/kuration/v1/runs/{_RUN_ID}/events", RunEventsHandler),
    (rf"/kuration/v1/runs/{_RUN_ID}/result", RunResultHandler),
    (rf"/kuration/v1/ws/runs/{_RUN_ID}", RunEventsWebSocket),
    (r"/kuration/v1/inspect", InspectHandler),
    (r"/kuration/v1/compare", CompareHandler),
    (r"/kuration/v1/stats", StatsHandler),
    (r"/kuration/v1/stats/examples", StatsExamplesHandler),
    (r"/kuration/v1/stats/html", StatsHtmlHandler),
    (r"/kuration/v1/stats/plot", StatsPlotHandler),
    (r"/kuration/v1/shards", ShardsHandler),
    (r"/kuration/v1/shards/arrays", ShardArraysHandler),
    (r"/kuration/v1/frames", FramesHandler),
]
