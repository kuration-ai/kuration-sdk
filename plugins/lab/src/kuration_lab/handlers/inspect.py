"""Artifact inspection endpoints: inspect, compare, stats (+capabilities), examples,
rendered HTML, and matplotlib plots.

Everything here is stateless — each request re-resolves its subject from the path. For big
artifacts that costs a re-scan per call; deliberate for v1 (loaders' snapshot-currency checks
make cached iterators hazardous, and correctness beats latency here).
"""
from __future__ import annotations

import io

from tornado import web

from .base import KurationAPIHandler, api_errors


def _stats_report(subject_path: str, body: dict):
    from kuration.core.stats import stats

    return stats(
        subject_path,
        hist_bins=body.get("hist_bins", 10),
        max_sources=body.get("max_sources", 3),
        max_ticks_per_source=body.get("max_ticks_per_source", 10_000),
        content=body.get("content", True),
        verify=body.get("verify", False),
    )


class InspectHandler(KurationAPIHandler):
    @web.authenticated
    @api_errors
    async def post(self):
        from kuration.core.inspection import inspect_dataset

        body = self.body_json()
        dataset = self.body_path(body, "dataset")
        inspection = await self.run_sync(inspect_dataset, dataset,
                                         verify=bool(body.get("verify", False)))
        self.finish_json(dict(inspection))


class CompareHandler(KurationAPIHandler):
    @web.authenticated
    @api_errors
    async def post(self):
        from kuration.core.inspection import compare_datasets

        body = self.body_json()
        left = self.body_path(body, "left")
        right = self.body_path(body, "right")
        comparison = await self.run_sync(compare_datasets, left, right,
                                         verify=bool(body.get("verify", False)))
        self.finish_json(dict(comparison))


class StatsHandler(KurationAPIHandler):
    """One round trip gives the UI the report *and* its capability discovery."""

    @web.authenticated
    @api_errors
    async def post(self):
        body = self.body_json()
        subject = self.body_path(body, "subject")

        def _run():
            from kuration.core.stats import stats_capabilities

            report = _stats_report(subject, body)
            return {"report": dict(report), "capabilities": stats_capabilities(report)}

        self.finish_json(await self.run_sync(_run))


class StatsExamplesHandler(KurationAPIHandler):
    @web.authenticated
    @api_errors
    async def post(self):
        body = self.body_json()
        subject = self.body_path(body, "subject")

        def _run():
            from kuration.core.stats import stats_examples

            report = _stats_report(subject, body)
            examples = stats_examples(
                report, mode=body.get("mode", "auto"), metric=body.get("metric"),
                compare=body.get("compare"), n=body.get("n", 6),
                frames=body.get("frames", 3))
            return dict(examples)

        self.finish_json(await self.run_sync(_run))


class StatsHtmlHandler(KurationAPIHandler):
    @web.authenticated
    @api_errors
    async def post(self):
        body = self.body_json()
        subject = self.body_path(body, "subject")

        def _run():
            from kuration.core.stats import render_stats_html

            report = _stats_report(subject, body)
            return render_stats_html(report, title=body.get("title", "kuration stats"))

        html = await self.run_sync(_run)
        self.finish_bytes(html, "text/html; charset=UTF-8")


class StatsPlotHandler(KurationAPIHandler):
    @web.authenticated
    @api_errors
    async def post(self):
        body = self.body_json()
        subject = self.body_path(body, "subject")

        def _run():
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            from kuration.core.stats import plot_stats

            report = _stats_report(subject, body)
            result = plot_stats(report, body.get("metric"), kind=body.get("kind"))
            figure = getattr(result, "figure", result)  # plotters may return an Axes
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=110)
            plt.close(figure)
            return buffer.getvalue()

        png = await self.run_sync(_run)
        self.finish_bytes(png, "image/png")
