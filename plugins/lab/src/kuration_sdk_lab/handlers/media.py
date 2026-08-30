"""Inline frame previews: decode a handful of evenly spaced frames from one source.

Mirrors the SDK's canonical bounded flow (``kuration.gameplay._stats._decode_frames``):
bounded tick read, ``linspace`` frame pick, ndarray-vs-FrameRef branch, PNG data URLs.
Needs the ``[video]`` extra when frames are lazy references; the ImportError from the SDK's
loud lazy import flows through the error envelope with its install hint.
"""
from __future__ import annotations

from tornado import web

from .base import KurationAPIHandler, api_errors


def _bounded_ticks(adapter, source, limit: int) -> list:
    ticks = []
    for tick in adapter.ticks(source):
        ticks.append(tick)
        if len(ticks) >= limit:
            break
    return ticks


class FramesHandler(KurationAPIHandler):
    @web.authenticated
    @api_errors
    async def post(self):
        body = self.body_json()
        job = body.get("job")
        if not isinstance(job, dict):
            raise web.HTTPError(400, reason="request needs a JSON-object 'job' field")
        if isinstance(body.get("job_dir"), str):
            from ..paths import normalize_job_paths

            job = normalize_job_paths(job, self.resolve_request_path(body["job_dir"]))
        source = body.get("source")
        count = int(body.get("count", 3))
        start = int(body.get("start", 0))
        end = int(body.get("end", 512))
        if not 1 <= count <= 12:
            raise web.HTTPError(400, reason="count must be in [1, 12]")

        def _run():
            import numpy as np

            from kuration.core.cli.job import resolve_job
            from kuration.core.media import media_provider_for

            from .._png import png_data_url

            resolved = resolve_job(job)
            the_source = source if source is not None else (
                resolved.sources[0] if resolved.sources else None)
            if the_source is None:
                raise ValueError("job has no sources; pass a 'source' field")
            media_names = list(getattr(resolved.schema, "media", {}) or {})
            if not media_names:
                raise ValueError("job schema declares no media; nothing to decode")
            name = body.get("media", media_names[0])
            if name not in media_names:
                raise ValueError(f"unknown media column {name!r}; available: {media_names}")
            ticks = _bounded_ticks(resolved.adapter, the_source, end)
            window = ticks[start:end]
            values = [t.media.get(name) for t in window if t.media.get(name) is not None]
            if not values:
                raise ValueError(f"no media values for {name!r} in ticks [{start}, {end})")
            indexes = np.linspace(0, len(values) - 1, num=min(count, len(values)), dtype=int)
            chosen = [values[int(i)] for i in indexes]
            if all(isinstance(v, np.ndarray) for v in chosen):
                frames = chosen
            else:
                provider = media_provider_for(resolved.adapter)
                if provider is None:
                    raise ValueError("adapter has no media provider; cannot decode lazy frames")
                frames = list(provider.decode_batch(chosen))
            return {"media": name, "source": str(the_source),
                    "frames": [png_data_url(f) for f in frames]}

        self.finish_json(await self.run_sync(_run))
