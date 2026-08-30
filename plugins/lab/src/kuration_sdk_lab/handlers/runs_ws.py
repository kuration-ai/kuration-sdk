"""WebSocket progress stream: replay from ``?since=<seq>`` then live tail.

The stream is driven by ``JobManager.stream`` (per-subscriber file tailing), so a reconnect
with the last seen ``seq`` is lossless and works even for runs that finished long ago or
across a server restart. The client can send ``{"op": "cancel"}`` on the same socket.
"""
from __future__ import annotations

import asyncio
import contextlib
import json

from jupyter_server.base.handlers import JupyterHandler
from jupyter_server.base.websocket import WebSocketMixin
from tornado import web, websocket


class RunEventsWebSocket(WebSocketMixin, JupyterHandler, websocket.WebSocketHandler):
    auth_resource = "kuration"

    @property
    def job_manager(self):
        return self.settings["kuration_sdk_lab"]["job_manager"]

    async def pre_get(self):
        if self.current_user is None:
            raise web.HTTPError(403)

    async def get(self, *args, **kwargs):
        await self.pre_get()
        await super().get(*args, **kwargs)

    def open(self, run_id: str):
        self._run_id = run_id
        try:
            since = int(self.get_query_argument("since", default="0"))
        except ValueError:
            since = 0
        self._pump = asyncio.ensure_future(self._stream(run_id, since))

    async def _stream(self, run_id: str, since: int):
        try:
            async for event in self.job_manager.stream(run_id, since=since):
                self.write_message(json.dumps(event, allow_nan=False))
        except FileNotFoundError:
            self.close(4404, f"unknown run: {run_id}")
            return
        except websocket.WebSocketClosedError:
            return
        self.close(1000, "run complete")

    async def on_message(self, message):
        try:
            payload = json.loads(message)
        except ValueError:
            return
        if isinstance(payload, dict) and payload.get("op") == "cancel":
            with contextlib.suppress(FileNotFoundError):
                await self.job_manager.cancel(self._run_id)

    def on_close(self):
        pump = getattr(self, "_pump", None)
        if pump is not None:
            pump.cancel()
