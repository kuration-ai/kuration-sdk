"""Shared handler base: auth, JSON body parsing, error envelope, executor offload.

Every blocking SDK call goes through ``run_sync`` (a small shared thread pool) so a slow
``stats()`` over a big artifact never freezes the Jupyter server's event loop, which also
serves kernels.
"""
from __future__ import annotations

import asyncio
import functools
import json

from jupyter_server.base.handlers import APIHandler
from tornado import web

from ..errors import error_envelope, status_for


def api_errors(method):
    """Convert SDK exceptions into the JSON error envelope with an actionable hint."""

    @functools.wraps(method)
    async def wrapper(self, *args, **kwargs):
        try:
            return await method(self, *args, **kwargs)
        except web.HTTPError:
            raise
        except Exception as error:  # noqa: BLE001 - envelope every SDK failure
            self.send_api_error(error)

    return wrapper


class KurationAPIHandler(APIHandler):
    auth_resource = "kuration"

    @property
    def lab(self) -> dict:
        return self.settings["kuration_lab"]

    @property
    def job_manager(self):
        return self.lab["job_manager"]

    def resolve_request_path(self, value: str) -> str:
        from ..paths import resolve_path

        return resolve_path(value, self.lab["root_dir"],
                            allow_absolute=self.lab["allow_absolute_paths"])

    def body_json(self) -> dict:
        try:
            body = json.loads(self.request.body or b"{}")
        except ValueError as error:
            raise web.HTTPError(400, reason=f"request body is not valid JSON: {error}") from None
        if not isinstance(body, dict):
            raise web.HTTPError(400, reason="request body must be a JSON object")
        return body

    def body_path(self, body: dict, key: str) -> str:
        value = body.get(key)
        if not isinstance(value, str) or not value.strip():
            raise web.HTTPError(400, reason=f"request needs a string {key!r} field")
        return self.resolve_request_path(value)

    async def run_sync(self, fn, /, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.lab["executor"],
                                          functools.partial(fn, *args, **kwargs))

    def finish_json(self, payload, status: int = 200) -> None:
        self.set_status(status)
        self.finish(json.dumps(payload, allow_nan=False))

    def finish_bytes(self, payload: bytes | str, content_type: str) -> None:
        # jupyter_server's APIHandler.finish force-resets Content-Type to application/json
        # unless it is passed explicitly.
        self.finish(payload, set_content_type=content_type)

    def send_api_error(self, error: BaseException) -> None:
        self.set_status(status_for(error))
        self.set_header("Content-Type", "application/json")
        self.finish(json.dumps(error_envelope(error)))
