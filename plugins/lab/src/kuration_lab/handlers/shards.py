"""Shard browsing: manifest entries (JSON) and array payloads (npz bytes).

The entry list is materialized inside the request — loaders' snapshot-currency guard forbids
holding iterators across requests. Integrity verification (whole-shard SHA-256) defaults off
for interactive browsing; the client exposes it as a toggle.
"""
from __future__ import annotations

import io

from tornado import web

from .base import KurationAPIHandler, api_errors


class ShardsHandler(KurationAPIHandler):
    @web.authenticated
    @api_errors
    async def post(self):
        from kuration.core.loaders import shard_entries

        body = self.body_json()
        dataset = self.body_path(body, "dataset")
        entries, container = await self.run_sync(shard_entries, dataset,
                                                 split=body.get("split"))
        self.finish_json({"entries": entries, "container": container})


class ShardArraysHandler(KurationAPIHandler):
    """One shard's arrays as an npz payload, optionally sliced along time."""

    @web.authenticated
    @api_errors
    async def post(self):
        body = self.body_json()
        dataset = self.body_path(body, "dataset")
        entry = body.get("entry")
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise web.HTTPError(400, reason="request needs an 'entry' object from /shards")
        start = body.get("start")
        stop = body.get("stop")

        def _run() -> bytes:
            import numpy as np

            from kuration.core.loaders import read_shard_arrays, shard_entries

            # Re-derive the trusted entry from the manifest instead of reading whatever
            # path the client sent: match on the entry's manifest identity.
            entries, container = shard_entries(dataset)
            wanted = {k: entry.get(k) for k in ("path", "source_id", "shard_index")}
            matched = next(
                (e for e in entries
                 if e.get("path") == wanted["path"]
                 or (e.get("source_id") == wanted["source_id"]
                     and e.get("shard_index") == wanted["shard_index"])),
                None)
            if matched is None:
                raise FileNotFoundError(
                    f"shard not found in dataset manifest: {wanted['path']!r}")
            arrays = read_shard_arrays(matched, container,
                                       modalities=body.get("modalities"),
                                       trim=body.get("trim", True),
                                       verify=bool(body.get("verify", False)))
            if start is not None or stop is not None:
                arrays = {name: a[start:stop] for name, a in arrays.items()}
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays)
            return buffer.getvalue()

        payload = await self.run_sync(_run)
        self.set_header("Content-Disposition", "attachment; filename=shard.npz")
        self.finish_bytes(payload, "application/x-npz")
