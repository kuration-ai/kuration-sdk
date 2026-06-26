"""Streaming shard writer — emit an admitted source's shards from a lazy tick iterator,
holding at most one shard's worth of records per (entity, span).

The out-of-core counterpart to `write_track`: instead of materializing a whole `Track` and
chunking it, it streams the source once, routes each tick's records into the right
(entity, span) accumulator, and flushes a shard the moment one fills. Memory is bounded by
`shard_length × (concurrent admitted entity-spans for the source)`, never the source length.
short_policy (keep/pad/drop) is applied to each accumulator's tail at end-of-stream.
"""
from __future__ import annotations

from ..io import exists as _exists
from .base import PassthroughEncoder
from .containers import CONTAINER_EXT, write_shard
from .shardwriter import _build_arrays


def stream_source_shards(out_dir: str, source_id: str, tick_iter, jobs: list[dict], *,
                         schema, spec, encoder=None, resume: bool = False) -> list[dict]:
    """`jobs`: [{eid, span_index, start, end, split, name_suffix, group}] for this source."""
    encoder = encoder or PassthroughEncoder()
    ext = CONTAINER_EXT[spec.container]
    shape = {m.name: list(m.shape) for m in spec.modalities}
    st: dict = {}
    for j in jobs:
        st[(j["eid"], j["span_index"])] = {
            "buf": [], "shard": 0, "entries": [], "start": j["start"], "end": j["end"],
            "split": j["split"], "eid": j["eid"], "suffix": j.get("name_suffix", ""),
            "group": j.get("group")}

    def _flush(key, pad_to=None):
        s = st[key]
        real_len = len(s["buf"])
        if real_len == 0:
            return
        piece, padded = s["buf"], False
        if pad_to and real_len < pad_to:
            piece, padded = s["buf"] + [None] * (pad_to - real_len), True
        ctx = {"schema": schema, "track": {"group": s["group"], "source_id": source_id,
                                           "entity_id": s["eid"]}}
        name = f"{source_id}_{s['eid']}{s['suffix']}".replace("/", "_")
        base = f"{out_dir}/{s['split']}/{name}_shard{s['shard']:04d}" + ("_padded" if padded else "")
        if resume and _exists(base + ext):
            path = base + ext
        else:
            path = write_shard(base, spec.container, _build_arrays(piece, spec, encoder, ctx), real_len)
        s["entries"].append({"path": path, "source_id": source_id, "entity_id": str(s["eid"]),
                             "split": s["split"], "shard_index": s["shard"], "n_steps": real_len,
                             "padded": padded, "shape": shape})
        s["shard"] += 1
        s["buf"] = []

    for t, tk in enumerate(tick_iter):
        for key, s in st.items():
            if s["start"] <= t < s["end"]:
                rec = dict(tk.entities.get(s["eid"]) or {})
                for gname, gval in (tk.globals or {}).items():
                    rec.setdefault(gname, gval)
                s["buf"].append(rec)
                if len(s["buf"]) == spec.shard_length:
                    _flush(key)
    for key, s in st.items():
        if s["buf"]:
            if spec.short_policy == "drop":
                continue
            _flush(key, pad_to=spec.shard_length if spec.short_policy == "pad" else None)

    return [e for s in st.values() for e in s["entries"]]
