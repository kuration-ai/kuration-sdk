"""Streaming shard writer — emit an admitted source's shards from a lazy tick iterator,
holding at most one shard's worth of records per (entity, span).

The out-of-core counterpart to `write_track`: instead of materializing a whole `Track` and
chunking it, it streams the source once, routes each tick's records into the right
(entity, span) accumulator, and flushes a shard the moment one fills. Memory is bounded by
`shard_length × (concurrent admitted entity-spans for the source)`, never the source length.
short_policy (keep/pad/drop) is applied to each accumulator's tail at end-of-stream.
"""
from __future__ import annotations

from ..io import exists as _exists, file_integrity
from .base import PassthroughEncoder
from .containers import container_extension, write_shard
from .shardwriter import _build_arrays, _safe_split, entity_id_token, track_name
from .track import PRESENT_KEY


def stream_source_shards(out_dir: str, source_id: str, tick_iter, jobs: list[dict], *,
                         schema, spec, encoder=None, resume: bool = False, media_provider=None,
                         resume_entries=None, registry=None, expected_ticks: int | None = None,
                         expected_entities=None) -> list[dict]:
    """`jobs`: [{eid, span_index, start, end, split, name_suffix, group}] for this source."""
    encoder = encoder or PassthroughEncoder()
    ext = container_extension(spec.container, registry=registry)
    shape = {m.name: list(m.shape) for m in spec.modalities}
    dtype = {m.name: m.dtype for m in spec.modalities}
    if spec.presence_mask:
        shape["_present"] = []
        dtype["_present"] = "bool"
    from ..provenance import publishable_source_id
    published_source_id = publishable_source_id(source_id)
    st: dict = {}
    for j in jobs:
        split = _safe_split(j["split"])
        st[(j["eid"], j["span_index"])] = {
            "buf": [], "shard": 0, "entries": [], "start": j["start"], "end": j["end"],
            "split": split, "eid": j["eid"], "suffix": j.get("name_suffix", ""),
            "group": j.get("group"), "meta": dict(j.get("meta") or {}), "consumed": 0}

    def _flush(key, pad_to=None):
        s = st[key]
        real_len = len(s["buf"])
        if real_len == 0:
            return
        piece, padded = s["buf"], False
        if pad_to and real_len < pad_to:
            piece, padded = s["buf"] + [None] * (pad_to - real_len), True
        ctx = {"schema": schema, "track": {**s["meta"], "group": s["group"],
                                             "source_id": source_id,
                                             "entity_id": s["eid"]},
               "media_provider": media_provider}
        name = track_name(published_source_id, s["eid"], s["suffix"])
        base = f"{out_dir}/{s['split']}/{name}_shard{s['shard']:04d}" + ("_padded" if padded else "")
        candidate = base + ext
        previous = (resume_entries or {}).get(candidate)
        reusable = False
        if resume and previous and _exists(candidate) and previous.get("sha256"):
            actual = file_integrity(candidate)
            reusable = (actual["sha256"] == previous["sha256"]
                        and actual["size_bytes"] == previous.get("size_bytes"))
        if reusable:
            path = candidate
            integrity = actual
        else:
            path = write_shard(base, spec.container, _build_arrays(piece, spec, encoder, ctx),
                               real_len, registry=registry)
            integrity = file_integrity(path)
        s["entries"].append({"path": path, "source_id": published_source_id,
                             "entity_id": entity_id_token(s["eid"]),
                             "split": s["split"], "track_id": name,
                             "shard_index": s["shard"], "n_steps": real_len,
                             "padded": padded, "shape": shape, "dtype": dtype,
                             **integrity})
        s["shard"] += 1
        s["buf"] = []

    observed_ticks = 0
    observed_entities = set()
    for t, tk in enumerate(tick_iter):
        observed_ticks = t + 1
        observed_entities.update(tk.entities)
        for key, s in st.items():
            if s["start"] <= t < s["end"]:
                rec = dict(tk.entities.get(s["eid"]) or {})
                for gname, gval in (tk.globals or {}).items():
                    rec.setdefault(gname, gval)
                for mname, ref in (tk.media or {}).items():
                    rec[f"media:{mname}"] = ref
                rec[PRESENT_KEY] = s["eid"] in tk.entities
                s["buf"].append(rec)
                s["consumed"] += 1
                if len(s["buf"]) == spec.shard_length:
                    _flush(key)
    expected_entity_set = (None if expected_entities is None else set(expected_entities))
    source_mismatch = []
    if expected_ticks is not None and observed_ticks != expected_ticks:
        source_mismatch.append(f"tick count {observed_ticks} != expected {expected_ticks}")
    if expected_entity_set is not None and observed_entities != expected_entity_set:
        source_mismatch.append(
            f"entity roster {sorted(map(str, observed_entities))} != expected "
            f"{sorted(map(str, expected_entity_set))}")
    invalid = [
        {"entity_id": entity_id_token(s["eid"]), "start": s["start"], "end": s["end"],
         "consumed": s["consumed"]}
        for s in st.values()
        if observed_ticks < s["end"] or s["consumed"] != s["end"] - s["start"]]
    if invalid or source_mismatch:
        raise ValueError(
            f"source {source_id!r} changed or ended early while writing selected spans: "
            f"observed {observed_ticks} ticks; source mismatch {source_mismatch}; "
            f"incomplete jobs {invalid}")
    for key, s in st.items():
        if s["buf"]:
            if spec.short_policy == "drop":
                continue
            _flush(key, pad_to=spec.shard_length if spec.short_policy == "pad" else None)

    return [e for s in st.values() for e in s["entries"]]
