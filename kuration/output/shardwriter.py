"""ShardWriter — chunk a track into shards, build modality arrays, apply short-policy, write.

Generic mechanics, no domain knowledge: the genuinely format-specific work (deriving a frame
or an action tensor for an `encoder:` modality) is delegated to the `OutputEncoder`.
short_policy generalizes the reference chunker:
  keep  — last shard is variable-length (its real tail)
  pad   — last shard padded to shard_length with zero rows (padded flag set)
  drop  — last short shard dropped (paper-strict, loses the tail)
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .base import OutputEncoder, PassthroughEncoder
from .containers import CONTAINER_EXT, write_shard
from .spec import OutputSpec
from .track import Track


def _chunks(records: list[dict], shard_length: int, short_policy: str):
    """Yield (piece, real_len, padded). For pad, `piece` is padded to shard_length with
    None rows; for keep, the tail piece is shorter; for drop, the tail is skipped."""
    n = len(records)
    n_full = n // shard_length
    for k in range(n_full):
        yield records[k * shard_length:(k + 1) * shard_length], shard_length, False
    rem = n - n_full * shard_length
    if rem == 0:
        return
    tail = records[n_full * shard_length:]
    if short_policy == "drop":
        return
    if short_policy == "pad":
        yield tail + [None] * (shard_length - rem), rem, True
    else:  # keep
        yield tail, rem, False


def _build_arrays(piece: list, spec: OutputSpec, encoder: OutputEncoder, ctx: dict) -> dict:
    L = len(piece)
    arrays = {m.name: m.zeros(L) for m in spec.modalities}
    enc_mods = [m for m in spec.modalities if m.src_kind == "encoder"]
    for t, rec in enumerate(piece):
        if rec is None:           # pad row — leave zeros
            continue
        enc_out = encoder.timestep(rec, ctx) if enc_mods else {}
        for m in spec.modalities:
            if m.src_kind == "field":
                val = rec.get(m.src_arg)
            elif m.src_kind == "segment":
                val = ctx["track"].get(m.src_arg)
            else:
                val = enc_out.get(m.src_arg)
            if val is not None:
                arrays[m.name][t] = val
    return arrays


def write_track(out_dir: str, track: Track, spec: OutputSpec, *,
                encoder: Optional[OutputEncoder] = None, schema=None,
                split: str = "all", resume: bool = False, name_suffix: str = "") -> list[dict]:
    """Write all shards for one track; returns manifest entries. With `resume=True`, a shard
    whose file already exists is left as-is (no re-encode) but still recorded in the manifest —
    chunk boundaries are deterministic, so the entry is reconstructable without rewriting.
    `name_suffix` disambiguates shard filenames across sub-track spans of one track."""
    from ..io import exists as _exists
    encoder = encoder or PassthroughEncoder()
    ctx = {"schema": schema, "track": {**track.meta, "source_id": track.source_id,
                                       "entity_id": track.entity_id}}
    name = f"{track.source_id}_{track.entity_id}{name_suffix}".replace("/", "_")
    ext = CONTAINER_EXT[spec.container]
    entries: list[dict] = []
    for k, (piece, real_len, padded) in enumerate(
            _chunks(track.records, spec.shard_length, spec.short_policy)):
        base = f"{out_dir}/{split}/{name}_shard{k:04d}" + ("_padded" if padded else "")
        if resume and _exists(base + ext):
            path = base + ext                       # already written — skip the re-encode
        else:
            path = write_shard(base, spec.container, _build_arrays(piece, spec, encoder, ctx),
                               real_len)
        entries.append({"path": path, "source_id": track.source_id,
                        "entity_id": str(track.entity_id), "split": split,
                        "shard_index": k, "n_steps": real_len, "padded": padded,
                        "shape": {m.name: list(m.shape) for m in spec.modalities}})
    return entries
