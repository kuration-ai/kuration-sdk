"""ShardWriter — chunk a track into shards, build modality arrays, apply short-policy, write.

Generic mechanics, no domain knowledge: the genuinely format-specific work (deriving a frame
or an action tensor for an `encoder:` modality) is delegated to the `OutputEncoder`.
short_policy generalizes the reference chunker:
  keep  — last shard is variable-length (its real tail)
  pad   — last shard padded to shard_length with zero rows (padded flag set)
  drop  — last short shard dropped (paper-strict, loses the tail)
"""
from __future__ import annotations

import hashlib
import re
from typing import Optional

import numpy as np

from .base import OutputEncoder, PassthroughEncoder
from .containers import container_extension, write_shard
from .spec import OutputSpec
from .track import PRESENT_KEY, Track


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


def track_name(source_id, entity_id, name_suffix: str = "") -> str:
    """Filesystem-safe, collision-free shard basename for one (source, entity, span).
    The readable prefix flattens '/' to '_', which can alias distinct pairs
    (('runs/a','b') vs ('runs','a_b')) — the content hash disambiguates."""
    raw = f"{source_id}_{entity_id}{name_suffix}"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip(". _") or "track"
    safe = safe[:96].rstrip(". ")
    h = hashlib.sha256(f"{source_id}\x00{entity_id}\x00{name_suffix}".encode()).hexdigest()[:12]
    return f"{safe}_{h}"


def _build_arrays(piece: list, spec: OutputSpec, encoder: OutputEncoder, ctx: dict) -> dict:
    L = len(piece)
    arrays = {m.name: m.zeros(L) for m in spec.modalities}
    if spec.presence_mask:
        arrays["_present"] = np.zeros(L, dtype=np.bool_)
    enc_mods = [m for m in spec.modalities if m.src_kind == "encoder"]
    clean_piece = [None if rec is None else
                   (rec if PRESENT_KEY not in rec else
                    {k: v for k, v in rec.items() if k != PRESENT_KEY}) for rec in piece]
    enc_batch = None
    if enc_mods and hasattr(encoder, "encode_batch"):
        enc_batch = encoder.encode_batch(clean_piece, ctx)
        if not isinstance(enc_batch, dict):
            raise ValueError("output encoder encode_batch must return a dict")
        for key, value in enc_batch.items():
            if np.asarray(value).shape[:1] != (L,):
                raise ValueError(f"output encoder batch {key!r} needs leading dimension {L}")
    media_values: dict[str, np.ndarray] = {}
    media_mods = [m for m in spec.modalities if m.src_kind == "media"]
    for m in media_mods:
        positions, refs = [], []
        for i, rec in enumerate(clean_piece):
            if rec is not None and rec.get(f"media:{m.src_arg}") is not None:
                positions.append(i)
                refs.append(rec[f"media:{m.src_arg}"])
        provider = ctx.get("media_provider")
        if refs and provider is None:
            raise ValueError(f"media:{m.src_arg} output needs the input adapter's media provider")
        decoded = provider.decode_batch(refs) if refs else np.empty((0, *m.shape), dtype=m.dtype)
        values = m.zeros(L)
        for pos, value in zip(positions, decoded):
            values[pos] = value
        media_values[m.name] = values
    for t, rec in enumerate(piece):
        if rec is None:           # pad row — leave zeros
            continue
        present = bool(rec.get(PRESENT_KEY, True))
        clean = rec if PRESENT_KEY not in rec else {k: v for k, v in rec.items() if k != PRESENT_KEY}
        if spec.presence_mask:
            arrays["_present"][t] = present
        enc_out = ({k: np.asarray(v)[t] for k, v in enc_batch.items()} if enc_batch is not None
                   else (encoder.timestep(clean, ctx) if enc_mods else {}))
        if not isinstance(enc_out, dict):
            raise ValueError(f"output encoder must return a dict, got {type(enc_out).__name__} "
                             f"for source {ctx['track'].get('source_id')!r} at row {t}")
        for m in spec.modalities:
            if m.src_kind == "field":
                val = clean.get(m.src_arg)
            elif m.src_kind == "vocab":
                raw = clean.get(m.src_arg)
                vocab = (spec.vocabs or {}).get(m.src_arg, {})
                val = vocab.get(str(raw)) if raw is not None else None
            elif m.src_kind == "segment":
                val = ctx["track"].get(m.src_arg)
            elif m.src_kind == "media":
                val = media_values[m.name][t]
            else:
                val = enc_out.get(m.src_arg)
            if val is None:
                is_global = bool(ctx.get("schema") and m.src_arg in ctx["schema"].globals)
                required = (m.src_kind == "segment" or is_global or present)
                if required:
                    raise ValueError(
                        f"missing value for output modality {m.name!r} "
                        f"({m.src_kind}:{m.src_arg}) in source "
                        f"{ctx['track'].get('source_id')!r}, entity "
                        f"{ctx['track'].get('entity_id')!r}, row {t}")
                continue
            try:
                arrays[m.name][t] = val
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"bad value for output modality {m.name!r} ({m.src_kind}:{m.src_arg}) "
                    f"at source {ctx['track'].get('source_id')!r}, entity "
                    f"{ctx['track'].get('entity_id')!r}, row {t}: expected dtype={m.dtype}, "
                    f"shape={m.shape}, got {val!r}") from e
    return arrays


def write_track(out_dir: str, track: Track, spec: OutputSpec, *,
                encoder: Optional[OutputEncoder] = None, schema=None,
                split: str = "all", resume: bool = False, name_suffix: str = "",
                media_provider=None) -> list[dict]:
    """Write all shards for one track; returns manifest entries. With `resume=True`, a shard
    whose file already exists is left as-is (no re-encode) but still recorded in the manifest —
    chunk boundaries are deterministic, so the entry is reconstructable without rewriting.
    `name_suffix` disambiguates shard filenames across sub-track spans of one track."""
    from ..io import exists as _exists
    if any(m.src_kind in {"field", "media", "vocab"} for m in spec.modalities):
        if schema is None:
            raise ValueError("write_track needs schema when OutputSpec contains field: modalities")
        spec.validate_schema(schema)
    encoder = encoder or PassthroughEncoder()
    ctx = {"schema": schema, "track": {**track.meta, "source_id": track.source_id,
                                       "entity_id": track.entity_id},
           "media_provider": media_provider}
    name = track_name(track.source_id, track.entity_id, name_suffix)
    ext = container_extension(spec.container)
    entries: list[dict] = []
    shape = {m.name: list(m.shape) for m in spec.modalities}
    dtype = {m.name: m.dtype for m in spec.modalities}
    if spec.presence_mask:
        shape["_present"] = []
        dtype["_present"] = "bool"
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
                        "shape": shape, "dtype": dtype})
    return entries
