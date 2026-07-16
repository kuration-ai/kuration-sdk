"""`Track` — one entity's per-timestep record sequence: the output unit.

A shard is a chunk of a track (the generalization of asg2diamond's per-POV episode: there a
clip = one player's frame sequence). Splitting a canonical `Tick` stream into tracks is
generic: one track per entity that appears in the source, each record = that entity's fields
at a timestep merged with the tick's globals (so a `field:` modality can pull either).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..buffer import Tick
from ..schema import SchemaSpec

PRESENT_KEY = "__kuration_entity_present__"


@dataclass
class Track:
    source_id: str
    entity_id: Any
    records: list[dict]              # records[t] = {field/global -> value} or {} if entity absent
    meta: dict = field(default_factory=dict)   # group, split, segment-derived scalars

    @property
    def n_steps(self) -> int:
        return len(self.records)


def tracks_from_ticks(schema: SchemaSpec, ticks: list[Tick], source_id: str) -> list[Track]:
    """One Track per entity seen in the source (full-source span). Globals are merged into
    every record so they're addressable as `field:` modalities."""
    entity_ids: list = []
    seen = set()
    for tk in ticks:
        for eid in tk.entities:
            if eid not in seen:
                seen.add(eid)
                entity_ids.append(eid)
    tracks: list[Track] = []
    for eid in entity_ids:
        records = []
        for tk in ticks:
            rec = dict(tk.entities.get(eid) or {})
            for gname, gval in (tk.globals or {}).items():
                rec.setdefault(gname, gval)
            for mname, ref in (tk.media or {}).items():
                rec[f"media:{mname}"] = ref
            rec[PRESENT_KEY] = eid in tk.entities
            records.append(rec)
        tracks.append(Track(source_id=source_id, entity_id=eid, records=records))
    return tracks
