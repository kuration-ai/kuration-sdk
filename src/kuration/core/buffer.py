"""`WindowBuffer` — one segment's window extracted into shared numpy buffers.

The generalization of a CS2-specific `WindowArrays`: storage is driven entirely by the
schema, not by hardcoded player fields. Every numeric field is `(T, E, D)` over a
per-window entity roster (E = number of distinct entities seen; E = 1 for single-stream
schemas), with absent cells NaN and a `valid (T, E)` presence mask so temporal ops
(`delta`/`speed`) stay NaN-safe across entities appearing/leaving mid-window.

All of a strategy's metrics evaluate against one shared buffer per segment, so the
per-window extraction cost is paid once, not once per metric — the amortization that
keeps a whole profile cheap. Adapters (`adapters/`) are what build these from raw records;
tests build them directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .schema import SchemaSpec


@dataclass
class Tick:
    """One timestep in the SDK's canonical record form — the interface between an
    `InputAdapter` (raw → canonical) and the generic buffer builder. An adapter's *only*
    real job is to produce a stream of these; everything downstream is schema-driven and
    domain-free.

      entities: {entity_id -> {field_name -> value}}   (value: number, [components], or str)
      events:   {stream -> [ {field -> value}, ... ]}
      globals:  {name -> scalar}
    """
    time: Any
    entities: dict
    events: dict = field(default_factory=dict)
    globals: dict = field(default_factory=dict)
    media: dict = field(default_factory=dict)  # name -> lazy media reference


@dataclass
class WindowBuffer:
    schema: SchemaSpec
    T: int
    E: int
    numeric: dict[str, np.ndarray]   # field name -> (T, E, D) float, NaN absent (scalar D=1)
    codes: dict[str, np.ndarray]     # categorical field -> (T, E) int, -1 absent
    vocabs: dict[str, list[str]]     # categorical field -> code -> string
    valid: np.ndarray                # (T, E) bool
    events: dict[str, list[list[dict]]] = field(default_factory=dict)  # stream -> per-tick events
    globals: dict[str, np.ndarray] = field(default_factory=dict)       # name -> (T,) float

    def numeric_field(self, name: str) -> np.ndarray:
        """(T, E, D) for a scalar/vector field; raises if not numeric/declared."""
        if name not in self.numeric:
            raise KeyError(f"numeric field {name!r} not in buffer")
        return self.numeric[name]

    def categorical_field(self, name: str) -> tuple[np.ndarray, list[str]]:
        if name not in self.codes:
            raise KeyError(f"categorical field {name!r} not in buffer")
        return self.codes[name], self.vocabs.get(name, [])


def empty_like(schema: SchemaSpec, T: int, E: int) -> WindowBuffer:
    """Allocate a zero/NaN buffer matching `schema` — a convenience for adapters/tests
    that then fill cells in. Numeric fields are NaN, codes -1, nothing valid yet."""
    E = max(E, 1)
    numeric: dict[str, np.ndarray] = {}
    codes: dict[str, np.ndarray] = {}
    vocabs: dict[str, list[str]] = {}
    for name, f in schema.fields.items():
        if f.kind == "categorical":
            codes[name] = np.full((T, E), -1, dtype=np.int64)
            vocabs[name] = []
        else:
            numeric[name] = np.full((T, E, f.width), np.nan, dtype=np.float64)
    globals_ = {g: np.full(T, np.nan, dtype=np.float64) for g in schema.globals}
    events: dict[str, list[list[dict]]] = {
        s: [[] for _ in range(T)] for s in schema.event_streams}
    return WindowBuffer(schema=schema, T=T, E=E, numeric=numeric, codes=codes,
                        vocabs=vocabs, valid=np.zeros((T, E), dtype=bool),
                        events=events, globals=globals_)


def build_window(schema: SchemaSpec, ticks: list[Tick]) -> WindowBuffer:
    """Build a `WindowBuffer` from a window of canonical `Tick`s — generic, schema-driven,
    domain-free. The entity roster is the union of entity ids seen in the window (so an
    entity appearing/leaving mid-window is handled), absent cells stay NaN/-1, and
    categorical strings are coded into a per-window vocab. This is the amortization point:
    one buffer per segment, shared by all of a profile's metrics."""
    T = len(ticks)
    roster: dict[Any, int] = {}
    for tk in ticks:
        for eid in tk.entities:
            roster.setdefault(eid, len(roster))
    w = empty_like(schema, T, len(roster))

    cat_codes: dict[str, dict[str, int]] = {n: {} for n, f in schema.fields.items()
                                            if f.kind == "categorical"}
    for ti, tk in enumerate(ticks):
        for eid, fvals in tk.entities.items():
            j = roster[eid]
            w.valid[ti, j] = True
            for fname, fs in schema.fields.items():
                val = fvals.get(fname)
                if val is None:
                    continue
                if fs.kind == "categorical":
                    voc = cat_codes[fname]
                    w.codes[fname][ti, j] = voc.setdefault(str(val), len(voc))
                elif fs.kind == "vector":
                    vector = np.asarray(val)
                    if vector.ndim != 1 or vector.shape[0] != fs.width:
                        raise ValueError(
                            f"field {fname!r} is a {fs.width}-dim vector but entity {eid!r} "
                            f"supplied shape {vector.shape} at tick {ti}")
                    w.numeric[fname][ti, j, :] = vector
                else:
                    w.numeric[fname][ti, j, 0] = float(val)
        for stream, evs in (tk.events or {}).items():
            if stream in w.events:
                w.events[stream][ti] = list(evs)
        for gname, gval in (tk.globals or {}).items():
            if gname in w.globals and gval is not None:
                w.globals[gname][ti] = float(gval)

    for fname, voc in cat_codes.items():
        arr: list = [None] * len(voc)
        for s, c in voc.items():
            arr[c] = s
        w.vocabs[fname] = arr
    return w
