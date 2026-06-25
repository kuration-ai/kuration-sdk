"""`SchemaSpec` — the declared vocabulary a dataset exposes to the metric DSL.

This is what makes the closed grammar (`dsl/grammar.py`) dataset-agnostic without
opening it up: instead of hardcoding which fields exist (the way a CS2-specific DSL
would), the grammar validates every source/map/filter/scope token against a schema
supplied as data. A schema is pure JSON, validated loudly at load, and embedded in a
run's `config.json` so a dataset reproduces from its own artifact.

Field `semantics` tags are the load-bearing idea: they let *universal* maps stay safe
without naming fields. A field tagged `position` is the only thing `speed` and the
spatial reduces will accept; `angular` drives wrap-around in `delta`/`angle_delta`;
`bitmask` (with a declared `bits` map) is what `bit:`/`flag:` decode; `max_rate` is the
per-field outlier clamp (the generalization of CS2's `sv_maxvelocity` teleport guard).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

_KINDS = {"scalar", "vector", "categorical"}
_SEMANTICS = {None, "position", "angular", "bitmask"}


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str                       # scalar | vector | categorical
    dims: tuple[str, ...] = ()      # vector component names, e.g. ("x","y","z")
    semantics: Optional[str] = None  # None | position | angular | bitmask
    bits: dict[str, int] = field(default_factory=dict)  # bitmask: NAME -> bit value
    max_rate: Optional[float] = None  # outlier clamp for speed/delta (units/sec)

    @property
    def width(self) -> int:
        """Number of numeric components (D in the (T,E,D) buffer). 1 for scalar."""
        return len(self.dims) if self.kind == "vector" else 1


def _field_from_dict(name: str, d: dict) -> FieldSpec:
    kind = d.get("kind")
    if kind not in _KINDS:
        raise ValueError(f"field {name!r}: kind must be one of {sorted(_KINDS)}, got {kind!r}")
    dims = tuple(d.get("dims") or ())
    semantics = d.get("semantics")
    if semantics not in _SEMANTICS:
        raise ValueError(
            f"field {name!r}: semantics must be one of {sorted(s for s in _SEMANTICS if s)} "
            f"or absent, got {semantics!r}")
    bits = {str(k): int(v) for k, v in (d.get("bits") or {}).items()}
    max_rate = d.get("max_rate")

    if kind == "vector" and not dims:
        raise ValueError(f"field {name!r}: vector fields need a non-empty 'dims' list")
    if kind != "vector" and dims:
        raise ValueError(f"field {name!r}: 'dims' is only valid for vector fields")
    if semantics == "position" and kind != "vector":
        raise ValueError(f"field {name!r}: semantics 'position' requires a vector field")
    if semantics == "position" and len(dims) < 2:
        raise ValueError(f"field {name!r}: 'position' needs >=2 dims (a spatial plane)")
    if semantics == "angular" and kind not in ("scalar", "vector"):
        raise ValueError(f"field {name!r}: semantics 'angular' requires scalar/vector")
    if semantics == "bitmask":
        if kind != "scalar":
            raise ValueError(f"field {name!r}: semantics 'bitmask' requires a scalar field")
        if not bits:
            raise ValueError(f"field {name!r}: bitmask fields need a non-empty 'bits' map")
    if bits and semantics != "bitmask":
        raise ValueError(f"field {name!r}: 'bits' is only valid on a bitmask field")
    return FieldSpec(name=name, kind=kind, dims=dims, semantics=semantics,
                     bits=bits, max_rate=max_rate)


@dataclass(frozen=True)
class SchemaSpec:
    name: str
    fields: dict[str, FieldSpec]
    entity_axis: Optional[str] = None          # e.g. "player"; None for single-stream data
    sample_hz: Optional[float] = None          # enables time-unit windowing + speed scaling
    event_streams: dict[str, dict] = field(default_factory=dict)  # name -> {fields: {...FieldSpec}}
    globals: dict[str, FieldSpec] = field(default_factory=dict)   # per-step scalar streams

    # ---- construction -------------------------------------------------------
    @staticmethod
    def from_dict(d: dict) -> "SchemaSpec":
        name = d.get("name")
        if not name:
            raise ValueError("schema needs a 'name'")
        raw_fields = d.get("fields") or {}
        if not raw_fields:
            raise ValueError(f"schema {name!r}: at least one field is required")
        fields = {n: _field_from_dict(n, fd) for n, fd in raw_fields.items()}

        entity_axis = d.get("entity_axis")
        sample_hz = d.get("sample_hz")
        if sample_hz is not None and not (isinstance(sample_hz, (int, float)) and sample_hz > 0):
            raise ValueError(f"schema {name!r}: sample_hz must be a positive number")

        event_streams: dict[str, dict] = {}
        for sn, sd in (d.get("event_streams") or {}).items():
            sfields = {fn: _field_from_dict(f"{sn}.{fn}", fd)
                       for fn, fd in (sd.get("fields") or {}).items()}
            if not sfields:
                raise ValueError(f"schema {name!r}: event stream {sn!r} needs fields")
            event_streams[sn] = {"fields": sfields}

        globals_ = {}
        for gn, gd in (d.get("globals") or {}).items():
            gf = _field_from_dict(gn, gd)
            if gf.kind != "scalar":
                raise ValueError(f"schema {name!r}: global {gn!r} must be scalar")
            globals_[gn] = gf

        # Namespaces must not collide — a source token resolves by prefix.
        clash = set()
        if entity_axis:
            clash.add(entity_axis)
        for ns in list(event_streams) + list(globals_):
            if ns in clash:
                raise ValueError(f"schema {name!r}: namespace {ns!r} declared more than once")
            clash.add(ns)

        return SchemaSpec(name=name, fields=fields, entity_axis=entity_axis,
                          sample_hz=sample_hz, event_streams=event_streams, globals=globals_)

    def to_dict(self) -> dict:
        """Round-trippable plain dict for embedding in config.json."""
        def fd(f: FieldSpec) -> dict:
            out = {"kind": f.kind}
            if f.dims:
                out["dims"] = list(f.dims)
            if f.semantics:
                out["semantics"] = f.semantics
            if f.bits:
                out["bits"] = dict(f.bits)
            if f.max_rate is not None:
                out["max_rate"] = f.max_rate
            return out
        d: dict = {"name": self.name, "fields": {n: fd(f) for n, f in self.fields.items()}}
        if self.entity_axis:
            d["entity_axis"] = self.entity_axis
        if self.sample_hz is not None:
            d["sample_hz"] = self.sample_hz
        if self.event_streams:
            d["event_streams"] = {sn: {"fields": {fn: fd(f) for fn, f in s["fields"].items()}}
                                  for sn, s in self.event_streams.items()}
        if self.globals:
            d["globals"] = {n: fd(f) for n, f in self.globals.items()}
        return d
