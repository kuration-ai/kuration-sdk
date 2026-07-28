"""`SchemaSpec` — the declared vocabulary a dataset exposes to the metric DSL.

This is what makes the closed grammar (`dsl/grammar.py`) dataset-agnostic without
opening it up: instead of hardcoding which fields exist (the way a CS2-specific DSL
would), the grammar validates every source/map/filter/scope token against a schema
supplied as data. A schema is pure JSON, validated loudly at load, and embedded in a
run's ``config.json`` as part of its auditable contract. Replay also requires the same immutable
or content-addressed sources and compatible component implementations.

Field `semantics` tags are the load-bearing idea: they let *universal* maps stay safe
without naming fields. A field tagged `position` is the only thing `speed` and the
spatial reduces will accept; `angular` drives wrap-around in `delta`/`angle_delta`;
`bitmask` (with a declared `bits` map) is what `bit:`/`flag:` decode; `max_rate` is the
per-field outlier clamp (the generalization of CS2's `sv_maxvelocity` teleport guard).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Optional

_KINDS = {"scalar", "vector", "categorical"}
_SEMANTICS = {None, "position", "angular", "bitmask"}
_MEDIA_KINDS = {"video-frame"}
_MEDIA_LAYOUTS = {"HWC", "CHW"}
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Dimension labels are user/domain vocabulary, not Python identifiers.  Keep them addressable
# by the DSL's ``field.dimension`` syntax while allowing useful labels such as
# ``mouse_x_-1000`` from discretized action spaces.
_DIMENSION = re.compile(r"^[^.\s]+$")


class FrozenDict(dict):
    """Small pickle-friendly immutable mapping used by frozen public specifications."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("specification mappings are immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable

    def __reduce__(self):
        return (type(self), (dict(self),))


def _identifier(value, *, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a non-empty DSL identifier "
                         "([A-Za-z_][A-Za-z0-9_]*)")
    return value


def _validated_bits(field_name: str, values) -> FrozenDict:
    try:
        raw = dict(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"field {field_name!r}: bits must be a mapping") from error
    bits = {}
    for bit_name, bit_value in raw.items():
        bit_name = _identifier(
            bit_name, label=f"field {field_name!r} bit name"
        )
        if type(bit_value) is not int or bit_value <= 0:
            raise ValueError(
                f"field {field_name!r}: bit {bit_name!r} must have a positive integer "
                f"value, got {bit_value!r}")
        bits[bit_name] = bit_value
    return FrozenDict(bits)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str                       # scalar | vector | categorical
    dims: tuple[str, ...] = ()      # vector component names, e.g. ("x","y","z")
    semantics: Optional[str] = None  # None | position | angular | bitmask
    bits: dict[str, int] = field(default_factory=dict)  # bitmask: NAME -> bit value
    max_rate: Optional[float] = None  # outlier clamp for speed/delta (units/sec)

    def __post_init__(self) -> None:
        _identifier(self.name, label="field name")
        if self.kind not in _KINDS:
            raise ValueError(f"field {self.name!r}: kind must be one of {sorted(_KINDS)}")
        dims = tuple(self.dims)
        for dim in dims:
            if not isinstance(dim, str) or not _DIMENSION.fullmatch(dim):
                raise ValueError(f"field {self.name!r} dimension must be non-empty and cannot "
                                 "contain dots or whitespace")
        if len(set(dims)) != len(dims):
            raise ValueError(f"field {self.name!r}: vector dimensions must be unique")
        if self.kind == "vector" and not dims:
            raise ValueError(f"field {self.name!r}: vector fields need non-empty dimensions")
        if self.kind != "vector" and dims:
            raise ValueError(f"field {self.name!r}: dimensions are only valid for vectors")
        if self.semantics not in _SEMANTICS:
            raise ValueError(f"field {self.name!r}: invalid semantics {self.semantics!r}")
        if self.semantics == "position" and (self.kind != "vector" or len(dims) < 2):
            raise ValueError(f"field {self.name!r}: position semantics need a >=2-D vector")
        if self.semantics == "angular" and self.kind not in {"scalar", "vector"}:
            raise ValueError(f"field {self.name!r}: angular semantics need scalar/vector")
        bits = _validated_bits(self.name, self.bits)
        if self.semantics == "bitmask" and (self.kind != "scalar" or not bits):
            raise ValueError(f"field {self.name!r}: bitmask semantics need a scalar and bits")
        if bits and self.semantics != "bitmask":
            raise ValueError(f"field {self.name!r}: bits are only valid for bitmasks")
        max_rate = self.max_rate
        if max_rate is not None:
            if (not isinstance(max_rate, (int, float)) or isinstance(max_rate, bool)
                    or not math.isfinite(float(max_rate)) or float(max_rate) <= 0):
                raise ValueError(f"field {self.name!r}: max_rate must be finite and positive")
            max_rate = float(max_rate)
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "bits", bits)
        object.__setattr__(self, "max_rate", max_rate)

    @property
    def width(self) -> int:
        """Number of numeric components (D in the (T,E,D) buffer). 1 for scalar."""
        return len(self.dims) if self.kind == "vector" else 1


@dataclass(frozen=True)
class MediaSpec:
    """Semantic description of an auxiliary media column.

    Media is discoverable from the schema, but is deliberately excluded from ``fields`` and
    therefore never copied into a :class:`WindowBuffer` or exposed to the metric grammar.
    """
    name: str
    kind: str
    dtype: str
    shape: tuple[int, ...]
    layout: str = "HWC"
    color_space: str = "RGB"
    timing: str = "pts"
    capabilities: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        import numpy as np
        _identifier(self.name, label="media name")
        if self.kind not in _MEDIA_KINDS:
            raise ValueError(f"media {self.name!r}: invalid kind {self.kind!r}")
        try:
            dtype = np.dtype(self.dtype)
        except (TypeError, ValueError) as error:
            raise ValueError(f"media {self.name!r}: invalid dtype {self.dtype!r}") from error
        if dtype.hasobject:
            raise ValueError(f"media {self.name!r}: object dtype is not portable or safe")
        shape = tuple(self.shape)
        if not shape or any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in shape):
            raise ValueError(f"media {self.name!r}: shape needs positive integer dimensions")
        if self.layout not in _MEDIA_LAYOUTS or len(self.layout) != len(shape):
            raise ValueError(f"media {self.name!r}: layout must match shape and be HWC or CHW")
        if not isinstance(self.color_space, str) or not self.color_space:
            raise ValueError(f"media {self.name!r}: color_space must be non-empty")
        if self.timing not in {"pts", "frame_index"}:
            raise ValueError(f"media {self.name!r}: invalid timing {self.timing!r}")
        caps = FrozenDict(dict(self.capabilities))
        if any(not isinstance(k, str) or not isinstance(v, bool) for k, v in caps.items()):
            raise ValueError(f"media {self.name!r}: capabilities must map names to booleans")
        object.__setattr__(self, "dtype", dtype.name)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "capabilities", caps)


def _media_from_dict(name: str, d: dict) -> MediaSpec:
    if not isinstance(d, dict):
        raise ValueError(f"media {name!r}: specification must be an object")
    unknown = sorted(set(d) - {"kind", "dtype", "shape", "layout", "color_space", "timing",
                               "capabilities"})
    if unknown:
        raise ValueError(f"media {name!r}: unknown key(s) {unknown}")
    import numpy as np
    kind = d.get("kind")
    if kind not in _MEDIA_KINDS:
        raise ValueError(f"media {name!r}: kind must be one of {sorted(_MEDIA_KINDS)}")
    try:
        dtype = np.dtype(d.get("dtype")).name
    except (TypeError, ValueError) as e:
        raise ValueError(f"media {name!r}: invalid dtype {d.get('dtype')!r}") from e
    shape = tuple(d.get("shape") or ())
    if not shape or any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in shape):
        raise ValueError(f"media {name!r}: shape needs positive integer dimensions")
    layout = d.get("layout", "HWC")
    if layout not in _MEDIA_LAYOUTS or len(layout) != len(shape):
        raise ValueError(f"media {name!r}: layout must match shape and be HWC or CHW")
    color = d.get("color_space", "RGB")
    timing = d.get("timing", "pts")
    if not isinstance(color, str) or not color or timing not in {"pts", "frame_index"}:
        raise ValueError(f"media {name!r}: invalid color_space/timing")
    caps = d.get("capabilities") or {}
    if not isinstance(caps, dict) or any(not isinstance(v, bool) for v in caps.values()):
        raise ValueError(f"media {name!r}: capabilities must map names to booleans")
    return MediaSpec(name, kind, dtype, shape, layout, color, timing, dict(caps))


def _field_from_dict(name: str, d: dict) -> FieldSpec:
    if not isinstance(d, dict):
        raise ValueError(f"field {name!r}: specification must be an object")
    unknown = sorted(set(d) - {"kind", "dims", "semantics", "bits", "max_rate"})
    if unknown:
        raise ValueError(f"field {name!r}: unknown key(s) {unknown}")
    kind = d.get("kind")
    if kind not in _KINDS:
        raise ValueError(f"field {name!r}: kind must be one of {sorted(_KINDS)}, got {kind!r}")
    dims = tuple(d.get("dims") or ())
    semantics = d.get("semantics")
    if semantics not in _SEMANTICS:
        raise ValueError(
            f"field {name!r}: semantics must be one of {sorted(s for s in _SEMANTICS if s)} "
            f"or absent, got {semantics!r}")
    bits = _validated_bits(name, d.get("bits") or {})
    max_rate = d.get("max_rate")
    if max_rate is not None:
        try:
            max_rate = float(max_rate)
        except (TypeError, ValueError):
            raise ValueError(f"field {name!r}: max_rate must be numeric, got {max_rate!r}")
        if not math.isfinite(max_rate) or max_rate <= 0:
            raise ValueError(f"field {name!r}: max_rate must be finite and > 0 "
                             f"(it is an outlier clamp), "
                             f"got {max_rate!r}")

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
    media: dict[str, MediaSpec] = field(default_factory=dict)     # lazy auxiliary columns

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("schema name must be a non-empty string")
        if self.entity_axis is not None:
            _identifier(self.entity_axis, label="entity_axis")
        if self.sample_hz is not None and (
                not isinstance(self.sample_hz, (int, float)) or isinstance(self.sample_hz, bool)
                or not math.isfinite(float(self.sample_hz)) or float(self.sample_hz) <= 0):
            raise ValueError(f"schema {self.name!r}: sample_hz must be finite and positive")
        fields = FrozenDict(dict(self.fields))
        globals_ = FrozenDict(dict(self.globals))
        media = FrozenDict(dict(self.media))
        for label, values, expected in (("field", fields, FieldSpec),
                                        ("global", globals_, FieldSpec),
                                        ("media", media, MediaSpec)):
            for name, value in values.items():
                _identifier(name, label=f"{label} name")
                if not isinstance(value, expected) or value.name != name:
                    raise ValueError(f"schema {self.name!r}: {label} {name!r} must be a matching "
                                     f"{expected.__name__}")
        if any(value.kind != "scalar" for value in globals_.values()):
            raise ValueError(f"schema {self.name!r}: globals must be scalar")
        streams = {}
        for stream, spec in dict(self.event_streams).items():
            _identifier(stream, label="event stream name")
            if not isinstance(spec, dict) or set(spec) != {"fields"}:
                raise ValueError(f"event stream {stream!r} needs only a fields mapping")
            stream_fields = FrozenDict(dict(spec["fields"]))
            if not stream_fields:
                raise ValueError(f"event stream {stream!r} needs fields")
            for name, value in stream_fields.items():
                _identifier(name, label=f"event stream {stream!r} field")
                if not isinstance(value, FieldSpec) or value.name != name:
                    raise ValueError(f"event stream {stream!r} field {name!r} must be a matching "
                                     "FieldSpec")
            streams[stream] = FrozenDict({"fields": stream_fields})
        namespaces = ([self.entity_axis] if self.entity_axis else []) + list(streams) + list(globals_)
        if len(namespaces) != len(set(namespaces)):
            raise ValueError(f"schema {self.name!r}: entity/event/global namespaces must be unique")
        if self.entity_axis is None:
            # A vector component is addressed as ``field.dimension`` in a single-stream
            # schema, so a field cannot also be an event/global namespace.
            collision = sorted(set(fields) & (set(globals_) | set(streams)))
            if collision:
                raise ValueError(f"schema {self.name!r}: single-stream fields and event/global "
                                 f"namespaces share unqualified names {collision}")
        if not fields and not media:
            raise ValueError(f"schema {self.name!r}: at least one field or media column is required")
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "globals", globals_)
        object.__setattr__(self, "media", media)
        object.__setattr__(self, "event_streams", FrozenDict(streams))
        if self.sample_hz is not None:
            object.__setattr__(self, "sample_hz", float(self.sample_hz))

    # ---- construction -------------------------------------------------------
    @staticmethod
    def from_dict(d: dict) -> "SchemaSpec":
        if not isinstance(d, dict):
            raise ValueError("schema must be an object")
        unknown = sorted(set(d) - {"name", "fields", "entity_axis", "sample_hz",
                                   "event_streams", "globals", "media"})
        if unknown:
            raise ValueError(f"unknown schema key(s) {unknown}")
        name = d.get("name")
        if not name:
            raise ValueError("schema needs a 'name'")
        raw_fields = d.get("fields") or {}
        raw_media = d.get("media") or {}
        if not raw_fields and not raw_media:
            raise ValueError(f"schema {name!r}: at least one field or media column is required")
        fields = {n: _field_from_dict(n, fd) for n, fd in raw_fields.items()}
        media = {n: _media_from_dict(n, md) for n, md in raw_media.items()}

        entity_axis = d.get("entity_axis")
        sample_hz = d.get("sample_hz")
        if sample_hz is not None and not (
                isinstance(sample_hz, (int, float)) and not isinstance(sample_hz, bool)
                and math.isfinite(sample_hz) and sample_hz > 0):
            raise ValueError(f"schema {name!r}: sample_hz must be a finite positive number")

        event_streams: dict[str, dict] = {}
        for sn, sd in (d.get("event_streams") or {}).items():
            if not isinstance(sd, dict) or set(sd) != {"fields"}:
                raise ValueError(f"schema {name!r}: event stream {sn!r} needs only 'fields'")
            sfields = {fn: _field_from_dict(fn, fd)
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
                          sample_hz=sample_hz, event_streams=event_streams, globals=globals_,
                          media=media)

    def to_dict(self) -> dict:
        """Round-trippable plain dict for embedding in config.json."""
        def fd(f: FieldSpec) -> dict:
            out: dict[str, object] = {"kind": f.kind}
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
        if self.media:
            d["media"] = {n: {"kind": m.kind, "dtype": m.dtype, "shape": list(m.shape),
                               "layout": m.layout, "color_space": m.color_space,
                               "timing": m.timing, "capabilities": dict(m.capabilities)}
                          for n, m in self.media.items()}
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
