"""`OutputSpec` — declared modalities for training shards.

The output analogue of `SchemaSpec`: it says *what* each training timestep contains and how
to derive it, without baking any format into the core. Each modality routes its per-timestep
value from one of three sources:

  field:<name>    pull the track entity's schema field straight from the record
  segment:<attr>  a per-track scalar (e.g. a precomputed numeric label)
  encoder:<hook>  call an OutputEncoder — where domain logic (video decode, action binning)
                  lives, as an application plugin, never in core

`shard_length` + `short_policy` (keep/pad/drop) generalize the chunking the reference
pipeline does; `container` picks the on-disk format. The whole point: the DIAMOND HDF5 layout
is just one app encoder + (npz/hdf5) container, not something the SDK knows about.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..schema import FrozenDict

_SHORT_POLICIES = {"keep", "pad", "drop"}
# npz/json are dep-free; hdf5 needs [array], parquet needs [parquet].
_CONTAINERS = {"npz", "json", "hdf5", "parquet"}
_SRC_KINDS = {"field", "segment", "encoder", "vocab", "media"}
_RESERVED_MODALITY_NAMES = {"_n_steps", "_present"}
_OUTPUT_KEYS = {"modalities", "shard_length", "short_policy", "container",
                "presence_mask", "vocabs"}
_MODALITY_KEYS = {"dtype", "shape", "source"}


def validate_modality_name(name: object, *, allow_presence_mask: bool = False) -> str:
    """Return a safe portable modality name or raise ``ValueError``.

    Modality names become container keys and, for plain-array exports, filename components.
    Keep one validation contract on both the write and read boundaries so a recommitted
    artifact cannot bypass the writer's path-safety rules.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("modality names must be non-empty strings")
    if name in {".", ".."}:
        raise ValueError(f"modality name {name!r} is not a safe path component")
    if any(character in name for character in ("/", "\\")):
        raise ValueError(f"modality name {name!r} must not contain path separators")
    if any(unicodedata.category(character) == "Cc" for character in name):
        raise ValueError(f"modality name {name!r} must not contain NUL or control characters")
    if name in _RESERVED_MODALITY_NAMES and not (
            allow_presence_mask and name == "_present"):
        raise ValueError(f"modality name {name!r} is reserved for container metadata")
    return name


@dataclass(frozen=True)
class Modality:
    name: str
    dtype: str
    shape: tuple          # per-timestep shape; () for scalar
    src_kind: str         # field | segment | encoder
    src_arg: str

    def __post_init__(self) -> None:
        validate_modality_name(self.name)
        if self.src_kind not in _SRC_KINDS:
            raise ValueError(f"modality {self.name!r}: source kind must be {sorted(_SRC_KINDS)}")
        if not isinstance(self.src_arg, str) or not self.src_arg:
            raise ValueError(f"modality {self.name!r}: source argument must not be empty")
        try:
            dtype = np.dtype(self.dtype)
        except (TypeError, ValueError) as error:
            raise ValueError(f"modality {self.name!r}: bad dtype {self.dtype!r}") from error
        if dtype.hasobject:
            raise ValueError(f"modality {self.name!r}: object dtypes are unsafe and unsupported")
        try:
            shape = tuple(self.shape)
        except TypeError as error:
            raise ValueError(f"modality {self.name!r}: shape must be an array of positive "
                             "integers") from error
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in shape):
            raise ValueError(f"modality {self.name!r}: shape dimensions must be positive integers")
        dtype_name = (f"U{dtype.itemsize // np.dtype('U1').itemsize}"
                      if dtype.kind == "U" else
                      f"S{dtype.itemsize}" if dtype.kind == "S" else dtype.name)
        object.__setattr__(self, "dtype", dtype_name)
        object.__setattr__(self, "shape", shape)

    def zeros(self, length: int) -> np.ndarray:
        return np.zeros((length, *self.shape), dtype=np.dtype(self.dtype))


@dataclass(frozen=True)
class OutputSpec:
    modalities: tuple
    shard_length: int
    short_policy: str = "keep"
    container: str = "npz"
    presence_mask: bool = True
    vocabs: Optional[dict] = None

    def __post_init__(self) -> None:
        modalities = tuple(self.modalities)
        if not modalities or any(not isinstance(modality, Modality) for modality in modalities):
            raise ValueError("output spec needs at least one Modality")
        names = [modality.name for modality in modalities]
        if len(names) != len(set(names)):
            raise ValueError("output modality names must be unique")
        if (not isinstance(self.shard_length, int) or isinstance(self.shard_length, bool)
                or self.shard_length < 1):
            raise ValueError("output spec needs an integer shard_length >= 1")
        if self.short_policy not in _SHORT_POLICIES:
            raise ValueError(f"short_policy must be {sorted(_SHORT_POLICIES)}")
        if not isinstance(self.container, str) or not self.container:
            raise ValueError("container must be a non-empty registered name")
        if not isinstance(self.presence_mask, bool):
            raise ValueError("presence_mask must be true or false")

        # Built-in formats have a deliberately narrow, tested round-trip matrix. Third-party
        # containers own their dtype contract and still benefit from Modality's object-safety
        # baseline without core second-guessing their implementation.
        if self.container in _CONTAINERS:
            allowed_kinds = {"b", "i", "u", "f"}
            if self.container in {"npz", "json"}:
                allowed_kinds.add("U")
            if self.container == "npz":
                allowed_kinds.add("S")
            for modality in modalities:
                dtype = np.dtype(modality.dtype)
                if dtype.kind not in allowed_kinds:
                    raise ValueError(
                        f"modality {modality.name!r}: dtype {dtype.name!r} is unsupported: "
                        f"container {self.container!r} does not safely round-trip this dtype; "
                        "use boolean, integer, unsigned integer, float, or a "
                        "container-supported fixed-width string dtype")

        raw_vocabs = {} if self.vocabs is None else self.vocabs
        if not isinstance(raw_vocabs, dict):
            raise ValueError("vocabs must be an object mapping field names to label/code maps")
        vocabs = {}
        for field_name, values in raw_vocabs.items():
            if not isinstance(field_name, str) or not field_name or not isinstance(values, dict):
                raise ValueError("vocabs must map non-empty field names to objects")
            normalized = {}
            for label, code in values.items():
                if not isinstance(label, str) or not label:
                    raise ValueError(f"vocabs[{field_name!r}] labels must be non-empty strings")
                if not isinstance(code, int) or isinstance(code, bool) or code < 0:
                    raise ValueError(f"vocabs[{field_name!r}][{label!r}] must be a non-negative "
                                     "integer")
                normalized[label] = code
            if len(set(normalized.values())) != len(normalized):
                raise ValueError(f"vocabs[{field_name!r}] codes must be unique")
            vocabs[field_name] = FrozenDict(normalized)
        object.__setattr__(self, "modalities", modalities)
        object.__setattr__(self, "vocabs", FrozenDict(vocabs))

    @staticmethod
    def from_dict(d: dict, *, registry=None) -> "OutputSpec":
        if not isinstance(d, dict):
            raise ValueError(f"output spec must be an object, got {type(d).__name__}")
        unknown = set(d) - _OUTPUT_KEYS
        if unknown:
            raise ValueError(f"output spec has unknown key(s) {sorted(unknown)}; expected "
                             f"{sorted(_OUTPUT_KEYS)}")
        container = d.get("container", "npz")
        if container not in _CONTAINERS:
            from ..plugins import load_plugins
            from ..registry import default_registry
            active_registry = registry or default_registry
            if registry is None:
                load_plugins()
            if container not in active_registry.names("container"):
                raise ValueError(f"unknown container {container!r}; registered: "
                                 f"{active_registry.names('container')}")
        raw = d.get("modalities")
        if not raw:
            raise ValueError("output spec needs at least one modality")
        mods = []
        for name, m in raw.items():
            if not isinstance(m, dict):
                raise ValueError(f"modality {name!r} must be an object")
            unknown_modality = set(m) - _MODALITY_KEYS
            if unknown_modality:
                raise ValueError(f"modality {name!r}: unknown key(s) "
                                 f"{sorted(unknown_modality)}; expected "
                                 f"{sorted(_MODALITY_KEYS)}")
            src = m.get("source", "")
            if ":" not in src:
                raise ValueError(f"modality {name!r}: source must be '<kind>:<arg>', got {src!r}")
            kind, arg = src.split(":", 1)
            if "dtype" not in m:
                raise ValueError(f"modality {name!r}: bad dtype None")
            mods.append(Modality(name=name, dtype=m["dtype"], shape=m.get("shape") or (),
                                 src_kind=kind, src_arg=arg))
        shard_length = d.get("shard_length", 0)
        if (not isinstance(shard_length, int) or isinstance(shard_length, bool)
                or shard_length < 1):
            raise ValueError("output spec needs an integer shard_length >= 1")
        short_policy = d.get("short_policy", "keep")
        if short_policy not in _SHORT_POLICIES:
            raise ValueError(f"short_policy must be {sorted(_SHORT_POLICIES)}")
        presence_mask = d.get("presence_mask", True)
        if not isinstance(presence_mask, bool):
            raise ValueError("presence_mask must be true or false")
        vocabs = {} if d.get("vocabs") is None else d.get("vocabs")
        return OutputSpec(modalities=tuple(mods), shard_length=shard_length,
                          short_policy=short_policy, container=container,
                          presence_mask=presence_mask, vocabs=vocabs)

    def validate_schema(self, schema) -> None:
        """Validate schema-backed modality sources before any data is read or written."""
        available = {**schema.fields, **schema.globals}
        for m in self.modalities:
            if m.src_kind == "vocab":
                fs = available.get(m.src_arg)
                if fs is None or fs.kind != "categorical":
                    raise ValueError(f"modality {m.name!r}: vocab source {m.src_arg!r} must be "
                                     "a categorical schema field")
                if m.src_arg not in (self.vocabs or {}):
                    raise ValueError(f"modality {m.name!r}: missing vocabs[{m.src_arg!r}]")
                dtype = np.dtype(m.dtype)
                if dtype.kind not in {"i", "u"}:
                    raise ValueError(f"modality {m.name!r}: vocab outputs need an integer dtype")
                codes = (self.vocabs or {})[m.src_arg].values()
                limit = np.iinfo(dtype)
                if any(code < limit.min or code > limit.max for code in codes):
                    raise ValueError(f"modality {m.name!r}: a vocabulary code does not fit "
                                     f"dtype {m.dtype}")
                continue
            if m.src_kind == "media":
                ms = schema.media.get(m.src_arg)
                if ms is None:
                    raise ValueError(f"modality {m.name!r}: media source {m.src_arg!r} is not "
                                     f"declared; available: {sorted(schema.media)}")
                if m.shape != ms.shape:
                    raise ValueError(f"modality {m.name!r}: shape {m.shape} does not match "
                                     f"media:{m.src_arg} shape {ms.shape}")
                if np.dtype(m.dtype) != np.dtype(ms.dtype):
                    raise ValueError(f"modality {m.name!r}: dtype {m.dtype} does not match "
                                     f"media:{m.src_arg} dtype {ms.dtype}")
                continue
            if m.src_kind != "field":
                continue
            fs = available.get(m.src_arg)
            if fs is None:
                raise ValueError(
                    f"modality {m.name!r}: field source {m.src_arg!r} is not declared in schema "
                    f"fields/globals; available: {sorted(available)}")
            expected = (fs.width,) if fs.kind == "vector" else ()
            if m.shape != expected:
                raise ValueError(
                    f"modality {m.name!r}: shape {m.shape} does not match schema field "
                    f"{m.src_arg!r} ({fs.kind}, expected {expected})")

    def to_dict(self) -> dict:
        out = {
            "shard_length": self.shard_length,
            "short_policy": self.short_policy,
            "container": self.container,
            "presence_mask": self.presence_mask,
            "modalities": {m.name: {"dtype": m.dtype, "shape": list(m.shape),
                                    "source": f"{m.src_kind}:{m.src_arg}"} for m in self.modalities},
        }
        if self.vocabs:
            out["vocabs"] = {name: dict(values) for name, values in self.vocabs.items()}
        return out
