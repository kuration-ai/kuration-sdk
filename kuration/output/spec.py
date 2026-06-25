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

from dataclasses import dataclass

import numpy as np

_SHORT_POLICIES = {"keep", "pad", "drop"}
# npz/json are dep-free; hdf5 needs [array], parquet needs [parquet].
_CONTAINERS = {"npz", "json", "hdf5", "parquet"}
_SRC_KINDS = {"field", "segment", "encoder"}


@dataclass(frozen=True)
class Modality:
    name: str
    dtype: str
    shape: tuple          # per-timestep shape; () for scalar
    src_kind: str         # field | segment | encoder
    src_arg: str

    def zeros(self, length: int) -> np.ndarray:
        return np.zeros((length, *self.shape), dtype=np.dtype(self.dtype))


@dataclass(frozen=True)
class OutputSpec:
    modalities: tuple
    shard_length: int
    short_policy: str = "keep"
    container: str = "npz"

    @staticmethod
    def from_dict(d: dict) -> "OutputSpec":
        raw = d.get("modalities")
        if not raw:
            raise ValueError("output spec needs at least one modality")
        mods = []
        for name, m in raw.items():
            src = m.get("source", "")
            if ":" not in src:
                raise ValueError(f"modality {name!r}: source must be '<kind>:<arg>', got {src!r}")
            kind, arg = src.split(":", 1)
            if kind not in _SRC_KINDS:
                raise ValueError(f"modality {name!r}: source kind must be {sorted(_SRC_KINDS)}")
            try:
                np.dtype(m["dtype"])
            except TypeError as e:
                raise ValueError(f"modality {name!r}: bad dtype {m.get('dtype')!r}") from e
            mods.append(Modality(name=name, dtype=m["dtype"], shape=tuple(m.get("shape") or ()),
                                 src_kind=kind, src_arg=arg))
        shard_length = int(d.get("shard_length", 0))
        if shard_length < 1:
            raise ValueError("output spec needs shard_length >= 1")
        short_policy = d.get("short_policy", "keep")
        if short_policy not in _SHORT_POLICIES:
            raise ValueError(f"short_policy must be {sorted(_SHORT_POLICIES)}")
        container = d.get("container", "npz")
        if container not in _CONTAINERS:
            raise ValueError(f"container must be {sorted(_CONTAINERS)}")
        return OutputSpec(modalities=tuple(mods), shard_length=shard_length,
                          short_policy=short_policy, container=container)

    def to_dict(self) -> dict:
        return {
            "shard_length": self.shard_length,
            "short_policy": self.short_policy,
            "container": self.container,
            "modalities": {m.name: {"dtype": m.dtype, "shape": list(m.shape),
                                    "source": f"{m.src_kind}:{m.src_arg}"} for m in self.modalities},
        }
