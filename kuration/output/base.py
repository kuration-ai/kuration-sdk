"""`OutputEncoder` — the output seam. Where domain logic for `encoder:` modalities lives.

Core ships only the passthrough encoder (modalities sourced from `field:`/`segment:` need no
code). Anything that *derives* a tensor — decode a video frame, bin an action vector — is an
application plugin implementing this one-method protocol. The SDK never imports such code.
"""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class OutputEncoder(Protocol):
    def timestep(self, record: dict, ctx: dict) -> dict[str, np.ndarray]:
        """Return {modality_name -> array} for one timestep's `encoder:` modalities.
        `ctx` carries {'schema', 'track'} (the track's metadata)."""
        ...


class PassthroughEncoder:
    """No `encoder:` modalities — every modality is field:/segment:. The default."""
    def timestep(self, record: dict, ctx: dict) -> dict:
        return {}


_ENCODERS: dict[str, Callable[..., OutputEncoder]] = {}


def register_encoder(name: str, factory: Callable[..., OutputEncoder]) -> None:
    if name in _ENCODERS:
        raise ValueError(f"encoder {name!r} already registered")
    _ENCODERS[name] = factory


def get_encoder(name: str, **kwargs) -> OutputEncoder:
    if name not in _ENCODERS:
        from ..plugins import load_plugins
        load_plugins()
    if name not in _ENCODERS:
        raise ValueError(f"unknown encoder {name!r}; registered: {sorted(_ENCODERS)}")
    return _ENCODERS[name](**kwargs)
