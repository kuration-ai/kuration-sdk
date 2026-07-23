"""`OutputEncoder` — the output seam. Where domain logic for `encoder:` modalities lives.

Core ships only the passthrough encoder (modalities sourced from `field:`/`segment:` need no
code). Anything that *derives* a tensor — decode a video frame, bin an action vector — is an
application plugin implementing this one-method protocol. The SDK never imports such code.
"""
from __future__ import annotations

from typing import Callable, Protocol, TypeAlias, runtime_checkable

import numpy as np


@runtime_checkable
class TimestepOutputEncoder(Protocol):
    def timestep(self, record: dict, ctx: dict) -> dict[str, np.ndarray]:
        """Return {modality_name -> array} for one timestep's `encoder:` modalities.
        `ctx` carries {'schema', 'track'} (the track's metadata)."""
        ...


@runtime_checkable
class BatchOutputEncoder(Protocol):
    def encode_batch(self, records: list, ctx: dict) -> dict[str, np.ndarray]:
        """Return arrays with a leading ``len(records)`` dimension."""
        ...


# Encoders implement either hook.  A batch-only encoder is a first-class, type-correct
# implementation (the writer already prefers it when both are present).
OutputEncoder: TypeAlias = TimestepOutputEncoder | BatchOutputEncoder


class PassthroughEncoder:
    """No `encoder:` modalities — every modality is field:/segment:. The default."""
    def timestep(self, record: dict, ctx: dict) -> dict:
        return {}


_ENCODERS: dict[str, Callable[..., OutputEncoder]] = {}


def register_encoder(name: str, factory: Callable[..., OutputEncoder], **metadata) -> None:
    """Register an output-encoder factory under a unique name.

    Duplicate names raise ``ValueError``; ``metadata`` becomes its component descriptor.
    """
    if name in _ENCODERS:
        raise ValueError(f"encoder {name!r} already registered")
    _ENCODERS[name] = factory
    from ..registry import default_registry
    default_registry.register("encoder", name, factory, replace=True, **metadata)


def get_encoder(name: str, *, registry=None, **kwargs) -> OutputEncoder:
    """Construct a named output encoder, loading plugins on demand.

    ``kwargs`` are passed to the factory. An unknown name raises ``ValueError``.
    """
    if registry is not None:
        from ..registry import default_registry
        if name not in registry.names("encoder") and registry is default_registry:
            from ..plugins import load_plugins
            load_plugins()
        return registry.get("encoder", name)(**kwargs)
    if name not in _ENCODERS:
        from ..plugins import load_plugins
        load_plugins()
    if name not in _ENCODERS:
        raise ValueError(f"unknown encoder {name!r}; registered: {sorted(_ENCODERS)}")
    return _ENCODERS[name](**kwargs)
