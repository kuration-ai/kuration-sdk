"""`InputAdapter` — the input seam. The one place domain knowledge about a raw format
enters the SDK.

An adapter's contract is deliberately tiny: turn a raw source into an ordered stream of
canonical `Tick`s (parse + group-by-timestep). Everything after that — building the
`(T,E,D)` buffer, segmentation, metrics, selection — is generic SDK code that never knows
which dataset it's looking at. The stream is an `Iterable`, so a future streaming segmenter
can consume it lazily; the in-memory path materializes it.

Adapters are registered by name so worker processes can reconstruct them (the SDK's
parallel feature pass rebuilds the adapter from its registered name + config rather than
pickling a live object — the same pattern the GCS client uses).
"""
from __future__ import annotations

from typing import Callable, Iterable, Protocol, runtime_checkable

from ..buffer import Tick
from ..schema import SchemaSpec


@runtime_checkable
class InputAdapter(Protocol):
    schema: SchemaSpec
    def ticks(self, source_uri: str) -> Iterable[Tick]: ...


_ADAPTERS: dict[str, Callable[..., InputAdapter]] = {}


def register_adapter(name: str, factory: Callable[..., InputAdapter]) -> None:
    if name in _ADAPTERS:
        raise ValueError(f"adapter {name!r} already registered")
    _ADAPTERS[name] = factory


def get_adapter(name: str, **kwargs) -> InputAdapter:
    if name not in _ADAPTERS:
        raise ValueError(f"unknown adapter {name!r}; registered: {sorted(_ADAPTERS)}")
    return _ADAPTERS[name](**kwargs)
