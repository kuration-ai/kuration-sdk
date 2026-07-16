"""Ordered execution backend contracts.

Backends return results in input order. Callers continue to key and canonically sort segment
results, so scheduling/completion order cannot affect selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol


@dataclass(frozen=True)
class ResourceSpec:
    num_cpus: float = 1.0
    num_gpus: float = 0.0
    memory: int | None = None


class ExecutionBackend(Protocol):
    def map_sources(self, items: Iterable, fn: Callable, resources: ResourceSpec = ResourceSpec()): ...
    def map_batches(self, items: Iterable, fn: Callable, resources: ResourceSpec = ResourceSpec()): ...


class LocalExecutionBackend:
    """In-process ordered execution backend."""

    def map_sources(self, items, fn, resources=ResourceSpec()):
        return [fn(item) for item in items]

    def map_batches(self, items, fn, resources=ResourceSpec()):
        return [fn(item) for item in items]


class RayExecutionBackend:
    """Optional Ray task backend with input-ordered result assembly."""

    def __init__(self, **init_kwargs):
        try:
            import ray
        except ImportError as e:
            raise ImportError("Ray execution needs the optional extra: "
                              "pip install 'kuration-sdk[ray]'") from e
        self.ray = ray
        if not ray.is_initialized():
            ray.init(**init_kwargs)

    def _map(self, items, fn, resources):
        options = {"num_cpus": resources.num_cpus, "num_gpus": resources.num_gpus}
        if resources.memory is not None:
            options["memory"] = resources.memory
        remote = self.ray.remote(fn).options(**options)
        refs = [remote.remote(item) for item in items]
        return self.ray.get(refs)

    def map_sources(self, items, fn, resources=ResourceSpec()):
        return self._map(items, fn, resources)

    def map_batches(self, items, fn, resources=ResourceSpec()):
        return self._map(items, fn, resources)


def get_execution_backend(name: str):
    """Construct the named ``local`` or ``ray`` execution backend."""
    if name == "local":
        return LocalExecutionBackend()
    if name == "ray":
        return RayExecutionBackend()
    raise ValueError("executor must be 'local' or 'ray'")
