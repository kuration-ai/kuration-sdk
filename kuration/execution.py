"""Ordered execution backend contracts.

Backends return results in input order. Callers continue to key and canonically sort segment
results, so scheduling/completion order cannot affect selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Protocol


@dataclass(frozen=True)
class ResourceSpec:
    num_cpus: float = 1.0
    num_gpus: float = 0.0
    memory: int | None = None


class ExecutionBackend(Protocol):
    def map_sources(self, items: Iterable, fn: Callable, resources: ResourceSpec = ResourceSpec()): ...
    def map_batches(self, items: Iterable, fn: Callable, resources: ResourceSpec = ResourceSpec()): ...
    def iter_batches(self, items: Iterable, fn: Callable, *, max_in_flight: int = 2,
                     workers: int = 1, resources: ResourceSpec = ResourceSpec()): ...


def _bounds(max_in_flight: int, workers: int) -> None:
    for name, value in (("max_in_flight", max_in_flight), ("workers", workers)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")


class LocalExecutionBackend:
    """In-process ordered execution backend."""

    def map_sources(self, items, fn, resources=ResourceSpec()):
        return [fn(item) for item in items]

    def map_batches(self, items, fn, resources=ResourceSpec()):
        return list(self.iter_batches(items, fn, resources=resources))

    def iter_batches(self, items, fn, *, max_in_flight=2, workers=1,
                     resources=ResourceSpec()):
        """Yield input-ordered results with bounded producer backpressure."""
        _bounds(max_in_flight, workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending: deque[Any] = deque()
            for item in items:
                pending.append(executor.submit(fn, item))
                if len(pending) >= max_in_flight:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()


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
        return list(self.iter_batches(items, fn, resources=resources))

    def iter_batches(self, items, fn, *, max_in_flight=2, workers=1,
                     resources=ResourceSpec()):
        """Submit at most ``max_in_flight`` Ray objects and assemble in input order."""
        _bounds(max_in_flight, workers)
        options = {"num_cpus": resources.num_cpus, "num_gpus": resources.num_gpus}
        if resources.memory is not None:
            options["memory"] = resources.memory
        remote = self.ray.remote(fn).options(**options)
        pending: deque[Any] = deque()
        for item in items:
            pending.append(remote.remote(item))
            if len(pending) >= max_in_flight:
                yield self.ray.get(pending.popleft())
        while pending:
            yield self.ray.get(pending.popleft())


def get_execution_backend(name: str):
    """Construct the named ``local`` or ``ray`` execution backend."""
    if name == "local":
        return LocalExecutionBackend()
    if name == "ray":
        return RayExecutionBackend()
    raise ValueError("executor must be 'local' or 'ray'")
