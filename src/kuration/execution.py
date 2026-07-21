"""Ordered execution backend contracts.

Backends return results in input order. Callers continue to key and canonically sort segment
results, so scheduling/completion order cannot affect selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import math
from typing import Any, Callable, Iterable, Protocol


@dataclass(frozen=True)
class ResourceSpec:
    num_cpus: float = 1.0
    num_gpus: float = 0.0
    memory: int | None = None

    def __post_init__(self):
        for name, value in (("num_cpus", self.num_cpus), ("num_gpus", self.num_gpus)):
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite non-negative number")
        if (self.memory is not None
                and (not isinstance(self.memory, int) or isinstance(self.memory, bool)
                     or self.memory < 1)):
            raise ValueError("memory must be a positive integer number of bytes or None")


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

    def map_sources(self, items, fn, resources=ResourceSpec(), *, workers=2,
                    max_in_flight=None):
        """Map sources with bounded Ray parallelism (two concurrent tasks by default)."""
        max_in_flight = workers if max_in_flight is None else max_in_flight
        return list(self.iter_batches(
            items, fn, workers=workers, max_in_flight=max_in_flight, resources=resources))

    def map_batches(self, items, fn, resources=ResourceSpec(), *, workers=2,
                    max_in_flight=None):
        """Map batches with bounded Ray parallelism (two concurrent tasks by default)."""
        max_in_flight = workers if max_in_flight is None else max_in_flight
        return list(self.iter_batches(
            items, fn, workers=workers, max_in_flight=max_in_flight, resources=resources))

    def iter_batches(self, items, fn, *, max_in_flight=2, workers=1,
                     resources=ResourceSpec()):
        """Submit at most ``min(workers, max_in_flight)`` Ray tasks in input order."""
        _bounds(max_in_flight, workers)
        submission_limit = min(max_in_flight, workers)
        options = {"num_cpus": resources.num_cpus, "num_gpus": resources.num_gpus}
        if resources.memory is not None:
            options["memory"] = resources.memory
        remote = self.ray.remote(fn).options(**options)
        pending: deque[Any] = deque()
        for item in items:
            pending.append(remote.remote(item))
            if len(pending) >= submission_limit:
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
