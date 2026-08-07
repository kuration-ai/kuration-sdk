"""Bounded local batch scheduling, backpressure, and stable assembly."""
from threading import Lock
import time

import pytest

from kuration import LocalExecutionBackend, RayExecutionBackend
from kuration.core.versioning import _major


def test_local_iter_batches_bounds_work_and_preserves_input_order():
    lock = Lock()
    active = 0
    peak = 0

    def work(value):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.005 * (4 - value % 4))
        with lock:
            active -= 1
        return value * 2

    result = list(LocalExecutionBackend().iter_batches(
        range(12), work, workers=3, max_in_flight=4))
    assert result == [value * 2 for value in range(12)]
    assert 1 < peak <= 3


def test_local_iter_batches_consumes_producer_only_to_in_flight_bound():
    produced = 0

    def items():
        nonlocal produced
        for value in range(5):
            produced += 1
            yield value

    iterator = LocalExecutionBackend().iter_batches(
        items(), lambda value: value, max_in_flight=2)
    assert next(iterator) == 0
    assert produced == 2
    assert list(iterator) == [1, 2, 3, 4]


@pytest.mark.parametrize("name,value", [("max_in_flight", 0), ("workers", 0)])
def test_local_iter_batches_validates_bounds(name, value):
    kwargs = {"max_in_flight": 2, "workers": 1, name: value}
    with pytest.raises(ValueError, match=name):
        list(LocalExecutionBackend().iter_batches([], lambda item: item, **kwargs))


def test_ray_iter_batches_bounds_submissions_by_workers_and_preserves_input_order():
    produced = 0

    class Remote:
        def __init__(self, fn):
            self.fn = fn

        def options(self, **_options):
            return self

        def remote(self, item):
            return self.fn(item)

    class FakeRay:
        def remote(self, fn):
            return Remote(fn)

        def get(self, ref):
            return ref

    def items():
        nonlocal produced
        for value in range(6):
            produced += 1
            yield value

    backend = object.__new__(RayExecutionBackend)
    backend.ray = FakeRay()
    iterator = backend.iter_batches(
        items(), lambda value: value * 3, max_in_flight=5, workers=2)
    assert next(iterator) == 0
    assert produced == 2
    assert list(iterator) == [3, 6, 9, 12, 15]


@pytest.mark.parametrize("method", ["map_sources", "map_batches"])
@pytest.mark.parametrize("kwargs,expected_in_flight", [({}, 2),
                                                       ({"workers": 3,
                                                         "max_in_flight": 5}, 3)])
def test_ray_convenience_maps_forward_parallelism(method, kwargs, expected_in_flight):
    submitted = 0
    submitted_at_get = []

    class Remote:
        def __init__(self, fn):
            self.fn = fn

        def options(self, **_options):
            return self

        def remote(self, item):
            nonlocal submitted
            submitted += 1
            return self.fn(item)

    class FakeRay:
        def remote(self, fn):
            return Remote(fn)

        def get(self, ref):
            submitted_at_get.append(submitted)
            return ref

    backend = object.__new__(RayExecutionBackend)
    backend.ray = FakeRay()

    assert getattr(backend, method)(range(5), lambda value: value * 2, **kwargs) == [
        0, 2, 4, 6, 8]
    assert submitted_at_get[0] == expected_in_flight


def test_real_ray_backend_roundtrip():
    ray = pytest.importorskip("ray")
    ray.shutdown()
    try:
        backend = RayExecutionBackend(num_cpus=1, include_dashboard=False,
                                      log_to_driver=False)
        assert backend.map_sources(["1.0", "2.3", "7.1"], _major) == [1, 2, 7]
    finally:
        ray.shutdown()
