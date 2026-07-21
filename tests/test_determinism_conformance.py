"""One golden fixture must be identical across supported physical execution choices."""
from __future__ import annotations

import json

import pytest

from kuration import canonical_hash, input


def _sources(tmp_path):
    paths = []
    for source in range(6):
        path = tmp_path / f"source-{source}.jsonl"
        rows = [
            {
                "t": tick,
                "signal": float((source + 1) * (tick % 7) + (tick % 3)),
                "quality": float((source * 5 + tick * 3) % 17),
                "label": "rare" if (source + tick) % 13 == 0 else "common",
            }
            for tick in range(30)
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths.append(str(path))
    return paths


def _pipeline(paths, *, workers=1, streaming=False, executor="local", batch_size=5):
    return (
        input(paths, time="t")
        .window(6, stride=3)
        .measure("signal_mean", "mean", of="signal")
        .measure("quality_std", "std", of="quality")
        .bin("signal_mean", bins=5)
        .bin("quality_std", bins=4)
        .tag("label", min_frac=0.1)
        .sample("uniform-bins", n=17)
        .options(
            workers=workers,
            streaming=streaming,
            executor=executor,
            batch_size=batch_size,
            seed=73,
            on_error="raise",
        )
    )


def _snapshot(result, *, include_pool=True):
    def segment(item):
        return {
            "id": [item.source_id, item.start_idx, item.end_idx],
            "features": {key: item.features[key] for key in sorted(item.features)},
            "bin": list(item.bin),
            "tags": sorted(item.tags),
            "split": getattr(item, "split", None),
        }

    snapshot = {
        "selected": [segment(item) for item in result.selected],
        "pool_size": len(result.all_segments),
        "errors": result.errors,
    }
    if include_pool:
        snapshot["pool"] = [segment(item) for item in sorted(
            result.all_segments, key=lambda value: (value.source_id, value.start_idx))]
    return snapshot


def _digest(result, *, include_pool=True):
    return canonical_hash(_snapshot(result, include_pool=include_pool))


@pytest.mark.parametrize("workers,batch_size", [(1, 2), (4, 9)])
def test_bounded_and_materialized_plans_have_the_same_deterministic_digest(
        tmp_path, workers, batch_size):
    paths = _sources(tmp_path)
    reference = _pipeline(paths).collect()
    pipeline = _pipeline(paths, workers=workers, streaming=True, batch_size=batch_size)
    bounded = pipeline._bounded_count_result(
        str(tmp_path / f"bounded-{workers}-{batch_size}"), pipeline._compile())
    assert _snapshot(bounded, include_pool=False) == _snapshot(reference, include_pool=False)
    assert _digest(bounded, include_pool=False) == _digest(reference, include_pool=False)


@pytest.mark.parametrize("workers,streaming,batch_size", [
    (1, False, 1),
    (1, True, 7),
    (4, False, 3),
    (4, True, 11),
])
def test_materialized_streaming_and_worker_modes_are_deterministic(
        tmp_path, workers, streaming, batch_size):
    paths = _sources(tmp_path)
    reference = _pipeline(paths).collect()
    candidate = _pipeline(
        paths, workers=workers, streaming=streaming, batch_size=batch_size).collect()
    assert _snapshot(candidate) == _snapshot(reference)
    assert _digest(candidate) == _digest(reference)


def test_real_ray_pipeline_matches_local_deterministic_digest(tmp_path):
    ray = pytest.importorskip("ray")
    paths = _sources(tmp_path)
    ray.shutdown()
    try:
        reference = _pipeline(paths).collect()
        candidate = _pipeline(paths, workers=2, executor="ray").collect()
        assert _snapshot(candidate) == _snapshot(reference)
        assert _digest(candidate) == _digest(reference)
    finally:
        ray.shutdown()
