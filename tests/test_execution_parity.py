"""Reference/materialized and bounded execution must select identical stable keys."""
from __future__ import annotations

import pytest

from kuration import input

pytest.importorskip("pyarrow")


def _keys(result):
    return [(segment.source_id, segment.start_idx, segment.end_idx)
            for segment in result.selected]


@pytest.mark.parametrize("seed", [0, 7, 29])
@pytest.mark.parametrize("batch_size", [1, 3, 11])
@pytest.mark.parametrize("normalize", ["quantile", "minmax", "zscore"])
def test_top_k_predicate_parity_across_partitioning(tmp_path, seed, batch_size, normalize):
    sessions = {
        f"source-{source}": [{"x": float(tick + source),
                              "quality": float((tick * 3 + source) % 11)}
                             for tick in range(18)]
        for source in range(4)
    }
    pipeline = (input(sessions).window(4, stride=2)
                .measure("level", "mean", of="x")
                .measure("quality", "mean", of="quality")
                .filter("quality", ">=", 3.0)
                .sample("top-k", n=9, weights={"level": 0.7, "quality": 0.3},
                        normalize=normalize)
                .options(seed=seed, batch_size=batch_size, on_error="raise"))
    reference = pipeline.collect()
    bounded = pipeline._bounded_count_result(
        str(tmp_path / f"top-{normalize}-{seed}-{batch_size}"), pipeline._compile())
    assert _keys(bounded) == _keys(reference)
    assert len(bounded.all_segments) == len(reference.all_segments)


@pytest.mark.parametrize("seed", [1, 13, 41])
@pytest.mark.parametrize("batch_size", [2, 5, 17])
def test_uniform_bin_parity_across_partitioning(tmp_path, seed, batch_size):
    rows = [{"x": float(index % 9), "y": float((index * 7) % 13)}
            for index in range(50)]
    pipeline = (input(rows).window(5, stride=2)
                .measure("x", "mean", of="x").measure("y", "std", of="y")
                .bin("x", bins=4).bin("y", bins=3)
                .sample("uniform-bins", n=12)
                .options(seed=seed, batch_size=batch_size, on_error="raise"))
    reference = pipeline.collect()
    bounded = pipeline._bounded_count_result(
        str(tmp_path / f"bins-{seed}-{batch_size}"), pipeline._compile())
    assert _keys(bounded) == _keys(reference)
