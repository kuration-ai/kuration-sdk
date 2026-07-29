"""Property checks for determinism and equivalent bounded/in-memory behavior."""
from __future__ import annotations

import copy
import warnings

import numpy as np
from hypothesis import given, strategies as st

from kuration import Segmenter, Tick, assign_splits, migrate_config, migrate_job
from kuration.core.binning import bin_codes


@given(
    n=st.integers(min_value=0, max_value=100),
    window=st.integers(min_value=1, max_value=20),
    stride=st.integers(min_value=1, max_value=20),
)
def test_streaming_and_materialized_segmentation_are_equivalent(n, window, stride):
    ticks = [Tick(float(i), {0: {"value": float(i)}}) for i in range(n)]
    segmenter = Segmenter(window, stride)
    materialized = segmenter.segments("source", ticks)
    streamed = list(segmenter.stream("source", iter(ticks)))
    assert [segment for segment, _ in streamed] == materialized
    assert [[tick.time for tick in values] for _, values in streamed] == [
        [tick.time for tick in ticks[segment.start_idx:segment.end_idx]]
        for segment in materialized
    ]


@given(
    groups=st.lists(st.one_of(st.none(), st.integers(0, 4)), min_size=0, max_size=50),
    test_frac=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_split_assignment_is_order_independent_and_has_global_quota(groups, test_frac, seed):
    source_groups = {f"source-{i:03d}": group for i, group in enumerate(groups)}
    reversed_groups = dict(reversed(list(source_groups.items())))
    split = assign_splits(source_groups, test_frac, seed)
    assert split == assign_splits(reversed_groups, test_frac, seed)
    assert set(split) == set(source_groups)
    assert sum(value == "test" for value in split.values()) == round(test_frac * len(groups))


@given(st.lists(st.floats(min_value=-1e9, max_value=1e9, allow_nan=False,
                          allow_infinity=False), min_size=0, max_size=200),
       st.integers(min_value=1, max_value=20))
def test_bin_codes_are_monotonic_and_bounded(values, n_bins):
    ordered = np.asarray(sorted(values), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        codes = bin_codes(ordered, n_bins)
    assert codes.shape == ordered.shape
    assert np.all(np.diff(codes) >= 0)
    assert np.all(codes >= 0)
    assert np.all(codes < n_bins)


json_scalars = st.one_of(st.none(), st.booleans(), st.integers(),
                         st.floats(allow_nan=False, allow_infinity=False), st.text())
json_values = st.recursive(json_scalars,
                           lambda children: st.one_of(st.lists(children, max_size=5),
                                                      st.dictionaries(st.text(), children,
                                                                      max_size=5)),
                           max_leaves=15)


@given(st.dictionaries(st.text(), json_values, max_size=10))
def test_job_migration_is_immutable_and_idempotent(job):
    job.pop("job_schema_version", None)
    original = copy.deepcopy(job)
    migrated = migrate_job(job)
    assert job == original
    assert migrate_job(migrated) == migrated


@given(st.dictionaries(st.text(), json_values, max_size=10))
def test_config_migration_is_immutable_and_idempotent(config):
    config.pop("artifact_format_version", None)
    original = copy.deepcopy(config)
    migrated = migrate_config(config)
    assert config == original
    assert migrate_config(migrated) == migrated
