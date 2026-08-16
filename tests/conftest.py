"""Shared fixtures for the smoke suite.

The fixtures produce a small deterministic synthetic dataset: three sessions
of 150 ticks, windowed at length 8 stride 4: that every test file reuses.
Kept in-memory so the suite has no filesystem or extras dependency beyond the
SDK itself.
"""
from __future__ import annotations

import pytest
from kuration import input as kuration_input


def _sessions() -> dict[str, list[dict]]:
    """Three synthetic sessions with a signal, a temp, and a rare fault flag.

    Session i's fault fires on ticks where ``step % 30 == i``, so there are ~5
    faults per session: enough for RareEventCoverage to have something to
    upweight without dominating the window pool.
    """
    return {
        f"session-{i}": [
            {
                "t": float(step),
                "signal": float(i * step % 7),
                "temp": float(20 + step % 10),
                "fault": 1 if step % 30 == i else 0,
            }
            for step in range(150)
        ]
        for i in range(3)
    }


@pytest.fixture
def base_pipeline():
    """Windowed SDK pipeline with no measurements applied yet.

    Yields 36 windows per session × 3 sessions = 108 windows total, well
    above any count= value used in the smoke suite.
    """
    return kuration_input(_sessions(), time="t", hz=10.0).window(8, stride=4)


@pytest.fixture
def measured_pipeline(base_pipeline):
    """Windowed pipeline with two features already measured.

    DiverseSubset expects features to be measured on the pipeline before it
    runs, so this fixture stages ``act`` and ``temp_mean`` for cluster-balanced
    sampling.
    """
    return (
        base_pipeline
        .measure("act", "std", of="signal")
        .measure("temp_mean", "mean", of="temp")
    )
