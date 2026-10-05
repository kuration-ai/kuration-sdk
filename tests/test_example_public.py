"""Offline contract tests for the pipeline-only public-data walkthroughs.

These run the examples' `--demo` path, so they never touch the network. What they pin is the
teaching claim of each script: the pipeline chain alone produces a real artifact, tag coverage
changes what survives selection, and splits are decided per source.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from examples.public_data import ett, person_activity
from kuration import iter_shards


# ---- ETT ---------------------------------------------------------------------
def test_ett_demo_curates_days_through_the_pipeline_api_only(tmp_path, capsys):
    source = ett.write_demo(tmp_path / "ETTh1-demo.csv", days=30)
    out = tmp_path / "out"
    # The real ETT dataset's `date` column is an ISO datetime string, not numeric epoch time --
    # expected to trip the non-numeric-time-column warning (F22); this pins that it's still
    # just a warning, not a failure, on real-shaped data.
    with pytest.warns(UserWarning, match="time column 'date' is not numeric"):
        result = ett.run(str(source), str(out), n=12)

    assert len(result["shards"]) == 12
    # A window is one day of hourly readings, and shards are written at that length.
    for arrays in iter_shards(str(out)):
        assert arrays["OT"].shape == (24,)
    assert (out / "config.json").exists()

    printed = capsys.readouterr().out
    assert "selection: 12/30" in printed          # 30 demo days -> 12 selected
    assert "inferred 7 fields" in printed


def test_ett_pipeline_needs_no_schema_adapter_or_strategy(tmp_path):
    """The point of the example: the chain is the entire SDK-facing surface."""
    source = ett.write_demo(tmp_path / "demo.csv", days=12)
    described = ett.build(str(source), n=4).describe()
    assert described["schema"]["sample_hz"] == ett.HOURLY
    assert [axis["metric"] for axis in described["strategy"]["binning"]["axes"]] == [
        "load", "temp_swing"]
    assert described["strategy"]["selection"]["method"] == "uniform-bins"


def test_ett_demo_data_varies_enough_to_bin(tmp_path):
    """A perfectly periodic generator makes every window identical and the axes useless."""
    source = ett.write_demo(tmp_path / "demo.csv", days=30)
    # ETT's real `date` column is an ISO datetime string, not numeric epoch time -- expected
    # to trip the non-numeric-time-column warning (F22).
    with pytest.warns(UserWarning, match="time column 'date' is not numeric"):
        result = ett.build(str(source), n=10).collect()
    loads = {round(segment.features["load"], 6) for segment in result.all_segments}
    assert len(loads) > 5, f"demo days are too uniform to bin: {sorted(loads)}"


# ---- person activity ----------------------------------------------------------
def test_person_activity_demo_runs_multi_source_and_splits_by_recording(tmp_path, capsys):
    sources = person_activity.write_demo(tmp_path / "recordings", recordings=6,
                                                seconds=30)
    out = tmp_path / "out"
    result = person_activity.run(f"{sources}/*.csv", str(out), n=20)

    per_split: dict[str, set] = {}
    for entry in result["shards"]:
        per_split.setdefault(entry["split"], set()).add(entry["source_id"])
    assert per_split["train"] and per_split["test"]
    # Leakage-freedom is a source-level guarantee, and it is the one worth pinning.
    assert not (per_split["train"] & per_split["test"])

    printed = capsys.readouterr().out
    assert "6 recordings" in printed
    assert "entity axis 'tag_id'" in printed
    # Only positions are emitted; the categorical columns describe windows, not tensors.
    for arrays in iter_shards(str(out)):
        assert set(arrays) == {"x", "y", "z", "_present"}
        break


def test_person_activity_tag_coverage_admits_every_activity(tmp_path):
    """Without a floor a rare label can vanish; with one it is guaranteed a share."""
    sources = person_activity.write_demo(tmp_path / "recordings", recordings=5,
                                                seconds=60)
    pattern = f"{sources}/*.csv"

    def activities(min_frac):
        result = (person_activity.input(
            pattern, time="timestamp", entity="tag_id", hz=person_activity.TICK_HZ)
            .window(2, unit="seconds")
            .measure("motion", "std", of="z")
            .tag("activity", min_frac=min_frac)
            .bin("motion", bins=6)
            .sample("equal", n=8)).collect()
        return {tag for segment in result.selected for tag in (segment.tags or [])}

    pool_tags = activities(0.02)
    assert "activity:falling" in pool_tags
    assert len(pool_tags) >= len(activities(None))


def test_person_activity_group_key_is_the_person(tmp_path):
    assert person_activity.person_of("data/person_activity/C04.csv") == "C"
    assert person_activity.person_of(Path("E01.csv").name) == "E"
