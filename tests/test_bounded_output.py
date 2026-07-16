"""Automatic bounded physical-plan selection for the supported count output path."""
import json

import pytest

from kuration import input

pytest.importorskip("pyarrow")


def _pipeline():
    return (input([{"x": float(value)} for value in range(12)])
            .window(2, stride=1)
            .measure("level", "mean", of="x")
            .sample("top-k", n=3, weights={"level": 1.0}))


def test_count_output_automatically_uses_bounded_checkpoint_plan(tmp_path):
    out = tmp_path / "dataset"
    result = _pipeline().output(str(out), categorical="omit")
    assert result["selected"] == 3
    config = json.loads((out / "config.json").read_text())
    assert config["execution"]["bounded"] is True
    assert config["execution"]["physical_plan"]["stages"][0]["mode"] == "pipeline_parallel"
    commits = list((out / ".kuration" / "checkpoints" / "stage000" / "commits").glob("*.json"))
    assert commits
    assert config["results"]["segments_total"] == 11


def test_bounded_count_output_is_deterministic_across_batch_partitions(tmp_path):
    first = _pipeline().output(str(tmp_path / "first"), categorical="omit")
    second = _pipeline().output(str(tmp_path / "second"), categorical="omit")
    keys = lambda result: [(entry["source_id"], entry["path"].split("_seg")[-1])
                           for entry in result["shards"]]
    assert keys(first) == keys(second)


def test_uniform_bins_automatic_bounded_result_matches_reference(tmp_path):
    pipeline = (input([{"x": float(value)} for value in range(30)])
                .window(2, stride=1).measure("level", "mean", of="x")
                .bin("level", bins=4).sample("uniform-bins", n=9).options(seed=23))
    strategy = pipeline._compile()
    reference = pipeline.collect()
    bounded = pipeline._bounded_count_result(str(tmp_path / "bounded"), strategy)
    assert [(s.source_id, s.start_idx) for s in bounded.selected] == [
        (s.source_id, s.start_idx) for s in reference.selected]
    assert len(bounded.all_segments) == len(reference.all_segments)


def test_undeclared_approximate_mode_fails_before_reading(tmp_path):
    with pytest.raises(ValueError, match="no declared approximate"):
        _pipeline().options(approximation="approximate").output(
            str(tmp_path / "approximate"), categorical="omit")


def test_bounded_count_remote_output_uses_local_temporary_spill():
    fsspec = pytest.importorskip("fsspec")
    fsspec.filesystem("memory").store.clear()
    result = _pipeline().output("memory://bounded-count", categorical="omit")
    assert result["selected"] == 3
    with fsspec.open("memory://bounded-count/config.json", "rt") as file:
        config = json.load(file)
    assert config["execution"]["bounded"] is True
    assert config["execution"]["physical_plan"]["spill_path"] is None


def test_bounded_count_reads_json_arrays_through_json_adapter(tmp_path):
    paths = []
    for source in range(2):
        path = tmp_path / f"source-{source}.json"
        path.write_text(json.dumps([{"t": tick, "x": float(source + tick)}
                                    for tick in range(6)]))
        paths.append(str(path))
    pipeline = (input(paths, time="t").window(2).measure("level", "mean", of="x")
                .sample("top-k", n=2, weights={"level": 1.0}))
    result = pipeline.output(str(tmp_path / "out"), categorical="omit")
    assert result["selected"] == 2
    assert result["shards"]
