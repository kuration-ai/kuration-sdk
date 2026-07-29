"""Bounded work paths are isolated from sources and cannot redirect through symlinks."""
from __future__ import annotations

import json
from pathlib import Path
import uuid

import pytest

from kuration import SchemaSpec, Tick, input
from kuration.core.manifest import dataset_state


pytest.importorskip("pyarrow")


SCHEMA = SchemaSpec.from_dict({
    "name": "bounded-path-safety",
    "entity_axis": "entity",
    "sample_hz": 1.0,
    "fields": {"x": {"kind": "scalar"}},
})


def _write_jsonl(path: Path) -> None:
    path.write_text("".join(
        json.dumps({"t": index, "x": float(index)}) + "\n"
        for index in range(8)
    ))


def _pipeline(source, budget: str, *, adapter=None):
    opened = (
        input([str(source)], time="t", hz=1)
        if adapter is None
        else input(adapter=adapter, sources=[str(source)], schema=SCHEMA)
    )
    measured = (
        opened.window(2, stride=2)
        .measure("mean_x", "mean", of="x")
    )
    if budget == "count":
        return measured.sample("top-k", n=2, weights={"mean_x": 1.0})
    return measured.sample("top-k", hours=4 / 3600, weights={"mean_x": 1.0})


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize("budget", ["count", "hours"])
@pytest.mark.parametrize("work_option", ["checkpoint_dir", "spill_dir"])
@pytest.mark.parametrize(
    "relationship",
    ["same", "work_contains_source", "source_contains_work"],
)
def test_custom_bounded_work_path_cannot_overlap_a_local_source(
        tmp_path, budget, work_option, relationship):
    if relationship == "work_contains_source":
        work_path = tmp_path / "source-tree"
        work_path.mkdir()
        source = work_path / "source.jsonl"
    else:
        source = tmp_path / "source.jsonl"
        work_path = source if relationship == "same" else source / "work"
    _write_jsonl(source)
    before = source.read_bytes()
    destination = tmp_path / f"out-{budget}-{work_option}"
    pipeline = _pipeline(source, budget).options(**{work_option: str(work_path)})

    with pytest.raises(ValueError, match=rf"{work_option} .*overlaps a source"):
        pipeline.output(destination, categorical="omit")

    assert source.read_bytes() == before
    assert dataset_state(str(destination)) == "absent"


@pytest.mark.parametrize("budget", ["count", "hours"])
@pytest.mark.parametrize(
    ("work_option", "component"),
    [("checkpoint_dir", "checkpoints"), ("spill_dir", "spill")],
)
def test_default_nested_work_symlink_cannot_redirect_to_a_victim(
        tmp_path, budget, work_option, component):
    source = tmp_path / f"source-{budget}-{component}.jsonl"
    _write_jsonl(source)
    destination = tmp_path / f"nested-{budget}-{component}"
    work_parent = destination / ".kuration"
    work_parent.mkdir(parents=True)
    victim = tmp_path / f"victim-{budget}-{component}"
    victim.mkdir()
    victim_file = victim / "keep.bin"
    victim_file.write_bytes(b"victim")
    try:
        (work_parent / component).symlink_to(victim, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ValueError, match=rf"{work_option} .*contains symlink"):
        _pipeline(source, budget).output(destination, categorical="omit")

    assert victim_file.read_bytes() == b"victim"
    assert dataset_state(str(destination)) == "absent"
    assert not (destination / "dataset_manifest.json").exists()


@pytest.mark.parametrize("budget", ["count", "hours"])
@pytest.mark.parametrize(
    ("work_option", "component"),
    [("checkpoint_dir", "checkpoints"), ("spill_dir", "spill")],
)
def test_force_preserves_committed_artifact_when_custom_work_preflight_fails(
        tmp_path, budget, work_option, component):
    source = tmp_path / f"force-{budget}.jsonl"
    _write_jsonl(source)
    destination = tmp_path / f"committed-{budget}"
    pipeline = _pipeline(source, budget)
    pipeline.output(destination, categorical="omit")
    before = _artifact_bytes(destination)

    victim = tmp_path / f"force-victim-{budget}"
    victim.mkdir()
    victim_file = victim / "keep.bin"
    victim_file.write_bytes(b"victim")
    custom_parent = tmp_path / f"custom-work-{budget}-{component}"
    custom_parent.mkdir()
    redirected = custom_parent / component
    try:
        redirected.symlink_to(victim, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ValueError, match=rf"{work_option} .*contains symlink"):
        pipeline.options(**{work_option: str(redirected)}).output(
            destination, categorical="omit", force=True)

    assert victim_file.read_bytes() == b"victim"
    assert dataset_state(str(destination)) == "committed"
    assert _artifact_bytes(destination) == before


class _RemoteIdAdapter:
    supports_streaming = True

    def ticks(self, _source):
        return [Tick(index, {0: {"x": float(index)}}) for index in range(8)]

    def iter_ticks(self, source):
        return iter(self.ticks(source))

    def source_fingerprint(self, _source):
        return "remote-test-source"


@pytest.mark.parametrize("budget", ["count", "hours"])
@pytest.mark.parametrize("work_option", ["checkpoint_dir", "spill_dir"])
def test_memory_work_root_cannot_overlap_a_remote_source(
        tmp_path, budget, work_option):
    fsspec = pytest.importorskip("fsspec")
    fs = fsspec.filesystem("memory")
    prefix = f"memory://bounded-path-safety-{uuid.uuid4().hex}"
    source = f"{prefix}/source"
    victim = f"{source}/victim.bin"
    fs.pipe(victim, b"victim")
    destination = tmp_path / f"remote-overlap-{budget}"
    pipeline = _pipeline(source, budget, adapter=_RemoteIdAdapter()).options(
        **{work_option: source})
    try:
        with pytest.raises(ValueError, match=rf"{work_option} .*overlaps a source"):
            pipeline.output(destination, categorical="omit")
        assert fs.cat(victim) == b"victim"
        assert dataset_state(str(destination)) == "absent"
    finally:
        fs.rm(prefix, recursive=True)
