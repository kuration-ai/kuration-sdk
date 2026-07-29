"""Persisted shard/checkpoint reuse requires content-stable source identities."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kuration import (
    OutputSpec,
    SchemaSpec,
    Segmenter,
    Tick,
    input,
    load_strategy,
    write_dataset,
)
from kuration.core.manifest import dataset_state


SCHEMA = SchemaSpec.from_dict({
    "name": "source-reuse",
    "entity_axis": "entity",
    "sample_hz": 1.0,
    "fields": {"x": {"kind": "scalar"}},
})
SPEC = OutputSpec.from_dict({
    "container": "npz",
    "shard_length": 2,
    "modalities": {
        "x": {"dtype": "float32", "shape": [], "source": "field:x"},
    },
})
STRATEGY = load_strategy({
    "name": "highest",
    "define": {"level": {"source": "entity.x", "reduce": "mean"}},
    "selection": {"method": "top-k", "weights": {"level": 1.0}},
}, SCHEMA)

_MODULE_VALUES = [1.0, 2.0, 3.0, 4.0]


class ModuleGlobalAdapter:
    """Intentionally weak: the logical ID does not capture module-global payload state."""

    schema = SCHEMA

    def ticks(self, _source):
        return [
            Tick(index, {0: {"x": value}})
            for index, value in enumerate(_MODULE_VALUES)
        ]

    def iter_ticks(self, source):
        yield from self.ticks(source)


def _write(destination: Path, *, resume: bool = False, force: bool = False,
           memory_budget: int | None = None):
    return write_dataset(
        str(destination),
        schema=SCHEMA,
        adapter=ModuleGlobalAdapter(),
        sources=["logical-source"],
        strategy=STRATEGY,
        segmenter=Segmenter(2, 2),
        spec=SPEC,
        train_hours=4 / 3600,
        on_error="raise",
        resume=resume,
        force=force,
        memory_budget=memory_budget,
    )


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _written_values(result) -> list[float]:
    values = [
        np.asarray(np.load(entry["path"])["x"]).reshape(-1)
        for entry in result["shards"]
    ]
    return np.concatenate(values).tolist()


def test_resume_rejects_weak_logical_source_before_publication(tmp_path):
    global _MODULE_VALUES
    destination = tmp_path / "resume"
    _MODULE_VALUES = [1.0, 2.0, 3.0, 4.0]
    _write(destination)
    before = _artifact_bytes(destination)

    _MODULE_VALUES = [101.0, 102.0, 103.0, 104.0]
    with pytest.raises(ValueError, match="resume=True.*immutable source identities"):
        _write(destination, resume=True)

    assert dataset_state(str(destination)) == "committed"
    assert _artifact_bytes(destination) == before


def test_weak_source_uses_non_checkpointed_fallback_on_forced_rerun(tmp_path):
    pytest.importorskip("pyarrow")
    global _MODULE_VALUES
    destination = tmp_path / "forced"

    _MODULE_VALUES = [1.0, 2.0, 3.0, 4.0]
    first = _write(destination)
    assert _written_values(first) == [1.0, 2.0, 3.0, 4.0]
    assert not (destination / ".kuration" / "checkpoints").exists()
    assert "execution" not in json.loads((destination / "config.json").read_text())

    _MODULE_VALUES = [101.0, 102.0, 103.0, 104.0]
    second = _write(destination, force=True)
    assert _written_values(second) == [101.0, 102.0, 103.0, 104.0]
    assert not (destination / ".kuration" / "checkpoints").exists()


def test_memory_bounded_output_rejects_weak_checkpoint_identity(tmp_path):
    pytest.importorskip("pyarrow")
    destination = tmp_path / "bounded"

    with pytest.raises(ValueError, match="memory-bounded.*immutable source identities"):
        _write(destination, memory_budget=100_000)

    assert dataset_state(str(destination)) == "absent"


def test_count_output_also_bypasses_checkpoints_for_weak_sources(tmp_path):
    pytest.importorskip("pyarrow")
    global _MODULE_VALUES
    destination = tmp_path / "count"
    pipeline = (
        input(
            adapter=ModuleGlobalAdapter(),
            schema=SCHEMA,
            sources=["logical-source"],
        )
        .window(2, stride=2)
        .measure("level", "mean", of="x")
        .sample("top-k", n=2, weights={"level": 1.0})
    )

    _MODULE_VALUES = [1.0, 2.0, 3.0, 4.0]
    first = pipeline.output(destination, spec=SPEC)
    assert _written_values(first) == [1.0, 2.0, 3.0, 4.0]
    assert "execution" not in json.loads((destination / "config.json").read_text())

    _MODULE_VALUES = [201.0, 202.0, 203.0, 204.0]
    second = pipeline.output(destination, spec=SPEC, force=True)
    assert _written_values(second) == [201.0, 202.0, 203.0, 204.0]
    assert not (destination / ".kuration" / "checkpoints").exists()
