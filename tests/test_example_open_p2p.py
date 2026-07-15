"""Offline contract tests for the Open P2P public-data example."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip(
    "google.protobuf",
    reason="Open P2P example tests require the optional p2p extra",
)

from kuration import iter_shards
from examples.public_data import open_p2p
from examples.public_data.adapters.open_p2p import (
    OpenP2PAnnotationAdapter,
    open_p2p_schema,
    parse_annotation,
    write_demo_annotation,
)


def test_minimal_proto_and_system_action_precedence(tmp_path):
    path = tmp_path / "annotation.proto"
    write_demo_annotation(path, recording=2)
    annotation = parse_annotation(str(path))
    assert annotation.metadata.frames_per_second == 20
    assert len(annotation.frame_annotations) == 160

    adapter = OpenP2PAnnotationAdapter(open_p2p_schema())
    adapter.check_source(str(path))
    ticks = adapter.ticks(str(path))
    assert len(ticks) == 160
    assert ticks[0].entities["player"]["action_source"] == "system"
    assert ticks[20 + 2].entities["player"]["instruction_present"] == 1
    assert ticks[40].entities["player"]["instruction_present"] == 1
    assert ticks[62].entities["player"]["instruction_present"] == 0


def test_open_p2p_pipeline_offline(tmp_path):
    sources = []
    for recording in range(4):
        path = tmp_path / f"recording_{recording}" / "annotation.proto"
        write_demo_annotation(path, recording=recording)
        sources.append(str(path))

    result = open_p2p.run(sources, str(tmp_path / "out"), n=12)
    assert result["selected"] == 12
    shards = list(iter_shards(tmp_path / "out"))
    assert len(shards) == 12
    assert shards[0]["frame_index"].shape == (40,)
    assert shards[0]["keys"].dtype.kind == "U"
    assert any(np.any(shard["instruction"] != "") for shard in shards)
