"""Offline contract tests for the TeaPearce public-data example."""
from __future__ import annotations

import numpy as np
import pytest

from kuration import iter_shards
from examples.public_data.adapters.teapearce import (
    ACTION_WIDTH,
    MOUSE_X_VALUES,
    MOUSE_Y_VALUES,
    TeaPearceHDF5Adapter,
    decode_action,
    teapearce_schema,
    write_demo_recording,
)
from examples.public_data import teapearce_csgo

def test_action_layout_and_decode():
    action = np.zeros(ACTION_WIDTH, dtype=np.float32)
    action[0] = action[3] = 1
    action[11] = 1
    action[13 + 4] = 1
    action[13 + len(MOUSE_X_VALUES) + 9] = 1
    decoded = decode_action(action)
    assert ACTION_WIDTH == 51
    assert decoded["keys_down"] == 2
    assert decoded["left_click"] == 1
    assert decoded["mouse_dx"] == MOUSE_X_VALUES[4]
    assert decoded["mouse_dy"] == MOUSE_Y_VALUES[9]
    with pytest.raises(ValueError, match="51"):
        decode_action(np.zeros(50))


def test_teapearce_adapter_and_pipeline_offline(tmp_path):
    pytest.importorskip("h5py")
    sources = []
    for recording in range(4):
        path = tmp_path / f"recording_{recording}.hdf5"
        write_demo_recording(path, recording=recording)
        sources.append(str(path))

    adapter = TeaPearceHDF5Adapter(teapearce_schema(), include_frames=True)
    adapter.check_source(sources[0])
    ticks = adapter.ticks(sources[0])
    assert len(ticks) == 256
    assert ticks[0].entities["player"]["action"].shape == (51,)
    assert ticks[0].entities["player"]["frame"].shape == (8, 12, 3)
    assert any(tick.events["combat"] for tick in ticks)

    result = teapearce_csgo.run(
        sources, str(tmp_path / "out"), n=12, include_frames=True)
    assert result["selected"] == 12
    shards = list(iter_shards(tmp_path / "out"))
    assert len(shards) == 12
    assert shards[0]["action"].shape == (96, 51)
    assert shards[0]["frame"].shape == (96, 8, 12, 3)
