"""Tests for AsgCollisionAdapter: reading ASG `data.json` match replays into one canonical
Tick stream per (match, player), and `discover_map_for_source`'s per-match map lookup.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from kuration.gameplay.collision.asg_adapter import ASG_SCHEMA, AsgCollisionAdapter


def _write_match(tmp_path, map_name="de_dust2"):
    data = {
        "header": {"map_name": map_name},
        "ticks": [
            {"tick": 100, "players": [
                {"id32": 111, "pos": {"x": 0.0, "y": 0.0, "z": 0.0}, "ang": {"yaw": 0.0, "pitch": 0.0},
                 "hp": 100, "btn_mask": 8, "flags": 1, "weapon": "AK-47"},
                {"id32": 222, "pos": {"x": 10.0, "y": 0.0, "z": 0.0}, "ang": {"yaw": 90.0, "pitch": 0.0},
                 "hp": 80, "btn_mask": 0, "flags": 1, "weapon": "Knife"},
            ]},
            {"tick": 104, "players": [
                {"id32": 111, "pos": {"x": 1.0, "y": 0.0, "z": 0.0}, "ang": {"yaw": 0.0, "pitch": 0.0},
                 "hp": 100, "btn_mask": 8, "flags": 1, "weapon": "AK-47"},
                {"id32": 222, "pos": {"x": 10.0, "y": 1.0, "z": 0.0}, "ang": {"yaw": 90.0, "pitch": 0.0},
                 "hp": 80, "btn_mask": 0, "flags": 1, "weapon": "Knife"},
            ]},
        ],
    }
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_ticks_yields_one_stream_per_player(tmp_path):
    path = _write_match(tmp_path)
    adapter = AsgCollisionAdapter()
    ticks = list(adapter.ticks(f"{path}::111"))
    assert len(ticks) == 2
    assert ticks[0].time == pytest.approx(100 / ASG_SCHEMA.sample_hz)
    np.testing.assert_allclose(ticks[0].entities["player"]["position"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(ticks[1].entities["player"]["position"], [1.0, 0.0, 0.0])
    assert ticks[0].entities["player"]["weapon"] == "AK-47"


def test_ticks_filters_to_the_requested_player_only(tmp_path):
    path = _write_match(tmp_path)
    adapter = AsgCollisionAdapter()
    ticks = list(adapter.ticks(f"{path}::222"))
    assert len(ticks) == 2
    assert ticks[0].entities["player"]["weapon"] == "Knife"


def test_check_source_raises_for_unknown_player(tmp_path):
    path = _write_match(tmp_path)
    adapter = AsgCollisionAdapter()
    adapter.check_source(f"{path}::111")
    with pytest.raises(ValueError):
        adapter.check_source(f"{path}::999")


def test_discover_map_for_source_covers_every_player(tmp_path):
    path = _write_match(tmp_path, map_name="de_mirage")
    mapping = AsgCollisionAdapter.discover_map_for_source([path])
    assert mapping == {f"{path}::111": "de_mirage", f"{path}::222": "de_mirage"}
