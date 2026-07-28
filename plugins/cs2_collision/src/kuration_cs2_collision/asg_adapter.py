"""InputAdapter for the ASG `data.json` format (tick-by-tick CS2 match replay JSON).

One kuration "source" = one (match, player) pair -- matching how the reference scripts'
own `by_player`/`tick_index` helpers already organize match data (one continuous tick
sequence per player), and how `plugins/action_state_consistency`/`plugins/pixel_shift`
treat a `segment_id`'s `source_id` as one continuous clip. This keeps windowing,
lookbehind, and the collision operator's per-source mesh lookup all operating on one
player's own uninterrupted tick sequence, with no cross-player mixing inside a window.

Only the raw physics fields the collision operator (and, generically, any other
operator) might need are emitted: `position`, `angle`, `hp`, `btn_mask`, `flags`,
`weapon`. No collision detection/classification logic lives here -- that is entirely
the `collision_operator` module's concern, kept upstream-format-agnostic on purpose.
"""
from __future__ import annotations

import json
from collections.abc import Iterable

import numpy as np

from kuration import FieldSpec, SchemaSpec
from kuration.buffer import Tick

ASG_SCHEMA = SchemaSpec(
    name="asg_cs2_collision",
    entity_axis="player",
    # ASG match data is sampled every 4 of 64 server ticks -> one sample every 62.5ms (16Hz),
    # matching the reference physics_utils.py's DT constant.
    sample_hz=16.0,
    fields={
        "position": FieldSpec(name="position", kind="vector", dims=("x", "y", "z"),
                              semantics="position"),
        "angle": FieldSpec(name="angle", kind="vector", dims=("yaw", "pitch"),
                           semantics="angular"),
        "hp": FieldSpec(name="hp", kind="scalar"),
        "btn_mask": FieldSpec(name="btn_mask", kind="scalar"),
        "flags": FieldSpec(name="flags", kind="scalar"),
        "weapon": FieldSpec(name="weapon", kind="categorical"),
    },
)

_SOURCE_SEP = "::"


def _map_name(data: dict) -> str:
    header = data.get("header", {})
    if "map_name" in header:
        return header["map_name"]
    return header.get("demo_file_header", {}).get("map_name", "unknown")


class AsgCollisionAdapter:
    """`InputAdapter` over ASG `data.json` files. `source_uri` is `"<path>::<id32>"`."""

    schema = ASG_SCHEMA

    def ticks(self, source_uri: str) -> Iterable[Tick]:
        path, id32_str = source_uri.rsplit(_SOURCE_SEP, 1)
        id32 = int(id32_str)
        with open(path) as f:
            data = json.load(f)

        seq = []
        for t in data.get("ticks", []):
            for p in t["players"]:
                if p["id32"] == id32:
                    seq.append((t["tick"], p))
        seq.sort(key=lambda x: x[0])

        for tick, p in seq:
            pos = p["pos"]
            ang = p["ang"]
            yield Tick(
                time=tick / self.schema.sample_hz,
                entities={
                    "player": {
                        "position": np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float64),
                        "angle": np.array([ang["yaw"], ang["pitch"]], dtype=np.float64),
                        "hp": float(p["hp"]),
                        "btn_mask": float(p["btn_mask"]),
                        "flags": float(p["flags"]),
                        "weapon": p.get("weapon", "unknown"),
                    },
                },
            )

    def check_source(self, source_uri: str) -> None:
        path, id32_str = source_uri.rsplit(_SOURCE_SEP, 1)
        id32 = int(id32_str)
        with open(path) as f:
            data = json.load(f)
        if not any(p["id32"] == id32 for t in data.get("ticks", []) for p in t["players"]):
            raise ValueError(f"player {id32} not found in {path!r}")

    @staticmethod
    def discover_map_for_source(paths) -> dict:
        """{"<path>::<id32>": map_name} for every player found across the given
        data.json paths -- directly usable as the collision operator's `map_for_source`."""
        mapping: dict = {}
        for path in paths:
            with open(path) as f:
                data = json.load(f)
            map_name = _map_name(data)
            seen = {p["id32"] for t in data.get("ticks", []) for p in t["players"]}
            for id32 in seen:
                mapping[f"{path}{_SOURCE_SEP}{id32}"] = map_name
        return mapping
