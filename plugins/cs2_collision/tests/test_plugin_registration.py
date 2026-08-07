"""Plugin-wide registration contract: the `kuration.plugins` entry point resolves to this
package's own objects in an isolated registry snapshot (not merely "some callable with the
right name"), and every registered component -- both adapters, the transform/operator
pair, and the directional analyzer -- actually surfaces through the real `kur-list`/
`kur-describe` CLI entry points, not just the Python registry API. Mirrors
`plugins/pixel_shift/tests/test_pixel_shift.py`'s `test_plugin_entry_point_and_isolated_
snapshot_contract` / `test_pixel_shift_discoverable_through_the_kur_list_cli` pattern --
a convention this plugin didn't have yet.
"""
from __future__ import annotations

import json
import sys
from importlib.metadata import entry_points

import pytest

from kuration import default_registry, list_components

from kuration_cs2_collision import register
from kuration_cs2_collision.asg_adapter import AsgCollisionAdapter
from kuration_cs2_collision.collision_operator import COMPONENT_NAME, CollisionMechanics, collision_mechanics
from kuration_cs2_collision.directional_collision_analyzer import (
    COMPONENT_NAME as DIRECTIONAL_COMPONENT_NAME,
)
from kuration_cs2_collision.directional_collision_analyzer import directional_collision_analyzer
from kuration_cs2_collision.teapearce_adapter import TeaPearceDust2Adapter

pytest.importorskip("trimesh")

register()

ADAPTER_NAME = "aintropy/cs2-collision-asg"
TEAPEARCE_ADAPTER_NAME = "aintropy/cs2-collision-teapearce"


def test_plugin_entry_point_and_isolated_snapshot_contract():
    installed = entry_points()
    candidates = (
        installed.select(group="kuration.plugins", name="cs2_collision")
        if hasattr(installed, "select")
        else [point for point in installed.get("kuration.plugins", ())
              if point.name == "cs2_collision"]
    )
    assert [point.value for point in candidates] == ["kuration_cs2_collision:register"]

    isolated = default_registry.snapshot()
    assert isolated.get("adapter", ADAPTER_NAME) is AsgCollisionAdapter
    assert isolated.get("adapter", TEAPEARCE_ADAPTER_NAME) is TeaPearceDust2Adapter
    assert isolated.get("transform", COMPONENT_NAME) is CollisionMechanics
    assert isolated.get("operator", COMPONENT_NAME) is collision_mechanics
    assert isolated.get("analyzer", DIRECTIONAL_COMPONENT_NAME) is directional_collision_analyzer


def test_every_registered_component_discoverable_through_the_kur_list_cli(monkeypatch, capsys):
    from kuration.cli.discovery import list_main

    expected_by_kind = {
        "adapters": {ADAPTER_NAME, TEAPEARCE_ADAPTER_NAME},
        "transforms": {COMPONENT_NAME},
        "operators": {COMPONENT_NAME},
        "analyzers": {DIRECTIONAL_COMPONENT_NAME},
    }
    for kind, expected_names in expected_by_kind.items():
        monkeypatch.setattr(sys, "argv", ["kur-list", kind, "--json"])
        list_main()
        rows = json.loads(capsys.readouterr().out)
        names = {row["name"] for row in rows}
        assert expected_names <= names, f"{kind}: missing {expected_names - names}"


def test_every_registered_component_describable_through_the_kur_describe_cli(monkeypatch, capsys):
    from kuration.cli.discovery import describe_main

    for kind, name in [
        ("adapter", ADAPTER_NAME), ("adapter", TEAPEARCE_ADAPTER_NAME),
        ("transform", COMPONENT_NAME), ("operator", COMPONENT_NAME),
        ("analyzer", DIRECTIONAL_COMPONENT_NAME),
    ]:
        monkeypatch.setattr(sys, "argv", ["kur-describe", kind, name, "--json"])
        describe_main()
        row = json.loads(capsys.readouterr().out)
        assert row["name"] == name
        assert row["package"] == "kuration-plugin-cs2-collision"
        assert row["parameters"]["additionalProperties"] is False


def test_directional_analyzer_registered_via_list_components_api():
    names = {row["name"] for row in list_components("analyzer")}
    assert DIRECTIONAL_COMPONENT_NAME in names
