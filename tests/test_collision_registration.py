"""Collision registration contract: the components register through the bundled
`kuration.gameplay:register` plugin entry point (there is no separate collision
distribution any more -- the code moved from `plugins/collision/` into
`src/kuration/gameplay/collision/`), resolve to this package's own objects in an
isolated registry snapshot, and every registered component -- both adapters, the
transform/operator pair, the directional analyzer, and its pipeline-preserving companion
operator -- surfaces through the real `kur-list`/`kur-describe` CLI entry points, not
just the Python registry API. Mirrors the pixel-shift/action-state contract tests.
"""
from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import entry_points

import pytest

from kuration import default_registry, list_components
from kuration.gameplay.collision import (
    ADAPTER_NAME,
    COMPONENT_NAME,
    DIRECTIONAL_COMPONENT_NAME,
    TEAPEARCE_ADAPTER_NAME,
    AsgCollisionAdapter,
    CollisionMechanics,
    TeaPearceDust2Adapter,
    attach_directional_collision,
    collision_mechanics,
    directional_collision_analyzer,
    register,
)

pytest.importorskip("trimesh")

register()


def test_collision_registers_through_the_bundled_gameplay_entry_point():
    installed = entry_points()
    candidates = (
        installed.select(group="kuration.plugins", name="gameplay")
        if hasattr(installed, "select")
        else [point for point in installed.get("kuration.plugins", ())
              if point.name == "gameplay"]
    )
    assert [point.value for point in candidates] == ["kuration.gameplay:register"]

    # A fresh interpreter that only ever calls core's plugin discovery must end up with
    # the collision components -- the exact path a job-spec/CLI run takes.
    code = f"""
from kuration import default_registry, load_plugins
assert "{COMPONENT_NAME}" not in default_registry.names("operator")
load_plugins()
assert "{COMPONENT_NAME}" in default_registry.names("transform")
assert "{COMPONENT_NAME}" in default_registry.names("operator")
assert "{DIRECTIONAL_COMPONENT_NAME}" in default_registry.names("analyzer")
assert "{DIRECTIONAL_COMPONENT_NAME}" in default_registry.names("operator")
assert "{ADAPTER_NAME}" in default_registry.names("adapter")
assert "{TEAPEARCE_ADAPTER_NAME}" in default_registry.names("adapter")
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_isolated_snapshot_resolves_to_this_packages_own_objects():
    isolated = default_registry.snapshot()
    assert isolated.get("adapter", ADAPTER_NAME) is AsgCollisionAdapter
    assert isolated.get("adapter", TEAPEARCE_ADAPTER_NAME) is TeaPearceDust2Adapter
    assert isolated.get("transform", COMPONENT_NAME) is CollisionMechanics
    assert isolated.get("operator", COMPONENT_NAME) is collision_mechanics
    assert isolated.get("analyzer", DIRECTIONAL_COMPONENT_NAME) is directional_collision_analyzer
    assert isolated.get("operator", DIRECTIONAL_COMPONENT_NAME) is attach_directional_collision


def test_every_registered_component_discoverable_through_the_kur_list_cli(monkeypatch, capsys):
    from kuration.core.cli.discovery import list_main

    expected_by_kind = {
        "adapters": {ADAPTER_NAME, TEAPEARCE_ADAPTER_NAME},
        "transforms": {COMPONENT_NAME},
        "operators": {COMPONENT_NAME, DIRECTIONAL_COMPONENT_NAME},
        "analyzers": {DIRECTIONAL_COMPONENT_NAME},
    }
    for kind, expected_names in expected_by_kind.items():
        monkeypatch.setattr(sys, "argv", ["kur-list", kind, "--json"])
        list_main()
        rows = json.loads(capsys.readouterr().out)
        names = {row["name"] for row in rows}
        assert expected_names <= names, f"{kind}: missing {expected_names - names}"


def test_every_registered_component_describable_through_the_kur_describe_cli(monkeypatch, capsys):
    from kuration.core.cli.discovery import describe_main

    for kind, name in [
        ("adapter", ADAPTER_NAME), ("adapter", TEAPEARCE_ADAPTER_NAME),
        ("transform", COMPONENT_NAME), ("operator", COMPONENT_NAME),
        ("analyzer", DIRECTIONAL_COMPONENT_NAME),
        ("operator", DIRECTIONAL_COMPONENT_NAME),
    ]:
        monkeypatch.setattr(sys, "argv", ["kur-describe", kind, name, "--json"])
        describe_main()
        row = json.loads(capsys.readouterr().out)
        assert row["name"] == name
        assert row["package"] == "kuration-sdk"
        assert row["extra"] == "collision"
        assert row["parameters"]["additionalProperties"] is False


def test_directional_analyzer_registered_via_list_components_api():
    names = {row["name"] for row in list_components("analyzer")}
    assert DIRECTIONAL_COMPONENT_NAME in names
