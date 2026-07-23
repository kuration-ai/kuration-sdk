"""Compositional operator registration, expansion, and provenance."""
from __future__ import annotations

import json
import sys

import pytest

from kuration import (
    Pipeline,
    Registry,
    default_registry,
    describe_component,
    input,
    list_components,
    register_operator,
)


PARAMETERS = {
    "type": "object",
    "properties": {
        "field": {"type": "string"},
        "bins": {"type": "integer", "minimum": 1},
    },
    "required": ["field"],
    "additionalProperties": False,
}


@register_operator(
    "test/balance-signal",
    description="Measure and balance a scalar signal",
    parameters=PARAMETERS,
    version="1",
)
def balance_signal(pipeline, *, field, bins=4):
    return pipeline.measure("signal_level", "mean", of=field).organize(
        "signal_level", bins=bins)


def _pipeline():
    return input([{"signal": float(i)} for i in range(12)]).window(4)


def test_registered_operator_expands_without_mutating_input():
    base = _pipeline()
    expanded = base.balance_signal(field="signal", bins=3)

    assert "signal_level" not in base.describe()["strategy"]["define"]
    assert "signal_level" in expanded.describe()["strategy"]["define"]
    assert expanded.describe()["strategy"]["binning"]["axes"] == [
        {"metric": "signal_level", "bins": 3}]

    call = expanded._ops[-1]
    assert call["op"] == "apply"
    assert call["args"] == ["test/balance-signal"]
    assert call["kwargs"] == {"field": "signal", "bins": 3}
    assert [item["op"] for item in call["expansion"]] == ["measure", "organize"]
    assert call["component"]["kind"] == "operator"
    assert call["component"]["fingerprint"]


def test_operator_parameters_are_eagerly_validated():
    with pytest.raises(ValueError, match="missing parameter"):
        _pipeline().balance_signal()
    with pytest.raises(ValueError, match="unknown parameter"):
        _pipeline().balance_signal(field="signal", typo=True)
    with pytest.raises(ValueError, match="JSON type integer"):
        _pipeline().balance_signal(field="signal", bins="three")
    with pytest.raises(ValueError, match="must be >= 1"):
        _pipeline().balance_signal(field="signal", bins=0)
    with pytest.raises(ValueError, match="unknown operator"):
        _pipeline().apply("test/absent")


def test_operator_must_return_a_derived_nonempty_pipeline():
    registry = Registry(default_registry)
    registry.register_operator("test/wrong-type", lambda pipeline: object())
    registry.register_operator("test/no-op", lambda pipeline: pipeline)
    registry.register_operator("test/wrong-root", lambda pipeline: input([{"x": 1}]).window(1))

    with pytest.raises(ValueError, match="must return a Pipeline"):
        _pipeline().using(registry).wrong_type()
    with pytest.raises(ValueError, match="must add at least one"):
        _pipeline().using(registry).no_op()
    with pytest.raises(ValueError, match="must derive"):
        _pipeline().using(registry).wrong_root()


def test_operator_supports_registry_overlays_and_discovery():
    registry = Registry(default_registry)

    def add_measure(pipeline, *, field):
        return pipeline.measure("local_mean", of=field)

    registry.register_operator("lab/local-mean", add_measure, parameters=PARAMETERS)
    p = _pipeline().using(registry).local_mean(field="signal")
    assert isinstance(p, Pipeline)
    assert "local_mean" in p.describe()["strategy"]["define"]

    names = {row["name"] for row in list_components("operator")}
    assert "test/balance-signal" in names
    assert describe_component("operator", "test/balance-signal")["version"] == "1"


def test_operator_provenance_reaches_written_config(tmp_path):
    (_pipeline()
     .balance_signal(field="signal", bins=3)
     .curate("equal", n=2)
     .output(tmp_path / "out"))
    config = json.loads((tmp_path / "out" / "config.json").read_text())
    call = next(item for item in config["pipeline"] if item["op"] == "apply")
    assert call["component"]["name"] == "test/balance-signal"
    assert [item["op"] for item in call["expansion"]] == ["measure", "organize"]


def test_operator_is_available_through_discovery_cli(monkeypatch, capsys):
    from kuration.core.cli.discovery import list_main

    monkeypatch.setattr(sys, "argv", ["kur-list", "operators", "--json"])
    list_main()
    rows = json.loads(capsys.readouterr().out)
    assert "test/balance-signal" in {row["name"] for row in rows}


def test_operator_methods_are_discoverable_and_keep_apply_compatibility():
    pipeline = _pipeline()

    assert "balance_signal" in dir(pipeline)
    assert "test__balance_signal" in dir(pipeline)
    direct = pipeline.test__balance_signal(field="signal", bins=3)
    compatibility = pipeline.apply("test/balance-signal", field="signal", bins=3)
    assert direct._ops == compatibility._ops


def test_ambiguous_operator_methods_have_qualified_alternatives():
    registry = Registry(default_registry)

    def alpha(pipeline):
        return pipeline.measure("alpha_value", "mean", of="signal")

    def beta(pipeline):
        return pipeline.measure("beta_value", "mean", of="signal")

    registry.register_operator("alpha/shared-name", alpha)
    registry.register_operator("beta/shared-name", beta)
    pipeline = _pipeline().using(registry)

    with pytest.raises(AttributeError, match="ambiguous.*alpha__shared_name"):
        pipeline.shared_name()

    alpha_result = pipeline.alpha__shared_name()
    beta_result = pipeline.beta__shared_name()
    assert "alpha_value" in alpha_result.describe()["strategy"]["define"]
    assert "beta_value" in beta_result.describe()["strategy"]["define"]


def test_qualified_operator_method_handles_a_core_method_name_collision():
    registry = Registry(default_registry)
    registry.register_operator(
        "custom/window",
        lambda pipeline, *, size: pipeline.window(size),
        parameters={
            "type": "object",
            "properties": {"size": {"type": "integer", "minimum": 1}},
            "required": ["size"],
            "additionalProperties": False,
        },
    )
    pipeline = input([{"signal": float(i)} for i in range(12)]).using(registry)

    assert pipeline.window(3)._window_desc["window"] == 3
    assert pipeline.custom__window(size=5)._window_desc["window"] == 5


def test_canonical_operator_alias_resolves_normalized_namespace_collisions():
    registry = Registry(default_registry)
    registry.register_operator(
        "alpha-one/shared",
        lambda pipeline: pipeline.measure("hyphen_value", "mean", of="signal"),
    )
    registry.register_operator(
        "alpha.one/shared",
        lambda pipeline: pipeline.measure("dot_value", "mean", of="signal"),
    )
    pipeline = _pipeline().using(registry)

    with pytest.raises(AttributeError, match="alpha_hone__shared.*alpha_done__shared"):
        pipeline.alpha_one__shared()

    hyphen = pipeline.alpha_hone__shared()
    dotted = pipeline.alpha_done__shared()
    assert "hyphen_value" in hyphen.describe()["strategy"]["define"]
    assert "dot_value" in dotted.describe()["strategy"]["define"]
