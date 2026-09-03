"""Compositional operator registration, expansion, and provenance."""
from __future__ import annotations

import ast
import json
import re
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


def test_operator_name_that_shadows_a_pipeline_attribute_is_rejected():
    """A method spelling that is a real Pipeline attribute (`report`, `window`, `schema`, ...) was
    silently shadowed — ``__getattr__`` never runs — so registration now rejects it."""
    registry = Registry(default_registry)
    for name in ("acme/report", "custom/window", "acme/schema", "filter"):
        with pytest.raises(ValueError, match="silently shadowed"):
            registry.register_operator(name, lambda pipeline: pipeline.head(1))
        assert name not in registry.names("operator")

    with pytest.raises(ValueError, match=r"\['report'\].*rename the component"):
        registry.register_operator("acme/report", lambda pipeline: pipeline.head(1))

    # A namespaced name whose local part does not collide stays callable under both spellings,
    # and the built-in attribute is untouched.
    registry.register_operator("acme/report-balance",
                               lambda pipeline: pipeline.measure("rb", "mean", of="signal"))
    pipeline = _pipeline().using(registry)
    assert "rb" in pipeline.report_balance().describe()["strategy"]["define"]
    assert "rb" in pipeline.acme__report_balance().describe()["strategy"]["define"]
    assert input([{"signal": float(i)} for i in range(12)]).window(3)._window_desc["window"] == 3


def test_plugin_check_reports_a_shadowed_operator_name():
    """``kur-plugin-check`` must name the collision, not only fail somewhere inside the import."""
    from kuration.core.plugin_validation import _descriptor_errors
    from kuration.core.registry import ComponentDescriptor

    def descriptor(name):
        return ComponentDescriptor(
            name=name, kind="operator", description="operator under check",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            package="kuration-acme", install_hint="pip install kuration-acme", version="1")

    errors = _descriptor_errors(descriptor("acme/stats"), expected_package="kuration-acme")
    assert any("silently shadowed" in message and "acme/stats" in message for message in errors)
    assert _descriptor_errors(descriptor("acme/stats-balance"),
                              expected_package="kuration-acme") == []


def test_plain_operator_name_outranks_a_namespaced_local_shorthand():
    """``smooth`` and ``acme/smooth`` both wanted ``pipeline.smooth()``; the complete name wins and
    the namespaced operator keeps its qualified spelling."""
    registry = Registry(default_registry)
    registry.register_operator(
        "smooth", lambda pipeline: pipeline.measure("plain_value", "mean", of="signal"))
    registry.register_operator(
        "acme/smooth", lambda pipeline: pipeline.measure("acme_value", "mean", of="signal"))
    pipeline = _pipeline().using(registry)

    assert "plain_value" in pipeline.smooth().describe()["strategy"]["define"]
    assert "acme_value" in pipeline.acme__smooth().describe()["strategy"]["define"]
    assert {"smooth", "acme__smooth"} <= set(dir(pipeline))


def _call_suggestion(pipeline, suggestion: str):
    """Call one spelling exactly as an ambiguity message suggests it."""
    match = re.fullmatch(r"apply\((.+)\)", suggestion)
    if match:
        return pipeline.apply(ast.literal_eval(match.group(1)))
    assert suggestion.endswith("()"), suggestion
    return getattr(pipeline, suggestion[:-2])()


def _suggestions(message: str) -> list[str]:
    return ast.literal_eval(re.search(r"use one of (\[.*\])", message).group(1))


def test_every_ambiguity_suggestion_is_a_spelling_that_resolves():
    """The message used to suggest the ambiguous spelling itself, leaving one operator with no
    working call. Every suggestion must resolve to the operator it names."""
    registry = Registry(default_registry)
    # 'a.b/c' shares its local shorthand ('c') with both others, its friendly qualified alias
    # ('a_b__c') with 'a-b/c', and its canonical alias ('a_db__c') with 'a_db/c' — so it has no
    # unique method spelling at all and must be suggested as apply().
    for name, measure in (("a.b/c", "dotted_value"), ("a-b/c", "hyphen_value"),
                          ("a_db/c", "encoded_value")):
        registry.register_operator(
            name, lambda pipeline, measure=measure: pipeline.measure(measure, "mean", of="signal"))
    pipeline = _pipeline().using(registry)

    with pytest.raises(AttributeError) as info:
        pipeline.c()
    suggestions = _suggestions(str(info.value))
    assert len(suggestions) == 3
    assert "apply('a.b/c')" in suggestions
    defined = set()
    for suggestion in suggestions:
        defined |= set(_call_suggestion(pipeline, suggestion).describe()["strategy"]["define"])
    assert {"dotted_value", "hyphen_value", "encoded_value"} <= defined

    with pytest.raises(AttributeError) as info:
        pipeline.a_db__c()
    for suggestion in _suggestions(str(info.value)):
        _call_suggestion(pipeline, suggestion)


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
