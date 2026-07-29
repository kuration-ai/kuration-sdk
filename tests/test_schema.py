"""SchemaSpec load/validation — loud failure on malformed schemas, round-trip fidelity."""
from __future__ import annotations

import pytest

from kuration.core.schema import FieldSpec, SchemaSpec

GOOD = {
    "name": "demo",
    "entity_axis": "agent",
    "sample_hz": 2.0,
    "fields": {
        "pos": {"kind": "vector", "dims": ["x", "y", "z"], "semantics": "position", "max_rate": 100},
        "ang": {"kind": "vector", "dims": ["yaw", "pitch"], "semantics": "angular"},
        "hp": {"kind": "scalar"},
        "btn": {"kind": "scalar", "semantics": "bitmask", "bits": {"FIRE": 1, "JUMP": 2}},
        "weapon": {"kind": "categorical"},
    },
    "event_streams": {"event": {"fields": {"type": {"kind": "categorical"}}}},
    "globals": {"alive": {"kind": "scalar"}},
}


def test_good_schema_loads_and_round_trips():
    s = SchemaSpec.from_dict(GOOD)
    assert s.entity_axis == "agent" and s.sample_hz == 2.0
    assert s.fields["pos"].width == 3 and s.fields["hp"].width == 1
    assert s.fields["btn"].bits["FIRE"] == 1
    # Round-trip: to_dict reloads to an equivalent schema.
    again = SchemaSpec.from_dict(s.to_dict())
    assert again.to_dict() == s.to_dict()


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d["fields"]["pos"].pop("dims"), "vector"),                       # vector needs dims
    (lambda d: d["fields"].__setitem__("bad", {"kind": "blob"}), "kind"),       # bad kind
    (lambda d: d["fields"]["btn"].pop("bits"), "bitmask"),                       # bitmask needs bits
    (lambda d: d["fields"].__setitem__("p2", {"kind": "scalar", "semantics": "position"}), "position"),
    (lambda d: d.__setitem__("globals", {"agent": {"kind": "scalar"}}), "more than once"),  # ns clash
    (lambda d: d.__setitem__("fields", {}), "at least one field"),
])
def test_bad_schema_raises(mutate, msg):
    import copy
    d = copy.deepcopy(GOOD)
    mutate(d)
    with pytest.raises(ValueError) as e:
        SchemaSpec.from_dict(d)
    assert msg in str(e.value)


@pytest.mark.parametrize("change,match", [
    ({"fields": {"bad.name": {"kind": "scalar"}}}, "identifier"),
    ({"fields": {"v": {"kind": "vector", "dims": ["x", "x"]}}}, "unique"),
    ({"fields": {"x": {"kind": "scalar", "max_rate": float("nan")}}}, "finite"),
    ({"fields": {"x": {"kind": "scalar", "typo": 1}}}, "unknown"),
])
def test_schema_rejects_dsl_incompatible_or_unknown_values(change, match):
    with pytest.raises(ValueError, match=match):
        SchemaSpec.from_dict({"name": "bad", **change})


def test_direct_constructors_validate_and_nested_mappings_are_immutable():
    with pytest.raises(ValueError, match="kind"):
        FieldSpec("x", "blob")
    schema = SchemaSpec.from_dict({"name": "immutable", "fields": {
        "mask": {"kind": "scalar", "semantics": "bitmask", "bits": {"ON": 1}}}})
    with pytest.raises(TypeError, match="immutable"):
        schema.fields["extra"] = FieldSpec("extra", "scalar")
    with pytest.raises(TypeError, match="immutable"):
        schema.fields["mask"].bits["OFF"] = 2


@pytest.mark.parametrize("name", [1, "", "with space", "with.dot"])
def test_bit_names_must_be_string_dsl_identifiers(name):
    with pytest.raises(ValueError, match="bit name"):
        FieldSpec(
            "mask",
            "scalar",
            semantics="bitmask",
            bits={name: 1},
        )


def test_single_stream_fields_are_valid_unqualified_dsl_sources():
    from kuration.core.dsl import compile_metric
    from kuration.core.buffer import Tick, build_window
    schema = SchemaSpec.from_dict({"name": "single", "fields": {"x": {"kind": "scalar"}}})
    metric = compile_metric({"source": "x", "reduce": "mean"}, schema)
    window = build_window(schema, [Tick(time=0, entities={0: {"x": 3.0}})])
    assert metric(window) == 3.0


def test_dimension_labels_allow_domain_punctuation_but_remain_dsl_addressable():
    from kuration.core.dsl import compile_metric
    schema = SchemaSpec.from_dict({"name": "actions", "fields": {
        "action": {"kind": "vector", "dims": ["mouse_x_-1000"]}}})
    compile_metric({"source": "action.mouse_x_-1000", "reduce": "mean"}, schema)
    with pytest.raises(ValueError, match="cannot contain dots"):
        FieldSpec("action", "vector", ("ambiguous.component",))
