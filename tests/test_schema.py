"""SchemaSpec load/validation — loud failure on malformed schemas, round-trip fidelity."""
from __future__ import annotations

import pytest

from kuration.schema import SchemaSpec

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
