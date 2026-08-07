"""Manifest ordering must preserve entity type identity."""
from __future__ import annotations

from kuration import SchemaSpec, Tick, input
from kuration.core.output.dataset import _admitted_order_key
from kuration.core.output.shardwriter import entity_id_token


def test_type_distinct_entity_ids_have_a_canonical_order():
    admitted = [("source", 1, 0), ("source", "1", 0)]

    forward = sorted(admitted, key=_admitted_order_key)
    reverse = sorted(reversed(admitted), key=_admitted_order_key)

    assert forward == reverse
    assert _admitted_order_key(admitted[0]) != _admitted_order_key(admitted[1])


class _ReversedEntityAdapter:
    def __init__(self, schema):
        self.schema = schema
        self._ticks = [
            Tick(time=index, entities={"1": {"x": 2.0}, 1: {"x": 1.0}})
            for index in range(2)
        ]

    def ticks(self, _source):
        return list(self._ticks)

    def iter_ticks(self, _source):
        return iter(self._ticks)


def test_count_manifest_orders_entities_by_portable_identity(tmp_path):
    schema = SchemaSpec.from_dict(
        {
            "name": "entity-order",
            "entity_axis": "entity",
            "sample_hz": 1.0,
            "fields": {"x": {"kind": "scalar"}},
        }
    )
    pipeline = (
        input(adapter=_ReversedEntityAdapter(schema), sources=["source"], schema=schema)
        .window(2)
        .measure("level", "mean", of="x")
        .sample("top-k", n=1, weights={"level": 1.0})
    )

    result = pipeline.output(str(tmp_path / "dataset"), categorical="omit")

    identities = [entry["entity_id"] for entry in result["shards"]]
    assert identities == sorted((entity_id_token(1), entity_id_token("1")))
