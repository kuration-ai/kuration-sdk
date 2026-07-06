# SchemaSpec — declaring a dataset's vocabulary

A `SchemaSpec` is pure data (JSON / dict) describing the fields the metric DSL may reference.
It is what makes the closed grammar dataset-agnostic without opening it up: the grammar
validates every token against the schema, and the resolved schema is embedded in
`config.json` for reproducibility.

```python
from kuration import SchemaSpec
schema = SchemaSpec.from_dict({...})
```

## Top-level keys

| key | required | meaning |
|---|---|---|
| `name` | yes | schema identifier (recorded in config). |
| `fields` | yes | the per-entity, per-timestep observables (see below). |
| `entity_axis` | no | name of the entity dimension (e.g. `"player"`, `"drone"`). Omit for single-stream data; then use one nominal entity. Field sources are written `<entity_axis>.<field>`. |
| `sample_hz` | no | sampling rate; enables seconds-based windowing and scales `speed`. |
| `event_streams` | no | sparse per-timestep events (see below). |
| `globals` | no | per-timestep scalar streams not attached to an entity. |

## Fields

Each field declares a `kind` and optional `semantics` — the tags that let *universal* maps
stay type-safe without the grammar naming any field:

```jsonc
"fields": {
  "pos":  {"kind": "vector", "dims": ["x","y","z"], "semantics": "position", "max_rate": 50},
  "ang":  {"kind": "vector", "dims": ["yaw","pitch"], "semantics": "angular"},
  "hp":   {"kind": "scalar"},
  "btn":  {"kind": "scalar", "semantics": "bitmask", "bits": {"FIRE": 1, "JUMP": 2}},
  "item": {"kind": "categorical"}
}
```

| `kind` | stored as | reduces it supports |
|---|---|---|
| `scalar` | `(T,E,1)` float | all numeric reduces |
| `vector` | `(T,E,D)` float (`D=len(dims)`) | needs a map to a scalar (`speed`/`angle_delta`/`.<dim>`), or a spatial reduce |
| `categorical` | `(T,E)` int codes + per-window vocab | `entropy` / `nunique` / `count` |

**`semantics` tags** (optional) gate which maps/reduces are legal:

| semantics | requires | unlocks |
|---|---|---|
| `position` | vector, ≥2 dims | `speed`, spatial reduces (`dispersion`/`hull_area`/`min_pair_dist`); `max_rate` clamps `speed` |
| `angular` | scalar or vector | `angle_delta`; `delta` wraps (degrees) |
| `bitmask` | scalar + `bits` map | `bit:<NAME>` map and `<field>:bit:<NAME>` filter |

`max_rate` (any numeric field) is the per-field outlier clamp for `speed` (drops teleport-like
spikes to NaN); it must be a positive number. Validation is loud: a vector without `dims`,
`position` on a scalar, `bitmask` without `bits`, a non-numeric `max_rate`, a namespace
collision — all raise at `from_dict`. (Adapters are held to the schema too: supplying a scalar
for a declared vector field raises instead of silently broadcasting across its dims.)

## Event streams & globals

```jsonc
"event_streams": { "grenade": {"fields": {"type": {"kind": "categorical"}}} },
"globals":       { "alive_count": {"kind": "scalar"} }
```

Events are referenced `grenade.type` (reduces: `rate`/`count`/`nunique`/`entropy`, optional
filter `==<value>`); globals are referenced bare (`alive_count`, reduces only).

## Single-stream datasets

If the data has no entity dimension, give a nominal `entity_axis` (e.g. `"series"`) and emit
one entity per timestep; everything else is identical (the entity dimension is simply `E=1`).

`schema.to_dict()` round-trips to an equivalent schema — this is what lands in `config.json`.
