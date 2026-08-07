# SchemaSpec — declaring a dataset's vocabulary

Schemas may also declare an auxiliary `media` mapping. Media columns (initially
`kind: video-frame`) carry dtype, tensor shape, layout, color space, timing, and provider
capabilities. A media-only schema is valid. Media is excluded from ordinary fields and the metric
DSL; batch transforms and outputs address it as `media:<name>`.

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
| `fields` | unless `media` is non-empty | the per-entity, per-timestep observables (see below). |
| `media` | unless `fields` is non-empty | auxiliary semantic media columns, excluded from ordinary metric buffers |
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

Bit names use the same portable DSL-identifier grammar as field names
(`[A-Za-z_][A-Za-z0-9_]*`); non-string names are rejected rather than stringified.

Bit values are exact positive integers. Booleans, floats, numeric strings, zero, and negative
values are rejected instead of being coerced into a different mask.

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

## Media columns

```jsonc
"media": {
  "frame": {
    "kind": "video-frame", "dtype": "uint8", "shape": [1080, 1920, 3],
    "layout": "HWC", "color_space": "RGB", "timing": "pts",
    "capabilities": {"seeking": true, "variable_frame_rate": true, "streaming": true}
  }
}
```

The initial kind is `video-frame`; layout is `HWC` or `CHW` and must match the shape rank.
Transforms declare required kind/layout/color/timing semantics and fail before decode when the
column is incompatible. Media-only schemas use `"fields": {}`. `.tag()` remains selection-label
coverage and is unrelated to semantic media types.

## Single-stream datasets

If the data has no entity dimension, give a nominal `entity_axis` (e.g. `"series"`) and emit
one entity per timestep; everything else is identical (the entity dimension is simply `E=1`).

`schema.to_dict()` round-trips to an equivalent schema — this is what lands in `config.json`.
