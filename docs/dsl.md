# The metric DSL — a closed grammar over the schema

A metric is a JSON spec describing a reduction over one segment's window of timesteps:

```
SOURCE  →  MAP (optional)  →  FILTER (optional)  →  REDUCE  [× SCOPE]
```

`compile_metric(spec, schema)` validates every token against the `SchemaSpec` and returns
`fn(WindowBuffer) -> float`. The grammar is **closed** — every token is enumerated — so a spec
is safe, JSON-serializable, and reproducible. Validation is at compile time: unknown fields,
ill-typed maps, vector-fed-to-scalar-reduce, and unknown spec keys (a typo'd `"fitler"` can't
silently drop a constraint) all raise immediately. The only spec keys are
`source` / `reduce` / `map` / `filter` / `scope` — and `map`/`filter`/`scope` are entity-source
only (`scope` also raises on globals/events; spatial reduces take a `filter` but no `map`).

```python
from kuration import compile_metric
fn = compile_metric({"source": "drone.pos", "map": "speed", "reduce": "mean"}, schema)
value = fn(window_buffer)   # usually called for you by FeatureEngine
```

## SOURCE

| form | resolves to |
|---|---|
| `<entity_axis>.<field>` | a whole field (scalar or vector) |
| `<entity_axis>.<field>.<dim>` | one vector component → scalar (e.g. `drone.pos.x`) |
| `<event_stream>.<field>` | an event field (e.g. `grenade.type`) |
| `<global>` | a per-timestep global scalar (e.g. `alive_count`) |

## MAP (optional)

Maps turn the source stream into something a reduce can consume. Universal maps are gated by
the field's `semantics`, so they're safe without naming fields:

| map | requires | effect |
|---|---|---|
| `abs` | any | absolute value |
| `delta` | any | consecutive difference (wraps if `angular`) |
| `speed` | `position` vector + `sample_hz` | per-step displacement magnitude × `sample_hz`; clamped by `max_rate`; compilation fails when the sampling rate is absent |
| `angle_delta` | `angular` | abs wrapped per-step change (summed over dims for a vector) |
| `bit:<NAME>` | `bitmask` field | extract bit `NAME` → 0/1 stream |

## FILTER (optional)

A per-cell mask over an entity field:

| form | example |
|---|---|
| `<field><op><value>` (`>`,`>=`,`<`,`<=`,`==`) | `"hp>0"`, `"battery<=20"`, `"item==AWP"` |
| `<field>:bit:<NAME>` | `"btn:bit:FIRE"` |

Comparison filters take **scalar** (or categorical, `==` only) fields — a vector field is
rejected at compile time (there is no `.<dim>` filter syntax). Filters also apply to spatial
reduces, narrowing which entities count as present (e.g. squad dispersion of the living:
`{"source": "player.pos", "reduce": "dispersion", "filter": "hp>0"}`).
Numeric comparison thresholds must be finite: `nan`, positive infinity, and negative infinity
are rejected while the metric is compiled.

## REDUCE

| group | reduces |
|---|---|
| numeric | `mean` `std` `var` `min` `max` `sum` `count` `rate` `entropy` `jerk` `p<q>` (e.g. `p95`) |
| spatial (need a `position` source) | `dispersion` `min_pair_dist` `hull_area` |
| categorical / event | `entropy` `nunique` `count` (events also: `rate`) |

`rate` = fraction of nonzero cells (pairs with `bit:` maps); `jerk` = mean abs consecutive
change per entity (acceleration/jitter) — gaps break adjacency, so a value jump across an
absent tick or a filtered-out cell is *not* counted as one step; `entropy` = Shannon over a
16-bin histogram (numeric) or category counts.

`dispersion` and `min_pair_dist` are computed in **every declared position dimension** (a 3D
`position` field is measured in 3D); `hull_area` is planar and uses the first two dimensions.

Event metrics count only events that **carry the referenced field** (an unrelated event in the
same stream isn't a `None` category), and an `==<value>` filter matches numeric event values
too (`"==5"` matches `code=5`).

## SCOPE (optional, default `pooled`)

- `pooled` — flatten all entity×time cells, then reduce.
- `per_entity_mean` — reduce per entity, then average across entities — honored for categorical
  and event sources as well as numeric ones.

Single-stream schemas (`E=1`) only need `pooled`.

## Examples

```jsonc
{"source": "drone.pos",  "map": "speed",       "reduce": "mean"}             // mean speed
{"source": "drone.pos",  "map": "speed",       "reduce": "p95"}              // burst speed
{"source": "drone.att",  "map": "angle_delta", "reduce": "mean"}             // mean turn rate
{"source": "player.btn", "map": "bit:FIRE",    "reduce": "rate"}             // fraction firing
{"source": "player.hp",  "reduce": "mean", "filter": "hp>0"}                 // mean hp of the living
{"source": "player.pos", "reduce": "dispersion"}                             // squad spread
{"source": "grenade.type", "reduce": "rate", "filter": "==Molotov"}          // molotov density
{"source": "player.weapon", "reduce": "nunique"}                             // weapon variety
```

## Escape hatch: registered Python metrics

Anything the grammar can't express is a plain function over a `WindowBuffer`:

```python
from kuration import register_metric

@register_metric("my_metric")
def my_metric(w):          # w: kuration.WindowBuffer
    return float(w.numeric_field("pos")[..., 2].mean())   # mean z, say
```

Strategies reference it by name like any DSL metric. A `WindowBuffer` exposes
`numeric_field(name) -> (T,E,D)`, `categorical_field(name) -> (codes, vocab)`, `valid (T,E)`,
`events`, and `globals`.
