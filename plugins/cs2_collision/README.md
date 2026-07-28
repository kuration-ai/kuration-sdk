# Kuration CS2 Collision

Detects and classifies CS2 wall collisions -- `head_on`, `angled`, `W_slide`, `WA_slide`,
`slide_end`, `slide_constrained` -- from per-tick position/view-angle telemetry, raycast
against the match's map geometry. Ported from the reference two-stage pipeline
(`action_state_analyzer.py`'s learned-baseline anomaly scan + `collision_classifier.py`'s
raycast classification, as fused by `collision_pipeline_fused.py`) into a Kuration
`BatchTransform` + compositional `operator` pair.

## Install

```bash
pip install kuration-plugin-cs2-collision
```

For repository development:

```bash
python -m pip install -e . -e plugins/cs2_collision[test]
python -m pytest plugins/cs2_collision/tests -q
```

## Format-agnostic by design

This operator only ever reads canonical schema fields -- `position` (vector,
`semantics="position"`), `angle` (vector, `dims=("yaw","pitch")`,
`semantics="angular"`), `hp`, `btn_mask`, `flags` -- never a dataset's raw field names.
Any `InputAdapter` that maps its own raw telemetry onto those same canonical names can
drive this same registered operator unmodified.

`asg_adapter.py` ships the first such adapter: it reads the ASG `data.json` match-replay
format (tick-by-tick, nested `header` + `ticks[].players[]`) and yields one Kuration
"source" per `(match, player)` pair -- one continuous tick sequence per player, matching
how the reference scripts' own `by_player`/`tick_index` helpers already organize match
data. `source_uri` is `"<data.json path>::<id32>"`.

A second adapter for the public `TeaPearce/CounterStrike_Deathmatch` dataset (which
carries the same position/angle/hp telemetry inside `frame_i_xaux` and the dataset's
`.npy` metadata files, once actually read -- see this plugin's design notes) is planned
but not yet implemented; the operator itself needs no changes to support it.

## What gets registered

- `aintropy/cs2-collision-asg` -- an `adapter`: ASG `data.json` -> canonical `Tick`s.
- `aintropy/cs2-collision` -- a `transform` (the actual per-window compute) and an
  `operator` (its fluent-pipeline wrapper) under the same name, mirroring
  `plugins/pixel_shift`'s pattern.

Discover without reading a dataset:

```bash
kur-list adapters
kur-list transforms
kur-list operators
kur-describe operator aintropy/cs2-collision --json
```

## Usage

```python
from kuration import default_registry, input
from kuration_cs2_collision import register
from kuration_cs2_collision.asg_adapter import AsgCollisionAdapter

register()

data_json_paths = [...]   # your ASG data.json files
map_for_source = AsgCollisionAdapter.discover_map_for_source(data_json_paths)
map_registry = {"de_dust2": ["/path/to/de_dust2.bsp"], ...}

pipeline = (
    input(adapter=AsgCollisionAdapter(), sources=list(map_for_source), schema=AsgCollisionAdapter.schema)
    .window(256, stride=128)
    .apply("aintropy/cs2-collision",
           map_for_source=map_for_source,
           map_registry=map_registry,
           dist_path="action_state_dist.json")
    .sample("equal", n=200)
)
```

## Output

One value per output, per window (segment) -- not per tick:

| output | meaning |
|---|---|
| `collision_count` | total confirmed collision ticks in this window |
| `head_on_count` / `angled_count` / `w_slide_count` / `wa_slide_count` / `slide_end_count` / `slide_constrained_count` | per-type counts |
| `max_anomaly_score` | highest z-score seen across every scored tick in the window (continuous informativeness signal, independent of whether a collision was confirmed) |
| `min_dist_to_wall_hu` | closest confirmed-collision proximity-ray distance in the window, or `-1.0` if none |

## Map geometry caching

Parsing raw `.bsp`/`.glb`/`.nav` files (Source-engine lump tables + texture-based
non-solid-face filtering for `.bsp`) is the expensive part, not building the raycast
BVH. `mesh_cache.py` caches the parsed, filtered `(vertices, faces)` arrays to an
on-disk `.npz` (`disk_cache_dir`, default `.kuration_cs2_collision_cache`), so repeated
runs across separate processes skip raw parsing entirely. Within one pipeline run, an
in-memory `{map_name: mesh}` dict on the transform instance gives free reuse across
every source sharing a map -- safe because Kuration's batch-transform stage runs
single-process/sequential with one transform instance reused for the whole run (see
`collision_operator.py`'s module docstring).

## Known limitations (first pass)

- The classifier's `ray_cache` (bucketed by quantized position/yaw) is scoped per
  window call rather than per whole source, unlike the reference scripts -- a
  performance-only difference, not a correctness one.
- Overlapping windows (`stride < window`) are not deduplicated the way
  `plugins/pixel_shift` deduplicates isolated-press instances via
  `ownership="rightmost-start"` -- a collision tick inside the overlap of two windows
  is currently counted in both. Worth adding the same ownership convention if
  overlapping windows are used for curation.
