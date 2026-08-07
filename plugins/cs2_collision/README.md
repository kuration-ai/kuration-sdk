# Kuration CS2 Collision

Detects and classifies CS2 wall collisions -- `head_on`, `angled`, `W_slide`, `WA_slide`,
`slide_end`, `slide_constrained` -- from per-tick position/view-angle telemetry, raycast
against the match's map geometry. Ported from the reference two-stage pipeline
(`action_state_analyzer.py`'s learned-baseline anomaly scan + `collision_classifier.py`'s
raycast classification, as fused by `collision_pipeline_fused.py`) into a Kuration
`BatchTransform` + compositional `operator` pair.

A second, independent algorithm -- an acceleration-ramp-residual `Analyzer` needing no
learned baseline -- is also included; see
[Directional-residual analyzer](#directional-residual-analyzer-alternative-algorithm)
below.

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

`teapearce_adapter.py` is the second: it reads the public
`TeaPearce/CounterStrike_Deathmatch` dataset's dust2 `.npy` metadata files (NOT the
`.hdf5` files `examples/public_data/adapters/teapearce.py` reads -- those only carry a
51-dim action vector, no position/angle at all). Each `.npy` file's `curr_vars` column
carries real `localpos1/2/3` / `viewangle_xy/vert` / `obs_health` / `tp_wasd` /
`gsi_weap_active` telemetry, scraped via memory-read + GSI. One source = one `.npy`
recording. `weapon` is a `categorical` field, read only by the directional analyzer
below (`CollisionMechanics` never asks for it) -- see
[Directional-residual analyzer](#directional-residual-analyzer-alternative-algorithm).

**Verified against real dust2 map geometry before trusting this data**: bounding-box
containment (the `.bsp`'s bounds fully contain every observed TeaPearce position), raw
raycast hit-rate at a plausible non-degenerate distance distribution (94/200 sampled
ticks hit geometry within 200 HU, distances spread 25.6-198.3 HU), and NAV-mesh
floor-height checks landing within ~5-8 units of ground level at both real respawn
events found in a 3,000-frame sample -- consistent with a genuinely aligned coordinate
system, not a coincidence. The recorded AK-47 velocity cap in that sample (215 HU/s)
also matched `WEAPON_CAPS['AK-47']` exactly, independently confirming CS:GO and CS2
share movement physics constants.

No onground or ducking signal exists in `curr_vars` (confirmed absent, not just
unread), so `TeaPearceDust2Adapter` reports `onground=True`/`ducking=False` always --
see the adapter's module docstring for why this is a safe simplification (ducking never
affects the actual collision-detection thresholds, only which learned-baseline bucket a
tick is z-scored against, and the ground-fallback lookup already handles an
always-ground assumption gracefully).

**dist_path is not yet TeaPearce-calibrated.** This plugin has no learn phase (see
`docs/customization.md`'s guidance on keeping the learn/fit step outside the SDK,
mirrored by the reference `action_state_analyzer.py learn` command). Running the
pipeline against TeaPearce data today means reusing an ASG-fit `action_state_dist.json`
as a placeholder -- it proves the pipeline mechanics end-to-end (validated: 21 windows
across 3 real recordings, 2 confirmed `slide_end` collisions, internally consistent
`max_anomaly_score`/`min_dist_to_wall_hu` values), but the anomaly *threshold* itself is
calibrated against ASG's player population, not TeaPearce's -- a real TeaPearce learn
run would very likely surface more, and more confidently-ranked, candidates.

## What gets registered

- `aintropy/cs2-collision-asg` -- an `adapter`: ASG `data.json` -> canonical `Tick`s.
- `aintropy/cs2-collision-teapearce` -- an `adapter`: TeaPearce dust2 `.npy` -> canonical `Tick`s.
- `aintropy/cs2-collision` -- a `transform` (the actual per-window compute) and an
  `operator` (its fluent-pipeline wrapper) under the same name, mirroring
  `plugins/pixel_shift`'s pattern.
- `aintropy/cs2-collision-directional` -- an `analyzer`: the physically-modeled,
  no-learned-baseline alternative described below.

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
map_registry = {"de_dust2": ["/path/to/de_dust2.bsp"]}   # add more maps as needed

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

### TeaPearce usage

```python
from kuration import input
from kuration_cs2_collision import register
from kuration_cs2_collision.teapearce_adapter import TEAPEARCE_SCHEMA, TeaPearceDust2Adapter

register()

npy_paths = [...]   # your TeaPearce dust2 .npy recordings (dust2 subset only)
map_for_source = {p: "de_dust2" for p in npy_paths}
map_registry = {"de_dust2": ["/path/to/de_dust2.bsp"]}  # a CS2 dust2 .bsp works -- verified above

pipeline = (
    input(adapter=TeaPearceDust2Adapter(), sources=npy_paths, schema=TEAPEARCE_SCHEMA)
    .window(128, stride=128)
    .apply("aintropy/cs2-collision",
           map_for_source=map_for_source,
           map_registry=map_registry,
           dist_path="action_state_dist.json")  # ASG-fit placeholder -- see caveat above
    .sample("top-k", n=100_000, weights={"collision_count": 1.0})
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

## Directional-residual analyzer (alternative algorithm)

`aintropy/cs2-collision-directional` is a second, independent collision-detection
algorithm, ported unmodified from the reference `directional_collision_detector.py`
script. It needs no learned per-condition baseline (no `action_state_dist.json`): for
every tick where a movement combo is held it computes the *expected* velocity from a
fitted per-combo/weapon acceleration-ramp model and flags a tick only when the shortfall
against the *actual* recorded velocity is large **and** concentrated specifically along a
nearby wall's normal -- not spread across all directions, which would just mean
"hasn't accelerated yet," "turning," or "running uphill." It checks only the lower bound
(missing expected motion); there is no upper-cap/speedhack logic.

It is a Kuration `Analyzer` (`register_analyzer`), not a `BatchTransform`: the algorithm
walks a whole continuous per-source tick sequence in order, carrying state (the velocity
when the current key-combo started, ticks since that change) that a fixed-size window
would corrupt at every boundary. See `kuration_action_state_consistency.AnomalyChecker`
for the sibling precedent this follows, including its `to_records()` flat-export
convention.

### Fitting the ramp model

The `{(combo_bits, weapon): (tau_ticks, cap_hu_s)}` table plays the same role here that
`action_state_dist.json` plays for `CollisionMechanics` above: fit it once, pooled across
many sources, and reuse it across every analysis run. A single source rarely has enough
clean presses of every combo+weapon pair to fit reliably -- in practice, pooling 190
whole TeaPearce matches still only produced a reliable fit for 1 of 8 modeled combos, so
callers should expect to lean on the fallback ramp (see below) for most combo/weapon
pairs unless the pooled source set is large.

`teapearce_adapter.load_directional_arrays(source_uri)` gets you the `(btn_mask,
velocity, hp, weapon)` shape `fit_ramp_models` wants directly from a TeaPearce `.npy`
recording (it flattens that same file's canonical `Tick` stream -- the exact one
`TeaPearceDust2Adapter` itself yields -- into plain arrays, since `Analyzer`s bypass the
Pipeline's windowed Tick machinery entirely):

```python
from kuration_cs2_collision import fit_ramp_models, load_directional_arrays, save_ramp_models

npy_paths = [...]  # your TeaPearce dust2 .npy recordings
sources_by_id = {p: load_directional_arrays(p) for p in npy_paths}
fit_sources = [(a["btn_mask"], a["velocity"], a["hp"], a["weapon"]) for a in sources_by_id.values()]

models = fit_ramp_models(fit_sources, min_ramps=5)   # combos below min_ramps clean presses are omitted
save_ramp_models(models, "ramp_models.json")
```

For any other telemetry source, build the same `(btn_mask, velocity, hp, weapon)` array
tuples yourself -- `load_directional_arrays` is TeaPearce-specific, not a general
requirement.

### Usage

Construct the analyzer through the SDK registry, the same way as every other analyzer
(see `kuration_action_state_consistency`'s own README for why: discovery + parameter
validation, not because anything here is TeaPearce-specific):

```python
import kuration
from kuration_cs2_collision import load_directional_arrays, register

register()

analyzer = kuration.get_analyzer(
    "aintropy/cs2-collision-directional",
    map_for_source={p: "de_dust2" for p in npy_paths},
    map_registry={"de_dust2": ["/path/to/de_dust2.bsp"]},
    ramp_models_path="ramp_models.json",
)

# Call once per source, with that source's COMPLETE, correctly-ordered tick arrays --
# the tracked state would be silently corrupted by a partial or reordered slice.
for path in npy_paths:
    analyzer.process_source(path, **load_directional_arrays(path))

records = analyzer.to_records()   # one row per input tick -- join straight back onto telemetry
summary = analyzer.summary()
```

Analyzers are deliberately not chainable onto the fluent `Pipeline` -- there is no
`.directional_collision_analyzer(...)` step to `.apply()` -- see `analyzers.py`'s module
docstring: "the SDK therefore discovers and validates analyzer factories but does not
schedule them as implicit pipeline stages: callers own the analyzer instance and drive
its capability-specific update methods explicitly."

### Output

`to_records()` returns exactly one row per tick ever passed to `process_source()`, in
call order:

| field | meaning |
|---|---|
| `source_id` / `tick` | which source and tick index this row is for |
| `collision` | `True` only when every gate below passed |
| `reason` | which gate the tick stopped at: `dead_or_skipped`, `not_modeled`, `fast_turn`, `residual_too_small`, `no_wall_hit`, `low_concentration`, or `flagged` |
| `combo` / `tracking_combo` | the raw held combo this tick vs. the combo still being tracked (differ during a `GRACE_TICKS`-tolerated brief dropout) |
| `residual_mag` | \|expected velocity - actual velocity\|, HU/s (`None` if never computed) |
| `concentration_ratio` | fraction of the residual aligned with the nearby wall's normal (`None` if no wall hit) |
| `dist_to_wall_hu` | raycast distance to the nearest wall hit within `proximity` (`None` if none) |

`summary()` returns `{sources_processed, total_ticks, total_collisions, per_source}`.

### Known limitations

- Only the 8 pure/diagonal WASD combos (`W`, `A`, `S`, `D`, `W+A`, `W+D`, `S+A`, `S+D`)
  are modeled -- `W+S`/`A+D` cancel to an ambiguous wish-direction and are never
  flagged (`reason="not_modeled"`), and duck/walk speed modifiers are not accounted for
  in the ramp model at all.
- No FOV-visibility check: a flagged tick's wall may be entirely outside the player's
  camera view at that moment, which can't be confirmed from video alone. Two heuristics
  (a point check and an edge-ray refinement) were tried and dropped -- both
  misclassified roughly half of manually-confirmed real collisions as off-screen, so
  neither was trustworthy enough to ship. `MIN_MISSING_SPEED` is kept at a conservative
  `40.0` HU/s rather than a looser `15.0` specifically because the extra ticks a looser
  threshold recovers skew heavily toward this unverifiable off-screen case (see the
  comment above `MIN_MISSING_SPEED` in `directional_collision_analyzer.py`).
- No collision-type classification (unlike `CollisionMechanics`'s
  `head_on`/`angled`/`*_slide` types) -- this analyzer only ever produces a boolean flag
  plus raw diagnostics.
- Validated by direct parity against the reference standalone script (bit-for-bit
  identical flagged ticks across 5 real TeaPearce recordings, 5,000 ticks total) rather
  than an independent from-scratch reimplementation -- see `tests/test_ramp_fitting.py`
  and `tests/test_directional_collision_analyzer.py` for the unit-level coverage that
  backs it going forward.
