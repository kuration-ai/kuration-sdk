# Collision detection

Detects and classifies CS2 wall collisions -- `head_on`, `angled`, `W_slide`, `WA_slide`,
`slide_end`, `slide_constrained` -- from per-tick position/view-angle telemetry, raycast
against the match's map geometry. Ported from the reference two-stage pipeline
(`action_state_analyzer.py`'s learned-baseline anomaly scan + `collision_classifier.py`'s
raycast classification, as fused by `collision_pipeline_fused.py`) into a built-in
gameplay `BatchTransform` + compositional `operator` pair under
`kuration.gameplay.collision`.

A second, independent algorithm -- an acceleration-ramp-residual tick-stream `Analyzer`
needing no learned baseline -- ships alongside it; see
[Directional-residual analyzer](#directional-residual-analyzer-alternative-algorithm).

## Install

```bash
pip install "kuration-sdk[collision]"
```

The extra pulls `trimesh` (geometry + ray BVH) and `rtree`. Both stay lazy: importing
`kuration.gameplay` never loads them; the first raycast does, loudly if they are missing.

## Format-agnostic by design

The operator only ever reads canonical schema fields -- `position` (vector,
`semantics="position"`), `angle` (vector, `dims=("yaw","pitch")`,
`semantics="angular"`), `hp`, `btn_mask`, `flags`, `velocity` -- never a dataset's raw
field names. Any `InputAdapter` that maps its own raw telemetry onto those same canonical
names can drive the same registered operator unmodified. Two adapters ship with it:

- `aintropy/collision-asg` -- reads the ASG `data.json` match-replay format
  (tick-by-tick, nested `header` + `ticks[].players[]`) and yields one Kuration "source"
  per `(match, player)` pair; `source_uri` is `"<data.json path>::<id32>"`. Velocity is
  synthesized from position deltas at the format's own known sample spacing.
- `aintropy/collision-teapearce` -- reads the public
  `TeaPearce/CounterStrike_Deathmatch` dust2 `.npy` metadata files (real
  position/angle/velocity/health telemetry scraped via memory-read + GSI; one source per
  recording). Velocity is the engine's own recorded `m_vecVelocity` -- an earlier
  position-delta derivation tuned for ASG's tick spacing overstated it ~3-4x here, which
  is exactly why velocity is an adapter responsibility, not a transform one. Frames with
  `obs_mode != 0` (spectating: the telemetry follows the observed player while the key
  input is still the recorder's) are marked with a skip flag and excluded by both
  algorithms.

Verification of the TeaPearce coordinate system against real dust2 map geometry
(bounding-box containment, raycast hit-rate/distance distributions, NAV-mesh floor-height
checks at real respawn events) is documented in `teapearce_adapter.py`'s module
docstring.

## What gets registered

Everything registers through the bundled `kuration.gameplay` plugin (no separate
distribution):

- `aintropy/collision-asg`, `aintropy/collision-teapearce` -- `adapter`s.
- `aintropy/collision` -- a `transform` (the per-window compute) and an `operator` (its
  pipeline wrapper, reachable as `.collision(...)`), mirroring
  [pixel shift](pixel-shift.md).
- `aintropy/collision-directional` -- an `analyzer` (tick-stream capable) and an
  `operator` (`.collision_directional(...)`, the pipeline-preserving companion that runs
  it and attaches the verdict to pipeline metadata), mirroring
  [action-state consistency](action-state-consistency.md).

All three are discoverable and inspectable from the CLI; see
[Command-line interface](#command-line-interface) at the end of this document.

## Usage

```python
from kuration import input
from kuration.gameplay.collision import AsgCollisionAdapter

data_json_paths = [...]   # your ASG data.json files
map_for_source = AsgCollisionAdapter.discover_map_for_source(data_json_paths)
map_registry = {"de_dust2": ["/path/to/de_dust2.bsp"]}   # add more maps as needed

pipeline = (
    input(adapter=AsgCollisionAdapter(), sources=list(map_for_source),
          schema=AsgCollisionAdapter.schema)
    .window(256, stride=128)
    .collision(map_for_source=map_for_source,
               map_registry=map_registry,
               dist_path="action_state_dist.json")
    .sample("equal", n=200)
)
```

`.collision(...)` is the registered operator's method alias;
`.apply("aintropy/collision", ...)` is the equivalent explicit form.

### Output

One value per output, per window (segment) -- not per tick:

| output | meaning |
|---|---|
| `collision_count` | total confirmed collision ticks in this window |
| `head_on_count` / `angled_count` / `w_slide_count` / `wa_slide_count` / `slide_end_count` / `slide_constrained_count` | per-type counts |
| `max_anomaly_score` | highest z-score seen across every scored tick in the window (continuous informativeness signal, independent of whether a collision was confirmed) |
| `min_dist_to_wall_hu` | closest confirmed-collision proximity-ray distance in the window, or `-1.0` if none |

With overlapping windows (`stride < window`) the transform declares
`ownership="rightmost-start"`: a window owns its steps up to where the next window
starts, so a collision tick inside an overlap is counted exactly once -- the same
convention pixel shift applies to press instances.

### The learned baseline (`dist_path`)

This capability has no learn phase: `dist_path` names an externally fitted
`action_state_dist.json` (see [customization](customization.md) on keeping learn/fit
steps outside the SDK). A baseline is calibrated against the player population it was
fit on -- reusing an ASG-fit baseline against TeaPearce data proves pipeline mechanics
but is not a calibrated anomaly detector for that population.

## Kinematics extraction

`kinematics.extract_kinematics(pm2, pm1, p0)` derives one tick's local-frame
forward/right/up velocity, turn rate, and constant-velocity pose-error features from itself
and its two predecessors. It is the one genuinely reusable seam inside the fused
scan-then-classify per-tick loop: no dependency on the learned baseline, map geometry, or
raycast classification that consume its output downstream, so anything else needing
frame-relative velocity/turn features from raw position+angle telemetry can call it directly.
The anomaly-gating and raycast-classification stages that follow remain fused in
`collision_operator.py` -- gating is a threshold immediately consumed by classification a few
lines later on the same tick, and splitting either into its own component would only invent an
artificial `Segment`-shaped handoff with no independent use.

`aintropy/collision-directional` (below) computes a conceptually similar but *different*
kinematics quantity -- an *expected* velocity from a fitted acceleration-ramp model, not a
frame-relative velocity from raw position deltas -- so it does not share `extract_kinematics`.

## Map geometry caching

Parsing raw `.bsp`/`.glb`/`.nav` files (Source-engine lump tables, displacement
surfaces, static props, texture-based non-solid-face filtering) is the expensive part,
not the raycast BVH. Parsed `(vertices, faces)` arrays are cached to an on-disk `.npz`
(`disk_cache_dir`, default `.kuration_collision_cache`), plus an in-memory per-run dict.
Cache entries record both the geometry parser version (`GEOMETRY_CACHE_VERSION`) and a
fingerprint of the raw geometry paths, sizes, and modification times. Entries are
re-parsed when either changes, so parser upgrades and map-file replacements cannot
silently keep serving stale geometry.

## Directional-residual analyzer (alternative algorithm)

`aintropy/collision-directional` is a second, independent collision-detection algorithm
that needs no learned baseline: for every tick where a movement combo is held it
computes the *expected* velocity from a fitted per-combo/weapon acceleration-ramp model
and flags a tick only when the shortfall against the *actual* recorded velocity is large
**and** concentrated along a nearby wall's normal -- not spread across all directions
(which would just mean "hasn't accelerated yet," "turning," or "running uphill"). It
checks only the lower bound (missing expected motion); there is no upper-cap/speedhack
logic.

It is an `Analyzer`, not a `BatchTransform`: the walk carries tick-to-tick state a fixed
window would corrupt at every boundary. Two ways to drive it:

### Pipeline-first

The analyzer implements the tick-stream contract, so the companion operator runs it in
one bounded pass over the pipeline's segmented tick stream -- windows arrive per source,
in order, ownership-deduplicated, and are stitched back into each source's continuous
tick sequence before the walk runs (non-contiguous coverage raises rather than walking
corrupted state):

```python
from kuration import input
from kuration.gameplay.collision import TeaPearceDust2Adapter
from kuration.gameplay.collision.teapearce_adapter import TEAPEARCE_SCHEMA

npy_paths = [...]  # TeaPearce dust2 .npy recordings

pipeline = (
    input(adapter=TeaPearceDust2Adapter(), sources=npy_paths, schema=TEAPEARCE_SCHEMA)
    .window(128, stride=128)
    .collision_directional(
        map_registry={"de_dust2": ["/path/to/de_dust2.bsp"]},
        default_map="de_dust2",          # or map_for_source={...} per source
        ramp_models_path="ramp_models.json",
    )
)
summary = pipeline.metadata["gameplay"]["directional_collision"]["result"]["summary"]
# summary["flagged_ticks"] maps source_id -> flagged tick indices (source coordinates)
```

The attachment retains the summary (including `flagged_ticks`), the component
descriptor, and the input-basis fingerprint, exactly like every `enrich()`-backed
operator. `pipeline.analyze("aintropy/collision-directional", {...})` returns the same
result without deriving a new pipeline.

### Direct (expert)

```python
import kuration
from kuration.gameplay.collision import load_directional_arrays

analyzer = kuration.get_analyzer(
    "aintropy/collision-directional",
    map_for_source={p: "de_dust2" for p in npy_paths},
    map_registry={"de_dust2": ["/path/to/de_dust2.bsp"]},
    ramp_models_path="ramp_models.json",
)
# Call once per source, with that source's COMPLETE, correctly-ordered tick arrays.
for path in npy_paths:
    analyzer.process_source(path, **load_directional_arrays(path))

records = analyzer.to_records()   # one row per input tick -- join back onto telemetry
summary = analyzer.summary()
```

`to_records()` returns exactly one row per tick, in call order:

| field | meaning |
|---|---|
| `source_id` / `tick` | which source and tick index this row is for (true source coordinates) |
| `collision` | `True` only when every gate below passed |
| `reason` | which gate the tick stopped at: `dead_or_skipped`, `not_modeled`, `fast_turn`, `residual_too_small`, `no_wall_hit`, `low_concentration`, or `flagged` |
| `combo` / `tracking_combo` | the raw held combo this tick vs. the combo still being tracked (differ during a `GRACE_TICKS`-tolerated brief dropout) |
| `residual_mag` | \|expected velocity - actual velocity\|, HU/s (`None` if never computed) |
| `concentration_ratio` | fraction of the residual aligned with the nearby wall's normal (`None` if no wall hit) |
| `dist_to_wall_hu` | raycast distance to the nearest wall hit within `proximity` (`None` if none) |

### Fitting the ramp model

The `{(combo_bits, weapon): (tau_ticks, cap_hu_s)}` table plays the role
`action_state_dist.json` plays for the windowed operator: fit once, pooled across many
sources, reuse everywhere. Unlike that baseline, the fit is small and pure-array enough
to ship in the SDK:

```python
from kuration.gameplay.collision import (
    fit_ramp_models, load_directional_arrays, save_ramp_models,
)

npy_paths = [...]
arrays = [load_directional_arrays(p) for p in npy_paths]
fit_sources = [(a["btn_mask"], a["velocity"], a["hp"], a["weapon"]) for a in arrays]

models = fit_ramp_models(fit_sources, min_ramps=5)
save_ramp_models(models, "ramp_models.json")
```

Any `(combo, weapon)` pair absent from the fitted table uses the analyzer's
`fallback_ramp` -- an explicit constructor parameter defaulting to
`(tau=3.0 ticks, cap=215.0 HU/s)` (the rifle-class run speed). The fallback is the
common path in practice: pooling 190 whole TeaPearce games produced a reliable fit for
only 1 of 8 modeled combos, so expect to lean on it unless the pooled source set is
large.

## Known limitations

- Only the 8 pure/diagonal WASD combos are modeled by the directional analyzer --
  `W+S`/`A+D` cancel to an ambiguous wish-direction and are never flagged
  (`reason="not_modeled"`), and duck/walk speed modifiers are not in the ramp model.
- No FOV-visibility check: a flagged tick's wall may be outside the player's camera view
  at that moment. Two heuristics were tried and dropped (documented above
  `MIN_MISSING_SPEED` in the source) -- both misclassified roughly half of
  manually-confirmed real collisions as off-screen, so the threshold is kept
  conservative instead.
- The directional analyzer produces a boolean flag plus diagnostics -- no collision-type
  classification (that is the windowed operator's job).
- The windowed operator's `ray_cache` is scoped per window call rather than per whole
  source (performance-only, not correctness).

## Command-line interface

Discover the registered components without reading a dataset:

```bash
kur-list operators
kur-describe operator aintropy/collision --json
kur-describe analyzer aintropy/collision-directional --json
```
