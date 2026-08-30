# collision-aware

Fit an unobstructed-movement model from real telemetry, flag ticks where velocity falls short toward a nearby wall, and upsample them for training.

[![aintropy/collision-directional operator](https://img.shields.io/badge/aintropy%2Fcollision--directional-operator-0D9488?style=for-the-badge)](./collision-aware.ipynb)
[![telemetry + map geometry, no pixel labels](https://img.shields.io/badge/telemetry_%2B_map_geometry-no_pixel_labels-0D9488?style=for-the-badge)](../../../docs/collision.md)
[![catches blind-spot and off-screen contacts](https://img.shields.io/badge/catches-blind--spot_and_off--screen_contacts-0D9488?style=for-the-badge)](../../../docs/collision.md)

[![per-player unobstructed velocity model](https://img.shields.io/badge/per--player-unobstructed_velocity_model-0D9488)](../../../docs/collision.md)
[![output: one row per tick](https://img.shields.io/badge/output-one_row_per_tick-0D9488)](../../../docs/collision.md)
[![directions: front, back, left, right, corner](https://img.shields.io/badge/directions-front_back_left_right_corner-0D9488)](../../../docs/collision.md)

[![Colab](https://img.shields.io/badge/Colab-open-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aintropy-solutions/kuration-sdk/blob/main/kuration-sdk-wrapper/notebooks/collision-aware/collision-aware.ipynb) [![Binder](https://img.shields.io/badge/Binder-launch-579ACA?logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/aintropy-solutions/kuration-sdk/HEAD?labpath=kuration-sdk-wrapper%2Fnotebooks%2Fcollision-aware%2Fcollision-aware.ipynb)

> **Launcher status.** The URLs above point at `aintropy-solutions/kuration-sdk`. Colab and Binder can only clone public repositories, so the badges activate once this repo goes public. In the meantime, clone and run the notebook locally with `pip install kuration-sdk[gameplay]`.

<details><summary><b>Contents</b></summary>

- [Setup](#setup)
- [The idea in one paragraph](#the-idea-in-one-paragraph)
- [Fit an unobstructed-movement model from synthetic telemetry](#fit-an-unobstructed-movement-model-from-synthetic-telemetry)
- [Compute the residual and flag collisions](#compute-the-residual-and-flag-collisions)
- [From ticks to records](#from-ticks-to-records)
- [Extending to real footage](#extending-to-real-footage)
- [Try it locally](#try-it-locally)
- [Related](#related)

</details>

## Setup

```python
%pip install --quiet 'kuration-sdk[gameplay]' matplotlib
```

See cell 2.

## The idea in one paragraph

Good players do not run into walls, so wall-contact events are systematically underrepresented in real gameplay. A world model trained on that footage learns walls are optional. The collision operator uses telemetry (velocity, button state, health, weapon) and map geometry to flag every tick where a player's actual velocity falls short of an unobstructed-movement model in the direction of a nearby wall. Those ticks then get upsampled during training. The check runs on telemetry and map geometry, not pixels, so collisions in blind spots (a wall behind the player, a crate off-screen) are caught the same way head-on collisions are.

## Fit an unobstructed-movement model from synthetic telemetry

In production, the ramp model is fit from real gameplay telemetry. The notebook synthesizes a small tick stream with a player pressing forward, then hitting a wall from tick 200 to tick 260. The unobstructed model learns what forward-key-plus-no-obstacle produces on its own: a ramp of 8 ticks toward a target speed of 250 u/s (both notebook demo constants).

```python
target_speed = 250.0
ramp_ticks = 8
ramp_curve = np.clip(np.arange(N_TICKS) / ramp_ticks, 0, 1) * target_speed
velocity = ramp_curve + noise
velocity[wall_mask] = 40 + np.random.normal(0, 3, size=wall_mask.sum())
```

See cell 5.

## Compute the residual and flag collisions

The unobstructed-movement model says: after 8 ticks of holding W, velocity should be near `target_speed`. The residual is `predicted - measured`. When the residual is large and there is a wall in front, the tick is flagged.

```python
COLLISION_THRESHOLD = 100.0
WALL_PROXIMITY = 50.0
collision = (residual > COLLISION_THRESHOLD) & (dist_to_wall < WALL_PROXIMITY)
```

See cell 7.

## From ticks to records

The real analyzer returns one row per tick with `collision`, `residual`, `dist_to_wall`, and `direction`. Downstream `.sample()`, `.bin()`, or `.filter()` compose on top of those records the same way they compose on top of any other measured feature.

```python
records = [
    {
        "tick": int(t),
        "collision": bool(collision[t]),
        "residual": float(residual[t]),
        "dist_to_wall": float(dist_to_wall[t]) if np.isfinite(dist_to_wall[t]) else None,
        "direction": "front",
    }
    for t in range(N_TICKS)
]
```

See cell 9.

## Extending to real footage

The full pipeline against the DIAMOND CS:GO dataset lives in the SDK. Supply the `.npy` telemetry arrays and a `.bsp` map file, register the operator, and let it walk every tick:

```python
import kuration
from kuration.gameplay.collision import fit_ramp_models, load_directional_arrays, save_ramp_models, register

register()
sources = [load_directional_arrays(p) for p in npy_paths]
models = fit_ramp_models([(s["btn_mask"], s["velocity"], s["hp"], s["weapon"]) for s in sources])
save_ramp_models(models, "ramp_models.json")

analyzer = kuration.get_analyzer(
    "aintropy/collision-directional",
    map_for_source={p: "de_dust2" for p in npy_paths},
    map_registry={"de_dust2": ["/path/to/de_dust2.bsp"]},
    ramp_models_path="ramp_models.json",
)

for path in npy_paths:
    analyzer.process_source(path, **load_directional_arrays(path))

records = analyzer.to_records()
```

The public `de_dust2` map file is on the Steam Community. The DIAMOND CS:GO telemetry is in the [`TeaPearce/CounterStrike_Deathmatch`](https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch) dataset.

See cell 10.

## Try it locally

```bash
git clone https://github.com/aintropy-solutions/kuration-sdk.git
cd kuration-sdk/kuration-sdk-wrapper/notebooks/collision-aware
uv pip install 'kuration-sdk[gameplay]' matplotlib jupyter
uv run jupyter lab collision-aware.ipynb
```

## Related

- Paired long-form post: [aintropy.ai/blog/collision-aware](https://aintropy.ai/blog/collision-aware)
- Back to the wrapper index: [`../../README.md`](../../README.md)
