"""InputAdapter for the TeaPearce/CounterStrike_Deathmatch dust2 `.npy` metadata format
(https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch).

Each `.npy` file (e.g. `dm_july2021_expert_1.npy`, from the `dataset_dm_expert_dust2_npy_*`
archives) is a plain object array of shape `(n_frames, 6)`. Column 1 is a `curr_vars` dict
per frame containing real, verified position/angle/velocity/health telemetry
(`localpos1/2/3`, `viewangle_xy/vert`, `vel_1/2/3`, `obs_health`, `tp_wasd`, `tp_lclick`,
...) scraped via memory-read + Game State Integration -- see `docs/collision.md` for the
verification performed against real dust2 map geometry (bounding-box containment, raw
raycast hit-rate/distance distribution, and NAV-mesh floor-height checks at both real
respawn events found in the data, all consistent with a correctly-aligned coordinate
system).

One kuration source = one `.npy` file (one continuous recording), mirroring
`asg_adapter.py`'s one-source-per-continuous-tick-sequence convention.

Known approximations, all discussed and accepted before writing this (see
`docs/collision.md`'s TeaPearce section for the full reasoning):
- `onground` is always reported True. No ground-truth onground signal exists in
  `curr_vars`; the collision operator's learned-baseline lookup already has a
  ground-fallback for exactly this situation, and ducking/onground never affect the
  actual collision-detection thresholds (proximity/min_approach/etc. are fixed
  regardless) -- only which learned-baseline bucket a tick's z-score is measured against.
- `ducking` is always reported False. No crouch instances were found across three
  checked real files (3,000 frames, all alive frames at exactly standing eye-height).
- `obs_mode != 0` (spectator/observer) frames are reported with `FL_NOCLIP` set, so both
  collision algorithms' `is_skip_flags()` gate excludes them. While observing, the
  recorded position/velocity telemetry follows the OBSERVED player and `obs_health` is
  that player's health (often full), while `tp_wasd` is still the recorder's own keys --
  exactly the telemetry/input mismatch both algorithms would otherwise score as phantom
  collisions. An earlier revision mapped `obs_mode != 0` to merely "not FL_ONGROUND",
  which does not skip anything -- the ground-fallback baseline lookup papered over it and
  the directional analyzer processed observer frames outright. There is no direct
  FL_FROZEN equivalent recorded beyond this.
- `dist_path` (the learned anomaly baseline) is NOT produced by this adapter or by any
  learn phase in this plugin -- the learn phase intentionally stays outside the SDK (see
  the reference `action_state_analyzer.py learn` command). Reusing the ASG-fit
  `action_state_dist.json` against TeaPearce data is a documented placeholder for
  proving pipeline mechanics, not a calibrated anomaly detector -- a real TeaPearce
  baseline would need its own learn run against TeaPearce-shaped tick data.

`weapon` (from `curr_vars`' `gsi_weap_active.name`, via Game State Integration) is
included as a `categorical` field even though `CollisionMechanics` never reads it --
`DirectionalCollisionAnalyzer`'s acceleration-ramp model is weapon-dependent (caps
differ per gun) and needs it. Like `tp_wasd`/`obs_mode` elsewhere in this file, it's read
defensively (`.get()`, `None` when absent) rather than required, since not every frame
carries an active-weapon payload (e.g. death/spectator frames).

`velocity` is reported directly from `curr_vars`' own `vel_1/2/3` (read from the game
engine's `m_vecVelocity`, independent of position) -- NOT derived from position deltas.
This matters: an earlier position-delta-based derivation (mirroring the reference
collision_classifier.py's hardcoded `*64` HU/s conversion, tuned for a different
dataset's tick spacing) overstated true velocity by roughly 3-4x on real TeaPearce data,
confirmed against this same `vel_mag` field as ground truth -- producing false-positive
wall-collision detections on frames that were visually confirmed to show no collision
at all (the player barely moving). Using the engine's own recorded velocity sidesteps
that whole sample-rate-conversion problem entirely.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from kuration import FieldSpec, SchemaSpec, Tick

from .physics import BIT_A, BIT_D, BIT_S, BIT_W, FL_NOCLIP, FL_ONGROUND

TEAPEARCE_SAMPLE_HZ = 16.0

TEAPEARCE_SCHEMA = SchemaSpec(
    name="teapearce_dust2_collision",
    entity_axis="player",
    # TeaPearce recordings are natively 16Hz -- matches physics.DT's assumed sample rate.
    sample_hz=TEAPEARCE_SAMPLE_HZ,
    fields={
        "position": FieldSpec(name="position", kind="vector", dims=("x", "y", "z"),
                              semantics="position"),
        "angle": FieldSpec(name="angle", kind="vector", dims=("yaw", "pitch"),
                           semantics="angular"),
        "hp": FieldSpec(name="hp", kind="scalar"),
        "btn_mask": FieldSpec(name="btn_mask", kind="scalar"),
        "flags": FieldSpec(name="flags", kind="scalar"),
        "velocity": FieldSpec(name="velocity", kind="vector", dims=("x", "y", "z")),
        "weapon": FieldSpec(name="weapon", kind="categorical"),
    },
)

_KEY_BITS = {"w": BIT_W, "s": BIT_S, "a": BIT_A, "d": BIT_D}


def _btn_mask(tp_wasd) -> float:
    mask = 0
    for key in (tp_wasd or []):
        mask |= _KEY_BITS.get(key, 0)
    return float(mask)


def _weapon_name(cv) -> str | None:
    active = cv.get("gsi_weap_active")
    if not active:
        return None
    return active.get("name")


class TeaPearceDust2Adapter:
    """`InputAdapter` over TeaPearce dust2 `.npy` files. `source_uri` is a filesystem path
    to one `.npy` file; each file is one continuous recording (one kuration source)."""

    schema = TEAPEARCE_SCHEMA

    def ticks(self, source_uri: str) -> Iterable[Tick]:
        data = np.load(source_uri, allow_pickle=True)
        n = data.shape[0]
        for i in range(n):
            cv = data[i, 1]
            position = np.array(
                [cv["localpos1"], cv["localpos2"], cv["localpos3"]], dtype=np.float64)
            angle = np.array([cv["viewangle_xy"], cv["viewangle_vert"]], dtype=np.float64)
            hp = float(cv.get("obs_health", 0.0))
            btn_mask = _btn_mask(cv.get("tp_wasd"))
            # Observer frames carry the OBSERVED player's telemetry against the recorder's
            # own key input -- mark them FL_NOCLIP so is_skip_flags() excludes them (see
            # the module docstring).
            flags = float(FL_ONGROUND) if cv.get("obs_mode", 0) == 0 else float(FL_NOCLIP)
            velocity = np.array(
                [cv.get("vel_1", 0.0), cv.get("vel_2", 0.0), cv.get("vel_3", 0.0)],
                dtype=np.float64)
            weapon = _weapon_name(cv)
            yield Tick(
                time=i / TEAPEARCE_SAMPLE_HZ,
                entities={"player": {
                    "position": position, "angle": angle, "hp": hp,
                    "btn_mask": btn_mask, "flags": flags, "velocity": velocity,
                    "weapon": weapon,
                }},
            )

    def check_source(self, source_uri: str) -> None:
        # allow_pickle=True object arrays can't be validly memory-mapped (the payload is
        # pickled, not laid out in a fixed-size mmap-able format), so this reads the whole
        # file -- an accepted cost for a pre-flight validation of these ~100-130MB files.
        data = np.load(source_uri, allow_pickle=True)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(
                f"{source_uri!r}: expected a (n_frames, >=2) array, got shape {data.shape}")
        missing = {"localpos1", "localpos2", "localpos3", "viewangle_xy", "viewangle_vert",
                  "obs_health"} - set(data[0, 1].keys())
        if missing:
            raise ValueError(f"{source_uri!r}: curr_vars missing expected keys {missing}")


def load_directional_arrays(source_uri: str) -> dict:
    """Flatten one recording's canonical `Tick` stream into the plain per-source numpy
    arrays `DirectionalCollisionAnalyzer.process_source()` and
    `ramp_fitting.fit_ramp_models()` need: `position`, `yaw`, `btn_mask`, `velocity`,
    `hp`, `flags`, `weapon`.

    This is the expert direct path for driving the analyzer (or a ramp fit) by hand with
    whole, correctly-ordered per-source arrays; the pipeline path is
    ``pipeline.enrich("aintropy/collision-directional", ...)`` (or the registered
    ``.collision_directional(...)`` operator), which streams the same fields through the
    analyzer's ``add_batch`` hook. Both are built on the exact same
    `TeaPearceDust2Adapter.ticks()` generator `CollisionMechanics` consumes via the
    windowed path -- so there is exactly one place that parses raw `curr_vars` fields,
    not two drifting copies of the same field-extraction logic.
    """
    ticks = list(TeaPearceDust2Adapter().ticks(source_uri))
    n = len(ticks)
    position = np.zeros((n, 3))
    yaw = np.zeros(n)
    btn_mask = np.zeros(n, dtype=np.int64)
    velocity = np.zeros((n, 3))
    hp = np.zeros(n)
    flags = np.zeros(n)
    weapon = np.empty(n, dtype=object)
    for i, tk in enumerate(ticks):
        player = tk.entities["player"]
        position[i] = player["position"]
        yaw[i] = player["angle"][0]
        btn_mask[i] = int(player["btn_mask"])
        velocity[i] = player["velocity"]
        hp[i] = player["hp"]
        flags[i] = player["flags"]
        weapon[i] = player["weapon"]
    return dict(position=position, yaw=yaw, btn_mask=btn_mask, velocity=velocity,
                hp=hp, flags=flags, weapon=weapon)
