"""Directional-residual collision analyzer: a user-owned, stateful Analyzer that flags
per-tick candidate wall collisions from position/velocity/angle/button telemetry plus map
geometry -- WITHOUT needing a learned per-condition baseline the way the sibling
`CollisionMechanics` BatchTransform does.

For every tick where a movement combo (W/S/A/D or a diagonal strafe combo) is held, this
computes the EXPECTED velocity vector (wish direction, from held keys + view yaw, at a
magnitude given by a fitted acceleration ramp for that combo+weapon -- see
ramp_fitting.py) and compares it to the ACTUAL recorded velocity vector. A tick is flagged
only when the shortfall (expected - actual) is large AND concentrated specifically along a
nearby wall's normal direction -- not spread across all directions, which would just mean
"hasn't accelerated yet," "turning," or "running uphill," none of which involve a wall.
This deliberately only checks the LOWER bound (missing expected motion) -- no upper-cap/
speedhack logic, since that's a different concern from collision detection.

Why an Analyzer, not a BatchTransform: `CollisionMechanics` operates on fixed-size windows
and returns one aggregate scalar per window. This algorithm needs the opposite -- it walks
a whole continuous per-source tick sequence IN ORDER, carrying state (the velocity when the
current key-combo started, ticks since that change, whether the tracked combo has briefly
dropped out) that can span far more ticks than any fixed window or lookbehind would
preserve without corrupting it at every window boundary. `Analyzer` is the SDK's shape for
exactly this: stateful and cross-segment (see kuration.gameplay.action_state_consistency's
AnomalyChecker for the sibling precedent this follows, including its `to_records()`
flat-export convention).

Two ways to drive it:

* **Pipeline-first**: the analyzer implements the tick-stream contract
  (`add_batch`), so ``pipeline.enrich("aintropy/collision-directional", {...})`` -- or the
  registered ``.collision_directional(...)`` operator wrapping it -- streams the declared
  canonical fields through it in one bounded pass. Windows arrive per source, in order,
  with ``ownership="rightmost-start"`` dedup, and each owned prefix is processed
  immediately while the walk state carries across windows. Overlapping or gapped windows
  that cannot reconstruct a continuous sequence raise instead of walking corrupted state.
* **Direct (expert)**: construct it (via ``kuration.get_analyzer`` or the class) and call
  `process_source()` once per source with that source's complete, correctly-ordered
  arrays -- e.g. from `teapearce_adapter.load_directional_arrays()`.
"""
from __future__ import annotations

import math

import numpy as np

from .collision_operator import _drop_entity_axis_row
from .mesh_cache import cast_proximity_rays, load_mesh_cached
from .physics import BIT_A, BIT_D, BIT_S, BIT_W, MAX_DYAW_LEARN, is_skip_flags, wrap_angle
from .ramp_fitting import ALL_MOVE_BITS, load_ramp_models

COMPONENT_NAME = "aintropy/collision-directional"

# The canonical fields the tick-stream driver feeds add_batch(), in declaration order.
# Roles are the registration tokens; add_batch resolves them through ctx.input_token()
# so a per-use ``inputs=`` remapping keeps working.
STREAM_INPUTS = ("field:position", "field:angle", "field:btn_mask", "field:velocity",
                 "field:hp", "field:flags", "field:weapon")

# Used for every (combo, weapon) pair absent from the fitted table: tau=3 ticks to reach
# ~63% of cap, cap=215 HU/s (the rifle-class run speed -- AK-47's exact cap, verified
# against real recorded velocity). Most pairs DO hit the fallback in practice: pooling
# 190 whole TeaPearce games produced a reliable fit for only 1 of 8 modeled combos, so
# this is a load-bearing default, not an edge case -- which is why it is an explicit,
# documented constructor parameter rather than (as an earlier revision had it) silently
# whatever model happened to be first in the fitted table's insertion order.
DEFAULT_FALLBACK_RAMP = (3.0, 215.0)

# How many consecutive ticks of an unmodeled combo (typically a brief key release/change
# right as the player reacts to hitting something) we tolerate before dropping collision
# tracking entirely. Without this, the exact moment of a real collision -- when a player
# often instinctively lets go of a key -- was the one moment the detector went blind, since
# it would immediately reset ticks_since_change/v0 using the already-collided velocity.
GRACE_TICKS = 3

# Minimum |expected_vel - actual_vel| (HU/s) for a tick to even be considered a candidate
# collision. Lowering this to 15 recovers more manually-confirmed real collisions, but the
# extra ticks it adds skew heavily toward collisions whose surface is outside the player's
# camera FOV -- i.e. exactly the ones nobody can verify from video. Kept at 40 until a
# reliable FOV-visibility check exists (two point/edge-ray approximations tried so far
# misclassified roughly half of confirmed true positives as off-screen, so neither was
# trustworthy enough to act on).
MIN_MISSING_SPEED = 40.0

# (forward, right) unit-ish components for each modeled combo -- the only combos with an
# unambiguous single wish-direction (W+S or A+D together cancel out; not modeled).
COMBO_FORWARD_RIGHT = {
    BIT_W: (1, 0), BIT_S: (-1, 0), BIT_A: (0, -1), BIT_D: (0, 1),
    BIT_W | BIT_A: (1, -1), BIT_W | BIT_D: (1, 1),
    BIT_S | BIT_A: (-1, -1), BIT_S | BIT_D: (-1, 1),
}


def _wish_direction(combo_bits: int, yaw_deg: float) -> np.ndarray:
    """World-space unit vector for the wish direction of a held combo, given view yaw.
    Inverts physics.to_local()'s world->local rotation (that matrix is its own inverse)."""
    forward_component, right_component = COMBO_FORWARD_RIGHT[combo_bits]
    mag = math.hypot(forward_component, right_component)
    forward = forward_component / mag
    right = right_component / mag
    yaw = math.radians(yaw_deg)
    dx = forward * math.cos(yaw) + right * math.sin(yaw)
    dy = forward * math.sin(yaw) - right * math.cos(yaw)
    return np.array([dx, dy])


DIRECTIONAL_COLLISION_PARAMETERS: dict = {
    "type": "object",
    "properties": {
        "map_for_source": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "source_id -> map_name (sources absent here fall back to "
                           "default_map)",
        },
        "default_map": {
            "type": "string",
            "description": "map_name for every source not listed in map_for_source -- the "
                           "single-map shorthand (e.g. an all-dust2 pipeline)",
        },
        "map_registry": {
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "string"}},
            "description": "map_name -> geometry file path(s) (.bsp / .glb / .nav)",
        },
        "ramp_models_path": {
            "type": "string",
            "description": "path to a fitted ramp-models JSON (see ramp_fitting.fit_ramp_models "
                           "+ save_ramp_models) -- the acceleration-model analog of the sibling "
                           "transform's dist_path",
        },
        "fallback_ramp": {
            "type": "array",
            "items": {"type": "number", "exclusiveMinimum": 0},
            "minItems": 2,
            "maxItems": 2,
            "description": "(tau_ticks, cap_hu_s) used for every combo/weapon pair absent "
                           "from the fitted table",
        },
        "disk_cache_dir": {"type": "string"},
        "proximity": {"type": "number", "minimum": 0},
        "min_missing_speed": {"type": "number", "minimum": 0},
        "concentration_ratio": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["map_registry", "ramp_models_path"],
    "additionalProperties": False,
}


class DirectionalCollisionAnalyzer:
    """Stateful accumulator (same pattern as action_state_consistency's AnomalyChecker):
    construct once per analysis run with map + ramp-model parameters, feed it every
    relevant source -- via the pipeline stream driver's `add_batch()` or by calling
    `process_source()` directly with COMPLETE, correctly-ordered canonical telemetry
    arrays -- then read `to_records()`/`summary()` once at the end.
    """

    def __init__(
        self, *, map_registry, ramp_models_path, map_for_source=None, default_map=None,
        fallback_ramp=DEFAULT_FALLBACK_RAMP,
        disk_cache_dir=".kuration_collision_cache",
        proximity=40.0, min_missing_speed=MIN_MISSING_SPEED, concentration_ratio=0.6,
    ):
        if not map_for_source and default_map is None:
            raise ValueError(
                "directional collision analysis needs map_for_source= (per-source map "
                "names) or default_map= (one map for every source) -- there is no "
                "geometry to raycast against otherwise")
        self.map_for_source = dict(map_for_source or {})
        self.default_map = default_map
        self.map_registry = {k: list(v) for k, v in map_registry.items()}
        self.ramp_models = load_ramp_models(ramp_models_path)
        tau, cap = fallback_ramp
        if not (tau > 0 and cap >= 0):
            raise ValueError(f"fallback_ramp must be (tau_ticks > 0, cap_hu_s >= 0), "
                             f"got {fallback_ramp!r}")
        self.fallback_ramp = (float(tau), float(cap))
        self.disk_cache_dir = disk_cache_dir
        self._thresholds = dict(
            proximity=proximity, min_missing_speed=min_missing_speed,
            concentration_ratio=concentration_ratio,
        )
        self._mesh_cache: dict = {}
        self._records: list = []
        # Stateful per-source stream walks. add_batch() processes each owned prefix
        # immediately and retains only this small cross-tick state, never the input arrays.
        self._stream_states: dict = {}
        self._stream_next_start: dict[str, int] = {}
        self._active_stream_source: str | None = None

    @staticmethod
    def _new_walk_state() -> dict:
        return {
            "active_combo": None,
            "ticks_since_change": 0,
            "ticks_since_modeled": 0,
            "v0": np.zeros(2),
            "prev_yaw": None,
            "ray_cache": {},
        }

    def _mesh_for(self, source_id: str):
        map_name = self.map_for_source.get(source_id, self.default_map)
        if map_name is None:
            raise ValueError(
                f"source {source_id!r} has no entry in map_for_source and no default_map "
                "was given")
        if map_name not in self.map_registry:
            raise ValueError(
                f"map {map_name!r} (for source {source_id!r}) has no geometry paths in "
                f"map_registry; known maps: {sorted(self.map_registry)}")
        mesh = self._mesh_cache.get(map_name)
        if mesh is None:
            mesh = load_mesh_cached(map_name, self.map_registry[map_name],
                                    disk_cache_dir=self.disk_cache_dir)
            self._mesh_cache[map_name] = mesh
        return mesh

    def process_source(self, source_id: str, *, position, yaw, btn_mask, velocity, hp,
                       flags, weapon, tick_offset: int = 0):
        """Walk one source's full continuous tick sequence in order, appending exactly one
        record per input tick (a `collision` boolean plus diagnostics) to the accumulator.
        Must be called with the COMPLETE, correctly-ordered tick arrays for this source --
        the tracked state (v0/active_combo/ticks_since_change) is carried tick-to-tick and
        would be silently corrupted by a partial or reordered slice.

        `tick_offset` shifts the reported `tick` indices only (the stream driver passes the
        first window's real start index, so records stay in true source coordinates even
        when segmentation does not begin at tick 0); it has no effect on the walk itself.
        """
        self._process_chunk(
            source_id,
            position=position,
            yaw=yaw,
            btn_mask=btn_mask,
            velocity=velocity,
            hp=hp,
            flags=flags,
            weapon=weapon,
            tick_offset=tick_offset,
            state=self._new_walk_state(),
        )

    def _process_chunk(self, source_id: str, *, position, yaw, btn_mask, velocity, hp,
                       flags, weapon, tick_offset: int, state: dict) -> None:
        """Process one contiguous owned slice while preserving the source's walk state."""
        mesh = self._mesh_for(source_id)
        proximity = self._thresholds["proximity"]
        min_missing_speed = self._thresholds["min_missing_speed"]
        concentration_ratio = self._thresholds["concentration_ratio"]

        n = len(hp)
        ray_cache = state["ray_cache"]
        active_combo = state["active_combo"]
        ticks_since_change = state["ticks_since_change"]
        ticks_since_modeled = state["ticks_since_modeled"]
        v0 = state["v0"]
        prev_yaw = state["prev_yaw"]

        for i in range(n):
            combo = int(btn_mask[i]) & ALL_MOVE_BITS
            alive_ok = hp[i] > 0 and not is_skip_flags(int(flags[i]))
            modeled = combo in COMBO_FORWARD_RIGHT
            actual_vel = velocity[i][:2]

            # A spinning camera rotates the wish-direction target under us every tick (it's
            # recomputed from the CURRENT yaw each time) -- real momentum can't instantly
            # follow, producing a residual that looks like a collision but is really just a
            # turn. Same guard collision_operator.py applies to its own velocity features.
            fast_turn = (prev_yaw is not None
                        and abs(wrap_angle(float(yaw[i]) - prev_yaw)) > MAX_DYAW_LEARN)
            prev_yaw = float(yaw[i])

            record = {
                "source_id": source_id, "tick": tick_offset + i, "collision": False, "reason": None,
                "combo": combo, "tracking_combo": None, "weapon": weapon[i],
                "residual_mag": None, "concentration_ratio": None, "dist_to_wall_hu": None,
            }

            if not alive_ok:
                active_combo, ticks_since_change, ticks_since_modeled = None, 0, 0
                record["reason"] = "dead_or_skipped"
                self._records.append(record)
                continue

            if modeled:
                if combo != active_combo:
                    # Retarget, don't reset-to-rest: v0 is whatever velocity the player
                    # actually had the instant the combo changed (often nonzero -- e.g.
                    # already sprinting forward when a strafe key gets added), not an
                    # assumed zero. The exponential relaxation below carries that real
                    # momentum forward toward the new target.
                    ticks_since_change = 0
                    v0 = actual_vel.copy()
                    active_combo = combo
                else:
                    ticks_since_change += 1
                ticks_since_modeled = 0
            elif active_combo is not None:
                ticks_since_modeled += 1
                if ticks_since_modeled > GRACE_TICKS:
                    active_combo = None
                else:
                    # Still within the grace window: keep the same ramp running rather
                    # than dropping tracking right as e.g. a key gets released on impact.
                    ticks_since_change += 1

            record["tracking_combo"] = active_combo

            if active_combo is None:
                record["reason"] = "not_modeled"
                self._records.append(record)
                continue
            if fast_turn:
                record["reason"] = "fast_turn"
                self._records.append(record)
                continue

            tau, cap = self.ramp_models.get((active_combo, weapon[i]), self.fallback_ramp)
            wish_dir = _wish_direction(active_combo, float(yaw[i]))
            target_vel = cap * wish_dir
            decay = math.exp(-ticks_since_change / tau)
            expected_vel = target_vel + (v0 - target_vel) * decay

            residual = expected_vel - actual_vel
            residual_mag = float(np.linalg.norm(residual))
            record["residual_mag"] = round(residual_mag, 1)
            if residual_mag < min_missing_speed:
                record["reason"] = "residual_too_small"
                self._records.append(record)
                continue

            pos = position[i]
            # Search toward v0 -- the player's actual velocity direction at the moment this
            # combo/press STARTED, before any collision in this window had a chance to act
            # on it -- not where they're looking (no physical relationship to wall
            # position: a player can look anywhere while moving in a fixed direction, e.g.
            # strafing while tracking an enemy) and not the CURRENT tick's actual velocity
            # (a wall already deflecting the player redirects it away from itself, hiding
            # exactly the wall that caused the deflection). v0 is captured once per press
            # rather than every tick, so unlike the previous tick's velocity it doesn't
            # keep drifting further off the true approach direction the deeper into a
            # multi-tick collision we are.
            if np.linalg.norm(v0) > 1.0:
                search_yaw = math.degrees(math.atan2(v0[1], v0[0]))
            else:
                search_yaw = math.degrees(math.atan2(wish_dir[1], wish_dir[0]))
            cache_key = (int(pos[0]) >> 3, int(pos[1]) >> 3, int(search_yaw / 5))
            if cache_key in ray_cache:
                hit, dist_hu, wall_normal = ray_cache[cache_key]
            else:
                hit, dist_hu, wall_normal = cast_proximity_rays(mesh, pos, search_yaw, proximity)
                ray_cache[cache_key] = (hit, dist_hu, wall_normal)
            if not hit or wall_normal is None:
                record["reason"] = "no_wall_hit"
                self._records.append(record)
                continue
            record["dist_to_wall_hu"] = round(dist_hu, 2)

            n2 = wall_normal[:2]
            residual_normal = float(np.dot(residual, n2))
            ratio = abs(residual_normal) / (residual_mag + 1e-9)
            record["concentration_ratio"] = round(ratio, 2)
            if ratio < concentration_ratio:
                record["reason"] = "low_concentration"  # shortfall spread across
                                                          # directions -- not wall-specific
                self._records.append(record)
                continue

            record["collision"] = True
            record["reason"] = "flagged"
            self._records.append(record)

        state.update(
            active_combo=active_combo,
            ticks_since_change=ticks_since_change,
            ticks_since_modeled=ticks_since_modeled,
            v0=v0,
            prev_yaw=prev_yaw,
        )

    # ── Tick-stream contract (what Pipeline.enrich()/analyze() drive) ──────────────────

    def add_batch(self, batch, ctx) -> None:
        """Accumulate one bounded batch of windows from the stream driver.

        Windows arrive per source, in order, and the registration declares
        ``ownership="rightmost-start"``, so each row's owned prefix (`owned_lengths`) is
        exactly the slice of the source's continuous tick sequence no other window owns.
        Each owned prefix is processed immediately while only the small cross-tick walk
        state is retained. A window that does not continue exactly where the previous one
        left off (stride > window leaves gaps the walk's carried state cannot survive)
        raises instead of walking a corrupted sequence.
        """
        typed = hasattr(ctx, "input_token")

        def tok(role):
            return ctx.input_token(role) if typed else role

        owned_lengths = getattr(ctx, "owned_lengths", None)
        if owned_lengths is None:
            owned_lengths = batch.get("owned_lengths")
        valid_steps = getattr(ctx, "valid_steps", None)
        if valid_steps is None:
            valid_steps = batch.get("valid_steps")
        columns = {role.split(":", 1)[1]: batch[tok(role)] for role in STREAM_INPUTS}

        for i, (source_id, start_idx, _end_idx) in enumerate(batch["segment_ids"]):
            if source_id != self._active_stream_source:
                if source_id in self._stream_states:
                    raise ValueError(
                        f"source {source_id!r} resumed after another source began -- stream "
                        "windows must be grouped by source")
                if self._active_stream_source is not None:
                    self._stream_states.pop(self._active_stream_source, None)
                    self._stream_next_start.pop(self._active_stream_source, None)
                self._active_stream_source = source_id
            valid = int(valid_steps[i]) if valid_steps is not None else None
            owned = int(owned_lengths[i]) if owned_lengths is not None else None
            start = int(start_idx)
            expected_start = self._stream_next_start.get(source_id, start)
            if start != expected_start:
                raise ValueError(
                    f"source {source_id!r}: window starting at {start} does not "
                    f"continue the previous window (expected start {expected_start}) -- "
                    "the directional walk carries tick-to-tick state and needs contiguous "
                    "coverage; use stride <= window")
            chunk = {}
            for name, batched in columns.items():
                row = batched[i]
                if name == "weapon":
                    # Categorical fields arrive as (T, E) object arrays of raw strings.
                    row = row[:, 0]
                else:
                    row = _drop_entity_axis_row(np.asarray(row), vector=True)
                    if name in ("btn_mask", "hp", "flags"):
                        row = row[:, 0]
                    elif name == "angle":
                        row = row[:, 0]  # yaw only; pitch plays no part in this walk
                if valid is not None:
                    row = row[:valid]
                chunk[name] = row if owned is None else row[:owned]
            length = len(chunk["hp"])
            state = self._stream_states.setdefault(source_id, self._new_walk_state())
            self._process_chunk(
                source_id,
                position=chunk["position"],
                yaw=chunk["angle"],
                btn_mask=chunk["btn_mask"],
                velocity=chunk["velocity"],
                hp=chunk["hp"],
                flags=chunk["flags"],
                weapon=chunk["weapon"],
                tick_offset=start,
                state=state,
            )
            self._stream_next_start[source_id] = start + length

    def _release_stream_state(self) -> None:
        """Drop cross-window state once a result surface finalizes the current stream."""
        self._stream_states.clear()
        self._stream_next_start.clear()
        self._active_stream_source = None

    # ── Result surface ──────────────────────────────────────────────────────────────────

    def to_records(self) -> list:
        """Flat per-tick export: one row for EVERY tick ever passed to `process_source()`
        (directly or via the stream driver), in call order, with a `collision` boolean
        column plus whatever diagnostics were available at the gate the tick stopped at
        (None for fields never reached)."""
        self._release_stream_state()
        return list(self._records)

    def summary(self) -> dict:
        """Aggregate counts -- satisfies the Analyzer protocol's summary() contract.

        `flagged_ticks` lists every flagged tick index per source (true source
        coordinates), so the pipeline path -- whose enrich() attachment retains exactly
        this mapping -- exposes which ticks fired, not just how many; the full per-tick
        diagnostics stay on `to_records()`.
        """
        self._release_stream_state()
        per_source: dict = {}
        flagged_ticks: dict = {}
        for r in self._records:
            s = per_source.setdefault(r["source_id"], {"ticks": 0, "collisions": 0})
            s["ticks"] += 1
            s["collisions"] += int(r["collision"])
            if r["collision"]:
                flagged_ticks.setdefault(r["source_id"], []).append(int(r["tick"]))
        return {
            "sources_processed": len(per_source),
            "total_ticks": len(self._records),
            "total_collisions": sum(int(r["collision"]) for r in self._records),
            "per_source": per_source,
            "flagged_ticks": flagged_ticks,
        }


def directional_collision_analyzer(**kwargs) -> DirectionalCollisionAnalyzer:
    """Construct the discoverable, user-owned directional-collision analyzer."""
    return DirectionalCollisionAnalyzer(**kwargs)


DIRECTIONAL_OPERATOR_PARAMETERS = {
    "type": "object",
    "properties": {
        **DIRECTIONAL_COLLISION_PARAMETERS["properties"],
        "batch_size": {"type": "integer", "minimum": 1},
        "inputs": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "metadata_key": {
            "type": "string",
            "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]*$",
            "default": "directional_collision",
        },
        "replace": {"type": "boolean", "default": False},
    },
    "required": ["map_registry", "ramp_models_path"],
    "additionalProperties": False,
}


def attach_directional_collision(
    pipeline,
    *,
    map_registry,
    ramp_models_path,
    map_for_source=None,
    default_map=None,
    fallback_ramp=None,
    disk_cache_dir=None,
    proximity=None,
    min_missing_speed=None,
    concentration_ratio=None,
    inputs=None,
    batch_size=None,
    metadata_key: str = "directional_collision",
    replace: bool = False,
):
    """Run directional collision detection and retain the verdict on a derived pipeline.

    The pipeline counterpart of driving the analyzer by hand: one bounded pass over this
    pipeline's segmented tick stream (each source's windows stitched back into its
    continuous tick sequence -- see the module docstring), with the summary (including
    `flagged_ticks` in true source coordinates) attached under
    ``metadata["gameplay"][metadata_key]``. Tick-stream only: this analyzer measures raw
    telemetry against map geometry, which a written dataset's manifest no longer carries.
    """
    resolved = pipeline.analysis_mode(None)
    if resolved != "stream":
        raise ValueError(
            "directional collision analysis is tick-stream only: it walks raw per-source "
            "telemetry against map geometry, which a written dataset does not preserve -- "
            "run it on the raw input pipeline, before write()")
    parameters = {
        "map_registry": map_registry,
        "ramp_models_path": ramp_models_path,
    }
    optional = {
        "map_for_source": map_for_source,
        "default_map": default_map,
        "fallback_ramp": fallback_ramp,
        "disk_cache_dir": disk_cache_dir,
        "proximity": proximity,
        "min_missing_speed": min_missing_speed,
        "concentration_ratio": concentration_ratio,
    }
    parameters.update({key: value for key, value in optional.items() if value is not None})
    return pipeline.enrich(
        COMPONENT_NAME,
        parameters,
        namespace="gameplay",
        key=metadata_key,
        replace=replace,
        mode="stream",
        inputs=inputs,
        batch_size=batch_size,
    )
