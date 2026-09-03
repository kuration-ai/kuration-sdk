"""The `aintropy/collision` batch transform + operator pair.

Ported from the reference `collision_pipeline_fused.py` pipeline (itself a fused,
in-memory-only combination of `action_state_analyzer.py`'s per-tick anomaly scan and
`collision_classifier.py`'s raycast classification) -- the reference scripts are
untouched; only their per-tick scan/classify math is reproduced here.

Format-agnostic by design: this transform only ever reads the canonical schema fields
`position` (vector, semantics="position") and `angle` (vector, dims=("yaw","pitch"),
semantics="angular") plus `hp`/`btn_mask`/`flags` (scalars). Any InputAdapter that maps
its own raw fields onto those same canonical names -- the ASG `data.json` adapter today,
a TeaPearce-style adapter later -- can drive this same operator unmodified. Per-map
geometry and the learned anomaly baseline (`dist_path`) are supplied as operator
parameters, not hardcoded, since both are dataset-specific.

Kuration's batch-transform stage runs single-process and sequential (never inside a
ProcessPoolExecutor -- that's only used for the metric-scoring stage), with one
transform instance constructed once per pipeline run and reused across every source in
that run. That's what makes the in-memory `self._mesh_cache` dict in `__init__` safe:
it is never silently reset mid-run the way it would be behind a process pool.
"""
from __future__ import annotations

import json
import math

import numpy as np

from .kinematics import extract_kinematics
from .mesh_cache import cast_proximity_rays, load_mesh_cached, wall_stopped
from .physics import (
    BIT_A,
    BIT_D,
    BIT_W,
    FEATURES,
    FL_DUCKING,
    FL_ONGROUND,
    MAX_DYAW_LEARN,
    MOVE_BITS,
    condition_key,
    is_skip_flags,
    z_score,
)

def _drop_entity_axis(arr, *, vector: bool):
    """Normalize one batch field array back to the pre-entity-axis shape this module was
    written against: `(n_segments, T, D)` for vector fields, `(n_segments, T)` for scalar
    fields. Real Pipeline batches now carry an explicit `(n_segments, T, E, D)` shape
    (E = entities) even for this plugin's always-single-entity-per-source schemas
    (`entity_axis="player"`, one player per source, by design); pre-existing unit tests
    still hand-build the pre-entity-axis shape directly, so this dispatches on `ndim`
    rather than assuming either shape unconditionally."""
    if arr is None:
        return None
    if vector:
        return arr[:, :, 0, :] if arr.ndim == 4 else arr
    return arr[:, :, 0, 0] if arr.ndim == 4 else arr


def _drop_entity_axis_row(arr, *, vector: bool):
    """Same normalization as `_drop_entity_axis`, but for a single PER-ROW array -- one
    fewer leading axis, as returned per-element by `ctx.before()`/`ctx.after()`'s tuple:
    `(T, D)` for vectors, `(T,)` for scalars, pre-entity-axis; new-format core provides
    `(T, E, D)` / `(T, E, 1)`."""
    if arr is None:
        return None
    if vector:
        return arr[:, 0, :] if arr.ndim == 3 else arr
    return arr[:, 0, 0] if arr.ndim == 3 else arr


COMPONENT_NAME = "aintropy/collision"

COLLISION_TYPES = ("head_on", "angled", "W_slide", "WA_slide", "slide_end", "slide_constrained")

OUTPUT_KEYS = (
    "collision_count",
    "head_on_count",
    "angled_count",
    "w_slide_count",
    "wa_slide_count",
    "slide_end_count",
    "slide_constrained_count",
    "max_anomaly_score",
    "min_dist_to_wall_hu",
)

_TYPE_TO_OUTPUT_KEY = {
    "head_on": "head_on_count",
    "angled": "angled_count",
    "W_slide": "w_slide_count",
    "WA_slide": "wa_slide_count",
    "slide_end": "slide_end_count",
    "slide_constrained": "slide_constrained_count",
}

COLLISION_PARAMETERS = {
    "type": "object",
    "properties": {
        "map_for_source": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "source_id -> map_name",
        },
        "map_registry": {
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "string"}},
            "description": "map_name -> geometry file path(s) (.bsp / .glb / .nav)",
        },
        "dist_path": {"type": "string", "description": "path to a fitted action_state_dist.json"},
        "disk_cache_dir": {"type": "string"},
        "min_score": {"type": "number", "minimum": 0},
        "proximity": {"type": "number", "minimum": 0},
        "min_approach": {"type": "number", "minimum": 0},
        "max_residual": {"type": "number", "minimum": 0},
        "slide_angle": {"type": "number", "minimum": 0, "maximum": 180},
        "max_slide_speed": {"type": "number", "minimum": 0},
        "min_release_accel": {"type": "number", "minimum": 0},
        "max_release_vn": {"type": "number", "minimum": 0},
    },
    "required": ["map_for_source", "map_registry", "dist_path"],
    "additionalProperties": False,
}

# Defaults mirror collision_pipeline_fused.py's argparse defaults exactly.
_DEFAULTS = dict(
    disk_cache_dir=".kuration_collision_cache",
    min_score=2.0,
    proximity=40.0,
    min_approach=250.0,
    max_residual=20.0,
    slide_angle=75.0,
    max_slide_speed=400.0,
    min_release_accel=100.0,
    max_release_vn=100.0,
)


def _score_and_classify_window(
    pos, yaw, pitch, hp, btn, flags, vel,
    pos_before, yaw_before, pitch_before, hp_before, btn_before, flags_before, vel_before,
    mesh, dist, *, min_score, proximity, min_approach, max_residual,
    slide_angle, max_slide_speed, min_release_accel, max_release_vn,
    owned_length=None,
) -> dict:
    """Score + classify every tick in one window (a slice of one source's continuous
    tick sequence). Ported from _scan_one_player (scoring) + _classify_one_player
    (raycast classification), fused into one per-tick pass since scanning happens over
    the same window arrays the classifier would otherwise need to re-look-up.

    `vel` must already be true HU/s velocity, supplied by the adapter (either a real
    recorded engine velocity, or one the adapter derived itself at its own known sample
    rate) -- this function never derives or scales velocity from position deltas itself.

    `owned_length` restricts the scored ticks to the window's OWNED prefix under the
    core's ``ownership="rightmost-start"`` convention (a window owns its steps up to where
    the next window starts, the same rule pixel-shift applies to press instances): with
    overlapping windows (stride < window), a collision tick inside the overlap would
    otherwise be counted once per covering window. The overlap ticks this window skips
    are exactly the next window's owned prefix -- and the next window scores them with
    full lookbehind context, so nothing is lost, only deduplicated. ``None`` (a direct
    call outside the pipeline) means the whole window is owned.
    """
    have_history = pos_before is not None
    T = len(hp)
    limit = T if owned_length is None else min(T, int(owned_length))

    def rec(i):
        if i < 0:
            if not have_history:
                return None
            j = i + 2
            return (pos_before[j], float(yaw_before[j]), float(pitch_before[j]),
                    float(hp_before[j]), int(btn_before[j]), int(flags_before[j]), vel_before[j])
        return (pos[i], float(yaw[i]), float(pitch[i]), float(hp[i]), int(btn[i]), int(flags[i]), vel[i])

    ray_cache: dict = {}
    max_anomaly = 0.0
    min_dist = math.inf
    counts = {t: 0 for t in COLLISION_TYPES}
    total = 0

    for i in range(limit):
        p0 = rec(i)
        pm1 = rec(i - 1)
        pm2 = rec(i - 2)
        if pm1 is None or pm2 is None:
            continue

        pos0, yaw0, pitch0, hp0, btn0, flags0, vel0 = p0
        _, _, _, hp1, _, flags1, vel1 = pm1
        _, _, _, hp2, _, flags2, _vel2 = pm2

        if hp0 <= 0 or hp1 <= 0 or hp2 <= 0:
            continue
        if is_skip_flags(flags0) or is_skip_flags(flags1) or is_skip_flags(flags2):
            continue

        f = extract_kinematics(pm2, pm1, p0)

        mb0 = btn0 & MOVE_BITS
        og0 = bool(flags0 & FL_ONGROUND)
        dk0 = bool(flags0 & FL_DUCKING)
        ckey = condition_key(mb0, og0, dk0)
        st = dist.get(ckey)
        if st is None and not og0:
            st = dist.get(condition_key(mb0, True, dk0))
        if st is None:
            continue

        fast_turn = f["dyaw10"] > MAX_DYAW_LEARN
        z_scores = {}
        for feat in FEATURES:
            if fast_turn and feat in ("v_fwd", "v_right", "dv_fwd", "dv_right"):
                z_scores[feat] = 0.0
            else:
                z_scores[feat] = abs(z_score(f[feat], st.get(feat)))
        score = max(z_scores.values())
        max_anomaly = max(max_anomaly, score)
        if score < min_score:
            continue
        anomalous = {feat for feat in FEATURES if z_scores[feat] >= min_score}

        if not ({"dv_fwd", "dv_right", "v_fwd", "speed_2d"} & anomalous):
            continue
        if not (btn0 & BIT_W):
            continue

        vel_curr = vel0
        vel_prev = vel1

        cache_key = (int(pos0[0]) >> 3, int(pos0[1]) >> 3, int(yaw0 / 5))
        if cache_key in ray_cache:
            hit, dist_hu, wall_normal = ray_cache[cache_key]
        else:
            hit, dist_hu, wall_normal = cast_proximity_rays(mesh, pos0, yaw0, proximity)
            ray_cache[cache_key] = (hit, dist_hu, wall_normal)
        if not hit:
            continue

        flagged, vn_prev_hus, vn_curr_hus = wall_stopped(
            vel_prev, vel_curr, wall_normal, min_approach, max_residual)
        sp_prev_hus = math.hypot(vel_prev[0], vel_prev[1])
        sp_curr_hus = math.hypot(vel_curr[0], vel_curr[1])

        is_wall_release = (
            not flagged and max_residual < sp_prev_hus < max_slide_speed
            and vn_prev_hus <= max_release_vn
        )
        if not flagged and not is_wall_release:
            continue

        approach_angle = (
            math.degrees(math.acos(min(1.0, vn_prev_hus / sp_prev_hus)))
            if sp_prev_hus > 0 else 90.0
        )

        if "d_yaw" in anomalous and ("d_pitch" in anomalous or is_wall_release):
            continue

        strafing = bool(btn0 & (BIT_A | BIT_D))
        if not is_wall_release:
            if approach_angle > 60.0 and "dv_right" not in anomalous and "v_right" not in anomalous:
                if not (strafing and ("dv_fwd" in anomalous or "v_fwd" in anomalous)):
                    continue

        if is_wall_release:
            collision_type = (
                "slide_end" if sp_curr_hus - sp_prev_hus >= min_release_accel
                else "slide_constrained"
            )
        elif approach_angle <= 30.0:
            collision_type = "head_on"
        elif approach_angle <= slide_angle:
            collision_type = "angled"
        elif btn0 & (BIT_A | BIT_D):
            collision_type = "WA_slide"
        else:
            collision_type = "W_slide"

        counts[collision_type] += 1
        total += 1
        min_dist = min(min_dist, dist_hu)

    out = {"collision_count": float(total), "max_anomaly_score": float(max_anomaly),
           "min_dist_to_wall_hu": float(min_dist) if min_dist != math.inf else -1.0}
    for t, key in _TYPE_TO_OUTPUT_KEY.items():
        out[key] = float(counts[t])
    return out


class CollisionMechanics:
    """Kuration batch transform: per-window collision counts/anomaly score."""

    def __init__(
        self, *, map_for_source, map_registry, dist_path,
        disk_cache_dir=_DEFAULTS["disk_cache_dir"],
        min_score=_DEFAULTS["min_score"],
        proximity=_DEFAULTS["proximity"],
        min_approach=_DEFAULTS["min_approach"],
        max_residual=_DEFAULTS["max_residual"],
        slide_angle=_DEFAULTS["slide_angle"],
        max_slide_speed=_DEFAULTS["max_slide_speed"],
        min_release_accel=_DEFAULTS["min_release_accel"],
        max_release_vn=_DEFAULTS["max_release_vn"],
    ):
        # No **kwargs catch-all here on purpose: transforms.py's _construct() probes
        # whether a factory's signature *binds* a `device=` kwarg to decide whether to
        # pass one, and a catch-all would make that probe succeed unintentionally
        # (see PixelShift, which avoids this the same way -- explicit params, no **kwargs).
        self.map_for_source = dict(map_for_source)
        self.map_registry = {k: list(v) for k, v in map_registry.items()}
        with open(dist_path) as f:
            self.dist = json.load(f)
        self.disk_cache_dir = disk_cache_dir
        self._thresholds = dict(
            min_score=min_score, proximity=proximity, min_approach=min_approach,
            max_residual=max_residual, slide_angle=slide_angle,
            max_slide_speed=max_slide_speed, min_release_accel=min_release_accel,
            max_release_vn=max_release_vn,
        )
        self._mesh_cache: dict = {}

    def _mesh_for(self, source_id: str):
        map_name = self.map_for_source[source_id]
        mesh = self._mesh_cache.get(map_name)
        if mesh is None:
            mesh = load_mesh_cached(map_name, self.map_registry[map_name],
                                    disk_cache_dir=self.disk_cache_dir)
            self._mesh_cache[map_name] = mesh
        return mesh

    def transform_batch(self, batch, ctx):
        # Address the batch by declared role (a per-use ``inputs=`` remapping can
        # legitimately rename tokens); plain-mapping contexts from direct callers keep
        # the registration tokens.
        typed = hasattr(ctx, "input_token")

        def tok(role):
            return ctx.input_token(role) if typed else role

        pos = _drop_entity_axis(batch[tok("field:position")], vector=True)
        angle = _drop_entity_axis(batch[tok("field:angle")], vector=True)
        hp = _drop_entity_axis(batch[tok("field:hp")], vector=False)
        btn = _drop_entity_axis(batch[tok("field:btn_mask")], vector=False)
        flags = _drop_entity_axis(batch[tok("field:flags")], vector=False)
        vel = _drop_entity_axis(batch[tok("field:velocity")], vector=True)

        if hasattr(ctx, "before"):
            pos_before_batch = [_drop_entity_axis_row(a, vector=True) for a in ctx.before(tok("field:position"))]
            angle_before_batch = [_drop_entity_axis_row(a, vector=True) for a in ctx.before(tok("field:angle"))]
            hp_before_batch = [_drop_entity_axis_row(a, vector=False) for a in ctx.before(tok("field:hp"))]
            btn_before_batch = [_drop_entity_axis_row(a, vector=False) for a in ctx.before(tok("field:btn_mask"))]
            flags_before_batch = [_drop_entity_axis_row(a, vector=False) for a in ctx.before(tok("field:flags"))]
            vel_before_batch = [_drop_entity_axis_row(a, vector=True) for a in ctx.before(tok("field:velocity"))]
        else:
            n = len(batch["segment_ids"])
            pos_before_batch = batch.get("lookbehind:field:position", [None] * n)
            angle_before_batch = batch.get("lookbehind:field:angle", [None] * n)
            hp_before_batch = batch.get("lookbehind:field:hp", [None] * n)
            btn_before_batch = batch.get("lookbehind:field:btn_mask", [None] * n)
            flags_before_batch = batch.get("lookbehind:field:flags", [None] * n)
            vel_before_batch = batch.get("lookbehind:field:velocity", [None] * n)

        # Hand-built contexts (unit tests, direct callers) may carry neither attribute --
        # fall back to the batch mapping, then to "the whole window is owned and valid".
        owned_lengths = getattr(ctx, "owned_lengths", None)
        if owned_lengths is None:
            owned_lengths = batch.get("owned_lengths")
        valid_steps = getattr(ctx, "valid_steps", None)
        if valid_steps is None:
            valid_steps = batch.get("valid_steps")

        rows = []
        for i, (source_id, _, _) in enumerate(batch["segment_ids"]):
            mesh = self._mesh_for(source_id)
            # Batch rows are padded to the widest window in the batch (pad is NaN, which
            # int()/comparison code below must never see); ``valid_steps`` is the honest
            # per-row extent. The owned prefix can never exceed it.
            valid = int(valid_steps[i]) if valid_steps is not None else len(hp[i])
            owned = (min(valid, int(owned_lengths[i]))
                     if owned_lengths is not None else None)
            ang = angle[i][:valid]
            ang_before = angle_before_batch[i]
            rows.append(_score_and_classify_window(
                pos[i][:valid], ang[:, 0], ang[:, 1], hp[i][:valid], btn[i][:valid],
                flags[i][:valid], vel[i][:valid],
                pos_before_batch[i],
                ang_before[:, 0] if ang_before is not None else None,
                ang_before[:, 1] if ang_before is not None else None,
                hp_before_batch[i], btn_before_batch[i], flags_before_batch[i],
                vel_before_batch[i],
                mesh, self.dist, **self._thresholds, owned_length=owned,
            ))
        return {key: np.array([r[key] for r in rows], dtype=np.float64) for key in OUTPUT_KEYS}


def collision_mechanics(pipeline, **params):
    """Compose the registered transform into a pipeline call."""
    return pipeline.transform(COMPONENT_NAME, **params)
