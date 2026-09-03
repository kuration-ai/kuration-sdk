"""Built-in gameplay CS2 wall-collision detection/classification.

Ports the reference `collision_pipeline_fused.py` pipeline (analyzer scan + classifier
raycast, fused) as a `BatchTransform` + compositional `operator` pair, plus a second,
independent directional-residual algorithm as a tick-stream `Analyzer` with a
pipeline-preserving companion operator. The transform is format-agnostic: it reads only
the canonical `position`/`angle`/`hp`/`btn_mask`/`flags`/`velocity` schema fields, so any
InputAdapter mapping its own raw fields onto that same schema can drive it -- see
`asg_adapter.py` (ASG `data.json`) and `teapearce_adapter.py` (TeaPearce dust2 `.npy`)
for the two shipped adapters. Heavy geometry dependencies (trimesh/rtree) stay lazy;
install ``kuration-sdk[collision]`` before executing the raycast paths.
"""
from __future__ import annotations

from typing import Any

from .asg_adapter import ASG_SCHEMA, AsgCollisionAdapter
from .collision_operator import (
    COLLISION_PARAMETERS,
    COMPONENT_NAME,
    OUTPUT_KEYS,
    CollisionMechanics,
    collision_mechanics,
)
from .directional_collision_analyzer import (
    DEFAULT_FALLBACK_RAMP,
    DIRECTIONAL_COLLISION_PARAMETERS,
    DIRECTIONAL_OPERATOR_PARAMETERS,
    STREAM_INPUTS,
    DirectionalCollisionAnalyzer,
    attach_directional_collision,
    directional_collision_analyzer,
)
from .directional_collision_analyzer import COMPONENT_NAME as DIRECTIONAL_COMPONENT_NAME
from .ramp_fitting import fit_ramp_models, load_ramp_models, save_ramp_models
from .teapearce_adapter import TEAPEARCE_SCHEMA, TeaPearceDust2Adapter, load_directional_arrays

ADAPTER_NAME = "aintropy/collision-asg"
TEAPEARCE_ADAPTER_NAME = "aintropy/collision-teapearce"


def register() -> None:
    """Idempotently register the adapters, transform/operator pair, and the directional
    analyzer with its pipeline-preserving companion operator."""
    from kuration import (
        default_registry,
        register_adapter,
        register_analyzer,
        register_batch_transform,
        register_operator,
    )

    from ...core._registration import needs_registration

    common: dict[str, Any] = {
        "version": "1",
        "package": "kuration-sdk",
        "stability": "experimental",
        "extra": "collision",
        "install_hint": 'pip install "kuration-sdk[collision]"',
        "documentation": "https://aintropy-ai.github.io/kuration-sdk/collision/",
    }
    if needs_registration(default_registry, "adapter", ADAPTER_NAME, AsgCollisionAdapter):
        register_adapter(
            ADAPTER_NAME,
            AsgCollisionAdapter,
            description="Read ASG data.json CS2 match replays as one Tick stream per "
                        "(match, player), yielding canonical position/angle/hp/btn_mask/flags",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            **common,
        )
    if needs_registration(default_registry, "adapter", TEAPEARCE_ADAPTER_NAME,
                          TeaPearceDust2Adapter):
        register_adapter(
            TEAPEARCE_ADAPTER_NAME,
            TeaPearceDust2Adapter,
            description="Read TeaPearce/CounterStrike_Deathmatch dust2 .npy metadata files as "
                        "one Tick stream per recording, yielding canonical "
                        "position/angle/hp/btn_mask/flags",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            **common,
        )
    if needs_registration(default_registry, "transform", COMPONENT_NAME, CollisionMechanics):
        register_batch_transform(
            COMPONENT_NAME,
            CollisionMechanics,
            description="Detect and classify CS2 wall collisions from per-tick position/angle "
                        "telemetry raycast against map geometry",
            outputs=OUTPUT_KEYS,
            inputs=("field:position", "field:angle", "field:hp", "field:btn_mask", "field:flags",
                    "field:velocity"),
            resources={"num_gpus": 0},
            lookbehind=2,
            # A window owns its steps up to where the next window starts, so a collision
            # tick inside a stride<window overlap is counted exactly once (the same
            # convention pixel-shift applies to press instances).
            ownership="rightmost-start",
            parameters=COLLISION_PARAMETERS,
            **common,
        )
    if needs_registration(default_registry, "operator", COMPONENT_NAME, collision_mechanics):
        register_operator(
            COMPONENT_NAME,
            collision_mechanics,
            description="Compose the CS2 collision-detection transform into a pipeline",
            parameters=COLLISION_PARAMETERS,
            **common,
        )
    if needs_registration(default_registry, "analyzer", DIRECTIONAL_COMPONENT_NAME,
                          directional_collision_analyzer):
        register_analyzer(
            DIRECTIONAL_COMPONENT_NAME,
            directional_collision_analyzer,
            description="Flag per-tick candidate wall collisions from position/velocity/angle/"
                        "button telemetry against map geometry, using a fitted per-combo/weapon "
                        "acceleration-ramp model instead of a learned anomaly baseline",
            parameters=DIRECTIONAL_COLLISION_PARAMETERS,
            # The tick-stream contract: windows arrive per source, in order, ownership-
            # deduplicated, and add_batch stitches them back into each source's continuous
            # tick sequence before the stateful walk runs (see the module docstring).
            inputs=STREAM_INPUTS,
            ownership="rightmost-start",
            capabilities={"stateful": True, "cross_segment": True, "tick_stream": True},
            **common,
        )
    if needs_registration(default_registry, "operator", DIRECTIONAL_COMPONENT_NAME,
                          attach_directional_collision):
        register_operator(
            DIRECTIONAL_COMPONENT_NAME,
            attach_directional_collision,
            description="Run directional collision detection over the tick stream and attach "
                        "the verdict to pipeline metadata",
            parameters=DIRECTIONAL_OPERATOR_PARAMETERS,
            capabilities={
                "stateful": True,
                "cross_segment": True,
                "pipeline_metadata": True,
            },
            **common,
        )


__all__ = [
    "ADAPTER_NAME",
    "ASG_SCHEMA",
    "COLLISION_PARAMETERS",
    "COMPONENT_NAME",
    "DEFAULT_FALLBACK_RAMP",
    "DIRECTIONAL_COLLISION_PARAMETERS",
    "DIRECTIONAL_COMPONENT_NAME",
    "DIRECTIONAL_OPERATOR_PARAMETERS",
    "OUTPUT_KEYS",
    "STREAM_INPUTS",
    "TEAPEARCE_ADAPTER_NAME",
    "TEAPEARCE_SCHEMA",
    "AsgCollisionAdapter",
    "CollisionMechanics",
    "DirectionalCollisionAnalyzer",
    "TeaPearceDust2Adapter",
    "attach_directional_collision",
    "collision_mechanics",
    "directional_collision_analyzer",
    "fit_ramp_models",
    "load_directional_arrays",
    "load_ramp_models",
    "register",
    "save_ramp_models",
]
