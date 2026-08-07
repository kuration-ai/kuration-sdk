"""Separately packaged CS2 wall-collision detection/classification extension for Kuration.

Ports the reference `collision_pipeline_fused.py` pipeline (analyzer scan + classifier
raycast, fused) as a `BatchTransform` + compositional `operator` pair. The transform is
format-agnostic: it reads only the canonical `position`/`angle`/`hp`/`btn_mask`/`flags`
schema fields, so any InputAdapter mapping its own raw fields onto that same schema can
drive it -- see `asg_adapter.py` for the ASG `data.json` -> canonical-Tick adapter.
"""
from __future__ import annotations

from .asg_adapter import ASG_SCHEMA, AsgCollisionAdapter
from .collision_operator import (
    COLLISION_PARAMETERS,
    COMPONENT_NAME,
    OUTPUT_KEYS,
    CollisionMechanics,
    collision_mechanics,
)
from .directional_collision_analyzer import DIRECTIONAL_COLLISION_PARAMETERS
from .directional_collision_analyzer import COMPONENT_NAME as DIRECTIONAL_COMPONENT_NAME
from .directional_collision_analyzer import DirectionalCollisionAnalyzer, directional_collision_analyzer
from .ramp_fitting import fit_ramp_models, load_ramp_models, save_ramp_models
from .teapearce_adapter import TEAPEARCE_SCHEMA, TeaPearceDust2Adapter, load_directional_arrays


ADAPTER_NAME = "aintropy/cs2-collision-asg"
TEAPEARCE_ADAPTER_NAME = "aintropy/cs2-collision-teapearce"


def register() -> None:
    """Idempotently register the adapters, transform/operator pair, and directional
    collision analyzer."""
    from kuration import (
        default_registry,
        register_adapter,
        register_analyzer,
        register_batch_transform,
        register_operator,
    )

    common = {
        "version": "1",
        "package": "kuration-plugin-cs2-collision",
        "stability": "experimental",
        "install_hint": "pip install kuration-plugin-cs2-collision",
        "documentation": "https://github.com/aintropy-ai/kuration-sdk/tree/main/"
                         "plugins/cs2_collision#usage",
    }
    if ADAPTER_NAME not in default_registry.names("adapter"):
        register_adapter(
            ADAPTER_NAME,
            AsgCollisionAdapter,
            description="Read ASG data.json CS2 match replays as one Tick stream per "
                        "(match, player), yielding canonical position/angle/hp/btn_mask/flags",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            **common,
        )
    if TEAPEARCE_ADAPTER_NAME not in default_registry.names("adapter"):
        register_adapter(
            TEAPEARCE_ADAPTER_NAME,
            TeaPearceDust2Adapter,
            description="Read TeaPearce/CounterStrike_Deathmatch dust2 .npy metadata files as "
                        "one Tick stream per recording, yielding canonical "
                        "position/angle/hp/btn_mask/flags",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            **common,
        )
    if COMPONENT_NAME not in default_registry.names("transform"):
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
            parameters=COLLISION_PARAMETERS,
            **common,
        )
    if COMPONENT_NAME not in default_registry.names("operator"):
        register_operator(
            COMPONENT_NAME,
            collision_mechanics,
            description="Compose the CS2 collision-detection transform into a fluent pipeline",
            parameters=COLLISION_PARAMETERS,
            **common,
        )
    if DIRECTIONAL_COMPONENT_NAME not in default_registry.names("analyzer"):
        register_analyzer(
            DIRECTIONAL_COMPONENT_NAME,
            directional_collision_analyzer,
            description="Flag per-tick candidate wall collisions from position/velocity/angle/"
                        "button telemetry against map geometry, using a fitted per-combo/weapon "
                        "acceleration-ramp model instead of a learned anomaly baseline",
            parameters=DIRECTIONAL_COLLISION_PARAMETERS,
            capabilities={"stateful": True, "cross_segment": True},
            **common,
        )


__all__ = [
    "ADAPTER_NAME",
    "ASG_SCHEMA",
    "COLLISION_PARAMETERS",
    "COMPONENT_NAME",
    "DIRECTIONAL_COLLISION_PARAMETERS",
    "DIRECTIONAL_COMPONENT_NAME",
    "OUTPUT_KEYS",
    "TEAPEARCE_ADAPTER_NAME",
    "TEAPEARCE_SCHEMA",
    "AsgCollisionAdapter",
    "CollisionMechanics",
    "DirectionalCollisionAnalyzer",
    "TeaPearceDust2Adapter",
    "collision_mechanics",
    "directional_collision_analyzer",
    "fit_ramp_models",
    "load_directional_arrays",
    "load_ramp_models",
    "register",
    "save_ramp_models",
]
