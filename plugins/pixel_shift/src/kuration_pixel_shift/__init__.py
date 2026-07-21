"""Separately packaged pixel-shift extension for Kuration."""

from __future__ import annotations

from .diamond_action_decode import (
    ALL_CONTROLS,
    BOUNDARY_POLICIES,
    find_isolated_press_sequences,
    release_transition_displacement,
    validate_actions,
    validate_frames,
)
from .pixel_shift_operator import (
    CONTROL_PARAMETERS,
    OUTPUT_KEYS,
    PixelShift,
    PixelShiftReport,
    pixel_shift,
    segment_scalars,
)

COMPONENT_NAME = "aintropy/pixel-shift"


def register() -> None:
    """Idempotently register the transform and compositional operator."""
    from kuration import default_registry, register_batch_transform, register_operator

    common = {
        "version": "1",
        "package": "kuration-plugin-pixel-shift",
        "stability": "experimental",
        "install_hint": "pip install kuration-plugin-pixel-shift",
    }
    if COMPONENT_NAME not in default_registry.names("transform"):
        register_batch_transform(
            COMPONENT_NAME,
            PixelShift,
            description="Per-segment pixel-shift statistics for isolated key/mouse presses",
            outputs=OUTPUT_KEYS,
            inputs=("media:frame", "field:action"),
            media_requirements={"kind": "video-frame", "layout": "HWC"},
            resources={"num_gpus": 0},
            lookbehind=1,
            lookahead=3,
            ownership="rightmost-start",
            parameters=CONTROL_PARAMETERS,
            **common,
        )
    if COMPONENT_NAME not in default_registry.names("operator"):
        register_operator(
            COMPONENT_NAME,
            pixel_shift,
            description="Measure pixel shift for every isolated key/mouse press",
            parameters=CONTROL_PARAMETERS,
            **common,
        )


__all__ = [
    "ALL_CONTROLS",
    "BOUNDARY_POLICIES",
    "COMPONENT_NAME",
    "PixelShift",
    "PixelShiftReport",
    "find_isolated_press_sequences",
    "pixel_shift",
    "register",
    "release_transition_displacement",
    "segment_scalars",
    "validate_actions",
    "validate_frames",
]
