"""Built-in gameplay pixel-shift operators for DIAMOND-compatible action streams."""

from __future__ import annotations

from .diamond_action_decode import (
    ALL_CONTROLS,
    BOUNDARY_POLICIES,
    find_isolated_press_sequences,
    release_transition_displacement,
    validate_actions,
    validate_boundary,
    validate_color_space,
    validate_frames,
)
from .pixel_shift_operator import (
    CONTROL_PARAMETERS,
    OUTPUT_KEYS,
    PixelShift,
    PixelShiftReport,
    iter_instances,
    pixel_shift,
    segment_scalars,
    validate_segment_inputs,
)

COMPONENT_NAME = "aintropy/pixel-shift"


def register() -> None:
    """Idempotently register the transform and compositional operator."""
    from kuration import default_registry, register_batch_transform, register_operator
    from ...core._registration import needs_registration

    common = {
        "version": "1",
        "package": "kuration-sdk",
        "stability": "experimental",
        "extra": "pixel-shift",
        "install_hint": 'pip install "kuration-sdk[pixel-shift]"',
        "documentation": "https://aintropy-ai.github.io/kuration-sdk/pixel-shift/",
    }
    if needs_registration(default_registry, "transform", COMPONENT_NAME, PixelShift):
        register_batch_transform(
            COMPONENT_NAME,
            PixelShift,
            description="Per-segment pixel-shift statistics for isolated key/mouse presses",
            outputs=OUTPUT_KEYS,
            inputs=("media:frame", "field:action"),
            media_requirements={"kind": "video-frame", "layout": "HWC"},
            resources={"num_gpus": 0},
            # Only the action stream needs a lookbehind row (to prove a press did not begin
            # before the window); transform_batch never reads a lookbehind frame, so declaring
            # the count per input avoids decoding one extra video frame per segment.
            lookbehind={"field:action": 1},
            lookahead=3,
            ownership="rightmost-start",
            parameters=CONTROL_PARAMETERS,
            **common,
        )
    if needs_registration(default_registry, "operator", COMPONENT_NAME, pixel_shift):
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
    "iter_instances",
    "pixel_shift",
    "register",
    "release_transition_displacement",
    "segment_scalars",
    "validate_actions",
    "validate_boundary",
    "validate_color_space",
    "validate_frames",
    "validate_segment_inputs",
]
