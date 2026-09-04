"""One pipeline: detect action-state inconsistency and keep the verdict attached.

`kuration.gameplay` registers `aintropy/action-state-consistency` as both the stateful analyzer
and a direct pipeline operator. `.action_state_consistency(...)` drives the analyzer once and
returns an ordinary `Pipeline` carrying the verdict in `.metadata`. The operator is dataset-mode
only: it audits an already-written, curated dataset's shards, so this writes one before reading
it back through `CuratedDatasetAdapter`. No CLI, detached side object, or separate analyzer
construction is required.

Frames come from a fake in-memory video provider (real per-frame pixel shift, no PyAV/codec
needed), so this runs anywhere.

    python examples/07_action_state_consistency.py
"""
from __future__ import annotations

import tempfile

import numpy as np

import kuration.gameplay  # noqa: F401 -- registers the analyzer and pixel-shift on import
from kuration import (
    CuratedDatasetAdapter,
    FrameRef,
    OutputSpec,
    SchemaSpec,
    VideoMetadata,
    input,
)
from kuration.gameplay.pixel_shift import diamond_action_decode as dad

ANALYZER_NAME = "aintropy/action-state-consistency"
ACTION_DIM = dad.ACTION_WIDTH
FRAME_SHAPE = (40, 60, 3)


class FakeVideoProvider:
    """Minimal MediaProvider: yields lazy frame refs over an in-memory frame stack."""

    capabilities = {"streaming": True, "variable_frame_rate": True}

    def __init__(self, frames: list[np.ndarray]):
        self.frames = frames

    def probe(self, uri):
        h, w = self.frames[0].shape[:2]
        return VideoMetadata(uri, w, h, len(self.frames), float(len(self.frames) - 1), 1.0)

    def iter_refs(self, uri, *, source_id=None):
        for i in range(len(self.frames)):
            yield FrameRef(uri, source_id or uri, i, float(i))

    def decode_batch(self, refs):
        return np.stack([self.frames[ref.frame_index] for ref in refs])


def _shifting_noise_frames(n: int, h: int = 40, w: int = 60, seed: int = 3) -> list[np.ndarray]:
    """Textured frames with a real, recoverable 1px/frame shift -- optical flow needs actual
    texture to find, unlike solid color."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h, w + n, 3), dtype=np.uint8)
    return [base[:, i:i + w] for i in range(n)]


def _action_vector(w_pressed: bool = False) -> np.ndarray:
    """One 51-dim DIAMOND action vector. Mouse bins must be strict one-hot even at rest,
    so every vector picks the nearest bin to zero mouse movement."""
    vec = np.zeros(ACTION_DIM, dtype=np.float64)
    if w_pressed:
        vec[dad.KEY_NAMES.index("W")] = 1.0
    mx_idx = int(np.argmin(np.abs(dad.MOUSE_X_POSSIBLES - 0.0)))
    my_idx = int(np.argmin(np.abs(dad.MOUSE_Y_POSSIBLES - 0.0)))
    vec[dad.MX_OFF + mx_idx] = 1.0
    vec[dad.MY_OFF + my_idx] = 1.0
    return vec


ACTION_COLUMNS = [f"a{i}" for i in range(ACTION_DIM)]
SCHEMA = SchemaSpec.from_dict({
    "name": "action-state-example",
    "fields": {"action": {"kind": "vector", "dims": ACTION_COLUMNS}},
})

N_FRAMES = 8
PRESS_AT = 3  # isolated W-press at frame 3, clear of both edges
actions = [_action_vector() for _ in range(N_FRAMES)]
actions[PRESS_AT] = _action_vector(w_pressed=True)
rows = [
    {"t": t, **{col: float(v) for col, v in zip(ACTION_COLUMNS, actions[t], strict=True)}}
    for t in range(N_FRAMES)
]
provider = FakeVideoProvider(_shifting_noise_frames(N_FRAMES))

# A baseline says: pressing W should shift pixels in the +1 direction, by some plausible
# magnitude. Get consistent and inconsistent verdicts by flipping expected_sign.
baselines = {"W": {"1": {"expected_sign": 1, "min": 0.0, "max": 1e9}}}

live_pipeline = (
    input({"clip1": rows}, schema=SCHEMA, fields={"action": ACTION_COLUMNS}, time="t")
    .join_video({"clip1": "fake.mp4"}, provider=provider, on="frame_index")
    .window(4, stride=2)
    .measure("a0_mean", "mean", of="action.a0")
)

# The dataset-mode operator audits written shards, so write frames+actions first.
out_dir = tempfile.mkdtemp(prefix="kuration_ex07_")
spec = OutputSpec.from_dict({
    "shard_length": 4,
    "container": "npz",
    "modalities": {
        "frames": {"dtype": "uint8", "shape": list(FRAME_SHAPE), "source": "media:frame"},
        "actions": {"dtype": "float64", "shape": [ACTION_DIM], "source": "field:action"},
    },
})
live_pipeline.sample("top-k", weights={"a0_mean": 1.0}, n=10).output(out_dir, spec=spec)

adapter = CuratedDatasetAdapter(out_dir)
pipeline = input(adapter=adapter, schema=adapter.schema, sources=adapter.sources())

checked = pipeline.action_state_consistency(
    baselines=baselines,
    controls=["W"],
)
result = checked.metadata["gameplay"]["action_state_consistency"]["result"]

print(f"analyzer: {ANALYZER_NAME}")
print(f"passed:   {result['passed']}")
print(f"summary:  {result['summary']}")
print(f"frame_space: {result['frame_space']} (cross_shard_dedup={result['cross_shard_dedup']})")
assert result["passed"] is True
assert result["summary"]["consistent"] == ["clip1"]

# Flip the expected sign: the same evidence now reads as inconsistent.
bad_baselines = {"W": {"1": {"expected_sign": -1, "min": 0.0, "max": 1e9}}}
bad_checked = pipeline.action_state_consistency(
    baselines=bad_baselines,
    controls=["W"],
)
bad_result = bad_checked.metadata["gameplay"]["action_state_consistency"]["result"]
print(f"\nwith flipped expected_sign -> passed: {bad_result['passed']}, "
      f"summary: {bad_result['summary']}")
assert bad_result["passed"] is False
assert bad_result["summary"]["inconsistent"] == ["clip1"]
