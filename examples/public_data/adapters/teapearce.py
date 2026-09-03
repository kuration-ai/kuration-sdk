"""Adapter for TeaPearce/CounterStrike_Deathmatch frame-keyed HDF5 recordings.

The upstream files do not use the SDK's ordinary array-per-field HDF5 convention. Instead,
every timestep is stored under four keys such as ``frame_42_y`` and
``frame_42_helperarr``. This example adapter translates that public contract into canonical
Ticks without adding Counter-Strike knowledge to the SDK core.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from kuration import SchemaSpec, Tick, local_copy


KEY_NAMES = ("w", "a", "s", "d", "space", "ctrl", "shift", "1", "2", "3", "r")
MOUSE_X_VALUES = (-1000.0, -500.0, -300.0, -200.0, -100.0, -60.0, -30.0, -20.0,
                  -10.0, -4.0, -2.0, 0.0, 2.0, 4.0, 10.0, 20.0, 30.0, 60.0,
                  100.0, 200.0, 300.0, 500.0, 1000.0)
MOUSE_Y_VALUES = (-200.0, -100.0, -50.0, -20.0, -10.0, -4.0, -2.0, 0.0, 2.0,
                  4.0, 10.0, 20.0, 50.0, 100.0, 200.0)
ACTION_DIMS = (
    *(f"key_{name}" for name in KEY_NAMES),
    "left_click", "right_click",
    *(f"mouse_x_{value:g}" for value in MOUSE_X_VALUES),
    *(f"mouse_y_{value:g}" for value in MOUSE_Y_VALUES),
)
ACTION_WIDTH = len(ACTION_DIMS)
SAMPLE_HZ = 16.0
_TARGET_KEY = re.compile(r"^frame_(\d+)_y$")


def teapearce_schema() -> SchemaSpec:
    """Schema for action curation; raw images remain an optional output-only modality."""
    return SchemaSpec.from_dict({
        "name": "teapearce-counterstrike-deathmatch",
        "entity_axis": "player",
        "sample_hz": SAMPLE_HZ,
        "fields": {
            "action": {"kind": "vector", "dims": list(ACTION_DIMS)},
            "keys_down": {"kind": "scalar"},
            "left_click": {"kind": "scalar"},
            "right_click": {"kind": "scalar"},
            "mouse_dx": {"kind": "scalar"},
            "mouse_dy": {"kind": "scalar"},
            "mouse_motion": {"kind": "scalar"},
            "kill": {"kind": "scalar"},
            "death": {"kind": "scalar"},
        },
        "event_streams": {
            "combat": {"fields": {"kind": {"kind": "categorical"}}},
        },
    })


def decode_action(target) -> dict:
    """Decode the upstream 51-value target while preserving the original vector."""
    action = np.asarray(target, dtype=np.float32).reshape(-1)
    if action.shape != (ACTION_WIDTH,):
        raise ValueError(
            f"TeaPearce frame target must contain {ACTION_WIDTH} values, got {action.shape}")

    x_start = len(KEY_NAMES) + 2
    y_start = x_start + len(MOUSE_X_VALUES)
    mouse_dx = MOUSE_X_VALUES[int(np.argmax(action[x_start:y_start]))]
    mouse_dy = MOUSE_Y_VALUES[int(np.argmax(action[y_start:]))]
    return {
        "action": action,
        "keys_down": float(action[:len(KEY_NAMES)].sum()),
        "left_click": float(action[len(KEY_NAMES)]),
        "right_click": float(action[len(KEY_NAMES) + 1]),
        "mouse_dx": float(mouse_dx),
        "mouse_dy": float(mouse_dy),
        "mouse_motion": float(math.hypot(mouse_dx, mouse_dy)),
    }


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "the TeaPearce example needs h5py: pip install \"kuration-sdk[array]\"") from exc
    return h5py


def _frame_indices(h5file) -> list[int]:
    indices = sorted(int(match.group(1)) for key in h5file.keys()
                     if (match := _TARGET_KEY.match(key)))
    if not indices:
        raise ValueError("no TeaPearce frame_<i>_y datasets found")
    expected = list(range(indices[-1] + 1))
    if indices != expected:
        raise ValueError("TeaPearce frame indices must be contiguous and start at zero; "
                         f"found {indices[:3]}...{indices[-3:]}")
    return indices


class TeaPearceHDF5Adapter:
    """Translate one upstream HDF5 recording into one canonical player Tick stream."""

    def __init__(self, schema: SchemaSpec | None = None, *, include_frames: bool = False):
        self.schema = schema or teapearce_schema()
        self.include_frames = include_frames

    def check_source(self, source_uri: str) -> None:
        h5py = _h5py()
        with local_copy(source_uri) as local_path, h5py.File(local_path, "r") as h5file:
            indices = _frame_indices(h5file)
            for index in indices:
                required = [f"frame_{index}_y", f"frame_{index}_helperarr"]
                if self.include_frames:
                    required.append(f"frame_{index}_x")
                missing = [key for key in required if key not in h5file]
                if missing:
                    raise ValueError(f"{source_uri!r} is missing datasets {missing}")
            if h5file[f"frame_{indices[0]}_y"].shape != (ACTION_WIDTH,):
                raise ValueError(f"{source_uri!r}: frame target shape must be ({ACTION_WIDTH},)")
            if h5file[f"frame_{indices[0]}_helperarr"].shape != (2,):
                raise ValueError(f"{source_uri!r}: helper array shape must be (2,)")

    def frame_shape(self, source_uri: str) -> tuple[int, ...]:
        h5py = _h5py()
        with local_copy(source_uri) as local_path, h5py.File(local_path, "r") as h5file:
            index = _frame_indices(h5file)[0]
            key = f"frame_{index}_x"
            if key not in h5file:
                raise ValueError(f"{source_uri!r} is missing dataset {key!r}")
            return tuple(h5file[key].shape)

    def ticks(self, source_uri: str) -> list[Tick]:
        return list(self.iter_ticks(source_uri))

    def iter_ticks(self, source_uri: str):
        h5py = _h5py()
        with local_copy(source_uri) as local_path, h5py.File(local_path, "r") as h5file:
            for index in _frame_indices(h5file):
                fields = decode_action(h5file[f"frame_{index}_y"][:])
                helper = np.asarray(h5file[f"frame_{index}_helperarr"][:]).reshape(-1)
                if helper.shape != (2,):
                    raise ValueError(f"frame_{index}_helperarr must have shape (2,)")
                fields["kill"] = float(helper[0])
                fields["death"] = float(helper[1])
                if self.include_frames:
                    fields["frame"] = np.asarray(h5file[f"frame_{index}_x"][:])

                combat = []
                if helper[0]:
                    combat.append({"kind": "kill"})
                if helper[1]:
                    combat.append({"kind": "death"})
                yield Tick(time=index / SAMPLE_HZ, entities={"player": fields},
                           events={"combat": combat})


class FrameEncoder:
    """Expose the adapter's output-only raw frame as an encoder modality."""

    def timestep(self, record: dict, ctx: dict) -> dict:
        frame = record.get("frame")
        return {"frame": np.asarray(frame)} if frame is not None else {}


def write_demo_recording(path: str | Path, *, recording: int, n_frames: int = 256,
                         frame_shape: tuple[int, int, int] = (8, 12, 3)) -> None:
    """Create a tiny structural fixture with no gameplay imagery or upstream data."""
    h5py = _h5py()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5file:
        for index in range(n_frames):
            action = np.zeros(ACTION_WIDTH, dtype=np.float32)
            action[(index // (7 + recording)) % 4] = 1
            for extra_key in range(recording % 3):
                action[4 + extra_key] = 1
            if index % (19 - recording) == 0:
                action[len(KEY_NAMES)] = 1
            x_index = (index // (5 + recording)) % len(MOUSE_X_VALUES)
            y_index = (index // (9 + recording)) % len(MOUSE_Y_VALUES)
            action[len(KEY_NAMES) + 2 + x_index] = 1
            action[len(KEY_NAMES) + 2 + len(MOUSE_X_VALUES) + y_index] = 1

            helper = np.zeros(2, dtype=np.float32)
            if index in {64 + 3 * recording, 192 - 2 * recording}:
                helper[0] = 1
            if index == 128 + recording:
                helper[1] = 1

            # Deliberately non-game imagery: flat 3-channel values only exercise transport.
            frame = np.full(frame_shape, (index + recording) % 255, dtype=np.uint8)
            h5file.create_dataset(f"frame_{index}_x", data=frame)
            h5file.create_dataset(f"frame_{index}_xaux", data=np.zeros(54, dtype=np.float32))
            h5file.create_dataset(f"frame_{index}_y", data=action)
            h5file.create_dataset(f"frame_{index}_helperarr", data=helper)
