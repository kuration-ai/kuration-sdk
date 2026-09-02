"""Minimal Open P2P protobuf adapter for action/text curation.

Only the upstream fields used by this example are declared in the local protobuf descriptor.
Protocol Buffers safely ignores the remaining fields, avoiding a dependency on Open P2P's full
training stack while retaining wire compatibility with ``annotation.proto``.
"""
from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path

import numpy as np

from kuration import SchemaSpec, Tick, local_copy


SAMPLE_HZ = 20.0


def open_p2p_schema() -> SchemaSpec:
    return SchemaSpec.from_dict({
        "name": "open-p2p-annotations",
        "entity_axis": "player",
        "sample_hz": SAMPLE_HZ,
        "fields": {
            "frame_index": {"kind": "scalar"},
            "keys_down": {"kind": "scalar"},
            "mouse_buttons": {"kind": "scalar"},
            "mouse_dx": {"kind": "scalar"},
            "mouse_dy": {"kind": "scalar"},
            "mouse_motion": {"kind": "scalar"},
            "scroll_motion": {"kind": "scalar"},
            "known_action": {"kind": "scalar"},
            "instruction_present": {"kind": "scalar"},
            "action_source": {"kind": "categorical"},
            "environment": {"kind": "categorical"},
            "instruction_state": {"kind": "categorical"},
        },
    })


def _protobuf_imports():
    try:
        from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    except ImportError as exc:
        raise ImportError(
            "the Open P2P example needs protobuf: pip install \"kuration-sdk[p2p]\"") from exc
    return descriptor_pb2, descriptor_pool, message_factory


@lru_cache(maxsize=1)
def video_annotation_class():
    """Build a wire-compatible subset of Open P2P's VideoAnnotation message."""
    descriptor_pb2, descriptor_pool, message_factory = _protobuf_imports()
    field_type = descriptor_pb2.FieldDescriptorProto
    file_proto = descriptor_pb2.FileDescriptorProto(
        name="kuration_open_p2p_minimal.proto",
        package="kuration.examples.open_p2p",
        syntax="proto3",
    )

    def message(name):
        item = file_proto.message_type.add()
        item.name = name
        return item

    def field(parent, name, number, type_, *, repeated=False, type_name=None):
        item = parent.field.add()
        item.name = name
        item.number = number
        item.label = (field_type.LABEL_REPEATED if repeated else field_type.LABEL_OPTIONAL)
        item.type = type_
        if type_name:
            item.type_name = f".kuration.examples.open_p2p.{type_name}"

    vec = message("Vec2Int")
    field(vec, "x", 1, field_type.TYPE_INT32)
    field(vec, "y", 2, field_type.TYPE_INT32)

    env = message("VideoAnnotationEnv")
    field(env, "env", 1, field_type.TYPE_STRING)
    field(env, "env_subtype", 2, field_type.TYPE_STRING)

    metadata = message("VideoAnnotationMetadata")
    field(metadata, "id", 1, field_type.TYPE_STRING)
    field(metadata, "frames_per_second", 5, field_type.TYPE_FLOAT)
    field(metadata, "env", 6, field_type.TYPE_MESSAGE, type_name="VideoAnnotationEnv")

    mouse = message("MouseAction")
    field(mouse, "mouse_delta_px", 3, field_type.TYPE_MESSAGE, type_name="Vec2Int")
    field(mouse, "scroll_delta_px", 4, field_type.TYPE_MESSAGE, type_name="Vec2Int")
    field(mouse, "buttons_down", 9, field_type.TYPE_STRING, repeated=True)

    keyboard = message("KeyboardAction")
    field(keyboard, "keys", 1, field_type.TYPE_STRING, repeated=True)

    action = message("LowLevelAction")
    field(action, "keyboard", 2, field_type.TYPE_MESSAGE, type_name="KeyboardAction")
    field(action, "is_known", 3, field_type.TYPE_BOOL)
    field(action, "mouse", 4, field_type.TYPE_MESSAGE, type_name="MouseAction")

    text = message("FrameTextAnnotation")
    field(text, "instruction", 1, field_type.TYPE_STRING)
    field(text, "duration", 3, field_type.TYPE_FLOAT)

    frame = message("FrameAnnotation")
    field(frame, "user_action", 1, field_type.TYPE_MESSAGE, type_name="LowLevelAction")
    field(frame, "system_action", 6, field_type.TYPE_MESSAGE, type_name="LowLevelAction")
    field(frame, "frame_text_annotation", 7, field_type.TYPE_MESSAGE,
          repeated=True, type_name="FrameTextAnnotation")
    field(frame, "frame_time", 8, field_type.TYPE_UINT64)

    video = message("VideoAnnotation")
    field(video, "metadata", 2, field_type.TYPE_MESSAGE,
          type_name="VideoAnnotationMetadata")
    field(video, "frame_annotations", 3, field_type.TYPE_MESSAGE,
          repeated=True, type_name="FrameAnnotation")

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_proto)
    descriptor = pool.FindMessageTypeByName("kuration.examples.open_p2p.VideoAnnotation")
    if hasattr(message_factory, "GetMessageClass"):
        return message_factory.GetMessageClass(descriptor)
    return message_factory.MessageFactory(pool).GetPrototype(descriptor)


def parse_annotation(source_uri: str):
    annotation = video_annotation_class()()
    with local_copy(source_uri) as local_path:
        annotation.ParseFromString(Path(local_path).read_bytes())
    return annotation


def _environment(metadata) -> str:
    env = metadata.env.env or "unknown"
    subtype = metadata.env.env_subtype
    return f"{env}/{subtype}" if subtype else env


def _chosen_action(frame):
    """Match upstream training behavior: a known system correction takes precedence."""
    if frame.system_action.is_known:
        return frame.system_action, "system"
    if frame.user_action.is_known:
        return frame.user_action, "user"
    return frame.user_action, "unknown"


class OpenP2PAnnotationAdapter:
    """One ``annotation.proto`` file becomes one canonical player Tick stream."""

    def __init__(self, schema: SchemaSpec | None = None, *, sample_hz: float = SAMPLE_HZ):
        self.schema = schema or open_p2p_schema()
        self.sample_hz = float(sample_hz)

    def check_source(self, source_uri: str) -> None:
        annotation = parse_annotation(source_uri)
        if not annotation.frame_annotations:
            raise ValueError(f"{source_uri!r} contains no frame annotations")
        fps = float(annotation.metadata.frames_per_second)
        if fps and not math.isclose(fps, self.sample_hz, rel_tol=0.05):
            raise ValueError(
                f"{source_uri!r} declares {fps:g} FPS but the schema uses {self.sample_hz:g}; "
                "construct the adapter/schema with the recording rate")

    def ticks(self, source_uri: str) -> list[Tick]:
        return list(self.iter_ticks(source_uri))

    def iter_ticks(self, source_uri: str):
        annotation = parse_annotation(source_uri)
        environment = _environment(annotation.metadata)
        active_instruction = ""
        instruction_until = 0

        for index, frame in enumerate(annotation.frame_annotations):
            for text in frame.frame_text_annotation:
                if text.instruction:
                    active_instruction = text.instruction
                    instruction_until = index + max(1, round(float(text.duration) * self.sample_hz))
                    break
            if index >= instruction_until:
                active_instruction = ""

            action, source = _chosen_action(frame)
            keys = tuple(action.keyboard.keys) if source != "unknown" else ()
            buttons = tuple(action.mouse.buttons_down) if source != "unknown" else ()
            dx = float(action.mouse.mouse_delta_px.x) if source != "unknown" else 0.0
            dy = float(action.mouse.mouse_delta_px.y) if source != "unknown" else 0.0
            scroll_x = float(action.mouse.scroll_delta_px.x) if source != "unknown" else 0.0
            scroll_y = float(action.mouse.scroll_delta_px.y) if source != "unknown" else 0.0
            fields = {
                "frame_index": float(index),
                "keys_down": float(len(keys)),
                "mouse_buttons": float(len(buttons)),
                "mouse_dx": dx,
                "mouse_dy": dy,
                "mouse_motion": float(math.hypot(dx, dy)),
                "scroll_motion": float(math.hypot(scroll_x, scroll_y)),
                "known_action": float(source != "unknown"),
                "instruction_present": float(bool(active_instruction)),
                "action_source": source,
                "environment": environment,
                "instruction_state": "present" if active_instruction else "absent",
                # Output-only strings: an encoder preserves them without making arbitrary
                # cross-game key vocabularies part of the core schema.
                "keys_text": "|".join(keys),
                "buttons_text": "|".join(buttons),
                "instruction_text": active_instruction,
            }
            yield Tick(time=index / self.sample_hz, entities={"player": fields})


class OpenP2PTextEncoder:
    """Preserve variable action labels and instructions as fixed-capacity Unicode arrays."""

    def timestep(self, record: dict, ctx: dict) -> dict:
        return {
            "keys": np.asarray(record.get("keys_text", "")),
            "buttons": np.asarray(record.get("buttons_text", "")),
            "instruction": np.asarray(record.get("instruction_text", "")),
        }


def write_demo_annotation(path: str | Path, *, recording: int, n_frames: int = 160) -> None:
    """Generate a tiny wire-compatible annotation with no video or upstream content."""
    annotation = video_annotation_class()()
    annotation.metadata.id = f"synthetic-{recording}"
    annotation.metadata.frames_per_second = SAMPLE_HZ
    environments = (("Roblox", "Rivals"), ("DOOM", ""), ("Quake", ""),
                    ("Roblox", "Hypershot"))
    annotation.metadata.env.env, annotation.metadata.env.env_subtype = environments[recording % 4]

    for index in range(n_frames):
        frame = annotation.frame_annotations.add()
        frame.frame_time = index
        is_system = recording >= 2 and index % (17 - recording) == 0
        is_unknown = index % 37 == 0 and not is_system
        action = frame.system_action if is_system else frame.user_action
        action.is_known = not is_unknown
        if not is_unknown:
            action.keyboard.keys.extend(["w"] + (["space"] if recording % 2 else []))
            action.mouse.mouse_delta_px.x = ((index * (recording + 1)) % 31) - 15
            action.mouse.mouse_delta_px.y = ((index * 2) % 17) - 8
            if index % (13 - recording) == 0:
                action.mouse.buttons_down.append("left")
        if index in {20 + recording, 100 - recording}:
            text = frame.frame_text_annotation.add()
            text.instruction = ("move toward the objective" if index < 50
                                else "look around and engage")
            text.duration = 1.5 + 0.25 * recording

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(annotation.SerializeToString())
