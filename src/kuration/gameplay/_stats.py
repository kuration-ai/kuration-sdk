"""Gameplay enrichment for the core lifecycle stats surface."""
from __future__ import annotations

import base64
import binascii
from collections import Counter
from html import escape
from itertools import islice
import math
import struct
from typing import Any
import zlib

import numpy as np


_SLAM_LABELS = {
    "slam_quality_score": "SLAM quality",
    "slam_tracking_confidence_mean": "Tracking confidence",
    "slam_tracking_gap_fraction": "Tracking gaps",
    "slam_translation_jump_fraction": "Translation jumps",
    "slam_rotation_jump_fraction": "Rotation jumps",
    "slam_step_scale_drift": "Scale drift",
    "slam_motion_efficiency": "Motion efficiency",
    "slam_path_length": "Path length",
    "slam_turn_degrees": "Camera turn",
}

_PIXEL_SHIFT_SUFFIXES = (
    "_shift_min",
    "_shift_max",
    "_shift_mean",
    "_n_instances",
    "_requested",
    "_shift_measured",
)


def _is_action_metric(name: str) -> bool:
    lowered = name.lower()
    return (
        "action" in lowered
        or "control" in lowered
        or lowered.endswith(_PIXEL_SHIFT_SUFFIXES)
    )


def _metric_label(name: str) -> str:
    if name in _SLAM_LABELS:
        return _SLAM_LABELS[name]
    for suffix in _PIXEL_SHIFT_SUFFIXES:
        if name.endswith(suffix):
            control = name.removesuffix(suffix).upper().replace("_", " ")
            measure = suffix.removeprefix("_").replace("_", " ")
            return f"{control} {measure}"
    return name


def _schema_for(subject):
    return getattr(subject, "schema", None)


def _metric_documents(report) -> dict[str, dict]:
    sections = report.get("sections") or {}
    selection = (sections.get("selection") or {}).get("metrics") or {}
    if selection:
        return selection
    return (sections.get("candidates") or {}).get("metrics") or {}


def _mean_pair(value: dict) -> tuple[float | None, float | None]:
    if "candidate" in value or "selected" in value:
        candidate = (value.get("candidate") or {}).get("mean")
        selected = (value.get("selected") or {}).get("mean")
        return candidate, selected
    return value.get("mean"), None


def _formatted_pair(value: dict) -> tuple[str, str]:
    candidate, selected = _mean_pair(value)
    if candidate is None:
        return "—", "mean"
    if selected is None:
        return f"{candidate:.4g}", "candidate mean"
    return f"{candidate:.4g} → {selected:.4g}", "candidate → selected mean"


def gameplay_stats(subject: Any, report) -> dict | None:
    """Return gameplay-specific signals when the subject exposes relevant data."""
    schema = _schema_for(subject)
    media = []
    actions = []
    if schema is not None:
        for name, spec in schema.media.items():
            if spec.kind == "video-frame":
                media.append(
                    {
                        "name": name,
                        "shape": list(spec.shape),
                        "dtype": spec.dtype,
                        "layout": spec.layout,
                        "color_space": spec.color_space,
                        "timing": spec.timing,
                    }
                )
        for name, spec in schema.fields.items():
            lowered = name.lower()
            if (
                "action" in lowered
                or "control" in lowered
                or spec.semantics == "bitmask"
            ):
                actions.append(
                    {
                        "name": name,
                        "kind": spec.kind,
                        "dimensions": list(spec.dims),
                        "semantics": spec.semantics,
                    }
                )

    metrics = _metric_documents(report)
    slam = {name: value for name, value in metrics.items() if name.startswith("slam_")}
    action_metrics = {
        name: value
        for name, value in metrics.items()
        if _is_action_metric(name)
    }
    modalities = (
        ((report.get("sections") or {}).get("contents") or {}).get("modalities") or {}
    )
    video_outputs = [
        name
        for name, value in modalities.items()
        if name.lower() in {"frame", "frames", "video", "image", "rgb"}
        or (
            len(value.get("shape") or ()) == 3
            and (value.get("shape") or [0])[-1] in {1, 3, 4}
        )
    ]
    slam_outputs = [
        name for name in modalities if name.startswith("slam_") or name.startswith("camera_")
    ]
    if not (media or actions or slam or action_metrics or video_outputs or slam_outputs):
        return None

    summary = []
    if media:
        details = ", ".join(
            f"{item['name']} {item['shape']} {item['color_space']}/{item['layout']}"
            for item in media
        )
        summary.append(
            {"label": "Video inputs", "value": len(media), "detail": details}
        )
    if video_outputs:
        rows = sum(int(modalities[name].get("rows", 0)) for name in video_outputs)
        summary.append(
            {
                "label": "Video output",
                "value": f"{rows} frames",
                "detail": ", ".join(video_outputs),
            }
        )
    if actions:
        summary.append(
            {
                "label": "Action fields",
                "value": len(actions),
                "detail": ", ".join(item["name"] for item in actions),
            }
        )
    for name in _SLAM_LABELS:
        if name in slam:
            value, detail = _formatted_pair(slam[name])
            summary.append(
                {"label": _SLAM_LABELS[name], "value": value, "detail": detail}
            )
    if slam_outputs and not slam:
        summary.append(
            {
                "label": "SLAM outputs",
                "value": len(slam_outputs),
                "detail": ", ".join(slam_outputs),
            }
        )
    if action_metrics:
        summary.append(
            {
                "label": "Action signals",
                "value": len(action_metrics),
                "detail": ", ".join(sorted(action_metrics)),
            }
        )
    return {
        "kind": "gameplay",
        "media": media,
        "actions": actions,
        "slam_metrics": sorted(slam),
        "action_metrics": sorted(action_metrics),
        "video_outputs": video_outputs,
        "slam_outputs": slam_outputs,
        "plots": ["gameplay"] if slam or action_metrics else [],
        "summary": summary,
    }


def plot_gameplay_stats(report, *, metric: str | None = None, ax=None):
    """Plot the most decision-relevant gameplay signals from a lifecycle report."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "gameplay stats plotting needs Matplotlib: "
            "pip install 'kuration-sdk[stats]'"
        ) from error

    metrics = _metric_documents(report)
    available = {
        name: value
        for name, value in metrics.items()
        if name.startswith("slam_")
        or _is_action_metric(name)
    }
    if metric is not None:
        if metric not in available:
            raise ValueError(
                f"unknown gameplay metric {metric!r}; available: {sorted(available)}"
            )
        names = [metric]
    else:
        preferred = [name for name in _SLAM_LABELS if name in available]
        names = (preferred or sorted(available))[:6]
    if not names:
        raise ValueError(
            "gameplay stats has no plottable action or SLAM metrics; "
            "use kuration.gameplay.compare_videos(...) for visual media"
        )
    if ax is None:
        figure, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    else:
        figure = ax.figure
    candidate = []
    selected = []
    has_selection = False
    for name in names:
        before, after = _mean_pair(available[name])
        candidate.append(0.0 if before is None else before)
        selected.append(0.0 if after is None else after)
        has_selection = has_selection or after is not None
    positions = list(range(len(names)))
    labels = [_metric_label(name) for name in names]
    if has_selection:
        width = 0.38
        ax.bar([value - width / 2 for value in positions], candidate, width, label="candidate")
        ax.bar([value + width / 2 for value in positions], selected, width, label="selected")
        ax.legend()
    else:
        ax.bar(positions, candidate)
    ax.set_xticks(positions, labels, rotation=25, ha="right")
    ax.set_ylabel("Mean")
    ax.set_title("Gameplay signals")
    return figure


def _png_data_url(frame, *, color_space: str) -> str:
    """Encode one HWC frame as an inline PNG data URL.

    Float frames are scaled by the documented heuristic: a float image whose finite peak is
    ``<= 1.0`` is treated as normalized [0, 1] and multiplied by 255 (clipping alone rendered
    such frames near-black); any wider float range is treated as 0-255 already. Non-finite
    float values become 0. uint8 input is used as-is.
    """
    image = np.asarray(frame)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] not in {1, 3, 4}:
        raise ValueError(f"gameplay frame must have HWC image shape, got {image.shape}")
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.dtype != np.uint8:
        if image.dtype.kind == "f":
            finite = image[np.isfinite(image)]
            peak = float(finite.max()) if finite.size else 0.0
            image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
            if peak <= 1.0:  # normalized [0, 1] frame — clipping alone would render it black
                image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    if color_space.upper() == "BGR" and image.shape[2] >= 3:
        image = image[..., [2, 1, 0, *range(3, image.shape[2])]]
    height, width = image.shape[:2]
    step = max(1, math.ceil(max(height / 180, width / 320)))
    image = np.ascontiguousarray(image[::step, ::step])
    height, width = image.shape[:2]
    color_type = 6 if image.shape[2] == 4 else 2
    raw = b"".join(b"\0" + image[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk("IHDR".encode(), struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
        + chunk("IDAT".encode(), zlib.compress(raw, level=6))
        + chunk("IEND".encode(), b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _provider_for(adapter):
    current = adapter
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        provider = getattr(current, "provider", None)
        if provider is not None:
            return provider
        current = getattr(current, "adapter", None)
    return None


def _bounded_ticks(adapter, source, end: int):
    if hasattr(adapter, "iter_ticks") and not getattr(adapter, "in_memory", False):
        return list(islice(adapter.iter_ticks(source), end))
    return list(islice(iter(adapter.ticks(source)), end))


def _action_summary(schema, ticks) -> str:
    fields = [
        name
        for name, spec in schema.fields.items()
        if "action" in name.lower()
        or "control" in name.lower()
        or spec.semantics == "bitmask"
    ]
    summaries = []
    for name in fields:
        values = [
            entity[name]
            for tick in ticks
            for entity in tick.entities.values()
            if name in entity and entity[name] is not None
        ]
        if not values:
            continue
        spec = schema.fields[name]
        if spec.kind == "categorical":
            top = Counter(map(str, values)).most_common(3)
            summaries.append(
                f"{name}: " + ", ".join(f"{value}×{count}" for value, count in top)
            )
        else:
            active = sum(bool(np.asarray(value).any()) for value in values)
            summaries.append(f"{name}: {active}/{len(values)} active")
    return "; ".join(summaries)


def _decode_frames(pipeline, item: dict, count: int, tick_cache: dict) -> tuple[list, str]:
    source = item.get("source")
    start = item.get("start_idx")
    end = item.get("end_idx")
    if source is None or start is None or end is None:
        return [], ""
    # Cache per source, not per (source, end): keying on `end` re-iterated the source from
    # index 0 for every distinct window end. Re-read only when a deeper window needs more
    # ticks than are cached and the source is not already exhausted.
    cached = tick_cache.get(source)
    if cached is None or (len(cached["ticks"]) < int(end) and not cached["exhausted"]):
        ticks = _bounded_ticks(pipeline.adapter, source, int(end))
        cached = {"ticks": ticks, "exhausted": len(ticks) < int(end)}
        tick_cache[source] = cached
    ticks = cached["ticks"]
    window = ticks[int(start):int(end)]
    if not window:
        return [], ""
    action = _action_summary(pipeline.schema, window)
    media = [
        (name, spec)
        for name, spec in pipeline.schema.media.items()
        if spec.kind == "video-frame"
    ]
    if not media:
        return [], action
    name, spec = media[0]
    values = [tick.media.get(name) for tick in window if tick.media.get(name) is not None]
    if not values:
        return [], action
    indexes = np.linspace(0, len(values) - 1, num=min(count, len(values)), dtype=int)
    chosen = [values[int(index)] for index in indexes]
    if all(isinstance(value, np.ndarray) for value in chosen):
        frames = chosen
    else:
        provider = _provider_for(pipeline.adapter)
        if provider is None:
            return [], action
        frames = list(provider.decode_batch(chosen))
    return [
        _png_data_url(frame, color_space=spec.color_space)
        for frame in frames
    ], action


def render_gameplay_examples(examples) -> str | None:
    """Add bounded gameplay frames, actions, and named SLAM diagnostics."""
    context = examples.__dict__.get("_stats_context") or {}
    pipeline = context.get("pipeline")
    groups = examples.get("groups") or {}
    if pipeline is None:
        has_slam = any(
            name.startswith("slam_")
            for items in groups.values()
            for item in items
            for name in (item.get("features") or {})
        )
        if not has_slam:
            return None
    tick_cache: dict[object, dict] = {}
    cards = []
    for group, items in groups.items():
        for item in items:
            frames: list[str] = []
            action = ""
            error = None
            if pipeline is not None:
                try:
                    frames, action = _decode_frames(
                        pipeline,
                        item,
                        int(examples.get("frames_per_example", 3)),
                        tick_cache,
                    )
                except (ImportError, OSError, ValueError, KeyError) as caught:
                    error = str(caught)
            frame_html = "".join(
                f"<img src=\"{source}\" alt=\"Gameplay evidence frame\">"
                for source in frames
            )
            features = item.get("features") or {}
            slam = [
                f"{_SLAM_LABELS.get(name, name)}={value:.4g}"
                for name, value in features.items()
                if name.startswith("slam_") and isinstance(value, (int, float))
            ][:6]
            pixel_shift = [
                f"{_metric_label(name)}={value:.4g}"
                for name, value in features.items()
                if name.lower().endswith(_PIXEL_SHIFT_SUFFIXES)
                and not name.lower().endswith("_requested")
                and isinstance(value, (int, float))
            ][:8]
            detail = " · ".join(
                [part for part in (action, *slam, *pixel_shift) if part]
            )
            if error:
                detail = (detail + " · " if detail else "") + f"media unavailable: {error}"
            if not frame_html and not detail:
                continue
            location = (
                f"{item.get('source')} [{item.get('start_idx')}, {item.get('end_idx')})"
            )
            cards.append(
                "<article class=\"kur-gameplay-evidence\">"
                f"<header><strong>{escape(str(group))}</strong> "
                f"{escape(location)}</header>"
                f"<div class=\"kur-frame-strip\">{frame_html}</div>"
                f"<p>{escape(detail)}</p>"
                "</article>"
            )
    if not cards:
        return None
    return (
        "<style>.kur-gameplay-grid{display:grid;grid-template-columns:"
        "repeat(auto-fit,minmax(20rem,1fr));gap:.8rem;margin-top:.8rem}"
        ".kur-gameplay-evidence{border:1px solid #8885;border-radius:.55rem;padding:.7rem}"
        ".kur-frame-strip{display:flex;gap:.3rem;margin-top:.5rem;overflow:auto}"
        ".kur-frame-strip img{min-width:0;max-width:32%;height:auto;object-fit:contain;"
        "border-radius:.3rem;background:#111}.kur-gameplay-evidence p{font-size:.86em;"
        "margin:.45rem 0 0;opacity:.85}</style>"
        "<section><h2>Gameplay evidence</h2>"
        f"<div class=\"kur-gameplay-grid\">{''.join(cards)}</div></section>"
    )
