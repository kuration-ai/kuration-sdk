"""Minimal stdlib PNG encoding for inline frame previews (no imaging dependency).

Modeled on the SDK's proven encoder (``kuration.gameplay._stats._png_data_url``) including
its float-scaling heuristic and preview downscale, but kept local: that module is private,
domain-scoped, and not part of the SDK's public surface.
"""
from __future__ import annotations

import base64
import binascii
import math
import struct
import zlib

import numpy as np

_MAX_PREVIEW = (180, 320)  # (height, width) preview budget before integer downsampling


def _as_rgb_uint8(frame) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] not in {1, 3, 4}:
        raise ValueError(f"frame must have HWC image shape, got {image.shape}")
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.dtype != np.uint8:
        if image.dtype.kind == "f":
            finite = image[np.isfinite(image)]
            peak = float(finite.max()) if finite.size else 0.0
            image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
            if peak <= 1.0:  # normalized [0, 1] frame — clipping alone renders it black
                image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def png_bytes(frame) -> bytes:
    image = _as_rgb_uint8(frame)
    height, width = image.shape[:2]
    step = max(1, math.ceil(max(height / _MAX_PREVIEW[0], width / _MAX_PREVIEW[1])))
    image = np.ascontiguousarray(image[::step, ::step])
    height, width = image.shape[:2]
    color_type = 6 if image.shape[2] == 4 else 2
    raw = b"".join(b"\0" + image[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


def png_data_url(frame) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(frame)).decode("ascii")
