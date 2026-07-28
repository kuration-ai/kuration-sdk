"""Strict JSON decoding shared by CLI documents and persisted metadata readers."""
from __future__ import annotations

import json
from typing import Any


class _NonFiniteJSONConstant(ValueError):
    """Internal marker for Python's otherwise-permissive JSON decoder."""


class _DuplicateJSONKey(ValueError):
    """Internal marker for ambiguous duplicate object names."""


def _reject_json_constant(value: str) -> None:
    raise _NonFiniteJSONConstant(value)


def _unique_json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJSONKey(key)
        value[key] = item
    return value


def parse_json(value: str | bytes | bytearray, *, label: str) -> Any:
    """Decode standards-compliant JSON without duplicate keys or non-finite numbers."""
    try:
        return json.loads(
            value,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {error}") from error
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {error}") from error
    except _NonFiniteJSONConstant as error:
        raise ValueError(
            f"{label} is not valid JSON: non-finite numeric token "
            f"{str(error)!r} is unsupported"
        ) from error
    except _DuplicateJSONKey as error:
        raise ValueError(
            f"{label} is ambiguous JSON: duplicate object key {str(error)!r}"
        ) from error


def parse_json_object(value: str | bytes | bytearray, *, label: str) -> dict:
    """Decode one strict JSON object and reject all other top-level values."""
    parsed = parse_json(value, label=label)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return parsed
