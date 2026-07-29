"""`InputAdapter` — the input seam. The one place domain knowledge about a raw format
enters the SDK.

An adapter's contract is deliberately tiny: turn a raw source into an ordered stream of
canonical `Tick`s (parse + group-by-timestep). Everything after that — building the
`(T,E,D)` buffer, segmentation, metrics, selection — is generic SDK code that never knows
which dataset it's looking at. Streaming-capable adapters expose `iter_ticks`; the compatible
`ticks` path materializes a list for callers that need random access.

Adapters are registered by name for declarative job construction. The parallel feature pass
pickles the configured adapter into spawned workers.
"""
from __future__ import annotations

from typing import Callable, Iterable, Protocol, runtime_checkable

import numpy as np

from ..buffer import Tick
from ..schema import SchemaSpec


@runtime_checkable
class InputAdapter(Protocol):
    schema: SchemaSpec
    def ticks(self, source_uri: str) -> Iterable[Tick]: ...


def validate_adapter_availability(adapter, schema: SchemaSpec, *, define: dict | None = None,
                                  tags_spec: dict | None = None, output_spec=None) -> None:
    """Reject plans that depend on schema values a configured adapter cannot emit.

    Third-party adapters that do not declare ``available_fields``/``available_globals`` retain
    the protocol's historical promise that they can emit the full schema. Built-in configured
    adapters expose these sets, allowing a typo or partial mapping to fail before data is read
    instead of becoming an all-missing buffer that reduces to a plausible zero.
    """
    available_fields = getattr(adapter, "available_fields", None)
    available_globals = getattr(adapter, "available_globals", None)
    if available_fields is None and available_globals is None:
        return
    available_fields = set(schema.fields if available_fields is None else available_fields)
    available_globals = set(schema.globals if available_globals is None else available_globals)
    required_fields: set[str] = set()
    required_globals: set[str] = set()
    for spec in (define or {}).values():
        if not isinstance(spec, dict):
            continue
        source = spec.get("source")
        if isinstance(source, str):
            parts = source.split(".")
            if schema.entity_axis is not None and len(parts) >= 2 and parts[0] == schema.entity_axis:
                required_fields.add(parts[1])
            elif source in schema.globals:
                required_globals.add(source)
        filter_expr = spec.get("filter")
        if isinstance(filter_expr, str):
            from ..dsl.grammar import _parse_filter
            field = _parse_filter(filter_expr)[1]
            if field in schema.fields:
                required_fields.add(field)
    for source in ((tags_spec or {}).get("from") or ()):
        if source in schema.fields:
            required_fields.add(source)
        elif source in schema.globals:
            required_globals.add(source)
    if output_spec is not None:
        for modality in output_spec.modalities:
            if modality.src_kind == "field":
                required_fields.add(modality.src_arg)
    missing_fields = sorted(required_fields - available_fields)
    missing_globals = sorted(required_globals - available_globals)
    if missing_fields or missing_globals:
        details = []
        if missing_fields:
            details.append(f"fields {missing_fields}")
        if missing_globals:
            details.append(f"globals {missing_globals}")
        raise ValueError(f"{type(adapter).__name__} does not map required " + " and ".join(details))


def iter_arrays_to_ticks(schema: SchemaSpec, arrs: dict, gl: dict = None):
    """Lazily build single-entity ticks from indexable per-field arrays."""
    gl = gl or {}
    overlap = sorted(set(arrs) & set(gl))
    if overlap:
        raise ValueError(f"arrays_to_ticks: names cannot be both fields and globals: {overlap}")
    arrays = {**arrs, **gl}

    def step_count(array) -> int:
        # Zarr 3 arrays deliberately do not implement ``len``. Array stores do expose
        # ``shape`` as metadata, while ordinary Python sequences still use ``len``.
        shape = getattr(array, "shape", None)
        return int(shape[0]) if shape else (0 if shape == () else len(array))

    n = step_count(next(iter(arrays.values()))) if arrays else 0
    mismatched = {k: step_count(a) for k, a in arrays.items() if step_count(a) != n}
    if mismatched:                       # silent truncation to the first field's length is worse
        raise ValueError(f"arrays_to_ticks: length mismatch — expected {n} steps, got "
                         f"{mismatched}")
    import math
    for fn, arr in arrs.items():
        if fn not in schema.fields:
            raise ValueError(f"arrays_to_ticks: undeclared schema field {fn!r}")
        shape = tuple(getattr(arr, "shape", np.asarray(arr).shape))
        width = math.prod(shape[1:]) if shape else 0
        expected = schema.fields[fn].width
        if width != expected:
            raise ValueError(f"arrays_to_ticks: field {fn!r} has per-step width {width}, "
                             f"expected {expected} for {schema.fields[fn].kind}")
    for gn, arr in gl.items():
        if gn not in schema.globals:
            raise ValueError(f"arrays_to_ticks: undeclared schema global {gn!r}")
        shape = tuple(getattr(arr, "shape", np.asarray(arr).shape))
        width = math.prod(shape[1:]) if shape else 0
        if width != 1:
            raise ValueError(f"arrays_to_ticks: global {gn!r} has per-step width {width}, "
                             "expected 1")
    for t in range(n):
        ent: dict[str, object] = {}
        for fn, arr in arrs.items():
            fs = schema.fields[fn]
            x = arr[t]
            if fs.kind == "vector":
                ent[fn] = tuple(float(v) for v in np.ravel(x))
            elif fs.kind == "categorical":
                ent[fn] = x.decode() if isinstance(x, bytes) else str(x)
            else:
                ent[fn] = float(np.ravel(x)[0]) if np.ndim(x) else float(x)
        g = {gn: float(np.ravel(arr[t])[0]) for gn, arr in gl.items()}
        yield Tick(time=t, entities={0: ent}, globals=g)


def arrays_to_ticks(schema: SchemaSpec, arrs: dict, gl: dict = None) -> list:
    """Materialized counterpart to `iter_arrays_to_ticks` for existing callers."""
    return list(iter_arrays_to_ticks(schema, arrs, gl))


def check_mapped_arrays(schema: SchemaSpec, fields: dict, globals_: dict, source_uri: str,
                        shape_of: Callable) -> None:
    """Metadata-level mirror of `arrays_to_ticks`'s run-time checks, shared by the array-store
    adapters' `check_source` hooks. `shape_of(key)` returns the stored array's shape, or None
    if the key doesn't exist — shapes are store metadata, so no array data is read. Checks:
    every mapped key exists, all first dims agree (step count), and each field's per-step
    width matches the schema (a wide array mapped to a scalar would silently truncate to its
    first component at run time — loud beats masked)."""
    import math
    overlap = sorted(set(fields) & set(globals_))
    if overlap:
        raise ValueError(f"source {source_uri!r}: names cannot be both fields and globals: "
                         f"{overlap}")
    keys = {**fields, **globals_}                # mapped name -> store key
    shapes, missing = {}, []
    for key in keys.values():
        shp = shape_of(key)
        if shp is None:
            missing.append(key)
        else:
            shapes[key] = tuple(shp)
    if missing:
        raise ValueError(f"source {source_uri!r} is missing mapped key(s) {sorted(missing)}")
    lens = {k: (s[0] if s else 0) for k, s in shapes.items()}
    if len(set(lens.values())) > 1:
        raise ValueError(f"source {source_uri!r}: mapped arrays disagree on step count: {lens}")
    for f, key in fields.items():
        fs = schema.fields[f]
        width = math.prod(shapes[key][1:])
        if width != fs.width:
            raise ValueError(f"source {source_uri!r}: array {key!r} has per-step width {width}, "
                             f"but field {f!r} ({fs.kind}) expects {fs.width}")


_ADAPTERS: dict[str, Callable[..., InputAdapter]] = {}


def register_adapter(name: str, factory: Callable[..., InputAdapter], **metadata) -> None:
    """Register an input-adapter factory under a unique name.

    ``metadata`` becomes its component descriptor. Duplicate names raise ``ValueError``.
    """
    from ..registry import default_registry
    default_registry.register("adapter", name, factory, **metadata)
    _ADAPTERS[name] = factory


def get_adapter(name: str, *, registry=None, **kwargs) -> InputAdapter:
    """Construct a named adapter, loading plugins on demand.

    ``kwargs`` are passed to the factory. An explicit ``registry`` bypasses the legacy global
    adapter table; an unknown name raises ``ValueError``.
    """
    from ..registry import default_registry
    active_registry = registry or default_registry
    if name not in active_registry.names("adapter") and active_registry is default_registry:
        from ..plugins import load_plugins
        load_plugins()
    return active_registry.get("adapter", name)(**kwargs)


def detect_adapter(source_uri: str, *, registry=None) -> str:
    """Return the uniquely highest-confidence registered adapter for a source.

    Adapter factories may expose ``detect_source(uri) -> bool | number``. A numeric result is
    a confidence score; ``True`` is score 1 and falsey values mean no match. Detection is
    intentionally separate from construction because many adapters need inferred schema and
    mapping parameters before they can be instantiated.
    """
    from ..registry import default_registry

    active_registry = registry or default_registry
    if active_registry is default_registry:
        from ..plugins import load_plugins

        load_plugins()
    matches: list[tuple[float, str]] = []
    for name in active_registry.names("adapter"):
        factory = active_registry.get("adapter", name)
        detector = getattr(factory, "detect_source", None)
        if not callable(detector):
            continue
        confidence = detector(str(source_uri))
        if isinstance(confidence, bool):
            score = 1.0 if confidence else 0.0
        elif isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            score = float(confidence)
        else:
            raise ValueError(
                f"adapter {name!r} detect_source() must return a boolean or number"
            )
        if not np.isfinite(score) or score < 0:
            raise ValueError(
                f"adapter {name!r} detect_source() returned invalid confidence {score!r}"
            )
        if score > 0:
            matches.append((score, name))
    if not matches:
        raise ValueError(
            f"no registered input adapter recognized {source_uri!r}; "
            "pass adapter= explicitly or install/register a matching adapter"
        )
    best = max(score for score, _name in matches)
    winners = sorted(name for score, name in matches if score == best)
    if len(winners) > 1:
        raise ValueError(
            f"input adapter detection for {source_uri!r} is ambiguous at confidence "
            f"{best:g}: {winners}; pass adapter= explicitly"
        )
    return winners[0]
