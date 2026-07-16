"""Device-neutral bounded batch transforms over segment windows and lazy media."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BatchTransform(Protocol):
    output_features: tuple[str, ...]
    def transform_batch(self, batch: Mapping[str, Any], ctx: dict) -> Mapping[str, Any]: ...


_TRANSFORM_META: dict[str, dict] = {}


def register_batch_transform(name: str, factory: Callable | None = None, *, outputs=(),
                             inputs=(), media_requirements=None, resources=None, **metadata):
    """Register a transform factory, directly or as a decorator.

    ``outputs`` is mandatory unless the component declares ``output_features``/``outputs``.
    Framework imports belong inside the factory or its execution method.
    """
    def decorate(component):
        declared = tuple(outputs or getattr(component, "output_features", ())
                         or getattr(component, "outputs", ()))
        if not declared or any(not isinstance(x, str) or not x for x in declared):
            raise ValueError("batch transform registration needs non-empty scalar output names")
        _TRANSFORM_META[name] = {"outputs": declared, "inputs": tuple(inputs),
                                 "media_requirements": dict(media_requirements or {}),
                                 "resources": dict(resources or {})}
        from .registry import default_registry
        default_registry.register_transform(name, component, **metadata)
        return component
    return decorate if factory is None else decorate(factory)


# Concise public spelling; retained beside the explicit name for discoverability.
register_transform = register_batch_transform


def get_transform(name: str, *, registry=None):
    """Return a registered transform factory, loading plugins before failing loudly."""
    from .plugins import load_plugins
    from .registry import default_registry
    registry = registry or default_registry
    if name not in registry.names("transform"):
        load_plugins()
    return registry.get("transform", name)


def _metadata(name, component):
    meta = dict(_TRANSFORM_META.get(name) or {})
    meta.setdefault("outputs", tuple(getattr(component, "output_features", ())
                                     or getattr(component, "outputs", ())))
    meta.setdefault("inputs", tuple(getattr(component, "inputs", ())))
    meta.setdefault("media_requirements", dict(getattr(component, "media_requirements", {}) or {}))
    meta.setdefault("resources", dict(getattr(component, "resources", {}) or {}))
    return meta


def resolve_transform(name, *, using, inputs, batch_size, device, params, schema, registry):
    from .plugins import load_plugins
    implementation = using if isinstance(using, str) and using in registry.names("transform") else name
    if implementation not in registry.names("transform"):
        load_plugins()
    if implementation not in registry.names("transform"):
        hint = f"; install the plugin that provides {implementation!r}"
        raise ValueError(f"unknown batch transform {implementation!r}; registered: "
                         f"{registry.names('transform')}{hint}")
    component = registry.get("transform", implementation)
    meta = _metadata(implementation, component)
    outputs = tuple(meta["outputs"])
    if not outputs:
        raise ValueError(f"transform {implementation!r} does not declare output_features")
    resolved_inputs = tuple(inputs or meta["inputs"])
    if not resolved_inputs:
        raise ValueError(f"transform {implementation!r} needs at least one input")
    for token in resolved_inputs:
        if token.startswith("media:"):
            column = token.split(":", 1)[1]
            if column not in schema.media:
                raise ValueError(f"transform {implementation!r}: media column {column!r} "
                                 f"is unavailable; declared: {sorted(schema.media)}")
            req = meta["media_requirements"].get(token, meta["media_requirements"])
            if isinstance(req, dict):
                spec = schema.media[column]
                for key in ("kind", "layout", "color_space", "timing", "dtype"):
                    if key in req and getattr(spec, key) != req[key]:
                        raise ValueError(f"transform {implementation!r}: {token} requires "
                                         f"{key}={req[key]!r}, got {getattr(spec, key)!r}")
        elif token.startswith("field:"):
            field = token.split(":", 1)[1]
            if field not in schema.fields and field not in schema.globals:
                raise ValueError(f"transform {implementation!r}: unknown field {field!r}")
        elif token.startswith("feature:"):
            pass
        else:
            raise ValueError(f"transform input {token!r} must use field:/feature:/media:")
    if batch_size is None:
        batch_size = meta["resources"].get("preferred_batch_size", 1)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("transform batch_size must be a positive integer")
    if not isinstance(device, str) or not device:
        raise ValueError("transform device must be a non-empty string")
    factory_params = dict(params)
    if using is not None and implementation == name:
        factory_params["using"] = using
    return {"name": name, "implementation": implementation, "outputs": outputs,
            "inputs": resolved_inputs, "batch_size": batch_size, "device": device,
            "params": factory_params, "resources": meta["resources"]}


def _construct(factory, params, device):
    try:
        return factory(device=device, **params)
    except TypeError as first:
        try:
            return factory(**params)
        except TypeError:
            if isinstance(factory, type):
                raise first
            return factory


def _field_window(ticks, field):
    values = []
    for tick in ticks:
        if field in tick.globals:
            values.append(tick.globals[field])
        else:
            entities = [v[field] for v in tick.entities.values() if field in v]
            values.append(entities[0] if len(entities) == 1 else entities)
    return np.asarray(values)


def apply_transforms(calls, segments, *, adapter, registry, progress=None):
    """Populate transform outputs in place, preserving canonical segment ordering."""
    if not calls:
        return segments
    from .media import media_provider_for
    from .progress import emit
    by_source: dict[Any, list] = {}
    for seg in segments:
        by_source.setdefault(seg.source_id, []).append(seg)
    tick_cache = {src: list(adapter.ticks(src)) for src in by_source}
    provider = media_provider_for(adapter)
    emit(progress, "transforms", "start", total=len(calls))
    for call_i, call in enumerate(calls, 1):
        factory = registry.get("transform", call["implementation"])
        transform = _construct(factory, call["params"], call["device"])
        ordered = [s for src in sorted(by_source, key=str) for s in by_source[src]]
        size = call["batch_size"]
        for offset in range(0, len(ordered), size):
            chunk = ordered[offset:offset + size]
            batch: dict[str, Any] = {"segment_ids": [(s.source_id, s.start_idx, s.end_idx)
                                                      for s in chunk]}
            for token in call["inputs"]:
                kind, arg = token.split(":", 1)
                windows = [tick_cache[s.source_id][s.start_idx:s.end_idx] for s in chunk]
                if kind == "field":
                    batch[token] = np.stack([_field_window(w, arg) for w in windows])
                elif kind == "feature":
                    batch[token] = np.asarray([s.features[arg] for s in chunk])
                else:
                    if provider is None:
                        raise ValueError(f"transform {call['name']!r} needs a media provider")
                    refs = [[tick.media.get(arg) for tick in w] for w in windows]
                    if any(ref is None for group in refs for ref in group):
                        raise ValueError(f"transform {call['name']!r}: missing media:{arg} reference")
                    decoded = [provider.decode_batch(group) for group in refs]
                    batch[token] = np.stack(decoded)
            ctx = {"device": call["device"], "schema": adapter.schema,
                   "transform": call["name"], "seed": call["params"].get("seed")}
            fn = getattr(transform, "transform_batch", transform)
            try:
                result = fn(batch, ctx)
            except TypeError:
                result = fn(batch)
            if not isinstance(result, Mapping):
                if len(call["outputs"]) != 1:
                    raise ValueError(f"transform {call['name']!r} must return an output mapping")
                result = {call["outputs"][0]: result}
            for output in call["outputs"]:
                vals = np.asarray(result.get(output))
                if vals.shape != (len(chunk),) or not np.all(np.isfinite(vals)):
                    raise ValueError(f"transform {call['name']!r} output {output!r} must be "
                                     f"{len(chunk)} finite scalars, got shape {vals.shape}")
                for seg, value in zip(chunk, vals):
                    seg.features[output] = float(value)
        emit(progress, "transforms", "advance", current=call_i, total=len(calls),
             source=call["name"], status="ok")
    emit(progress, "transforms", "complete", current=len(calls), total=len(calls))
    return segments
