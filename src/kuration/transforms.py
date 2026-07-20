"""Device-neutral bounded batch transforms over segment windows and lazy media."""
from __future__ import annotations

from collections import deque
import inspect
from itertools import islice
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
        if len(set(declared)) != len(declared):
            raise ValueError(f"batch transform registration has duplicate outputs {declared!r}")
        meta = {"outputs": declared, "inputs": tuple(inputs),
                "media_requirements": dict(media_requirements or {}),
                "resources": dict(resources or {})}
        from .registry import default_registry
        # Registry registration can fail (most commonly on a duplicate name).  Publish the
        # companion execution metadata only after that succeeds so discovery and execution
        # cannot observe two different components under the same name.
        default_registry.register_transform(name, component, **metadata)
        _TRANSFORM_META[name] = meta
        return component
    return decorate if factory is None else decorate(factory)


# Concise public spelling; retained beside the explicit name for discoverability.
register_transform = register_batch_transform


def get_transform(name: str, *, registry=None):
    """Return a registered transform factory, loading plugins before failing loudly."""
    from .plugins import load_plugins
    from .registry import default_registry
    registry = registry or default_registry
    if name not in registry.names("transform") and registry is default_registry:
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


def resolve_transform(name, *, using, inputs, batch_size, device, params, schema, registry,
                      available_features=()):
    from .plugins import load_plugins
    implementation = using if isinstance(using, str) and using in registry.names("transform") else name
    from .registry import default_registry
    if implementation not in registry.names("transform") and registry is default_registry:
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
            feature = token.split(":", 1)[1]
            if not feature or feature not in set(available_features):
                raise ValueError(
                    f"transform {implementation!r}: feature input {feature!r} has no earlier "
                    f"producer; available: {sorted(available_features)}")
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


def _binds(callable_, *args, **kwargs) -> bool:
    """Return whether a Python callable accepts arguments without executing user code."""
    try:
        inspect.signature(callable_).bind(*args, **kwargs)
    except TypeError:
        return False
    except (ValueError, AttributeError) as error:
        raise TypeError(
            f"cannot inspect callable {callable_!r}; plugin callables need an inspectable "
            "Python signature"
        ) from error
    return True


def _construct(factory, params, device):
    if _binds(factory, device=device, **params):
        return factory(device=device, **params)
    if _binds(factory, **params):
        return factory(**params)
    if not isinstance(factory, type):
        # A registered transform function is already the runnable component.
        if _binds(factory, {}, {}):
            return factory
    raise TypeError(
        f"transform factory {factory!r} accepts neither (device=..., **params) nor **params"
    )


def _invoke(transform, batch, ctx):
    """Invoke a transform exactly once using its declared one- or two-argument shape."""
    fn = getattr(transform, "transform_batch", transform)
    if _binds(fn, batch, ctx):
        return fn(batch, ctx)
    if _binds(fn, batch):
        return fn(batch)
    raise TypeError(
        f"transform {fn!r} must accept (batch, ctx) or the compatibility form (batch)"
    )


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
    """Populate transform outputs while bounding raw-window memory.

    Segment metadata still represents the logical pool, but ticks and decoded media can be much
    larger. Re-read sources as streams and retain only the overlap needed for one transform batch
    instead of caching every source and every segment window.
    """
    if not calls:
        return segments
    by_source: dict[Any, list] = {}
    for seg in segments:
        by_source.setdefault(seg.source_id, []).append(seg)
    ordered = [s for src in sorted(by_source, key=str) for s in by_source[src]]
    runner = _TransformRunner(calls, adapter=adapter, registry=registry)
    batch_size = max(call["batch_size"] for call in calls)
    from .progress import emit
    emit(progress, "transforms", "start", total=len(calls))

    for source in sorted(by_source, key=str):
        pending_segments, pending_windows = [], []
        source_segments = sorted(by_source[source], key=lambda item: item.start_idx)
        for segment, window in _iter_segment_windows(adapter, source, source_segments):
            pending_segments.append(segment)
            pending_windows.append(window)
            if len(pending_segments) == batch_size:
                runner.apply(pending_segments, pending_windows, progress=None)
                pending_segments, pending_windows = [], []
        if pending_segments:
            runner.apply(pending_segments, pending_windows, progress=None)

    for call_i, call in enumerate(calls, 1):
        emit(progress, "transforms", "advance", current=call_i, total=len(calls),
             source=call["name"], status="ok")
    emit(progress, "transforms", "complete", current=len(calls), total=len(calls))
    return ordered


def _iter_segment_windows(adapter, source, segments):
    """Yield indexed segment windows while retaining at most their active overlap."""
    supports_streaming = getattr(adapter, "supports_streaming", True)
    if callable(supports_streaming):
        supports_streaming = supports_streaming()
    iterator = getattr(adapter, "iter_ticks", None)
    use_iterator = (callable(iterator) and supports_streaming is not False
                    and not getattr(adapter, "in_memory", False))
    if use_iterator:
        assert callable(iterator)
        ticks = iterator(source)
    else:
        ticks = iter(adapter.ticks(source))
    buffered: deque[Any] = deque()
    buffer_start = 0
    position = 0
    for segment in segments:
        if segment.start_idx < buffer_start:
            raise ValueError(f"segments for source {source!r} are not in start-index order")
        try:
            while buffer_start < segment.start_idx and buffered:
                buffered.popleft()
                buffer_start += 1
            while position < segment.start_idx:
                next(ticks)  # discard gaps between selected windows without caching them
                position += 1
                buffer_start = position
            while position < segment.end_idx:
                buffered.append(next(ticks))
                position += 1
        except StopIteration as error:
            raise ValueError(
                f"source {source!r} changed before transforms: segment ends at "
                f"{segment.end_idx}, but only {position} ticks are available") from error
        width = segment.end_idx - segment.start_idx
        yield segment, list(islice(buffered, 0, width))


def apply_transforms_to_windows(calls, segments, windows, *, adapter, registry, progress=None):
    """Apply transforms to already-bounded windows without rereading or caching a source."""
    return _TransformRunner(calls, adapter=adapter, registry=registry).apply(
        segments, windows, progress=progress)


class _TransformRunner:
    """Long-lived resolved transform instances for one physical stage worker."""

    def __init__(self, calls, *, adapter, registry):
        from .media import media_provider_for
        self.calls = tuple(calls)
        self.adapter = adapter
        self.provider = media_provider_for(adapter)
        self.resolved = tuple(
            (call, _construct(registry.get("transform", call["implementation"]),
                              call["params"], call["device"]))
            for call in self.calls)

    def apply(self, segments, windows, *, progress=None):
        if not self.calls:
            return segments
        if len(segments) != len(windows):
            raise ValueError("segments and windows must have identical lengths")
        provider = self.provider
        adapter = self.adapter
        from .progress import emit
        emit(progress, "transforms", "start", total=len(self.calls))
        for call_i, (call, transform) in enumerate(self.resolved, 1):
            size = call["batch_size"]
            for offset in range(0, len(segments), size):
                chunk = segments[offset:offset + size]
                chunk_windows = windows[offset:offset + size]
                batch: dict[str, Any] = {"segment_ids": [
                    (s.source_id, s.start_idx, s.end_idx) for s in chunk]}
                for token in call["inputs"]:
                    kind, arg = token.split(":", 1)
                    if kind == "field":
                        batch[token] = np.stack([_field_window(w, arg) for w in chunk_windows])
                    elif kind == "feature":
                        batch[token] = np.asarray([s.features[arg] for s in chunk])
                    else:
                        if provider is None:
                            raise ValueError(f"transform {call['name']!r} needs a media provider")
                        refs = [[tick.media.get(arg) for tick in w] for w in chunk_windows]
                        if any(ref is None for group in refs for ref in group):
                            raise ValueError(
                                f"transform {call['name']!r}: missing media:{arg} reference")
                        batch[token] = np.stack(
                            [provider.decode_batch(group) for group in refs])
                ctx = {"device": call["device"], "schema": adapter.schema,
                       "transform": call["name"], "seed": call["params"].get("seed")}
                result = _invoke(transform, batch, ctx)
                if not isinstance(result, Mapping):
                    if len(call["outputs"]) != 1:
                        raise ValueError(
                            f"transform {call['name']!r} must return an output mapping")
                    result = {call["outputs"][0]: result}
                for output in call["outputs"]:
                    vals = np.asarray(result.get(output))
                    if vals.shape != (len(chunk),) or not np.all(np.isfinite(vals)):
                        raise ValueError(f"transform {call['name']!r} output {output!r} must be "
                                         f"{len(chunk)} finite scalars, got shape {vals.shape}")
                    for seg, value in zip(chunk, vals):
                        seg.features[output] = float(value)
            emit(progress, "transforms", "advance", current=call_i, total=len(self.calls),
                 source=call["name"], status="ok")
        emit(progress, "transforms", "complete", current=len(self.calls), total=len(self.calls))
        return segments
