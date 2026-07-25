"""Device-neutral bounded batch transforms over segment windows and lazy media."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import inspect
from itertools import islice
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class TransformContext(Mapping[str, Any]):
    """Typed metadata and boundary context for one transform batch.

    ``before(token)`` and ``after(token)`` return one optional array per batch row. Source-edge
    rows are ``None``. The mapping interface preserves existing ``ctx["schema"]`` style access.
    Legacy boundary values also remain in the batch mapping during the release-candidate
    compatibility window.
    """

    device: str | None
    schema: Any
    transform: str
    seed: Any = None
    ownership: str | None = None
    lookbehind: Mapping[str, tuple[Any | None, ...]] = field(default_factory=dict)
    lookahead: Mapping[str, tuple[Any | None, ...]] = field(default_factory=dict)
    owned_lengths: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.device is not None and (not isinstance(self.device, str) or not self.device):
            raise ValueError("transform context device must be a non-empty string or None")
        if not isinstance(self.transform, str) or not self.transform:
            raise ValueError("transform context name must be a non-empty string")
        if self.ownership not in {None, "rightmost-start"}:
            raise ValueError("transform context ownership must be None or 'rightmost-start'")
        for name in ("lookbehind", "lookahead"):
            values = getattr(self, name)
            if not isinstance(values, Mapping) or any(
                    not isinstance(token, str) or not token
                    or not isinstance(rows, (list, tuple))
                    for token, rows in values.items()):
                raise ValueError(f"transform context {name} must map input tokens to row sequences")
            frozen = MappingProxyType({token: tuple(rows) for token, rows in values.items()})
            object.__setattr__(self, name, frozen)
        if self.owned_lengths is not None:
            lengths = np.asarray(self.owned_lengths, dtype=np.int64).copy()
            if lengths.ndim != 1 or np.any(lengths < 0):
                raise ValueError("transform context owned_lengths must be non-negative 1-D values")
            lengths.setflags(write=False)
            object.__setattr__(self, "owned_lengths", lengths)

    def before(self, token: str) -> tuple[Any | None, ...]:
        """Return per-row lookbehind arrays for a declared input token."""
        try:
            return self.lookbehind[token]
        except KeyError as error:
            raise KeyError(f"no lookbehind context was declared for {token!r}") from error

    def after(self, token: str) -> tuple[Any | None, ...]:
        """Return per-row lookahead arrays for a declared input token."""
        try:
            return self.lookahead[token]
        except KeyError as error:
            raise KeyError(f"no lookahead context was declared for {token!r}") from error

    def __getitem__(self, key: str) -> Any:
        if key not in {"device", "schema", "transform", "seed", "ownership",
                       "lookbehind", "lookahead", "owned_lengths"}:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(("device", "schema", "transform", "seed", "ownership",
                     "lookbehind", "lookahead", "owned_lengths"))

    def __len__(self) -> int:
        return 8


@runtime_checkable
class BatchTransform(Protocol):
    output_features: tuple[str, ...]
    def transform_batch(self, batch: Mapping[str, Any],
                        ctx: TransformContext) -> Mapping[str, Any]: ...


_TRANSFORM_META: dict[str, dict] = {}


def register_batch_transform(name: str, factory: Callable | None = None, *, outputs=(),
                             inputs=(), media_requirements=None, resources=None,
                             lookbehind: int = 0, lookahead: int = 0,
                             ownership: str | None = None, **metadata):
    """Register a transform factory, directly or as a decorator.

    ``outputs`` is mandatory unless the component declares ``output_features``/``outputs``.
    Framework imports belong inside the factory or its execution method.
    """
    def decorate(component):
        for label, value in (("lookbehind", lookbehind), ("lookahead", lookahead)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"batch transform {label} must be a non-negative integer")
        if ownership not in {None, "rightmost-start"}:
            raise ValueError("batch transform ownership must be None or 'rightmost-start'")
        declared = tuple(outputs or getattr(component, "output_features", ())
                         or getattr(component, "outputs", ()))
        if not declared or any(not isinstance(x, str) or not x for x in declared):
            raise ValueError("batch transform registration needs non-empty scalar output names")
        if len(set(declared)) != len(declared):
            raise ValueError(f"batch transform registration has duplicate outputs {declared!r}")
        meta = {"outputs": declared, "inputs": tuple(inputs),
                "media_requirements": dict(media_requirements or {}),
                "resources": dict(resources or {}), "lookbehind": lookbehind,
                "lookahead": lookahead, "ownership": ownership}
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
    meta.setdefault("lookbehind", int(getattr(component, "lookbehind", 0)))
    meta.setdefault("lookahead", int(getattr(component, "lookahead", 0)))
    meta.setdefault("ownership", getattr(component, "ownership", None))
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
    for label in ("lookbehind", "lookahead"):
        value = meta[label]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"transform {implementation!r} {label} must be a non-negative integer")
    if meta["ownership"] not in {None, "rightmost-start"}:
        raise ValueError(
            f"transform {implementation!r} ownership must be None or 'rightmost-start'")
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
            "params": factory_params, "resources": meta["resources"],
            "lookbehind": meta["lookbehind"], "lookahead": meta["lookahead"],
            "ownership": meta["ownership"]}


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


def _field_window(ticks, field, schema):
    """Build one schema-shaped transform input without roster-dependent rank changes."""
    if field in schema.globals:
        return np.asarray(
            [tick.globals.get(field, np.nan) for tick in ticks], dtype=np.float64)
    if field not in schema.fields:
        raise ValueError(f"transform input field {field!r} is not declared by the schema")

    spec = schema.fields[field]
    if spec.kind != "categorical":
        from .buffer import build_window
        # Numeric entity fields always use (T,E,D), including a one-entity scalar field.
        return build_window(schema, list(ticks)).numeric_field(field)

    roster: dict[Any, int] = {}
    for tick in ticks:
        for entity in tick.entities:
            roster.setdefault(entity, len(roster))
    values = np.full((len(ticks), max(1, len(roster))), None, dtype=object)
    for tick_index, tick in enumerate(ticks):
        for entity, fields in tick.entities.items():
            if field in fields:
                values[tick_index, roster[entity]] = fields[field]
    return values


def _stack_field_windows(windows):
    """Pad variable time/entity extents before adding the transform batch dimension."""
    if not windows:
        return np.asarray([])
    ndim = windows[0].ndim
    if any(value.ndim != ndim for value in windows):
        raise ValueError("transform field windows changed rank within one batch")
    shape = tuple(max(value.shape[axis] for value in windows) for axis in range(ndim))
    dtype = np.result_type(*(value.dtype for value in windows))
    fill = None if dtype.kind == "O" else np.nan
    out = np.full((len(windows), *shape), fill, dtype=dtype)
    for row, value in enumerate(windows):
        out[(row, *(slice(0, width) for width in value.shape))] = value
    return out


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
    lookbehind = max((call.get("lookbehind", 0) for call in calls), default=0)
    lookahead = max((call.get("lookahead", 0) for call in calls), default=0)
    from .progress import emit
    emit(progress, "transforms", "start", total=len(calls))

    for source in sorted(by_source, key=str):
        pending_segments, pending_windows, pending_contexts = [], [], []
        source_segments = sorted(by_source[source], key=lambda item: item.start_idx)
        for index, (segment, window, before, after) in enumerate(_iter_segment_windows(
                adapter, source, source_segments, lookbehind=lookbehind,
                lookahead=lookahead)):
            next_start = (source_segments[index + 1].start_idx
                          if index + 1 < len(source_segments) else segment.end_idx)
            owned_length = min(segment.end_idx, next_start) - segment.start_idx
            pending_segments.append(segment)
            pending_windows.append(window)
            pending_contexts.append({"lookbehind": before, "lookahead": after,
                                     "owned_length": owned_length})
            if len(pending_segments) == batch_size:
                runner.apply(pending_segments, pending_windows, contexts=pending_contexts,
                             progress=None)
                pending_segments, pending_windows, pending_contexts = [], [], []
        if pending_segments:
            runner.apply(pending_segments, pending_windows, contexts=pending_contexts,
                         progress=None)

    for call_i, call in enumerate(calls, 1):
        emit(progress, "transforms", "advance", current=call_i, total=len(calls),
             source=call["name"], status="ok")
    emit(progress, "transforms", "complete", current=len(calls), total=len(calls))
    return ordered


def _iter_segment_windows(adapter, source, segments, *, lookbehind=0, lookahead=0):
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
        retain_start = max(0, segment.start_idx - lookbehind)
        if retain_start < buffer_start:
            raise ValueError(f"segments for source {source!r} are not in start-index order")
        try:
            while buffer_start < retain_start and buffered:
                buffered.popleft()
                buffer_start += 1
            while position < retain_start:
                next(ticks)  # discard gaps between selected windows without caching them
                position += 1
                buffer_start = position
            while position < segment.end_idx + lookahead:
                buffered.append(next(ticks))
                position += 1
        except StopIteration as error:
            if position < segment.end_idx:
                raise ValueError(
                    f"source {source!r} changed before transforms: segment ends at "
                    f"{segment.end_idx}, but only {position} ticks are available") from error
        width = segment.end_idx - segment.start_idx
        offset = segment.start_idx - buffer_start
        window = list(islice(buffered, offset, offset + width))
        before = list(islice(buffered, max(0, offset - lookbehind), offset))
        after = list(islice(buffered, offset + width, offset + width + lookahead))
        yield segment, window, before, after


def apply_transforms_to_windows(calls, segments, windows, *, adapter, registry, contexts=None,
                                progress=None):
    """Apply transforms to already-bounded windows without rereading or caching a source."""
    return _TransformRunner(calls, adapter=adapter, registry=registry).apply(
        segments, windows, contexts=contexts, progress=progress)


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

    def apply(self, segments, windows, *, contexts=None, progress=None):
        if not self.calls:
            return segments
        if len(segments) != len(windows):
            raise ValueError("segments and windows must have identical lengths")
        needs_context = any(call.get("lookbehind", 0) or call.get("lookahead", 0)
                            or call.get("ownership") for call in self.calls)
        if contexts is None and needs_context:
            raise ValueError("boundary-aware transforms require extracted window contexts")
        if contexts is None:
            contexts = [{"lookbehind": [], "lookahead": [],
                         "owned_length": segment.end_idx - segment.start_idx}
                        for segment in segments]
        if len(contexts) != len(segments):
            raise ValueError("transform contexts and segments must have identical lengths")
        provider = self.provider
        adapter = self.adapter
        from .progress import emit
        emit(progress, "transforms", "start", total=len(self.calls))
        for call_i, (call, transform) in enumerate(self.resolved, 1):
            size = call["batch_size"]
            for offset in range(0, len(segments), size):
                chunk = segments[offset:offset + size]
                chunk_windows = windows[offset:offset + size]
                chunk_contexts = contexts[offset:offset + size]
                lookbehind_inputs: dict[str, tuple[Any | None, ...]] = {}
                lookahead_inputs: dict[str, tuple[Any | None, ...]] = {}
                batch: dict[str, Any] = {"segment_ids": [
                    (s.source_id, s.start_idx, s.end_idx) for s in chunk]}
                owned_lengths = None
                if call.get("ownership") == "rightmost-start":
                    owned_lengths = np.asarray(
                        [context["owned_length"] for context in chunk_contexts], dtype=np.int64)
                    batch["owned_lengths"] = owned_lengths
                for token in call["inputs"]:
                    kind, arg = token.split(":", 1)
                    if kind == "field":
                        batch[token] = _stack_field_windows(
                            [_field_window(w, arg, adapter.schema) for w in chunk_windows])
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
                    if kind == "feature":
                        continue
                    for direction, amount in (("lookbehind", call.get("lookbehind", 0)),
                                              ("lookahead", call.get("lookahead", 0))):
                        if not amount:
                            continue
                        tick_groups = []
                        for context in chunk_contexts:
                            ticks = context[direction]
                            ticks = ticks[-amount:] if direction == "lookbehind" else ticks[:amount]
                            tick_groups.append(ticks)
                        if kind == "field":
                            values = [_field_window(group, arg, adapter.schema) if group else None
                                      for group in tick_groups]
                        else:
                            if provider is None:
                                raise ValueError(
                                    f"transform {call['name']!r} needs a media provider")
                            ref_groups = [[tick.media.get(arg) for tick in group]
                                          for group in tick_groups]
                            if any(ref is None for group in ref_groups for ref in group):
                                raise ValueError(
                                    f"transform {call['name']!r}: missing media:{arg} context")
                            values = [provider.decode_batch(group) if group else None
                                      for group in ref_groups]
                        batch[f"{direction}:{token}"] = values
                        target = lookbehind_inputs if direction == "lookbehind" \
                            else lookahead_inputs
                        target[token] = tuple(values)
                ctx = TransformContext(
                    device=call["device"], schema=adapter.schema, transform=call["name"],
                    seed=call["params"].get("seed"), ownership=call.get("ownership"),
                    lookbehind=lookbehind_inputs, lookahead=lookahead_inputs,
                    owned_lengths=owned_lengths)
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
