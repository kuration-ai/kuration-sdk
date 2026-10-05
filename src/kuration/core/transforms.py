"""Device-neutral bounded batch transforms over segment windows and lazy media."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import inspect
from itertools import islice
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Protocol, runtime_checkable

import numpy as np

_CONTEXT_KEYS = ("device", "schema", "transform", "seed", "ownership", "lookbehind",
                 "lookahead", "owned_lengths", "inputs", "valid_steps")


@dataclass(frozen=True)
class TransformContext(Mapping[str, Any]):
    """Typed metadata and boundary context for one transform batch.

    ``before(token)`` and ``after(token)`` return one optional array per batch row. Source-edge
    rows are ``None``. The mapping interface preserves existing ``ctx["schema"]`` style access.
    Legacy boundary values also remain in the batch mapping during the release-candidate
    compatibility window.

    ``inputs`` maps each input role a transform declared at registration to the token actually
    resolved for this use, in declaration order, so a transform reads its batch through
    ``input_token(role)`` instead of hardcoding registration tokens that a per-use ``inputs=``
    remapping can legitimately replace. ``valid_steps`` gives the real timestep count of every
    batch row, so a ragged batch's ``field:`` padding is observable rather than silent.
    """

    device: str | None
    schema: Any
    transform: str
    seed: Any = None
    ownership: str | None = None
    lookbehind: Mapping[str, tuple[Any | None, ...]] = field(default_factory=dict)
    lookahead: Mapping[str, tuple[Any | None, ...]] = field(default_factory=dict)
    owned_lengths: np.ndarray | None = None
    inputs: Mapping[str, str] = field(default_factory=dict)
    valid_steps: np.ndarray | None = None

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
        if not isinstance(self.inputs, Mapping) or any(
                not isinstance(role, str) or not role
                or not isinstance(token, str) or not token
                for role, token in self.inputs.items()):
            raise ValueError(
                "transform context inputs must map declared input roles to resolved tokens")
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        for name in ("owned_lengths", "valid_steps"):
            values = getattr(self, name)
            if values is None:
                continue
            lengths = np.asarray(values, dtype=np.int64).copy()
            if lengths.ndim != 1 or np.any(lengths < 0):
                raise ValueError(f"transform context {name} must be non-negative 1-D values")
            lengths.setflags(write=False)
            object.__setattr__(self, name, lengths)

    def input_token(self, role: str) -> str:
        """Return the token resolved for one declared input role.

        A per-use ``inputs=`` remapping replaces the tokens a transform declared at
        registration, so an implementation must address its batch by role, not by the
        registration token.
        """
        try:
            return self.inputs[role]
        except KeyError as error:
            raise KeyError(f"transform {self.transform!r} declared no input role {role!r}; "
                           f"declared roles: {sorted(self.inputs)}") from error

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
        if key not in set(_CONTEXT_KEYS):
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(_CONTEXT_KEYS)

    def __len__(self) -> int:
        return len(_CONTEXT_KEYS)


@runtime_checkable
class BatchTransform(Protocol):
    output_features: tuple[str, ...]
    def transform_batch(self, batch: Mapping[str, Any],
                        ctx: TransformContext) -> Mapping[str, Any]: ...


_TRANSFORM_META: dict[str, dict] = {}


def _validated_boundary(label: str, value, roles, *, prefix: str):
    """Validate one boundary declaration: a shared count, or one count per input role.

    A per-role mapping keeps a transform from paying for context it never reads — the runner
    only extracts (and, for media, only decodes) the roles that declare a nonzero amount.
    """
    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"{prefix} {label} mapping must name at least one input")
        for role, amount in value.items():
            if not isinstance(role, str) or not role:
                raise ValueError(f"{prefix} {label} keys must be declared input tokens")
            if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
                raise ValueError(f"{prefix} {label} must be a non-negative integer")
        unknown = sorted(set(value) - set(roles)) if roles else []
        if unknown:
            raise ValueError(f"{prefix} {label} names undeclared input(s) {unknown}")
        return dict(value)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{prefix} {label} must be a non-negative integer")
    return value


def register_batch_transform(name: str, factory: Callable | None = None, *, outputs=(),
                             inputs=(), media_requirements=None, resources=None,
                             lookbehind: int | Mapping[str, int] = 0,
                             lookahead: int | Mapping[str, int] = 0,
                             ownership: str | None = None, **metadata):
    """Register a transform factory, directly or as a decorator.

    ``outputs`` is mandatory unless the component declares ``output_features``/``outputs``.
    ``lookbehind``/``lookahead`` are either one count shared by every input or a
    ``{declared input token: count}`` mapping for context only some inputs need.
    Framework imports belong inside the factory or its execution method.
    """
    def decorate(component):
        for label, value in (("lookbehind", lookbehind), ("lookahead", lookahead)):
            _validated_boundary(label, value, tuple(inputs), prefix="batch transform")
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
    for label in ("lookbehind", "lookahead"):
        # A component may declare either one count or a per-input mapping as an attribute; the
        # mapping form must survive here to be validated (and resolved) by resolve_transform.
        declared = getattr(component, label, 0)
        meta.setdefault(label, declared if isinstance(declared, Mapping) else int(declared))
    meta.setdefault("ownership", getattr(component, "ownership", None))
    return meta


def resolve_stream_inputs(meta, *, inputs, batch_size, device, schema, label, kind,
                          available_features=()):
    """Resolve one bounded-batch component's declared inputs and boundary context.

    Shared by :func:`resolve_transform` and the analyzer seam
    (:func:`kuration.core.analyzers.resolve_analyzer`) so both reach the *same* batch
    construction, media decode, and boundary-context machinery. ``label`` names the component
    in errors (``"transform 'acme/x'"``) and ``kind`` is the component kind word.
    """
    if meta["ownership"] not in {None, "rightmost-start"}:
        raise ValueError(f"{label} ownership must be None or 'rightmost-start'")
    declared_roles = tuple(meta["inputs"])
    resolved_inputs = tuple(inputs or declared_roles)
    if not resolved_inputs:
        raise ValueError(f"{label} needs at least one input")
    # A per-use remapping substitutes tokens for the roles the component declared, so it must
    # supply exactly one token of the same kind per role: the implementation reads its batch by
    # role (ctx.input_token) and a silently dropped or re-kinded role would only fail mid-run.
    if declared_roles and len(resolved_inputs) != len(declared_roles):
        raise ValueError(
            f"{label} declares {len(declared_roles)} input role(s) "
            f"{list(declared_roles)}; inputs= must supply one token per role, got "
            f"{list(resolved_inputs)}")
    input_roles = dict(zip(declared_roles or resolved_inputs, resolved_inputs, strict=True))
    for role, token in input_roles.items():
        token_kind = token.split(":", 1)[0] if ":" in token else None
        # A token with no usable prefix is reported by the token validation below, which owns
        # that message; this check is only about a role being remapped across kinds.
        if token_kind in {"field", "feature", "media"} and token_kind != role.split(":", 1)[0]:
            raise ValueError(f"{label}: input role {role!r} cannot be "
                             f"remapped to {token!r} of a different kind")
    boundary = {}
    for name in ("lookbehind", "lookahead"):
        declared = _validated_boundary(name, meta[name], declared_roles, prefix=label)
        if isinstance(declared, dict):
            boundary[name] = {input_roles[role]: amount
                              for role, amount in declared.items() if amount}
        else:
            boundary[name] = ({token: declared for token in resolved_inputs}
                              if declared else {})
    for role, token in input_roles.items():
        if token.startswith("media:"):
            column = token.split(":", 1)[1]
            if column not in schema.media:
                raise ValueError(f"{label}: media column {column!r} "
                                 f"is unavailable; declared: {sorted(schema.media)}")
            # Per-input requirements are keyed by the declared role, so a remapped token is
            # still held to the semantics its role asked for.
            req = meta["media_requirements"].get(role, meta["media_requirements"])
            if isinstance(req, dict):
                spec = schema.media[column]
                for key in ("kind", "layout", "color_space", "timing", "dtype"):
                    if key in req and getattr(spec, key) != req[key]:
                        raise ValueError(f"{label}: {token} requires "
                                         f"{key}={req[key]!r}, got {getattr(spec, key)!r}")
        elif token.startswith("field:"):
            field = token.split(":", 1)[1]
            if field not in schema.fields and field not in schema.globals:
                raise ValueError(f"{label}: unknown field {field!r}")
        elif token.startswith("feature:"):
            feature = token.split(":", 1)[1]
            if not feature or feature not in set(available_features):
                raise ValueError(
                    f"{label}: feature input {feature!r} has no earlier "
                    f"producer; available: {sorted(available_features)}")
        else:
            raise ValueError(f"{kind} input {token!r} must use field:/feature:/media:")
    if batch_size is None:
        batch_size = meta["resources"].get("preferred_batch_size", 1)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError(f"{kind} batch_size must be a positive integer")
    if not isinstance(device, str) or not device:
        raise ValueError(f"{kind} device must be a non-empty string")
    # "lookbehind"/"lookahead" stay the single tick counts every caller (window retention,
    # planning, provenance) already reads; the *_tokens maps carry the per-input detail.
    return {"inputs": resolved_inputs, "input_roles": input_roles,
            "batch_size": batch_size, "device": device,
            "resources": meta["resources"],
            "lookbehind": max(boundary["lookbehind"].values(), default=0),
            "lookahead": max(boundary["lookahead"].values(), default=0),
            "lookbehind_tokens": boundary["lookbehind"],
            "lookahead_tokens": boundary["lookahead"],
            "ownership": meta["ownership"]}


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
    resolved = resolve_stream_inputs(
        meta, inputs=inputs, batch_size=batch_size, device=device, schema=schema,
        label=f"transform {implementation!r}", kind="transform",
        available_features=available_features)
    factory_params = dict(params)
    if using is not None and implementation == name:
        factory_params["using"] = using
    return {"name": name, "implementation": implementation, "outputs": outputs,
            "params": factory_params, **resolved}


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


def _field_window(ticks, field, schema, *, buffer_cache=None):
    """Build one schema-shaped transform input without roster-dependent rank changes.

    ``build_window`` builds every schema field for the whole window, not just ``field`` --
    it's documented as "the amortization point: one buffer per segment, shared by all of a
    profile's metrics". Measures already get that amortization; transforms didn't, so a job
    with N transforms reading fields rebuilt the same whole-window buffer N times per window.
    ``buffer_cache``, keyed by the window's identity, lets `build_batch` share one build across
    every transform in the same batch pass.
    """
    if field in schema.globals:
        return np.asarray(
            [tick.globals.get(field, np.nan) for tick in ticks], dtype=np.float64)
    if field not in schema.fields:
        raise ValueError(f"transform input field {field!r} is not declared by the schema")

    spec = schema.fields[field]
    if spec.kind != "categorical":
        from .buffer import build_window
        # Numeric entity fields always use (T,E,D), including a one-entity scalar field.
        if buffer_cache is None:
            return build_window(schema, list(ticks)).numeric_field(field)
        key = id(ticks)
        buf = buffer_cache.get(key)
        if buf is None:
            buf = build_window(schema, list(ticks))
            buffer_cache[key] = buf
        return buf.numeric_field(field)

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
    """Pad variable time/entity extents before adding the transform batch dimension.

    Padding is NaN (``None`` for object dtype) and every transform is told how many timesteps
    each row really has, through ``ctx.valid_steps`` and the ``valid_steps`` batch entry, so a
    ragged batch cannot silently average pad cells into a measurement.
    """
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


def apply_transforms(calls, segments, *, adapter, registry, progress=None,
                     on_error="raise", errors=None):
    """Populate transform outputs while bounding raw-window memory.

    Segment metadata still represents the logical pool, but ticks and decoded media can be much
    larger. Re-read sources as streams and retain only the overlap needed for one transform batch
    instead of caching every source and every segment window. With ``on_error="skip"``, a
    transform failure removes that source's entire segment set and appends a portable diagnostic
    to ``errors`` when supplied, matching the feature and bounded-execution error policy.
    """
    if not calls:
        return segments
    if on_error not in {"raise", "skip"}:
        raise ValueError("on_error must be 'raise' or 'skip'")
    by_source: dict[Any, list] = {}
    for seg in segments:
        by_source.setdefault(seg.source_id, []).append(seg)
    runner = _TransformRunner(calls, adapter=adapter, registry=registry)
    batch_size = max(call["batch_size"] for call in calls)
    lookbehind = max((call.get("lookbehind", 0) for call in calls), default=0)
    lookahead = max((call.get("lookahead", 0) for call in calls), default=0)
    from .progress import emit
    emit(progress, "transforms", "start", total=len(calls))

    successful = []
    failures = 0
    for source in sorted(by_source, key=str):
        source_segments = sorted(by_source[source], key=lambda item: item.start_idx)
        try:
            pending_segments, pending_windows, pending_contexts = [], [], []
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
        except Exception as error:
            if on_error == "raise":
                raise
            import sys
            from .scoring import _redact_diagnostic

            failures += 1
            # A component may keep mutable/model state. Once a source raises, do not let
            # partially mutated state contaminate the next source that the skip policy retains.
            runner = _TransformRunner(calls, adapter=adapter, registry=registry)
            safe_source = _redact_diagnostic(source)
            safe_error = _redact_diagnostic(repr(error))
            print(f"  [skip] transform source {safe_source!r}: {safe_error}", file=sys.stderr)
            if errors is not None:
                errors.append({
                    "source": safe_source,
                    "error": safe_error,
                    "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
                    "message": _redact_diagnostic(str(error)),
                    "phase": "transforms",
                })
            continue
        successful.extend(source_segments)

    if failures and isinstance(segments, list):
        segments[:] = successful
    if failures and not successful:
        raise RuntimeError(
            "segment pool is EMPTY: every source failed transform execution")

    for call_i, call in enumerate(calls, 1):
        emit(progress, "transforms", "advance", current=call_i, total=len(calls),
             source=call["name"], status="partial" if failures else "ok")
    emit(progress, "transforms", "complete", current=len(calls), total=len(calls),
         detail=f"{failures} source(s) skipped" if failures else None)
    return successful


def source_tick_stream(adapter, source):
    """Return the laziest tick iterator an adapter supports for one source.

    Streaming-capable, file-backed adapters expose ``iter_ticks``; in-memory and
    materializing adapters fall back to ``ticks``. Every bounded consumer (transform runner,
    analyzer stream driver) resolves the stream here so none of them re-decides it.
    """
    supports_streaming = getattr(adapter, "supports_streaming", True)
    if callable(supports_streaming):
        supports_streaming = supports_streaming()
    iterator = getattr(adapter, "iter_ticks", None)
    if (callable(iterator) and supports_streaming is not False
            and not getattr(adapter, "in_memory", False)):
        return iterator(source)
    return iter(adapter.ticks(source))


def _iter_segment_windows(adapter, source, segments, *, lookbehind=0, lookahead=0):
    """Yield indexed segment windows while retaining at most their active overlap."""
    ticks = source_tick_stream(adapter, source)
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


def build_batch(call, segments, windows, contexts, *, adapter, provider, buffer_cache=None):
    """Build one bounded input batch and its typed context for a resolved component call.

    The single place ``field:``/``feature:``/``media:`` tokens become arrays: field windows are
    padded with an honest ``valid_steps`` companion, media references are decoded through the
    adapter's provider in one bounded ``decode_batch`` per token, and declared per-token
    lookbehind/lookahead context is extracted only for the roles that asked for it. Both the
    transform runner and the analyzer stream driver go through here, so there is exactly one
    decode path.

    ``buffer_cache``, when the caller is applying more than one component to the same windows
    (`_TransformRunner.apply`), lets every component's ``field:`` tokens share one
    `build_window` per window instead of rebuilding it per component.
    """
    label = call.get("label") or f"transform {call['name']!r}"
    lookbehind_inputs: dict[str, tuple[Any | None, ...]] = {}
    lookahead_inputs: dict[str, tuple[Any | None, ...]] = {}
    # Field windows are padded to the batch's widest extent, so every component gets the real
    # per-row timestep count and can ignore the pad instead of averaging it.
    valid_steps = np.asarray([len(window) for window in windows], dtype=np.int64)
    batch: dict[str, Any] = {
        "segment_ids": [(s.source_id, s.start_idx, s.end_idx) for s in segments],
        "valid_steps": valid_steps}
    owned_lengths = None
    if call.get("ownership") == "rightmost-start":
        owned_lengths = np.asarray([context["owned_length"] for context in contexts],
                                   dtype=np.int64)
        batch["owned_lengths"] = owned_lengths
    for token in call["inputs"]:
        kind, arg = token.split(":", 1)
        if kind == "field":
            batch[token] = _stack_field_windows(
                [_field_window(window, arg, adapter.schema, buffer_cache=buffer_cache)
                 for window in windows])
        elif kind == "feature":
            batch[token] = np.asarray([s.features[arg] for s in segments])
        else:
            if provider is None:
                raise ValueError(f"{label} needs a media provider")
            refs = [[tick.media.get(arg) for tick in window] for window in windows]
            if any(ref is None for group in refs for ref in group):
                raise ValueError(f"{label}: missing media:{arg} reference")
            batch[token] = np.stack([provider.decode_batch(group) for group in refs])
        if kind == "feature":
            continue
        for direction in ("lookbehind", "lookahead"):
            # Per-input counts avoid extracting (and decoding) boundary context for an input
            # the component never reads; a plain integer declaration still applies to every
            # input.
            per_token = call.get(f"{direction}_tokens")
            amount = (per_token.get(token, 0) if per_token is not None
                      else call.get(direction, 0))
            if not amount:
                continue
            tick_groups = []
            for context in contexts:
                ticks = context[direction]
                ticks = ticks[-amount:] if direction == "lookbehind" else ticks[:amount]
                tick_groups.append(ticks)
            if kind == "field":
                values = [_field_window(group, arg, adapter.schema) if group else None
                          for group in tick_groups]
            else:
                if provider is None:
                    raise ValueError(f"{label} needs a media provider")
                ref_groups = [[tick.media.get(arg) for tick in group] for group in tick_groups]
                if any(ref is None for group in ref_groups for ref in group):
                    raise ValueError(f"{label}: missing media:{arg} context")
                values = [provider.decode_batch(group) if group else None
                          for group in ref_groups]
            batch[f"{direction}:{token}"] = values
            target = lookbehind_inputs if direction == "lookbehind" else lookahead_inputs
            target[token] = tuple(values)
    ctx = TransformContext(
        device=call["device"], schema=adapter.schema, transform=call["name"],
        seed=(call.get("params") or {}).get("seed"), ownership=call.get("ownership"),
        lookbehind=lookbehind_inputs, lookahead=lookahead_inputs,
        owned_lengths=owned_lengths,
        inputs=call.get("input_roles") or {token: token for token in call["inputs"]},
        valid_steps=valid_steps)
    return batch, ctx


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
        # Scoped to this one `apply()` invocation (the same bounded `windows` batch every call
        # below shares), not to the whole source -- `build_window` is expensive enough that
        # rebuilding it per call made cost scale with the number of transforms regardless of
        # what they computed, but caching it across the WHOLE source would defeat the bounded-
        # memory point of batching in the first place.
        buffer_cache: dict[int, Any] = {}
        for call_i, (call, transform) in enumerate(self.resolved, 1):
            size = call["batch_size"]
            for offset in range(0, len(segments), size):
                chunk = segments[offset:offset + size]
                chunk_windows = windows[offset:offset + size]
                chunk_contexts = contexts[offset:offset + size]
                batch, ctx = build_batch(call, chunk, chunk_windows, chunk_contexts,
                                         adapter=adapter, provider=provider,
                                         buffer_cache=buffer_cache)
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
                records = result.get("records")
                if records is not None:
                    if len(records) != len(chunk):
                        raise ValueError(
                            f"transform {call['name']!r} records must have {len(chunk)} "
                            f"entries (one per segment), got {len(records)}")
                    for seg, row in zip(chunk, records):
                        seg.records[call["name"]] = row
            emit(progress, "transforms", "advance", current=call_i, total=len(self.calls),
                 source=call["name"], status="ok")
        emit(progress, "transforms", "complete", current=len(self.calls), total=len(self.calls))
        return segments
