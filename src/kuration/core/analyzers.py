"""Discovery, construction, and driving for stateful, direct-API analyzers.

Unlike a batch transform, an analyzer may accumulate evidence across segments before it
can return a result, so the SDK never schedules one as an implicit selection stage: the
caller owns the instance and its update lifecycle. Two drivers make that lifecycle
reachable without writing the loop by hand, and both are what ``Pipeline.enrich()`` and
``Pipeline.analyze()`` call:

* :func:`run_dataset_analyzer` — the artifact path, driving an analyzer's ``check_dataset``
  hook over a written dataset (also what ``kur-check --analyzer`` uses).
* :func:`run_stream_analyzer` — the tick-stream path, driving an analyzer's ``add_batch``
  hook over segmented raw sources with lazy media, in one bounded pass. It declares its
  inputs with the same ``field:``/``media:`` tokens a batch transform declares and reuses
  the *same* bounded batch/decode machinery (``transforms.build_batch``), so there is one
  media decode path in the SDK, not two.

On the tick-stream path a segment id is the real ``(source_id, start_idx, end_idx)`` of the
source window, so an analyzer's documented cross-window de-duplication works in true source
coordinates. The artifact path cannot offer that (a manifest records no source offset); an
analyzer that measures positions is expected to say so — see
``ActionStateConsistencyChecker.check_dataset``'s frame-space declaration.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class Analyzer(Protocol):
    """Minimum result contract for a user-owned stateful analyzer."""

    def summary(self) -> Any: ...


@runtime_checkable
class DatasetAnalyzer(Analyzer, Protocol):
    """Optional contract for analyzers that can be driven by ``kur-check``."""

    def check_dataset(self, dataset: Any, **options: Any) -> Mapping[str, Any]: ...


@runtime_checkable
class StreamAnalyzer(Analyzer, Protocol):
    """Optional contract for analyzers that can be driven over a raw tick stream.

    ``add_batch`` receives exactly what a batch transform's ``transform_batch`` receives — a
    batch mapping keyed by the analyzer's declared input tokens plus ``segment_ids`` and
    ``valid_steps``, and a :class:`~kuration.core.transforms.TransformContext` whose
    ``input_token(role)``/``before``/``after`` resolve the declared roles — but it accumulates
    state instead of returning per-segment scalars, so it must return ``None``.

    An analyzer may additionally implement ``stream_result()`` returning a mapping (with a
    boolean ``passed``) merged into the driver's result, mirroring ``check_dataset``.
    """

    def add_batch(self, batch: Mapping[str, Any], ctx: Any) -> None: ...


#: Execution metadata for analyzers that declare bounded batch inputs, kept beside the
#: registry exactly as ``transforms._TRANSFORM_META`` is: the registry stores discoverable
#: descriptors, this stores how to *feed* the component.
_ANALYZER_META: dict[str, dict] = {}


def register_analyzer(name: str, factory: Callable | None = None, *, inputs=(),
                      media_requirements=None, resources=None,
                      lookbehind: int | Mapping[str, int] = 0,
                      lookahead: int | Mapping[str, int] = 0,
                      ownership: str | None = None, **metadata):
    """Register a discoverable analyzer factory, directly or as a decorator.

    ``inputs`` is optional: an analyzer that only implements ``check_dataset`` needs none.
    Declaring ``field:``/``media:`` tokens (plus optional ``lookbehind``/``lookahead`` counts,
    shared or per input) is what makes the analyzer drivable over a tick stream by
    :func:`run_stream_analyzer` and ``Pipeline.analyze()``. Declarations are validated here so
    a malformed one fails at import, not inside a long run.
    """

    def decorate(component: Callable):
        from .registry import default_registry
        from .transforms import _validated_boundary

        declared = tuple(inputs)
        if any(not isinstance(token, str) or not token for token in declared):
            raise ValueError("analyzer input tokens must be non-empty strings")
        bad = [token for token in declared if not token.startswith(("field:", "media:"))]
        if bad:
            # No ``feature:``: the tick-stream driver is a single bounded pass over raw ticks
            # with no feature stage that could produce one.
            raise ValueError(f"analyzer input(s) {bad} must use field:/media:")
        if len(set(declared)) != len(declared):
            raise ValueError(f"analyzer registration has duplicate inputs {list(declared)!r}")
        for label, value in (("lookbehind", lookbehind), ("lookahead", lookahead)):
            _validated_boundary(label, value, declared, prefix="analyzer")
        if ownership not in {None, "rightmost-start"}:
            raise ValueError("analyzer ownership must be None or 'rightmost-start'")
        if (lookbehind or lookahead or ownership or media_requirements) and not declared:
            raise ValueError("analyzer boundary/media declarations need declared inputs")
        # Registry registration can fail (most commonly on a duplicate name). Publish the
        # companion execution metadata only after that succeeds, so discovery and execution
        # cannot observe two different components under the same name.
        default_registry.register_analyzer(name, component, **metadata)
        _ANALYZER_META[name] = {
            "inputs": declared, "media_requirements": dict(media_requirements or {}),
            "resources": dict(resources or {}), "lookbehind": lookbehind,
            "lookahead": lookahead, "ownership": ownership}
        return component

    if factory is None:
        return decorate
    return decorate(factory)


def analyzer_metadata(name: str, analyzer: Any = None) -> dict:
    """Return one analyzer's declared bounded-batch execution metadata.

    Registration metadata wins; an analyzer instance may instead declare the same names as
    attributes (``inputs``, ``lookbehind``, ``lookahead``, ``ownership``,
    ``media_requirements``, ``resources``), which is how a directly constructed analyzer stays
    drivable.
    """
    meta = dict(_ANALYZER_META.get(name) or {})
    meta.setdefault("inputs", tuple(getattr(analyzer, "inputs", ()) or ()))
    meta.setdefault("media_requirements",
                    dict(getattr(analyzer, "media_requirements", {}) or {}))
    meta.setdefault("resources", dict(getattr(analyzer, "resources", {}) or {}))
    for label in ("lookbehind", "lookahead"):
        declared = getattr(analyzer, label, 0)
        meta.setdefault(label, declared if isinstance(declared, Mapping) else int(declared))
    meta.setdefault("ownership", getattr(analyzer, "ownership", None))
    return meta


def resolve_analyzer(name: str, analyzer: Any, *, schema, inputs=None, batch_size=None,
                     device: str = "cpu") -> dict:
    """Validate one analyzer's declared inputs against a schema and return its call plan.

    The plan has the same shape a resolved transform call has, because the analyzer stream
    driver feeds it to the same batch builder.
    """
    meta = analyzer_metadata(name, analyzer)
    label = f"analyzer {name!r}"
    if not meta["inputs"] and not inputs:
        raise ValueError(
            f"{label} declares no field:/media: inputs, so it cannot be driven over a tick "
            "stream; register it with inputs=(...) (see docs/customization.md) or pass "
            "inputs=(...) at the call site")
    from .transforms import resolve_stream_inputs

    resolved = resolve_stream_inputs(
        meta, inputs=inputs, batch_size=batch_size, device=device, schema=schema,
        label=label, kind="analyzer")
    feature_inputs = [token for token in resolved["inputs"] if token.startswith("feature:")]
    if feature_inputs:
        # The stream driver deliberately runs no feature pass: it is a single bounded pass over
        # raw ticks, so there is no earlier producer for a segment feature.
        raise ValueError(
            f"{label} declares feature input(s) {feature_inputs}; the tick-stream analyzer "
            "path has no feature pass to produce them — measure from field:/media: inputs")
    return {"name": name, "label": label, "params": {}, **resolved}


def _hook_arguments(parameters, options) -> tuple[dict, dict]:
    """Validate the two JSON-object argument groups every analyzer driver accepts."""
    if parameters is not None and not isinstance(parameters, Mapping):
        raise ValueError("analyzer parameters must be a JSON object")
    if options is not None and not isinstance(options, Mapping):
        raise ValueError("analyzer options must be a JSON object")
    return dict(parameters or {}), dict(options or {})


def get_analyzer(name: str, *, registry=None, **params) -> Analyzer:
    """Construct a named analyzer after validating its descriptor parameters.

    Plugin entry points are loaded on demand for the default registry. The returned object
    is intentionally user-owned; callers drive its documented update methods and call
    ``summary()`` when all relevant segments have been supplied.
    """
    from .plugins import load_plugins
    from .registry import default_registry, validate_parameters

    registry = registry or default_registry
    if name not in registry.names("analyzer") and registry is default_registry:
        load_plugins()
    if name not in registry.names("analyzer"):
        raise ValueError(f"unknown analyzer {name!r}; registered: {registry.names('analyzer')}")
    descriptor = registry.descriptor("analyzer", name)
    validate_parameters(descriptor, params)
    analyzer = registry.get("analyzer", name)(**params)
    if not isinstance(analyzer, Analyzer):
        raise ValueError(
            f"analyzer {name!r} factory must return an object with a callable summary()"
        )
    return analyzer


def run_dataset_analyzer(
    name: str,
    dataset: Any,
    *,
    parameters: Mapping[str, Any] | None = None,
    options: Mapping[str, Any] | None = None,
    registry=None,
) -> dict:
    """Construct and run a CLI-compatible analyzer against a written dataset.

    Analyzer construction parameters and dataset-driving options are deliberately separate:
    the former are validated against the registered descriptor, while the latter belong to
    the analyzer's ``check_dataset`` hook (for example, action-state modality names).
    """
    params, hook_options = _hook_arguments(parameters, options)
    analyzer = get_analyzer(name, registry=registry, **params)
    return drive_dataset_analyzer(analyzer, name, dataset, options=hook_options)


def drive_dataset_analyzer(analyzer: Any, name: str, dataset: Any, *,
                           options: Mapping[str, Any] | None = None) -> dict:
    """Run an already-constructed analyzer's ``check_dataset`` hook and validate its result."""
    _, hook_options = _hook_arguments(None, options)
    if not isinstance(analyzer, DatasetAnalyzer):
        raise ValueError(
            f"analyzer {name!r} cannot run on a written dataset because it has no callable "
            "check_dataset(dataset, **options) hook"
        )
    raw = analyzer.check_dataset(dataset, **hook_options)
    if not isinstance(raw, Mapping):
        raise ValueError(f"analyzer {name!r} check_dataset() must return a mapping")
    result = dict(raw)
    if not isinstance(result.get("passed"), bool):
        raise ValueError(
            f"analyzer {name!r} check_dataset() must return a boolean 'passed' field"
        )
    result.setdefault("summary", analyzer.summary())
    result["name"] = name
    return result


def run_stream_analyzer(
    name: str,
    *,
    schema,
    adapter,
    sources: Sequence[str],
    segmenter,
    parameters: Mapping[str, Any] | None = None,
    options: Mapping[str, Any] | None = None,
    inputs=None,
    batch_size: int | None = None,
    device: str = "cpu",
    on_error: str = "raise",
    progress=None,
    registry=None,
    analyzer: Any = None,
) -> dict:
    """Drive a stream-capable analyzer over a segmented tick stream in one bounded pass.

    Every window of every source is measured in ``(source_id, start_idx)`` order, with the
    analyzer's declared lookbehind/lookahead context and lazily decoded media supplied by the
    same bounded batch machinery the transform runner uses. Residency is one batch of windows
    plus their halo, never the pool, so this is safe under ``streaming=True``.

    ``analyzer=`` drives an instance the caller already owns (and keeps its accumulated
    evidence afterwards); otherwise the named analyzer is constructed from ``parameters``.
    """
    if on_error not in {"skip", "raise"}:
        raise ValueError("on_error must be 'skip' or 'raise'")
    params, hook_options = _hook_arguments(parameters, options)
    if analyzer is not None and params:
        raise ValueError("pass analyzer= or parameters=, not both: an already constructed "
                         "analyzer cannot be re-parameterized")
    if analyzer is None:
        analyzer = get_analyzer(name, registry=registry, **params)
    if not isinstance(analyzer, StreamAnalyzer):
        hint = (" it does implement check_dataset, so run it on a written dataset "
                "(Pipeline.output(path) then analyze the artifact, or kur-check)"
                if isinstance(analyzer, DatasetAnalyzer) else "")
        raise ValueError(
            f"analyzer {name!r} cannot run on a tick stream because it has no callable "
            f"add_batch(batch, ctx) hook;{hint or ' see docs/customization.md'}")
    call = resolve_analyzer(name, analyzer, schema=schema, inputs=inputs,
                            batch_size=batch_size, device=device)
    from .media import media_provider_for
    from .progress import emit

    provider = media_provider_for(adapter)
    ordered_sources = sorted((str(source) for source in sources), key=str)
    errors: list[dict] = []
    segments_analyzed = 0
    sources_analyzed = 0
    emit(progress, "analyze", "start", total=len(ordered_sources))
    for index, source in enumerate(ordered_sources, 1):
        from .scoring import _redact_diagnostic

        safe_source = _redact_diagnostic(source)
        try:
            segments_analyzed += _drive_stream_source(
                analyzer, call, source, schema=schema, adapter=adapter, provider=provider,
                segmenter=segmenter, options=hook_options)
        except Exception as error:  # skip-and-log, exactly like the feature pass
            if on_error != "skip":
                raise
            import sys

            safe_error = _redact_diagnostic(repr(error))
            print(f"  [skip] source {safe_source!r}: {safe_error}", file=sys.stderr)
            errors.append({"source": safe_source,
                           "error": safe_error,
                           "error_type": type(error).__name__,
                           "message": _redact_diagnostic(str(error)),
                           "phase": "analyze"})
            emit(progress, "analyze", "advance", current=index,
                 total=len(ordered_sources), source=safe_source, status="skipped")
            continue
        sources_analyzed += 1
        emit(progress, "analyze", "advance", current=index, total=len(ordered_sources),
             source=safe_source, status="ok")
    emit(progress, "analyze", "complete", current=len(ordered_sources),
         total=len(ordered_sources), detail=f"{segments_analyzed} segment(s)")
    if ordered_sources and not segments_analyzed:
        # An analyzer that saw nothing "passes" vacuously, which is the exact failure mode a
        # runnable example once caught in the feature pass. Refuse to report it as a verdict.
        raise ValueError(
            f"analyzer {name!r} measured NO segments: {len(ordered_sources)} source(s) "
            "yielded no windows"
            + (f" ({len(errors)} skipped; first error: {errors[0]['error']})" if errors
               else " (is every source shorter than the segment window?)"))

    result: dict = {"name": name, "mode": "tick-stream",
                    "sources_analyzed": sources_analyzed,
                    "segments_analyzed": segments_analyzed,
                    "inputs": list(call["inputs"]), "errors": errors,
                    "summary": analyzer.summary()}
    extra = getattr(analyzer, "stream_result", None)
    if callable(extra):
        raw = extra()
        if not isinstance(raw, Mapping):
            raise ValueError(f"analyzer {name!r} stream_result() must return a mapping")
        if not isinstance(raw.get("passed"), bool):
            raise ValueError(
                f"analyzer {name!r} stream_result() must return a boolean 'passed' field")
        result.update(dict(raw))
        result["name"], result["mode"] = name, "tick-stream"
    return result


def _drive_stream_source(analyzer, call, source, *, schema, adapter, provider, segmenter,
                         options) -> int:
    """Feed one source's windows to an analyzer in batches, holding at most one batch."""
    from .transforms import build_batch, source_tick_stream

    size = call["batch_size"]
    pending: list[tuple] = []
    analyzed = 0

    def flush() -> int:
        if not pending:
            return 0
        segments = [item[0] for item in pending]
        windows = [item[1] for item in pending]
        contexts = [item[2] for item in pending]
        batch, ctx = build_batch(call, segments, windows, contexts, adapter=adapter,
                                 provider=provider)
        returned = analyzer.add_batch(batch, ctx, **options)
        if returned is not None:
            raise ValueError(
                f"analyzer {call['name']!r} add_batch() must accumulate state and return None, "
                f"got {type(returned).__name__}")
        count = len(segments)
        pending.clear()
        return count

    previous: tuple | None = None
    for segment, window, before, after in segmenter.stream_with_context(
            source, source_tick_stream(adapter, source),
            lookbehind=call["lookbehind"], lookahead=call["lookahead"]):
        if previous is not None:
            # Ownership is resolved exactly as the transform runner resolves it: a window owns
            # the steps before the next window starts. The last window of a source owns all of
            # its own steps, which is why finalizing lags one segment behind.
            previous[2]["owned_length"] = (
                min(previous[0].end_idx, segment.start_idx) - previous[0].start_idx)
            pending.append(previous)
            if len(pending) == size:
                analyzed += flush()
        previous = (segment, window, {"lookbehind": before, "lookahead": after,
                                      "owned_length": segment.end_idx - segment.start_idx})
    if previous is not None:
        pending.append(previous)
    analyzed += flush()
    return analyzed


SPLIT_LEAKAGE_PARAMETERS = {
    "type": "object",
    "properties": {
        "check_groups": {
            "type": "boolean",
            "default": True,
            "description": "Also flag groups whose sources cross split boundaries.",
        },
    },
    "additionalProperties": False,
}


class SplitLeakageAnalyzer:
    """Audit source and optional group assignments across dataset splits.

    Repeated observations of the same assignment are allowed because a source may produce
    many segments or shards. A source observed in more than one split is leakage. When
    ``check_groups`` is enabled, a group spanning splits or one source claiming multiple
    groups is also a failure.
    """

    def __init__(self, *, check_groups: bool = True):
        if not isinstance(check_groups, bool):
            raise ValueError("check_groups must be true or false")
        self.check_groups = check_groups
        self._observations = 0
        self._assignments: set[tuple[str, str, str | None]] = set()
        self._source_splits: dict[str, set[str]] = defaultdict(set)
        self._source_groups: dict[str, set[str]] = defaultdict(set)
        self._group_splits: dict[str, set[str]] = defaultdict(set)
        self._group_sources: dict[str, set[str]] = defaultdict(set)

    def add_assignment(self, source_id: str, split: str, *, group: str | None = None) -> None:
        """Record one source/split observation and its optional leakage group."""
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string")
        if not isinstance(split, str) or not split:
            raise ValueError("split must be a non-empty string")
        if group is not None and (not isinstance(group, str) or not group):
            raise ValueError("group must be a non-empty string or None")

        self._observations += 1
        self._assignments.add((source_id, split, group))
        self._source_splits[source_id].add(split)
        if group is not None:
            self._source_groups[source_id].add(group)
            self._group_splits[group].add(split)
            self._group_sources[group].add(source_id)

    def add_assignments(
        self,
        source_splits: Mapping[str, str],
        *,
        source_groups: Mapping[str, str] | None = None,
    ) -> None:
        """Record an SDK-style ``{source_id: split}`` mapping in one call."""
        if not isinstance(source_splits, Mapping):
            raise ValueError("source_splits must be a mapping")
        if source_groups is not None and not isinstance(source_groups, Mapping):
            raise ValueError("source_groups must be a mapping or None")
        if source_groups is not None:
            unknown = set(source_groups) - set(source_splits)
            if unknown:
                raise ValueError(
                    f"source_groups contains source(s) absent from source_splits: {sorted(unknown)!r}"
                )
        for source_id, split in source_splits.items():
            group = None if source_groups is None else source_groups.get(source_id)
            self.add_assignment(source_id, split, group=group)

    def summary(self) -> dict:
        """Return a deterministic, JSON-safe leakage report."""
        source_leaks = [
            {"source_id": source_id, "splits": sorted(splits)}
            for source_id, splits in sorted(self._source_splits.items())
            if len(splits) > 1
        ]
        group_leaks = []
        source_group_conflicts = []
        if self.check_groups:
            group_leaks = [
                {
                    "group": group,
                    "splits": sorted(splits),
                    "sources": sorted(self._group_sources[group]),
                }
                for group, splits in sorted(self._group_splits.items())
                if len(splits) > 1
            ]
            source_group_conflicts = [
                {"source_id": source_id, "groups": sorted(groups)}
                for source_id, groups in sorted(self._source_groups.items())
                if len(groups) > 1
            ]
        splits = sorted({split for values in self._source_splits.values() for split in values})
        failures = source_leaks or group_leaks or source_group_conflicts
        return {
            "passed": not failures,
            "check_groups": self.check_groups,
            "observations": self._observations,
            "unique_assignments": len(self._assignments),
            "sources": len(self._source_splits),
            "groups": len(self._group_splits),
            "splits": splits,
            "source_leaks": source_leaks,
            "group_leaks": group_leaks,
            "source_group_conflicts": source_group_conflicts,
        }

    def check_dataset(self, dataset, **options) -> dict:
        """Audit the source/split assignments recorded in a curated manifest."""
        if options:
            raise ValueError(
                f"kuration/split-leakage does not accept dataset options: {sorted(options)}"
            )
        from .loaders import shard_entries

        entries, _container = shard_entries(dataset)
        for entry in entries:
            source_id = entry.get("source_id")
            split = entry.get("split")
            if source_id is None or split is None:
                raise ValueError(
                    "split-leakage needs every manifest shard to contain source_id and split"
                )
            group = entry.get("group")
            self.add_assignment(
                str(source_id),
                str(split),
                group=None if group is None else str(group),
            )
        summary = self.summary()
        return {"passed": summary["passed"], "summary": summary}


@register_analyzer(
    "kuration/split-leakage",
    description="Audit source and group assignments for train/test split leakage",
    parameters=SPLIT_LEAKAGE_PARAMETERS,
    capabilities={"stateful": True, "cross_segment": True},
    documentation="https://aintropy-ai.github.io/kuration-sdk/customization/#stateful-analyzers",
    version="1",
)
def _split_leakage_analyzer(*, check_groups: bool = True) -> SplitLeakageAnalyzer:
    return SplitLeakageAnalyzer(check_groups=check_groups)
