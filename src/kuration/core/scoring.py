"""Parallel, fault-tolerant feature pass — shared by curate / experiment / write_dataset.

The single place sources are read, segmented and scored. Parallelism is per-source over a
process pool: each worker rebuilds the `FeatureEngine` in-process (the compiled DSL metrics
are closures and can't be pickled, so they're recompiled from the JSON `define` + schema —
both picklable; the adapter and segmenter pickle as-is). A source that raises is
**skipped-and-logged** (loud: stderr line per skip + a summary + collected in the returned
errors), never silently dropped. Determinism holds: the parent sorts the pooled segments by
`(source_id, start_idx)`, so the result is identical regardless of worker count or completion
order.

The pool explicitly uses `spawn` to avoid unsafe forks from multithreaded runtimes and to behave
consistently across Python versions. Custom registered metrics must therefore live in an
importable module; a metric registered only in `__main__` is not visible to workers and fails
loudly.
"""
from __future__ import annotations

from itertools import chain
import multiprocessing as mp
from collections import deque
from pathlib import Path
import re
import sys
import traceback
from urllib.parse import urlsplit
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from .features import FeatureEngine


_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s'\"<>]+")
_QUOTED_ABSOLUTE_PATH = re.compile(
    r"(?P<quote>['\"])(?P<path>(?:[A-Za-z]:[\\/]|\\\\|/)[^'\"]+)(?P=quote)"
)
_WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z]:[\\/]|\\\\)[^\s'\"<>]+")
_POSIX_PATH = re.compile(r"(?<![A-Za-z0-9_:])/[^\s'\"<>]+")


def _publish_diagnostic_reference(value: str) -> str:
    from .provenance import publishable_source_id
    return f"<source:{publishable_source_id(value)}>"


def _redact_diagnostic(value):
    """Redact credentials and host-specific paths before diagnostics enter artifacts."""
    text = str(value)
    home = str(Path.home())
    if home and home != "/":
        text = text.replace(home, "<home>")

    def clean(match):
        raw = match.group(0)
        try:
            urlsplit(raw)
            return _publish_diagnostic_reference(raw)
        except (TypeError, ValueError):
            return "<redacted-url>"

    text = _URL.sub(clean, text)

    def clean_quoted_path(match):
        quote = match.group("quote")
        return f"{quote}{_publish_diagnostic_reference(match.group('path'))}{quote}"

    text = _QUOTED_ABSOLUTE_PATH.sub(clean_quoted_path, text)
    text = _WINDOWS_PATH.sub(
        lambda match: _publish_diagnostic_reference(match.group(0)), text)
    return _POSIX_PATH.sub(
        lambda match: _publish_diagnostic_reference(match.group(0)), text)


def _cached_group_callback(group_of):
    """Wrap ``group_of`` so each source's successful value is computed once per terminal run."""
    values: dict[Any, Any] = {}
    if group_of is None:
        return None

    def cached(source):
        if source not in values:
            values[source] = group_of(source)
        return values[source]

    return cached


def _entities_in_order(ticks) -> list:
    seen, out = set(), []
    for tk in ticks:
        for eid in tk.entities:
            if eid not in seen:
                seen.add(eid)
                out.append(eid)
    return out


def _adapter_availability(adapter):
    """Return declared mapped field/global names, following transparent adapter wrappers."""
    current, seen = adapter, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        fields = getattr(current, "available_fields", None)
        globals_ = getattr(current, "available_globals", None)
        if fields is not None or globals_ is not None:
            return (None if fields is None else set(fields),
                    None if globals_ is None else set(globals_))
        current = getattr(current, "adapter", None)
    return None, None


def _preflight_adapter_inputs(schema, adapter, define, tags_spec, required_fields=()):
    """Reject declared metric/tag inputs that a partially mapped adapter cannot provide."""
    available_fields, available_globals = _adapter_availability(adapter)
    if available_fields is None and available_globals is None:
        return
    need_fields, need_globals = set(required_fields), set()
    from .dsl.grammar import _parse_filter
    for spec in define.values():
        source = spec.get("source", "")
        entity_source = False
        if "." in source:
            namespace, rest = source.split(".", 1)
            if schema.entity_axis and namespace == schema.entity_axis:
                need_fields.add(rest.split(".", 1)[0])
                entity_source = True
        elif source in schema.fields:  # unqualified field in a single-stream schema
            need_fields.add(source)
            entity_source = True
        elif source in schema.globals:
            need_globals.add(source)
        if entity_source and spec.get("filter"):
            parsed = _parse_filter(spec["filter"])
            need_fields.add(parsed[1])
    tag_froms = tags_spec.get("from") if isinstance(tags_spec, dict) else tags_spec
    for source in tag_froms or ():
        if "." not in source and source in schema.fields:
            need_fields.add(source)
    missing_fields = sorted(need_fields - available_fields) if available_fields is not None else []
    missing_globals = (sorted(need_globals - available_globals)
                       if available_globals is not None else [])
    if missing_fields or missing_globals:
        details = []
        if missing_fields:
            details.append(f"fields {missing_fields} (mapped: {sorted(available_fields)})")
        if missing_globals:
            details.append(f"globals {missing_globals} (mapped: {sorted(available_globals)})")
        raise ValueError(
            f"{type(adapter).__name__} cannot execute the declared strategy: required "
            + "; ".join(details))


def _error_payload(error: Exception, *, phase: str = "scoring") -> dict[str, str]:
    return {
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": str(error),
        "repr": repr(error),
        "traceback": traceback.format_exc(),
        "phase": phase,
    }


def _score_one(args):
    """Worker: read one source, segment, compute features. Returns a status tuple so a
    failure is data (skipped in the parent), not a pool-killing exception."""
    (schema, define, registered, tags_spec, adapter, segmenter, source, group, streaming,
     plugin_modules, registry) = args
    try:
        from .plugins import load_plugins
        # The parent already resolved the execution registry. Workers reconstruct only those
        # modules; rescanning ambient entry points here would break explicit job isolation.
        load_plugins(list(plugin_modules), include_entry_points=False)
        engine = FeatureEngine(schema, define=define, registered=registered, tags_spec=tags_spec,
                               registry=registry)
        if streaming:
            # Stream windows from a lazy tick iterator: never hold more than one window. A
            # wrapper observes every tick as it passes the deque, so total length + entity
            # roster (for the output track fill) come from the SAME single pass.
            n, order, seen = 0, [], set()

            def tracked(it):
                nonlocal n
                for tk in it:
                    n += 1
                    for eid in tk.entities:
                        if eid not in seen:
                            seen.add(eid)
                            order.append(eid)
                    yield tk

            segs = []
            tick_iter = (adapter.iter_ticks(source) if hasattr(adapter, "iter_ticks")
                         else adapter.ticks(source))
            for seg, win in segmenter.stream(source, tracked(tick_iter)):
                seg.group = group
                engine.score_window(seg, win)
                segs.append(seg)
            return ("ok", source, segs, [(eid, n) for eid in order])

        ticks = list(adapter.ticks(source))
        segs = segmenter.segments(source, ticks)
        for s in segs:
            s.group = group
        engine.compute(ticks, segs)
        track_lens = [(eid, len(ticks)) for eid in _entities_in_order(ticks)]
        return ("ok", source, segs, track_lens)
    except Exception as error:
        return ("err", source, _error_payload(error), None)


def _score_prepared(item):
    """Execute a prepared source item or forward a parent-side preflight result."""
    ready, payload = item
    return payload if ready else _score_one(payload)


def score_sources(*, schema, adapter, sources, define, registered, segmenter,
                  tags_spec=None, group_of=None, tags_of=None, workers: int = 1,
                  on_error: str = "skip", streaming: bool = False, progress=None,
                  executor: str = "local", registry=None, required_fields=()):
    """Return (all_segments_sorted, errors, source_tracks).

      errors:        [{"source", "error"}] for skipped sources (on_error='skip')
      source_tracks: {source_id -> [(entity_id, n_steps)]} — the output unit lengths, so the
                     track fill needs no second read of the data.
    `on_error`: 'skip' (log + continue) or 'raise'. `group_of(source)` is evaluated in the
    parent (callables needn't be picklable); `tags_of` is applied in the parent post-hoc.
    """
    if streaming and getattr(adapter, "supports_streaming", True) is False:
        raise ValueError(f"{type(adapter).__name__} cannot provide bounded-memory streaming; "
                         "use an uncompressed array store such as HDF5/Zarr or streaming=False")
    if on_error not in ("skip", "raise"):
        raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError(f"workers must be a positive integer, got {workers!r}")
    if not isinstance(streaming, bool):
        raise ValueError("streaming must be true or false")
    if executor not in {"local", "ray"}:
        raise ValueError("executor must be 'local' or 'ray'")
    try:
        source_total = len(sources)
    except TypeError:
        source_iterator = iter(sources)
        try:
            first_source = next(source_iterator)
        except StopIteration as error:
            raise ValueError("score_sources needs at least one source") from error
        sources = chain((first_source,), source_iterator)
    else:
        if source_total == 0:
            raise ValueError("score_sources needs at least one source")
    from .registry import default_registry
    registry = registry or default_registry
    _preflight_adapter_inputs(schema, adapter, define, tags_spec, required_fields)
    from .plugins import load_plugins, loaded_plugin_modules
    missing_metrics = [name for name in registered if name not in registry.names("metric")]
    if missing_metrics and registry is default_registry:
        load_plugins()
        missing_metrics = [name for name in missing_metrics
                           if name not in registry.names("metric")]
    if missing_metrics:
        raise ValueError(f"unknown registered metric(s) {sorted(missing_metrics)}; registered: "
                         f"{registry.names('metric')}")
    metric_modules = {
        getattr(registry.get("metric", name), "__module__", "")
        for name in registered if name in registry.names("metric")}
    metric_modules.discard("")
    # An isolated registry must not reconstruct unrelated ambient entry-point plugins in
    # workers.  Import only the modules that provide metrics used by this scoring stage;
    # preserve the historical full-plugin propagation for the process-global registry.
    ambient_modules = set(loaded_plugin_modules()) if registry is default_registry else set()
    plugin_modules = tuple(sorted(ambient_modules | metric_modules))
    worker_modules = plugin_modules if executor == "ray" or workers > 1 else ()
    worker_registry = registry
    if executor == "ray" or workers > 1:
        # Do not pickle the entire process registry: unrelated application components may be
        # local closures even when every metric needed by this stage is importable.  A minimal
        # execution registry preserves overlay resolution without coupling workers to plugins
        # they never execute.
        from .registry import Registry
        worker_registry = Registry()
        for name in registered:
            worker_registry.register_metric(name, registry.get("metric", name))
    from .progress import emit
    try:
        total = len(sources)
    except TypeError:
        total = None
    emit(progress, "scoring", "start", total=total)
    source_count = 0
    seen_sources: set[str] = set()

    def prepared_items():
        nonlocal source_count
        for src in sources:
            source_count += 1
            if not isinstance(src, str) or not src:
                raise ValueError("sources must contain only non-empty strings")
            if src in seen_sources:
                raise ValueError(f"sources must be unique; duplicate source ID {src!r}")
            seen_sources.add(src)
            try:
                group = group_of(src) if group_of else None
            except Exception as error:
                if on_error == "raise":
                    raise RuntimeError(
                        f"group_of failed for source {src!r}: {error!r}") from error
                yield True, ("err", src, _error_payload(error, phase="group_of"), None)
                continue
            yield False, (schema, define, registered, tags_spec, adapter, segmenter, src,
                          group, streaming, worker_modules, worker_registry)

    if executor == "ray":
        from .execution import RayExecutionBackend, ResourceSpec
        backend = RayExecutionBackend()
        results = backend.iter_batches(
            prepared_items(), _score_prepared, max_in_flight=max(1, workers * 2),
            workers=workers, resources=ResourceSpec(num_cpus=1))
    elif workers and workers > 1:
        context = mp.get_context("spawn")

        def process_results():
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor_pool:
                pending: deque[Any] = deque()
                max_in_flight = workers * 2
                for ready, payload in prepared_items():
                    pending.append((ready, payload if ready else executor_pool.submit(
                        _score_one, payload)))
                    if len(pending) >= max_in_flight:
                        is_ready, item = pending.popleft()
                        yield item if is_ready else item.result()
                while pending:
                    is_ready, item = pending.popleft()
                    yield item if is_ready else item.result()

        results = process_results()
    else:
        results = (_score_prepared(item) for item in prepared_items())

    all_segments: list[Any] = []
    errors: list[dict] = []
    source_tracks: dict[Any, Any] = {}
    done = 0
    for status, src, payload, track_lens in results:
        done += 1
        emit(progress, "scoring", "advance", current=done, total=total,
             source=src, status=status)
        if status == "ok":
            if tags_of:
                try:
                    for segment in payload:
                        segment.tags = frozenset(tags_of(segment))
                except Exception as error:
                    payload = _error_payload(error, phase="tags_of")
                    if on_error == "raise":
                        raise RuntimeError(
                            f"tags_of failed for source {src!r}: {error!r}") from error
                    status = "err"
            if status != "ok":
                pass
            else:
                all_segments.extend(payload)
                source_tracks[src] = track_lens
                continue
        if status == "err":
            if on_error == "raise":
                raise RuntimeError(
                    f"{payload.get('phase', 'scoring')} source {src!r} failed with "
                    f"{payload['type']}: {payload['message']}\n{payload['traceback']}"
                )
            safe_source = _redact_diagnostic(src)
            safe_error = _redact_diagnostic(payload["repr"])
            print(f"  [skip] source {safe_source!r}: {safe_error}", file=sys.stderr)
            errors.append({
                "source": safe_source if isinstance(src, str) else src,
                "error": safe_error,
                "error_type": payload["type"],
                "message": _redact_diagnostic(payload["message"]),
                "phase": payload.get("phase", "scoring"),
            })
    if errors:
        print(f"  [skip] {len(errors)} source(s) skipped due to errors", file=sys.stderr)
    if source_count and not all_segments:
        # An empty pool silently propagates into an empty-but-valid-looking dataset (the
        # vacuous-success failure mode) — total failure must raise even under on_error='skip'.
        raise RuntimeError(
            f"segment pool is EMPTY: {source_count} source(s) yielded no segments"
            + (f" ({len(errors)} skipped; first error: {errors[0]['error']})" if errors
               else " (no errors — is every source shorter than the segment window?)"))

    all_segments.sort(key=lambda s: (s.source_id, s.start_idx))   # determinism
    emit(progress, "scoring", "complete", current=source_count, total=source_count,
         detail=f"{len(all_segments)} segments, {len(errors)} skipped")
    return all_segments, errors, source_tracks
