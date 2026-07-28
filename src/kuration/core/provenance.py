"""Canonical run provenance and component/source fingerprints."""
from __future__ import annotations

import hashlib
import inspect
import json
import marshal
import re
from types import CodeType
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit, urlunsplit


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json_value(value: object, *, label: str) -> object:
    """Validate and normalize a value using the exact JSON contract artifacts persist."""
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{label} must be strict JSON data (finite numbers, strings, booleans, null, "
            f"lists, and string-keyed objects): {error}"
        ) from error
    return json.loads(encoded)


def _normalized_code(code: CodeType) -> CodeType:
    """Remove host-path/line placement while retaining bytecode and column positions."""
    constants = tuple(
        _normalized_code(value) if isinstance(value, CodeType) else value
        for value in code.co_consts
    )
    return code.replace(
        co_consts=constants,
        co_filename="<kuration-component>",
        co_firstlineno=1,
    )


def _code_hash(obj) -> str | None:
    source = None
    try:
        source = inspect.getsource(obj)
    except (OSError, TypeError):
        pass
    code = getattr(obj, "__code__", None)
    if source is None and code is None:
        return None
    payload = bytearray()
    if source is not None:
        payload.extend(b"source\0")
        payload.extend(source.encode())
    if isinstance(code, CodeType):
        # Source inspection returns the complete physical line for lambdas. The normalized
        # code object's position table keeps two lambdas on that line from colliding without
        # leaking the checkout path into artifact identity.
        payload.extend(b"\0code\0")
        payload.extend(marshal.dumps(_normalized_code(code)))
    return _sha(bytes(payload))


def _closure_values(fn) -> tuple:
    values = []
    for cell in fn.__closure__ or ():
        try:
            values.append(cell.cell_contents)
        except ValueError:
            values.append("<empty closure cell>")
    return tuple(values)


def _canonical_state(value, active: set[int] | None = None):
    """Convert common callable/plugin state to a hash-seed-independent tagged tree."""
    active = set() if active is None else active
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", str(value))
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, Path):
        return ("path", str(value))

    identity = id(value)
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    if identity in active:
        return ("cycle", type_name)
    active.add(identity)
    try:
        if isinstance(value, dict):
            entries = [
                (_canonical_state(key, active), _canonical_state(item, active))
                for key, item in value.items()
            ]
            entries.sort(key=lambda entry: json.dumps(
                entry, ensure_ascii=True, separators=(",", ":")))
            return ("dict", tuple(entries))
        if isinstance(value, list):
            return ("list", tuple(_canonical_state(item, active) for item in value))
        if isinstance(value, tuple):
            return ("tuple", tuple(_canonical_state(item, active) for item in value))
        if isinstance(value, (set, frozenset)):
            entries = [_canonical_state(item, active) for item in value]
            entries.sort(key=lambda entry: json.dumps(
                entry, ensure_ascii=True, separators=(",", ":")))
            return (type(value).__name__, tuple(entries))
        if inspect.ismodule(value):
            return ("module", value.__name__, str(getattr(value, "__version__", "")))
        if inspect.isfunction(value):
            return (
                "function",
                value.__module__,
                value.__qualname__,
                _code_hash(value),
                _canonical_state({
                    "defaults": value.__defaults__,
                    "kwdefaults": value.__kwdefaults__,
                    "closure": tuple(
                        item for item in _closure_values(value) if item is not value),
                }, active),
            )
        if inspect.ismethod(value):
            return (
                "method",
                _canonical_state(value.__func__, active),
                _canonical_state(value.__self__, active),
            )
        if inspect.isclass(value):
            return ("type", value.__module__, value.__qualname__)
        if type(value).__module__.startswith("numpy"):
            if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "tobytes"):
                return (
                    "numpy-array",
                    str(value.dtype),
                    tuple(int(size) for size in value.shape),
                    _sha(value.tobytes(order="C")),
                )
            if hasattr(value, "item"):
                return ("numpy-scalar", _canonical_state(value.item(), active))
        state = getattr(value, "__dict__", None)
        if state is not None:
            return ("object", type_name, _canonical_state(state, active))
        slots = []
        for cls in type(value).__mro__:
            declared = getattr(cls, "__slots__", ())
            declared = (declared,) if isinstance(declared, str) else declared
            for name in declared:
                if name not in {"__dict__", "__weakref__"} and hasattr(value, name):
                    slots.append((name, getattr(value, name)))
        if slots:
            return ("slots", type_name, _canonical_state(dict(slots), active))
        try:
            reduced = value.__reduce_ex__(5)
        except Exception:
            reduced = None
        if reduced is not None and reduced is not value:
            return ("reduced", type_name, _canonical_state(reduced, active))
        # Do not persist an address-bearing repr. The type still distinguishes opaque values;
        # callers needing stateful reuse must expose a source/component fingerprint.
        rendered = re.sub(r"0x[0-9a-fA-F]+", "0x…", repr(value))
        return ("opaque", type_name, rendered)
    finally:
        active.remove(identity)


def _state_hash(value) -> str:
    """Hash configured state deterministically across mapping order and hash randomization."""
    canonical = _canonical_state(value)
    encoded = json.dumps(
        canonical, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()
    return _sha(encoded)


def _referenced_global(value) -> object:
    """Capture behaviorally relevant global identity without recursively walking modules."""
    if inspect.ismodule(value):
        return {
            "kind": "module",
            "name": value.__name__,
            "version": str(getattr(value, "__version__", "")),
        }
    if inspect.isfunction(value):
        return {
            "kind": "function",
            "type": f"{value.__module__}.{value.__qualname__}",
            "code_sha256": _code_hash(value),
            "defaults": value.__defaults__,
            "kwdefaults": value.__kwdefaults__,
        }
    if inspect.isbuiltin(value) or inspect.isclass(value):
        return {
            "kind": "symbol",
            "type": f"{value.__module__}.{value.__qualname__}",
        }
    return value


def _function_state(fn) -> dict:
    referenced_globals = {
        name: _referenced_global(fn.__globals__[name])
        for name in sorted(set(fn.__code__.co_names))
        if name in fn.__globals__ and fn.__globals__[name] is not fn
    }
    return {"defaults": fn.__defaults__, "kwdefaults": fn.__kwdefaults__,
            "closure": _closure_values(fn), "globals": referenced_globals}


def component_identity(obj) -> dict | None:
    """Stable-enough identity for configured adapters, encoders, and callbacks."""
    if obj is None:
        return None
    is_fn = inspect.isfunction(obj)
    is_method = inspect.ismethod(obj)
    target = obj if is_fn else obj.__func__ if is_method else type(obj)
    out = {"type": f"{target.__module__}.{target.__qualname__}", "code_sha256": _code_hash(target)}
    if is_fn:
        out["state_sha256"] = _state_hash(_function_state(obj))
    elif is_method:
        out["state_sha256"] = _state_hash(
            {"function": _function_state(obj.__func__), "self": obj.__self__})
    else:
        out["state_sha256"] = _state_hash(obj)
    return out


def _portable_basename(path: str) -> str:
    """Return a basename for either host-native or Windows-style path text."""
    stripped = path.rstrip("/\\")
    if "\\" in stripped:
        return PureWindowsPath(stripped).name
    return Path(stripped).name


def _source_reference(source: str) -> dict[str, str]:
    """Portable, publishable source reference without credentials or host-local directories."""
    if "://" in source and not source.startswith("file://"):
        parsed = urlsplit(source)
        host = parsed.hostname or ""
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        visible = urlunsplit((parsed.scheme, host, parsed.path, "", ""))
        return {"source": visible, "source_id_sha256": _sha(visible.encode())}
    path = source[7:] if source.startswith("file://") else source
    if "/" in path or "\\" in path:
        return {"source": _portable_basename(path),
                "source_id_sha256": _sha(source.encode())}
    return {"source": source}


def _reference_with_evidence(reference: dict[str, str], evidence: object) -> dict[str, str]:
    """Bind an obscured source ID to portable evidence rather than a host-local spelling."""
    payload = json.dumps(
        {"source": reference["source"], "evidence": evidence},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return {**reference, "source_id_sha256": _sha(payload)}


def publishable_source_id(source: object) -> str:
    """Return a readable artifact ID without publishing credentials or host-local paths."""
    text = str(source)
    if "://" in text and not text.startswith("file://"):
        parsed = urlsplit(text)
        label = _portable_basename(parsed.path) or parsed.hostname or "source"
        path_like = True
        host = parsed.hostname or ""
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        identity = urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    else:
        path = text[7:] if text.startswith("file://") else text
        path_like = text.startswith("file://") or "/" in path or "\\" in path
        label = _portable_basename(path) if path_like else text
        identity = text
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip(". _") or "source"
    safe = safe[:72].rstrip(". _") or "source"
    if not path_like and safe == text and len(text) <= 72:
        return text
    return f"{safe}-{_sha(identity.encode())[:12]}"


def source_descriptor(source, adapter) -> dict:
    """Best available source identity without reading the source payload a second time."""
    custom = getattr(adapter, "source_fingerprint", None)
    reference = _source_reference(str(source))
    if custom is not None:
        fingerprint = strict_json_value(
            custom(source), label=f"source_fingerprint({str(source)!r})")
        return {
            **_reference_with_evidence(reference, fingerprint),
            "fingerprint": fingerprint,
            "strength": "adapter",
        }

    from .io import is_remote
    s = str(source)
    if is_remote(s):
        from .io import _fs
        fs = _fs(s)
        info = fs.info(s)
        keep = ("size", "mtime", "updated", "etag", "ETag", "checksum", "version_id", "generation")
        if info.get("type") == "directory":
            details = fs.find(s, detail=True)
            remote_entries = [(str(path), *(str(meta[k]) for k in keep if k in meta))
                              for path, meta in sorted(details.items())
                              if meta.get("type") != "directory"]
            encoded = json.dumps(remote_entries, sort_keys=True,
                                 separators=(",", ":")).encode()
            return {**reference, "metadata": {"is_dir": True,
                                                "entries": len(remote_entries),
                                                "tree_sha256": _sha(encoded)},
                    "strength": "remote-metadata"}
        metadata = {k: str(info[k]) for k in keep if info.get(k) is not None}
        strong = any(info.get(key) not in (None, "")
                     for key in ("etag", "ETag", "checksum", "version_id", "generation"))
        return {**reference, "metadata": metadata,
                "strength": "remote-version" if strong else "remote-metadata"}
    p = Path(s[7:] if s.startswith("file://") else s)
    if p.exists():
        st = p.stat()
        if p.is_dir():
            local_entries: list[tuple[str, int, str]] = []
            for child in sorted(p.rglob("*")):
                if child.is_file():
                    child_stat = child.stat()
                    digest = hashlib.sha256()
                    with child.open("rb") as stream:
                        while chunk := stream.read(1024 * 1024):
                            digest.update(chunk)
                    local_entries.append((child.relative_to(p).as_posix(), child_stat.st_size,
                                          digest.hexdigest()))
            encoded = json.dumps(local_entries, separators=(",", ":")).encode()
            tree_digest = _sha(encoded)
            return {**_reference_with_evidence(reference, tree_digest),
                    "metadata": {"is_dir": True,
                                                "entries": len(local_entries),
                                                "tree_sha256": tree_digest},
                    "strength": "content"}
        digest = hashlib.sha256()
        with p.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        content_digest = digest.hexdigest()
        return {**_reference_with_evidence(reference, content_digest),
                "metadata": {"size": st.st_size,
                                           "content_sha256": content_digest,
                                           "is_dir": False}, "strength": "content"}
    return {**reference, "metadata": {"non_path": True}, "strength": "identifier"}


def rebase_source_descriptor(descriptor: dict, source: object) -> dict:
    """Attach snapshot evidence to the caller-visible source reference.

    Importers read private immutable snapshots rather than reopening a caller-controlled
    pathname after validation.  The snapshot descriptor therefore has the right evidence but
    the wrong (temporary) source name.  Rebasing preserves the evidence and strength while
    deriving the public source ID from the original, credential-scrubbed reference.
    """
    body = {
        key: value
        for key, value in descriptor.items()
        if key not in {"source", "source_id_sha256"}
    }
    metadata = body.get("metadata")
    if "fingerprint" in body:
        evidence = body["fingerprint"]
    elif isinstance(metadata, dict) and "content_sha256" in metadata:
        evidence = metadata["content_sha256"]
    elif isinstance(metadata, dict) and "tree_sha256" in metadata:
        evidence = metadata["tree_sha256"]
    else:
        evidence = body
    return {
        **_reference_with_evidence(_source_reference(str(source)), evidence),
        **body,
    }


def source_descriptors(sources, adapter) -> list[dict]:
    """Return canonical descriptors in the order used by run provenance."""
    return sorted((source_descriptor(source, adapter) for source in sources),
                  key=lambda value: (value["source"], value.get("source_id_sha256", "")))


_REUSABLE_SOURCE_STRENGTHS = frozenset({"adapter", "content", "remote-version"})


def source_identities_allow_reuse(descriptors) -> bool:
    """Whether descriptors prove enough source stability for persisted-state reuse."""
    return all(
        isinstance(descriptor, dict)
        and descriptor.get("strength") in _REUSABLE_SOURCE_STRENGTHS
        for descriptor in descriptors
    )


def require_reusable_source_identities(descriptors, *, purpose: str) -> None:
    """Fail closed when persisted results would be authorized by mutable identifiers."""
    weak = sorted({
        str(descriptor.get("strength", "missing"))
        if isinstance(descriptor, dict) else "invalid"
        for descriptor in descriptors
        if not isinstance(descriptor, dict)
        or descriptor.get("strength") not in _REUSABLE_SOURCE_STRENGTHS
    })
    if weak:
        raise ValueError(
            f"{purpose} requires immutable source identities, but source descriptor "
            f"strength(s) {weak} cannot prove content stability. Implement "
            "adapter.source_fingerprint(source), use content-addressed local inputs, or "
            "use remote objects with immutable version/ETag metadata."
        )


def assert_sources_unchanged(sources, adapter, expected: list[dict]) -> None:
    """Fail before publication if a fingerprintable input changed during the run."""
    current = source_descriptors(sources, adapter)
    if current != list(expected):
        raise ValueError(
            "one or more input sources changed between planning and output; shards were left "
            "uncommitted—rerun against an immutable snapshot or versioned source")


def sdk_version() -> str:
    try:
        return version("kuration-sdk")
    except PackageNotFoundError:
        return "unknown"


def build_run_signature(*, schema, strategy, spec, segmenter, sources, adapter, encoder,
                        seed, sample_hz, train_hours, test_hours, span_len, clip_aggregate,
                        group_of=None, tags_of=None, transforms=(), registry=None) -> dict:
    from .plugins import plugin_provenance
    from .registry import default_registry
    from .versioning import ARTIFACT_FORMAT_VERSION
    registry = registry or strategy.registry or default_registry
    refs = [("method", strategy.method), ("container", spec.container)]
    refs += [("filter", f["type"]) for f in (strategy.filters or [])]
    refs += [("scorer", name) for name, _out, _params in (strategy.scorers or [])]
    refs += [("transform", call["implementation"]) for call in transforms]
    # Registered metrics are executable plugin components too.  Their descriptor fingerprint
    # must participate in resume identity even when an entry-point initializer lives in a
    # different module from the metric implementation.
    refs += [("metric", name) for name in sorted(strategy.referenced_metrics())
             if name not in strategy.define and name in registry.names("metric")]
    extensions = []
    for kind, name in refs:
        try:
            extensions.append(registry.descriptor(kind, name).to_dict())
        except ValueError:
            pass
    payload = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "sdk_version": sdk_version(),
        "schema": schema.to_dict(),
        "strategy": strategy.resolved,
        "output": spec.to_dict(),
        "segmenter": {"window": segmenter.window, "stride": segmenter.stride},
        "seed": seed,
        "sample_hz": sample_hz,
        "budgets": {"train_hours": train_hours, "test_hours": test_hours},
        "span_len": span_len,
        "clip_aggregate": clip_aggregate,
        "adapter": component_identity(adapter),
        "encoder": component_identity(encoder),
        "group_of": component_identity(group_of),
        "tags_of": component_identity(tags_of),
        "transforms": list(transforms),
        "plugins": plugin_provenance(registry=registry),
        "extensions": extensions,
        "sources": source_descriptors(sources, adapter),
    }
    normalized = strict_json_value(payload, label="run provenance")
    assert isinstance(normalized, dict)
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"fingerprint": _sha(encoded), "payload": normalized}
