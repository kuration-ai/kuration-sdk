"""Canonical run provenance and component/source fingerprints."""
from __future__ import annotations

import hashlib
import inspect
import json
import marshal
import pickle
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _code_hash(obj) -> str | None:
    try:
        return _sha(inspect.getsource(obj).encode())
    except (OSError, TypeError):
        code = getattr(obj, "__code__", None)
        return _sha(marshal.dumps(code)) if code is not None else None


def _state_hash(value) -> str:
    """Best-effort stable state hash. Repr is a conservative fallback for unpicklable plugin
    state; it is preferable to omitting state and incorrectly approving a stale resume."""
    try:
        return _sha(pickle.dumps(value, protocol=5))
    except Exception:
        return _sha(repr(value).encode())


def _function_state(fn) -> dict:
    closure = []
    for cell in fn.__closure__ or ():
        try:
            closure.append(cell.cell_contents)
        except ValueError:                     # empty closure cell
            closure.append("<empty closure cell>")
    return {"defaults": fn.__defaults__, "kwdefaults": fn.__kwdefaults__,
            "closure": tuple(closure)}


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


def source_descriptor(source, adapter) -> dict:
    """Best available source identity without reading the source payload a second time."""
    custom = getattr(adapter, "source_fingerprint", None)
    if custom is not None:
        return {"source": str(source), "fingerprint": custom(source)}

    from .io import is_remote
    s = str(source)
    if is_remote(s):
        from .io import _fs
        info = _fs(s).info(s)
        keep = ("size", "mtime", "updated", "etag", "ETag", "checksum", "version_id", "generation")
        return {"source": s, "metadata": {k: str(info[k]) for k in keep if k in info}}
    p = Path(s[7:] if s.startswith("file://") else s)
    if p.exists():
        st = p.stat()
        return {"source": s, "metadata": {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
                                             "is_dir": p.is_dir()}}
    return {"source": s, "metadata": {"non_path": True}}


def sdk_version() -> str:
    try:
        return version("kuration-sdk")
    except PackageNotFoundError:
        return "unknown"


def build_run_signature(*, schema, strategy, spec, segmenter, sources, adapter, encoder,
                        seed, sample_hz, train_hours, test_hours, span_len, clip_aggregate,
                        group_of=None, tags_of=None) -> dict:
    from .plugins import plugin_provenance
    from .registry import default_registry
    from .versioning import ARTIFACT_FORMAT_VERSION
    registry = strategy.registry or default_registry
    refs = [("method", strategy.method), ("container", spec.container)]
    refs += [("filter", f["type"]) for f in (strategy.filters or [])]
    refs += [("scorer", name) for name, _out, _params in (strategy.scorers or [])]
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
        "plugins": plugin_provenance(),
        "extensions": extensions,
        "sources": sorted((source_descriptor(s, adapter) for s in sources),
                          key=lambda d: d["source"]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return {"fingerprint": _sha(encoded), "payload": payload}
