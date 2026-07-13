"""Canonical run provenance and component/source fingerprints."""
from __future__ import annotations

import hashlib
import inspect
import json
import pickle
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _code_hash(obj) -> str | None:
    try:
        return _sha(inspect.getsource(obj).encode())
    except (OSError, TypeError):
        return None


def component_identity(obj) -> dict | None:
    """Stable-enough identity for configured adapters, encoders, and callbacks."""
    if obj is None:
        return None
    target = obj if inspect.isfunction(obj) else type(obj)
    out = {"type": f"{target.__module__}.{target.__qualname__}", "code_sha256": _code_hash(target)}
    if not inspect.isfunction(obj):
        try:
            out["state_sha256"] = _sha(pickle.dumps(obj, protocol=5))
        except Exception:
            out["state_sha256"] = _sha(repr(vars(obj) if hasattr(obj, "__dict__") else obj).encode())
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
    payload = {
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
        "sources": sorted((source_descriptor(s, adapter) for s in sources),
                          key=lambda d: d["source"]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return {"fingerprint": _sha(encoded), "payload": payload}
