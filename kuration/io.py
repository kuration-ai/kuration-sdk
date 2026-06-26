"""Path-agnostic I/O: local by default, any fsspec backend (gs:// / s3:// / az:// / …) when a
URL scheme is present.

`fsspec` is the optional `[cloud]` extra; local paths need nothing. Specific backends
(`gcsfs`, `s3fs`, `adlfs`, …) are the caller's to install. Libraries that demand a real
filesystem path (h5py, np.load, pyarrow) go through `local_copy` (read) / `staged_write`
(write), which stage a remote object to a tempfile — so every reader/writer in the SDK is
backend-agnostic without special-casing each cloud.
"""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def is_remote(path) -> bool:
    p = str(path)
    return "://" in p and not p.startswith("file://")


def _local(path) -> str:
    p = str(path)
    return p[7:] if p.startswith("file://") else p


def _fs(path):
    try:
        import fsspec
    except ImportError as e:
        raise ImportError(
            f"remote path {path!r} needs fsspec: pip install -e '.[cloud]' (plus the backend, "
            f"e.g. gcsfs for gs://, s3fs for s3://, adlfs for az://)") from e
    return fsspec.filesystem(str(path).split("://", 1)[0])


@contextmanager
def open_file(path, mode: str = "rb"):
    if is_remote(path):
        import fsspec  # error surfaced by _fs if missing
        with fsspec.open(str(path), mode) as f:
            yield f
    else:
        p = _local(path)
        if any(m in mode for m in "wax"):
            Path(p).parent.mkdir(parents=True, exist_ok=True)
        with open(p, mode) as f:
            yield f


def read_text(path) -> str:
    with open_file(path, "rt") as f:
        return f.read()


def write_text(path, text: str) -> None:
    with open_file(path, "wt") as f:
        f.write(text)


def write_bytes(path, data: bytes) -> None:
    with open_file(path, "wb") as f:
        f.write(data)


def exists(path) -> bool:
    if is_remote(path):
        return _fs(path).exists(str(path))
    return Path(_local(path)).exists()


def glob(pattern) -> list[str]:
    """List paths matching a glob, local or remote (sorted). Remote results keep their scheme,
    so each is a usable URL. `**` is recursive in both cases."""
    if is_remote(pattern):
        fs = _fs(pattern)
        scheme = str(pattern).split("://", 1)[0]
        return sorted(str(m) if "://" in str(m) else f"{scheme}://{m}"
                      for m in fs.glob(str(pattern)))
    from glob import glob as _g
    return sorted(_g(_local(pattern), recursive=True))


@contextmanager
def local_copy(path):
    """Yield a real local path for reading; a remote object is downloaded to a tempfile."""
    if is_remote(path):
        fs = _fs(path)
        suffix = os.path.splitext(str(path))[1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.close()
        try:
            fs.get_file(str(path), tmp.name)
            yield tmp.name
        finally:
            os.unlink(tmp.name)
    else:
        yield _local(path)


@contextmanager
def staged_write(path):
    """Yield a real local path to write to; on success upload it to the (possibly remote) dest."""
    if is_remote(path):
        fs = _fs(path)
        suffix = os.path.splitext(str(path))[1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.close()
        try:
            yield tmp.name
            fs.put_file(tmp.name, str(path))
        finally:
            os.unlink(tmp.name)
    else:
        p = _local(path)
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        yield p
