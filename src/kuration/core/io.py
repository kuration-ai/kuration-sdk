"""Path-agnostic I/O: local by default, any fsspec backend (gs:// / s3:// / az:// / …) when a
URL scheme is present.

`fsspec` is the optional `[cloud]` extra; local paths need nothing. Specific backends
(`gcsfs`, `s3fs`, `adlfs`, …) are the caller's to install. Libraries that demand a real
filesystem path (h5py, np.load, pyarrow) go through `local_copy` (read) / `staged_write`
(write), which stage a remote object to a tempfile — so every reader/writer in the SDK is
backend-agnostic without special-casing each cloud.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
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
            f"remote path {path!r} needs fsspec: pip install 'kuration-sdk[cloud]' "
            "(plus the backend, "
            f"e.g. gcsfs for gs://, s3fs for s3://, adlfs for az://)") from e
    return fsspec.filesystem(str(path).split("://", 1)[0])


@contextmanager
def open_file(path, mode: str = "rb"):
    if is_remote(path):
        try:
            import fsspec
        except ImportError as error:
            raise ImportError(f"remote path {path!r} needs fsspec and the matching backend; "
                              "install kuration-sdk[cloud]") from error
        with fsspec.open(str(path), mode) as f:
            yield f
    else:
        p = _local(path)
        if any(m in mode for m in "wax"):
            Path(p).parent.mkdir(parents=True, exist_ok=True)
        with open(p, mode) as f:
            yield f


def read_text(path) -> str:
    """Read UTF-8-compatible text from a local path or fsspec URL."""
    with open_file(path, "rt") as f:
        return f.read()


def write_text(path, text: str) -> None:
    """Atomically publish UTF-8 text to a local path or staged remote destination."""
    with staged_write(path) as temporary:
        with open(temporary, "wt", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())


def write_bytes(path, data: bytes) -> None:
    """Atomically publish bytes to a local path or staged remote destination."""
    with staged_write(path) as temporary:
        with open(temporary, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())


def exists(path) -> bool:
    """Return whether a local path or fsspec URL exists."""
    if is_remote(path):
        return _fs(path).exists(str(path))
    return Path(_local(path)).exists()


def file_integrity(path, *, chunk_size: int = 1024 * 1024) -> dict[str, int | str]:
    """Return a path-agnostic SHA-256 and byte size without materializing the payload."""
    digest = hashlib.sha256()
    size = 0
    with open_file(path, "rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": size}


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
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
    else:
        yield _local(path)


@contextmanager
def staged_write(path):
    """Yield a real local path to write to; on success upload it to the (possibly remote) dest.
    Local destinations use atomic replace. Remote destinations upload a complete unique attempt
    before replacement and never pre-delete the prior generation."""
    if is_remote(path):
        fs = _fs(path)
        suffix = os.path.splitext(str(path))[1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.close()
        remote_attempt = f"{str(path)}.kuration-attempt-{uuid.uuid4().hex}"
        try:
            yield tmp.name
            # Upload under a unique invisible-to-manifest name, then publish only the completed
            # object. Object-store moves may be copy+delete, but consumers can never observe a
            # partially uploaded final payload.
            fs.put_file(tmp.name, remote_attempt)
            # Never pre-delete the last good generation. Backends that cannot replace an existing
            # key must fail while leaving it intact rather than creating a destructive gap.
            fs.mv(remote_attempt, str(path))
        finally:
            if fs.exists(remote_attempt):
                fs.rm(remote_attempt)
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
    else:
        p = _local(path)
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        # A securely created same-directory file keeps the real extension (np.savez appends
        # '.npz' otherwise) and makes os.replace atomic without cross-filesystem surprises.
        root, ext = os.path.splitext(p)
        handle = tempfile.NamedTemporaryFile(dir=str(Path(p).parent),
                                             prefix=f".{Path(root).name}.", suffix=ext,
                                             delete=False)
        tmp_p = handle.name
        handle.close()
        try:
            yield tmp_p
            os.replace(tmp_p, p)
            # Persist the directory entry where the platform supports directory fsync.
            try:
                directory_fd = os.open(str(Path(p).parent), os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        finally:
            if os.path.exists(tmp_p):
                os.unlink(tmp_p)
