"""Path-agnostic I/O for local paths and supported cloud object stores.

`fsspec` is the optional `[cloud]` extra; local paths need nothing. Specific backends
(`gcsfs`, `s3fs`, `adlfs`, …) are the caller's to install. Libraries that demand a real
filesystem path (h5py, np.load, pyarrow) go through `local_copy` (read) / `staged_write`
(write), which stage a remote object to a tempfile. Mutations are deliberately limited to
object-store protocols whose namespace has no symlink traversal; hierarchical fsspec backends
such as FTP/SFTP fail closed.
"""
from __future__ import annotations

import errno
import hashlib
import os
import shutil
import stat
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path


def _fsync_file(path: str) -> None:
    """Flush a completed local file before its name is published."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    """Best-effort directory-entry durability on platforms that support it."""
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def is_remote(path) -> bool:
    p = str(path)
    return "://" in p and not p.startswith("file://")


def _local(path) -> str:
    p = str(path)
    return p[7:] if p.startswith("file://") else p


def _trusted_root_alias(path: str) -> str:
    """Resolve only immutable, root-owned aliases directly below ``/``.

    macOS exposes ordinary temporary paths through root-owned ``/tmp`` and ``/var`` symlinks.
    Rejecting those aliases makes secure descriptor traversal unusable, while following arbitrary
    user-controlled ancestors recreates the race this traversal prevents. A root-owned link in a
    root-owned, non-writable filesystem root cannot be swapped by the dataset writer, so rewrite
    that narrow system alias before the component-by-component no-follow walk.
    """
    candidate = os.path.abspath(_local(path))
    for _ in range(8):
        parts = Path(candidate).parts
        if len(parts) < 2:
            return candidate
        first = os.path.join(os.path.sep, parts[1])
        try:
            link_status = os.lstat(first)
            root_status = os.lstat(os.path.sep)
        except OSError:
            return candidate
        if not stat.S_ISLNK(link_status.st_mode):
            return candidate
        if (
            link_status.st_uid != 0
            or root_status.st_uid != 0
            or root_status.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            return candidate
        target = os.readlink(first)
        if not os.path.isabs(target):
            target = os.path.join(os.path.sep, target)
        candidate = os.path.normpath(os.path.join(target, *parts[2:]))
    raise ValueError(f"local path {str(path)!r} crosses too many trusted root aliases")


_HAS_POSIX_DESCRIPTOR_OPEN = (
    os.name == "posix"
    and all(hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK"))
    and os.open in getattr(os, "supports_dir_fd", ())
)


def _local_directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _local_file_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOCTTY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _open_local_directory_descriptor(path: str, *, label: str) -> int:
    """Walk an absolute directory path without following any component."""
    candidate = _trusted_root_alias(path)
    flags = _local_directory_flags()
    descriptor = os.open(os.path.sep, flags)
    cursor = os.path.sep
    try:
        for component in Path(candidate).parts[1:]:
            cursor = os.path.join(cursor, component)
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ValueError(
                        f"{label} crosses symlink or non-directory ancestor {cursor!r}"
                    ) from error
                raise
            previous_descriptor = descriptor
            descriptor = next_descriptor
            os.close(previous_descriptor)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_local_regular_descriptor(path: str, *, label: str) -> tuple[int, os.stat_result]:
    """Open one regular file relative to a no-follow chain of pinned ancestors."""
    candidate = os.path.abspath(_local(path))
    parent, name = os.path.split(candidate)
    if not name:
        raise ValueError(f"{label} must name a regular file")
    directory_descriptor = _open_local_directory_descriptor(parent, label=label)
    flags = _local_file_flags()
    try:
        try:
            descriptor = os.open(name, flags, dir_fd=directory_descriptor)
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ValueError(f"{label} must not be a symlink") from error
            if error.errno in {errno.ENODEV, errno.ENXIO}:
                raise ValueError(f"{label} must be a regular file") from error
            raise
    finally:
        os.close(directory_descriptor)
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode):
        os.close(descriptor)
        raise ValueError(f"{label} must be a regular file")
    return descriptor, opened


def _fallback_local_regular_descriptor(
    path: str,
    *,
    label: str,
) -> tuple[int, os.stat_result]:
    """Best-effort identity pin for verified readers without POSIX openat/O_NOFOLLOW.

    Callers that lack an expected content digest must reject this fallback; committed artifact
    and checkpoint readers validate SHA-256 and size against their committed metadata after the
    snapshot is copied.
    """
    candidate = os.path.abspath(_local(path))
    before = os.lstat(candidate)
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{label} must not be a symlink")
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular file")
    flags = _local_file_flags() & ~getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(candidate, flags)
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
    ):
        os.close(descriptor)
        raise ValueError(f"{label} changed while it was being opened")
    return descriptor, opened


def _opened_local_regular_descriptor(
    path: str,
    *,
    label: str,
) -> tuple[int, os.stat_result]:
    if _HAS_POSIX_DESCRIPTOR_OPEN:
        return _open_local_regular_descriptor(path, label=label)
    return _fallback_local_regular_descriptor(path, label=label)


def _same_file_generation(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def _normalize_explicit_path(path, *, label: str) -> str:
    """Normalize a destination/work root without collapsing a bare URI into a local path."""
    value = str(path)
    if not value.strip():
        raise ValueError(f"{label} must be an explicit non-empty path")
    if value.startswith("file://"):
        local = value[7:].rstrip("/") or "/"
        resolved = Path(local).expanduser().resolve(strict=False)
        if resolved == Path(resolved.anchor):
            raise ValueError(f"{label} must not be the local filesystem root")
        return f"file://{local}"
    if "://" in value:
        scheme, remainder = value.split("://", 1)
        remainder = remainder.strip("/")
        if not scheme or not remainder:
            raise ValueError(f"{label} must name a path below the remote backend root")
        if any(component in {"", ".", ".."} for component in remainder.split("/")):
            raise ValueError(
                f"{label} contains an ambiguous remote path component"
            )
        return f"{scheme}://{remainder}"
    normalized = value.rstrip("/") or os.path.sep
    resolved = Path(normalized).expanduser().resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise ValueError(f"{label} must not be the local filesystem root")
    return normalized


def _fs(path):
    try:
        import fsspec
    except ImportError as e:
        raise ImportError(
            f"remote path {path!r} needs fsspec: pip install 'kuration-sdk[cloud]' "
            "(plus the backend, "
            f"e.g. gcsfs for gs://, s3fs for s3://, adlfs for az://)") from e
    return fsspec.filesystem(str(path).split("://", 1)[0])


_REMOTE_MUTATION_SCHEMES = frozenset({
    "abfs", "abfss", "adl", "az", "gcs", "gs", "memory", "s3", "s3a",
})


def _ensure_remote_mutation_supported(path) -> None:
    """Reject backends whose rename/link/conditional-create safety is not guaranteed."""
    if not is_remote(path):
        return
    scheme = str(path).split("://", 1)[0].lower()
    if scheme not in _REMOTE_MUTATION_SCHEMES:
        raise ValueError(
            f"remote mutation protocol {scheme!r} is unsupported; writes and deletion are "
            "limited to s3://, gs:// or gcs://, az:// or abfs://, and equivalent supported "
            "object-store URLs because hierarchical fsspec backends can traverse symlinks"
        )


@contextmanager
def open_file(path, mode: str = "rb"):
    if is_remote(path):
        try:
            fs = _fs(path)
        except ImportError as error:
            raise ImportError(f"remote path {path!r} needs fsspec and the matching backend; "
                              "install kuration-sdk[cloud]") from error
        # Open the exact backend path. Older fsspec ``OpenFile`` wrappers interpret glob
        # metacharacters even for a single file, so a literal ``[`` in a checkpoint root can
        # otherwise make an existing object unreadable or select an adjacent namespace.
        with fs.open(str(path), mode) as f:
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
    """Publish UTF-8 text with atomic local replacement or a complete staged remote object."""
    with staged_write(path) as temporary:
        with open(temporary, "wt", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())


def write_bytes(path, data: bytes) -> None:
    """Publish bytes with atomic local replacement or a complete staged remote object."""
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


def _copy_pinned_directory(
    source_descriptor: int,
    destination: Path,
    *,
    label: str,
) -> None:
    """Recursively copy a directory using only names relative to pinned descriptors."""
    opened_directory = os.fstat(source_descriptor)
    if not stat.S_ISDIR(opened_directory.st_mode):
        raise ValueError(f"{label} must be a directory")
    with os.scandir(source_descriptor) as iterator:
        names = sorted(entry.name for entry in iterator)
    for name in names:
        before = os.stat(name, dir_fd=source_descriptor, follow_symlinks=False)
        target = destination / name
        if stat.S_ISLNK(before.st_mode):
            raise ValueError(f"{label} contains symlink entry {name!r}")
        if stat.S_ISDIR(before.st_mode):
            child_descriptor = os.open(
                name,
                _local_directory_flags(),
                dir_fd=source_descriptor,
            )
            try:
                child_opened = os.fstat(child_descriptor)
                if (
                    not stat.S_ISDIR(child_opened.st_mode)
                    or (child_opened.st_dev, child_opened.st_ino)
                    != (before.st_dev, before.st_ino)
                ):
                    raise ValueError(
                        f"{label} directory entry {name!r} changed while it was opened"
                    )
                target.mkdir()
                _copy_pinned_directory(
                    child_descriptor,
                    target,
                    label=f"{label}/{name}",
                )
            finally:
                os.close(child_descriptor)
            continue
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} contains non-regular entry {name!r}")
        descriptor = os.open(name, _local_file_flags(), dir_fd=source_descriptor)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            ):
                raise ValueError(
                    f"{label} file entry {name!r} changed while it was opened"
                )
            with os.fdopen(descriptor, "rb") as source, target.open("xb") as output:
                descriptor = -1
                shutil.copyfileobj(source, output, length=1024 * 1024)
                copied = os.fstat(source.fileno())
                output.flush()
                os.fsync(output.fileno())
            if not _same_file_generation(opened, copied):
                raise ValueError(f"{label} file entry {name!r} changed while it was copied")
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    after_directory = os.fstat(source_descriptor)
    if (
        opened_directory.st_dev,
        opened_directory.st_ino,
        opened_directory.st_mtime_ns,
        opened_directory.st_ctime_ns,
    ) != (
        after_directory.st_dev,
        after_directory.st_ino,
        after_directory.st_mtime_ns,
        after_directory.st_ctime_ns,
    ):
        raise ValueError(f"{label} changed while it was copied")


@contextmanager
def local_tree_snapshot(path):
    """Yield a private copy of one local directory without following source symlinks."""
    if is_remote(path):
        raise ValueError("remote directory snapshots require a backend-specific versioned API")
    local_path = os.path.abspath(_local(path))
    with tempfile.TemporaryDirectory(prefix="kuration-tree-snapshot-") as temporary_root:
        snapshot = Path(temporary_root) / (Path(local_path).name or "source")
        if _HAS_POSIX_DESCRIPTOR_OPEN:
            descriptor = _open_local_directory_descriptor(
                local_path,
                label=f"local snapshot source {str(path)!r}",
            )
            try:
                snapshot.mkdir()
                _copy_pinned_directory(
                    descriptor,
                    snapshot,
                    label=f"local snapshot source {str(path)!r}",
                )
            finally:
                os.close(descriptor)
        else:
            raise ValueError(
                "safe local directory snapshots require POSIX descriptor-relative open "
                "support; copy this directory to an immutable file-backed source first"
            )
        yield str(snapshot)


@contextmanager
def local_snapshot(path):
    """Copy one file into a private local snapshot and yield the immutable snapshot path.

    Unlike :func:`local_copy`, local inputs are copied too. POSIX local paths are opened through
    a component-by-component no-follow directory-descriptor walk, then copied from the pinned
    regular-file descriptor. An ancestor or leaf replacement therefore cannot redirect the
    bytes later verified and decoded, and an in-place mutation during the copy fails closed.
    Other platforms receive only a leaf-identity-checked fallback; callers there must bind the
    resulting snapshot to an expected digest (unverified import paths reject that fallback).
    """
    suffix = os.path.splitext(str(path))[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    try:
        if is_remote(path):
            with open_file(path, "rb") as source:
                shutil.copyfileobj(source, tmp, length=1024 * 1024)
        else:
            descriptor, opened = _opened_local_regular_descriptor(
                str(path), label=f"local snapshot source {str(path)!r}"
            )
            try:
                with os.fdopen(descriptor, "rb") as source:
                    descriptor = -1
                    shutil.copyfileobj(source, tmp, length=1024 * 1024)
                    copied = os.fstat(source.fileno())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            if not _same_file_generation(opened, copied):
                raise ValueError(
                    f"local snapshot source {str(path)!r} changed while it was copied"
                )
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        yield tmp_path
    finally:
        if not tmp.closed:
            tmp.close()
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@contextmanager
def staged_write(path):
    """Yield a real local path to write to; on success upload it to the (possibly remote) dest.
    Local destinations use atomic replace. Remote destinations upload a complete unique attempt
    before replacement and never pre-delete the prior generation."""
    if is_remote(path):
        _ensure_remote_mutation_supported(path)
        if any(character in str(path) for character in "*?["):
            raise ValueError(
                f"remote destination {str(path)!r} contains a glob metacharacter; fsspec "
                "mutation APIs do not consistently treat those names literally"
            )
        fs = _fs(path)
        suffix = os.path.splitext(str(path))[1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.close()
        remote_attempt = f"{str(path)}.kuration-attempt-{uuid.uuid4().hex}"
        attempt_published = False
        try:
            yield tmp.name
            _fsync_file(tmp.name)
            # Upload under a unique invisible-to-manifest name, then publish only the completed
            # object. Object-store moves may be copy+delete, but consumers can never observe a
            # partially uploaded final payload.
            fs.put_file(tmp.name, remote_attempt)
            # Never pre-delete the last good generation. Backends that cannot replace an existing
            # key must fail while leaving it intact rather than creating a destructive gap.
            fs.mv(remote_attempt, str(path))
            attempt_published = True
        finally:
            if not attempt_published and fs.exists(remote_attempt):
                fs.rm_file(fs._strip_protocol(remote_attempt))
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
    else:
        p = _local(path)
        parent = Path(p).parent
        missing_directories: list[Path] = []
        cursor = parent
        while not cursor.exists():
            missing_directories.append(cursor)
            cursor = cursor.parent
        parent.mkdir(parents=True, exist_ok=True)
        # A securely created same-directory file keeps the real extension (np.savez appends
        # '.npz' otherwise) and makes os.replace atomic without cross-filesystem surprises.
        root, ext = os.path.splitext(p)
        handle = tempfile.NamedTemporaryFile(dir=str(parent),
                                             prefix=f".{Path(root).name}.", suffix=ext,
                                             delete=False)
        tmp_p = handle.name
        handle.close()
        try:
            yield tmp_p
            _fsync_file(tmp_p)
            os.replace(tmp_p, p)
            # Persist the replacement and any parent directories created for this publication.
            _fsync_directory(parent)
            for directory in missing_directories:
                _fsync_directory(directory.parent)
        finally:
            if os.path.exists(tmp_p):
                os.unlink(tmp_p)
