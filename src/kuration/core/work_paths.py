"""Preflight isolation for bounded checkpoint and reducer-spill work paths."""
from __future__ import annotations

import os
from pathlib import Path

from .io import _local, _normalize_explicit_path, is_remote


def _normalize_path(value: str, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a non-empty path string")
    return _normalize_explicit_path(value, label=label)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _symlink_error(label: str, work_path: str, link: Path) -> ValueError:
    return ValueError(
        f"{label} {work_path!r} contains symlink {str(link)!r}; bounded work paths must "
        "not redirect checkpoint or spill writes outside their intended root"
    )


def _validate_local_symlinks(work_path: str, *, label: str, out_dir: str) -> None:
    """Reject symlinks at/inside a work root and nested redirects below the output root."""
    if is_remote(work_path):
        return
    work = Path(os.path.abspath(_local(work_path)))

    # Defaults, and explicit work paths placed beneath the output, have a stronger containment
    # boundary: every existing component below the destination must remain a real directory.
    if not is_remote(out_dir):
        output = Path(os.path.abspath(_local(out_dir)))
        if _is_relative_to(work, output):
            current = output
            for component in work.relative_to(output).parts:
                current = current / component
                if current.is_symlink():
                    raise _symlink_error(label, work_path, current)
            resolved_output = output.resolve(strict=False)
            resolved_work = work.resolve(strict=False)
            if not _is_relative_to(resolved_work, resolved_output):
                raise ValueError(
                    f"{label} {work_path!r} resolves outside output root {out_dir!r}"
                )

    if work.is_symlink():
        raise _symlink_error(label, work_path, work)
    if not work.is_dir():
        return

    # Existing checkpoint identities and stage/data/commit namespaces are reusable state. A
    # symlink anywhere in that tree could make validation or a later atomic replace cross the
    # declared work-root boundary.
    for directory, directory_names, file_names in os.walk(work, followlinks=False):
        parent = Path(directory)
        for name in [*directory_names, *file_names]:
            candidate = parent / name
            if candidate.is_symlink():
                raise _symlink_error(label, work_path, candidate)


def resolve_bounded_work_paths(
    *,
    out_dir: str,
    sources,
    adapter,
    checkpoint_dir: str | None,
    spill_dir: str | None,
) -> tuple[str, str | None]:
    """Resolve and validate the exact checkpoint/spill roots before bounded execution writes."""
    output = _normalize_path(str(out_dir), label="out_dir")
    checkpoint = _normalize_path(
        checkpoint_dir
        if checkpoint_dir is not None
        else f"{output}/.kuration/checkpoints",
        label="checkpoint_dir",
    )
    spill = (
        _normalize_path(spill_dir, label="spill_dir")
        if spill_dir is not None
        else None if is_remote(output)
        else _normalize_path(f"{output}/.kuration/spill", label="spill_dir")
    )

    from .manifest import validate_source_destination_separation

    for label, path in (("checkpoint_dir", checkpoint), ("spill_dir", spill)):
        if path is None:
            continue
        try:
            validate_source_destination_separation(path, sources, adapter=adapter)
        except ValueError as error:
            raise ValueError(
                f"{label} {path!r} overlaps a source; choose a separate bounded work path. "
                f"{error}"
            ) from error
        if is_remote(path) and any(character in path for character in "*?["):
            raise ValueError(
                f"{label} {path!r} contains a glob metacharacter; remote work paths must "
                "not contain '*', '?', or '['"
            )
        if is_remote(path):
            from .io import _ensure_remote_mutation_supported
            _ensure_remote_mutation_supported(path)
        _validate_local_symlinks(path, label=label, out_dir=output)

    if spill is not None and is_remote(spill):
        raise ValueError(
            "bounded disk-backed reducers require a local spill_dir; choose a local directory"
        )
    return checkpoint, spill
