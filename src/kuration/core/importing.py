"""Smart registered-adapter import into Kuration's artifact format."""
from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator, Mapping
from functools import wraps
from typing import Any, NoReturn, Sequence

import numpy as np

from .io import (
    file_integrity,
    is_remote,
    local_snapshot,
    local_tree_snapshot,
    write_text,
)
from .manifest import (
    begin_dataset,
    clear_dataset_resume_marker,
    clear_dataset_shards,
    commit_dataset,
    dataset_state,
    pin_writing_dataset,
    portable_shard_entries,
)
from .output.containers import validate_container_availability, write_shard
from .output.shardwriter import _safe_split, entity_id_token, track_name
from .output.spec import OutputSpec, validate_modality_name
from .provenance import (
    publishable_source_id,
    rebase_source_descriptor,
    source_descriptor,
)
from .versioning import ARTIFACT_FORMAT_VERSION


@dataclass(frozen=True)
class _PinnedImportSource:
    """One caller-visible source and the private immutable path adapters must read."""

    logical_path: str
    read_path: str


def _reject_generic_remote_import(path: str) -> NoReturn:
    raise ValueError(
        "remote import source "
        f"{publishable_source_id(path)!r} cannot be generation-pinned through the "
        "generic fsspec read API; download the exact immutable object generation "
        "to a local file first, or use a local mount that preserves that version"
    )


def _require_secure_local_import_support() -> None:
    from . import io as io_module

    if not io_module._HAS_POSIX_DESCRIPTOR_OPEN:
        raise ValueError(
            "local import requires POSIX descriptor-relative open support so source ancestry "
            "can be pinned before validation; on this platform, copy and validate the source "
            "with a trusted content-digest workflow before converting it"
        )


@contextmanager
def _pin_import_sources(paths: Sequence[str]) -> Iterator[list[_PinnedImportSource]]:
    """Stage each input once, before validation, and retain it through publication.

    Local files are copied from one already-open stream. Local directory stores are copied into
    a private tree before an adapter sees them. Generic fsspec reads cannot bind a GET to the
    metadata generation returned by ``info()``, so remote sources fail closed instead of relying
    on a vulnerable before/after metadata comparison.
    """
    pinned: list[_PinnedImportSource] = []
    with ExitStack() as stack:
        for path in paths:
            if is_remote(path):
                _reject_generic_remote_import(path)

            _require_secure_local_import_support()
            local_path = Path(path[7:] if path.startswith("file://") else path)
            if local_path.is_dir():
                read_path = stack.enter_context(local_tree_snapshot(path))
            else:
                read_path = stack.enter_context(local_snapshot(path))
            pinned.append(_PinnedImportSource(path, read_path))
        yield pinned


def _pinned_source_descriptor(source: _PinnedImportSource) -> dict:
    """Describe exactly the staged bytes while retaining the logical source reference."""
    descriptor = rebase_source_descriptor(
        source_descriptor(source.read_path, object()),
        source.logical_path,
    )
    return descriptor


def _align_pipeline_sources(
    roots: Sequence[_PinnedImportSource],
    read_paths: Sequence[str],
) -> list[_PinnedImportSource]:
    """Map adapter-expanded snapshot members back to their caller-visible tree paths."""
    aligned: list[_PinnedImportSource] = []
    for read_path in read_paths:
        match = next((root for root in roots if root.read_path == read_path), None)
        if match is not None:
            aligned.append(match)
            continue
        read = Path(read_path)
        for root in roots:
            root_read = Path(root.read_path)
            try:
                relative = read.relative_to(root_read)
            except ValueError:
                continue
            logical_prefix = "file://" if root.logical_path.startswith("file://") else ""
            logical_root = Path(
                root.logical_path[7:] if logical_prefix else root.logical_path
            )
            aligned.append(_PinnedImportSource(
                f"{logical_prefix}{logical_root / relative}",
                read_path,
            ))
            break
        else:
            raise ValueError(
                "internal import error: adapter exposed a source outside its pinned snapshot"
            )
    return aligned


def _discover_npz_sources(sources: Sequence[str | Path]) -> list[str]:
    discovered: list[str] = []
    for raw in sources:
        source = str(raw)
        if is_remote(source):
            if not source.lower().endswith(".npz"):
                raise ValueError(f"remote import source must be an .npz file, got {source!r}")
            discovered.append(source)
            continue
        path = Path(source)
        if path.is_dir():
            discovered.extend(str(candidate) for candidate in sorted(path.rglob("*.npz")))
        elif path.is_file() and path.suffix.lower() == ".npz":
            discovered.append(str(path))
        elif not path.exists():
            raise FileNotFoundError(f"import source not found: {source}")
        else:
            raise ValueError(f"import source must be an .npz file or directory, got {source!r}")
    unique = list(dict.fromkeys(discovered))
    if not unique:
        raise ValueError("no .npz files found in the import sources")
    return unique


def _source_name(path: str) -> str:
    if is_remote(path):
        from urllib.parse import urlsplit
        clean = urlsplit(path).path.rstrip("/").rsplit("/", 1)[-1]
    else:
        clean = path.rstrip("/").rsplit("/", 1)[-1]
    return clean[:-4] if clean.lower().endswith(".npz") else clean


def _load_adapter_arrays(factory, path: str) -> tuple[dict[str, np.ndarray], int]:
    loader = getattr(factory, "import_arrays", None)
    if not callable(loader):
        raise ValueError(f"adapter {factory!r} does not expose import_arrays()")
    try:
        loaded = loader(path)
    except ValueError as error:
        if "object arrays" in str(error).lower() or "allow_pickle" in str(error).lower():
            raise ValueError(
                f"cannot safely import {path!r}; object arrays and pickled values are unsupported"
            ) from error
        raise
    if not isinstance(loaded, Mapping):
        raise ValueError("adapter import_arrays() must return a mapping of modality arrays")
    arrays: dict[str, np.ndarray] = {}
    raw_n_steps = None
    found_n_steps = False
    for name, value in loaded.items():
        if not isinstance(name, str):
            raise ValueError(
                f"import source {path!r} array names must be strings, got {name!r}"
            )
        if name == "_n_steps":
            if found_n_steps:
                raise ValueError(f"import source {path!r} repeats reserved _n_steps metadata")
            raw_n_steps = np.asarray(value)
            found_n_steps = True
            continue
        try:
            validate_modality_name(name, allow_presence_mask=name == "_present")
        except ValueError as error:
            raise ValueError(
                f"import source {path!r} has invalid array name {name!r}: {error}"
            ) from error
        if name in arrays:
            raise ValueError(f"import source {path!r} repeats array name {name!r}")
        arrays[name] = np.asarray(value)
    if not arrays:
        raise ValueError(f"import source {path!r} contains no arrays")
    for name, array in arrays.items():
        if array.ndim < 1:
            raise ValueError(
                f"import source {path!r} array {name!r} needs a leading row dimension"
            )
        if array.dtype.hasobject or array.dtype.fields is not None:
            raise ValueError(
                f"import source {path!r} array {name!r} uses unsupported dtype {array.dtype}"
            )
    lengths = {name: int(array.shape[0]) for name, array in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"import source {path!r} arrays disagree on row count: {lengths}")
    stored_rows = next(iter(lengths.values()))
    if raw_n_steps is None:
        n_steps = stored_rows
    else:
        if raw_n_steps.ndim != 0 or raw_n_steps.dtype.kind not in "iu":
            raise ValueError(f"import source {path!r} _n_steps must be an integer scalar")
        n_steps = int(raw_n_steps)
        if not 0 <= n_steps <= stored_rows:
            raise ValueError(
                f"import source {path!r} _n_steps={n_steps} is outside [0, {stored_rows}]"
            )
    if n_steps == 0:
        raise ValueError(f"import source {path!r} has no rows")
    return {name: array[:n_steps] for name, array in arrays.items()}, n_steps


def _prepare_destination(out: str, *, force: bool) -> str:
    """Read-only destination guard performed before any source is materialized.

    A committed dataset and a half-written one need different messages: the second is not
    readable at all, so "already contains a dataset" misleads a user who is simply retrying a
    failed import. Destructive replacement is deferred until all source/output preflight passes.
    """
    state = dataset_state(out)
    if state == "committed" and not force:
        raise FileExistsError(
            f"output {out!r} already contains a committed Kuration dataset; pass force=True "
            "to replace it"
        )
    if state == "writing" and not force:
        raise FileExistsError(
            f"output {out!r} holds a partially written Kuration dataset: a previous import "
            "failed part-way and never committed, so the artifact cannot be read; pass "
            "force=True to discard the leftovers and re-import"
        )
    if state == "corrupt" and not force:
        raise FileExistsError(
            f"output {out!r} contains damaged or incomplete Kuration metadata/shards; pass "
            "force=True to replace it or choose a fresh out directory"
        )
    return state


def _begin_import_destination(out: str, *, state: str, force: bool) -> list[str]:
    """Enter the writing state, then perform the explicitly forced cleanup."""
    begin_dataset(out)
    removed = clear_dataset_shards(out) if force and state != "absent" else []
    clear_dataset_resume_marker(out)
    pin_writing_dataset(out)
    return removed


def _contract(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {"shape": list(array.shape[1:]), "dtype": str(array.dtype)}
        for name, array in sorted(arrays.items())
    }


def _import_array_dataset(
    sources: list[_PinnedImportSource],
    out: str | Path,
    *,
    adapter_name: str,
    factory,
    source_id: str | None = None,
    split: str = "train",
    shard_length: int | None = None,
    container: str = "npz",
    force: bool = False,
    destination_state: str | None = None,
    registry=None,
) -> dict:
    """Import through an adapter's lossless ``import_arrays`` hook."""
    logical_paths = [source.logical_path for source in sources]
    if source_id is not None:
        if len(sources) != 1:
            raise ValueError("source_id can only be used when importing one .npz file")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string")
    if shard_length is not None and (
        not isinstance(shard_length, int)
        or isinstance(shard_length, bool)
        or shard_length < 1
    ):
        raise ValueError("shard_length must be a positive integer or None")
    if not isinstance(force, bool):
        raise ValueError("force must be true or false")
    split = _safe_split(split)
    from .io import _normalize_explicit_path
    out = _normalize_explicit_path(out, label="out")
    from .manifest import validate_source_destination_separation
    validate_source_destination_separation(out, logical_paths)
    validate_container_availability(container, registry=registry)
    if destination_state is None:
        destination_state = _prepare_destination(out, force=force)

    source_ids = [source_id or _source_name(path) for path in logical_paths]
    if any(not name for name in source_ids):
        raise ValueError("every import source must produce a non-empty source ID")
    duplicates = sorted({name for name in source_ids if source_ids.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"import sources produce duplicate source IDs {duplicates}; rename the files "
            "or import them separately with source_id="
        )

    expected_contract = None
    source_records = []
    prepared_sources = []
    total_steps = 0
    max_source_steps = 0
    for pinned_source, current_source_id in zip(sources, source_ids):
        path = pinned_source.read_path
        logical_path = pinned_source.logical_path
        source_before = _pinned_source_descriptor(pinned_source)
        arrays, n_steps = _load_adapter_arrays(factory, path)
        source_after = _pinned_source_descriptor(pinned_source)
        if source_after != source_before:
            raise ValueError(
                f"private snapshot for import source {logical_path!r} changed while it was "
                "being read"
            )
        current_contract = _contract(arrays)
        if expected_contract is None:
            expected_contract = current_contract
        elif current_contract != expected_contract:
            raise ValueError(
                f"import source {logical_path!r} has a different modality contract: "
                f"expected {expected_contract}, got {current_contract}"
            )
        if "_present" in arrays:
            present = arrays["_present"]
            if present.shape != (n_steps,) or present.dtype != np.dtype("bool"):
                raise ValueError(
                    f"import source {logical_path!r} _present must have shape ({n_steps},) "
                    "and dtype bool"
                )
        candidate_modalities = {
            name: {**values, "source": f"encoder:import_{name}"}
            for name, values in current_contract.items()
            if name != "_present"
        }
        if not candidate_modalities:
            raise ValueError(f"import source {logical_path!r} contains no data modalities")
        OutputSpec.from_dict({
            "container": container,
            "shard_length": shard_length or n_steps,
            "short_policy": "keep",
            "presence_mask": "_present" in arrays,
            "modalities": candidate_modalities,
        }, registry=registry)
        prepared_sources.append(
            (pinned_source, current_source_id, n_steps, current_contract, source_after))
        source_records.append(source_after)
        total_steps += n_steps
        max_source_steps = max(max_source_steps, n_steps)

    # Recheck every source immediately before crossing the publication boundary. The write pass
    # below still verifies each source after reading, but a change already visible here must not
    # invalidate an existing forced destination.
    for pinned_source, _source_id, _n_steps, _contract_value, expected_source in (
            prepared_sources):
        if _pinned_source_descriptor(pinned_source) != expected_source:
            raise ValueError(
                f"private snapshot for import source {pinned_source.logical_path!r} changed "
                "after preflight"
            )

    removed_shards = _begin_import_destination(
        out, state=destination_state, force=force)
    entries: list[dict] = []
    for pinned_source, current_source_id, expected_steps, current_contract, expected_source in (
            prepared_sources):
        path = pinned_source.read_path
        logical_path = pinned_source.logical_path
        arrays, n_steps = _load_adapter_arrays(factory, path)
        source_after = _pinned_source_descriptor(pinned_source)
        if source_after != expected_source:
            raise ValueError(
                f"private snapshot for import source {logical_path!r} changed after preflight"
            )
        if n_steps != expected_steps or _contract(arrays) != current_contract:
            raise ValueError(
                f"private snapshot for import source {logical_path!r} changed its array "
                "contract after preflight")
        width = shard_length or n_steps
        published_source_id = publishable_source_id(current_source_id)
        basename = track_name(published_source_id, 0)
        for shard_index, start in enumerate(range(0, n_steps, width)):
            end = min(start + width, n_steps)
            piece = {name: array[start:end] for name, array in arrays.items()}
            base = f"{out}/{split}/{basename}_shard{shard_index:04d}"
            written = write_shard(
                base, container, piece, end - start, registry=registry)
            entries.append({
                "path": written,
                "source_id": published_source_id,
                "entity_id": entity_id_token(0),
                "split": split,
                "track_id": basename,
                "shard_index": shard_index,
                "n_steps": end - start,
                "padded": False,
                "shape": {
                    name: values["shape"] for name, values in current_contract.items()
                },
                "dtype": {
                    name: values["dtype"] for name, values in current_contract.items()
                },
                **file_integrity(written),
            })

    for pinned_source, _source_id, _n_steps, _contract_value, expected_source in (
            prepared_sources):
        if _pinned_source_descriptor(pinned_source) != expected_source:
            raise ValueError(
                f"private snapshot for import source {pinned_source.logical_path!r} changed "
                "during publication; shards were left uncommitted"
            )

    expected_contract = expected_contract or {}
    presence_mask = "_present" in expected_contract
    modalities = {
        name: {**values, "source": f"encoder:import_{name}"}
        for name, values in expected_contract.items()
        if name != "_present"
    }
    results = {
        "shards": len(entries),
        "sources": len(sources),
        "steps": total_steps,
        "sources_skipped": 0,
    }
    fingerprint_payload = json.dumps(
        {
            "kind": "adapter-array-import",
            "adapter": adapter_name,
            "sources": source_records,
            "source_id": source_id,
            "container": container,
            "split": split,
            "shard_length": shard_length,
            "presence_mask": presence_mask,
            "modalities": modalities,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    config = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "output": {
            "container": container,
            "shard_length": shard_length or max_source_steps,
            "short_policy": "keep",
            "presence_mask": presence_mask,
            "modalities": modalities,
        },
        "results": results,
        "import": {"adapter": adapter_name, "mode": "arrays", "sources": source_records},
        "provenance": {
            "kind": "import",
            "fingerprint": hashlib.sha256(fingerprint_payload).hexdigest(),
        },
        "errors": [],
    }
    portable = portable_shard_entries(out, entries)
    write_text(
        f"{out}/dataset_manifest.json",
        json.dumps({
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "commit_protocol": "sidecar-v1",
            "count": len(entries),
            "shards": portable,
        }, indent=2),
    )
    write_text(f"{out}/config.json", json.dumps(config, indent=2))
    commit_dataset(out)
    return {
        "dataset": out,
        "adapter": adapter_name,
        **results,
        "modalities": sorted(expected_contract),
        **({"replaced_shards": len(removed_shards)} if removed_shards else {}),
    }


def _given_options(**options) -> list[str]:
    """Names of the options the caller actually set (an empty mapping counts as unset)."""
    return sorted(
        name for name, value in options.items()
        if value is not None and not (isinstance(value, Mapping) and not value)
    )


def _load_schema(schema):
    if schema is None:
        return None
    from .schema import SchemaSpec

    if isinstance(schema, SchemaSpec):
        return schema
    if isinstance(schema, (str, Path)):
        schema = json.loads(Path(schema).read_text())
    if not isinstance(schema, dict):
        raise ValueError("schema must be a SchemaSpec, JSON object, path, or None")
    return SchemaSpec.from_dict(schema)


def _source_ids(paths: list[str], source_id: str | None) -> list[str]:
    if source_id is not None:
        if len(paths) != 1:
            raise ValueError("source_id can only be used when the adapter resolves one source")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string")
        return [source_id]
    ids = [_source_name(path) for path in paths]
    duplicates = sorted({name for name in ids if ids.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"adapter sources produce duplicate source IDs {duplicates}; rename the inputs "
            "or import them separately with source_id="
        )
    return ids


def _import_tick_dataset(
    pipeline,
    pinned_sources: list[_PinnedImportSource],
    out: str | Path,
    *,
    adapter_name: str,
    source_id: str | None,
    split: str,
    shard_length: int | None,
    container: str,
    force: bool,
    destination_state: str | None = None,
) -> dict:
    """Materialize every canonical tick from a configured input adapter."""
    from .output.shardwriter import write_track
    from .output.track import tracks_from_ticks

    schema = pipeline.schema
    adapter = pipeline.adapter
    paths = list(pipeline.sources)
    logical_paths = [source.logical_path for source in pinned_sources]
    if paths != [source.read_path for source in pinned_sources]:
        raise ValueError("internal import error: pipeline sources do not match pinned snapshots")
    from .io import _normalize_explicit_path
    out = _normalize_explicit_path(out, label="out")
    from .manifest import validate_source_destination_separation
    validate_source_destination_separation(out, logical_paths, adapter=adapter)
    if destination_state is None:
        destination_state = _prepare_destination(out, force=force)
    if schema.event_streams:
        # `tracks_from_ticks` carries fields, globals and media into records; there is no
        # event kind in the output-modality grammar, so an event stream would vanish from the
        # shards without a trace. Refuse instead of writing a lossy artifact.
        raise ValueError(
            f"import cannot materialize event streams to shards yet; schema "
            f"{schema.name!r} declares event stream(s) "
            f"{sorted(schema.event_streams)}. Import the per-timestep fields only (drop the "
            "event_streams block from schema=), or curate the source instead of importing it."
        )
    ids = _source_ids(logical_paths, source_id)
    tracks = []
    source_records = []
    tick_count = 0
    for path, pinned_source, current_source_id in zip(paths, pinned_sources, ids):
        logical_path = pinned_source.logical_path
        before = _pinned_source_descriptor(pinned_source)
        iterator = (
            adapter.iter_ticks(path)
            if callable(getattr(adapter, "iter_ticks", None))
            else adapter.ticks(path)
        )
        ticks = list(iterator)
        if not ticks:
            raise ValueError(
                f"adapter {adapter_name!r} produced no ticks for {logical_path!r}"
            )
        after = _pinned_source_descriptor(pinned_source)
        if after != before:
            raise ValueError(
                f"private snapshot for import source {logical_path!r} changed while it was "
                "being read"
            )
        current_tracks = tracks_from_ticks(schema, ticks, current_source_id)
        if not current_tracks:
            raise ValueError(
                f"adapter {adapter_name!r} produced no entities for {logical_path!r}"
            )
        tracks.extend(current_tracks)
        source_records.append(after)
        tick_count += len(ticks)

    categorical_vocabs: dict[str, dict[str, int]] = {}
    for name, field in schema.fields.items():
        if field.kind != "categorical":
            continue
        labels = sorted({
            str(record[name])
            for track in tracks
            for record in track.records
            if record.get(name) is not None
        })
        if not labels:
            raise ValueError(f"categorical field {name!r} contains no values")
        categorical_vocabs[name] = {label: index for index, label in enumerate(labels)}

    resolved_shard_length = shard_length or max(track.n_steps for track in tracks)
    modalities = {}
    for name, field in {**schema.fields, **schema.globals}.items():
        if field.kind == "categorical":
            modalities[name] = {
                "dtype": "int64",
                "shape": [],
                "source": f"vocab:{name}",
            }
        else:
            modalities[name] = {
                "dtype": "float64",
                "shape": [field.width] if field.kind == "vector" else [],
                "source": f"field:{name}",
            }
    for name, media in schema.media.items():
        modalities[name] = {
            "dtype": media.dtype,
            "shape": list(media.shape),
            "source": f"media:{name}",
        }
    spec = OutputSpec.from_dict({
        "container": container,
        "shard_length": resolved_shard_length,
        "short_policy": "keep",
        "presence_mask": True,
        "modalities": modalities,
        "vocabs": categorical_vocabs,
    })
    validate_container_availability(
        spec.container, registry=getattr(pipeline, "_registry", None))
    for pinned_source, expected_source in zip(pinned_sources, source_records):
        if _pinned_source_descriptor(pinned_source) != expected_source:
            raise ValueError(
                f"private snapshot for import source {pinned_source.logical_path!r} changed "
                "after preflight"
            )
    removed_shards = _begin_import_destination(
        out, state=destination_state, force=force)
    entries = []
    for track in tracks:
        entries.extend(write_track(
            out,
            track,
            spec,
            schema=schema,
            split=split,
            media_provider=getattr(adapter, "provider", None),
            registry=getattr(pipeline, "_registry", None),
        ))

    for pinned_source, expected_source in zip(pinned_sources, source_records):
        if _pinned_source_descriptor(pinned_source) != expected_source:
            raise ValueError(
                f"private snapshot for import source {pinned_source.logical_path!r} changed "
                "during publication; shards were left uncommitted"
            )

    results = {
        "shards": len(entries),
        "sources": len(paths),
        "tracks": len(tracks),
        "steps": sum(track.n_steps for track in tracks),
        "ticks": tick_count,
        "sources_skipped": 0,
    }
    signature_payload = {
        "kind": "adapter-tick-import",
        "adapter": adapter_name,
        "sources": source_records,
        "source_id": source_id,
        "schema": schema.to_dict(),
        "output": spec.to_dict(),
        "split": split,
    }
    fingerprint = hashlib.sha256(json.dumps(
        signature_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()).hexdigest()
    config = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "schema": schema.to_dict(),
        "output": spec.to_dict(),
        "results": results,
        "import": {
            "adapter": adapter_name,
            "mode": "ticks",
            "sources": source_records,
        },
        "provenance": {"kind": "import", "fingerprint": fingerprint},
        "errors": [],
    }
    write_text(
        f"{out}/dataset_manifest.json",
        json.dumps({
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "commit_protocol": "sidecar-v1",
            "count": len(entries),
            "shards": portable_shard_entries(out, entries),
        }, indent=2),
    )
    write_text(f"{out}/config.json", json.dumps(config, indent=2))
    commit_dataset(out)
    return {
        "dataset": out,
        "adapter": adapter_name,
        **results,
        "modalities": sorted([*modalities, "_present"]),
        **({"replaced_shards": len(removed_shards)} if removed_shards else {}),
    }


def _leased_import(method):
    @wraps(method)
    def locked(sources, out, *args, **kwargs):
        raw_sources = [sources] if isinstance(sources, (str, Path)) else list(sources)
        from .manifest import (
            dataset_writer_lease,
            validate_source_destination_separation,
        )
        validate_source_destination_separation(
            str(out), [str(source) for source in raw_sources])
        with dataset_writer_lease(str(out)):
            return method(raw_sources, out, *args, **kwargs)
    return locked


@_leased_import
def import_dataset(
    sources: Sequence[str | Path] | str | Path,
    out: str | Path,
    *,
    adapter: str | None = None,
    schema=None,
    fields: Mapping[str, Any] | None = None,
    time: str | None = None,
    entity: str | None = None,
    hz: float | None = None,
    kinds: Mapping[str, str] | None = None,
    adapter_options: Mapping[str, Any] | None = None,
    source_id: str | None = None,
    split: str = "train",
    shard_length: int | None = None,
    container: str = "npz",
    force: bool = False,
    registry=None,
) -> dict:
    """Detect a registered adapter and import all of its data into Kuration.

    Array adapters can expose ``import_arrays(source)`` for shape/dtype-preserving tensor
    import. Other adapters are constructed through the ordinary :func:`kuration.input` path,
    including its conservative schema and field-mapping inference.
    """
    raw_sources = [sources] if isinstance(sources, (str, Path)) else list(sources)
    if not raw_sources:
        raise ValueError("import_dataset needs at least one source")
    if adapter is not None and (not isinstance(adapter, str) or not adapter):
        raise ValueError("adapter must be a non-empty registered name or None")
    if fields is not None and not isinstance(fields, Mapping):
        raise ValueError("fields must be a mapping or None")
    if kinds is not None and not isinstance(kinds, Mapping):
        raise ValueError("kinds must be a mapping or None")
    if adapter_options is not None and not isinstance(adapter_options, Mapping):
        raise ValueError("adapter_options must be a mapping or None")
    if shard_length is not None and (
        not isinstance(shard_length, int)
        or isinstance(shard_length, bool)
        or shard_length < 1
    ):
        raise ValueError("shard_length must be a positive integer or None")
    if not isinstance(force, bool):
        raise ValueError("force must be true or false")
    split = _safe_split(split)
    from .io import _normalize_explicit_path
    out_text = _normalize_explicit_path(out, label="out")
    # Existing outputs are rejected before plugin loading, adapter detection, source discovery,
    # schema-file reads, or any other input materialization. A typo in --out must be cheap and
    # side-effect free.
    destination_state = _prepare_destination(out_text, force=force)
    for raw_source in raw_sources:
        if is_remote(str(raw_source)):
            # Plugin loading and adapter detection are allowed to execute third-party code.
            # Reject unsupported generic remote reads before either can inspect the source.
            _reject_generic_remote_import(str(raw_source))
    _require_secure_local_import_support()
    schema = _load_schema(schema)

    from .adapters.base import detect_adapter
    from .plugins import load_plugins
    from .registry import default_registry

    active_registry = registry or default_registry
    if active_registry is default_registry:
        load_plugins()

    # A directory of episode NPZ files is a convenient multi-source input even though an NPZ
    # adapter itself recognizes files rather than directories.
    directory_npz = (
        adapter in {None, "npz"}
        and all(not is_remote(str(item)) and Path(item).is_dir() for item in raw_sources)
        and all(list(Path(item).rglob("*.npz")) for item in raw_sources)
    )
    paths = (
        _discover_npz_sources(raw_sources)
        if directory_npz
        else [str(item) for item in raw_sources]
    )
    for path in paths:
        if is_remote(path):
            # Adapter detectors are plugin code and may inspect a source. Reject before detection
            # so generic remote data is never read without a generation-bound request.
            _reject_generic_remote_import(path)
    if adapter is None:
        try:
            detected = {path: detect_adapter(path, registry=active_registry) for path in paths}
        except ValueError as error:
            # Directory-shaped adapters (partitioned Parquet, Zarr) only detect locally: a
            # remote prefix would cost a recursive listing per candidate adapter.
            remote_dirs = [path for path in paths
                           if is_remote(path) and not Path(path).suffix]
            if not remote_dirs:
                raise
            raise ValueError(
                f"{error} Remote directories/prefixes {remote_dirs} are not auto-detected "
                "(detection never lists a remote tree); pass adapter= explicitly, e.g. "
                "adapter='parquet-dataset' with schema=."
            ) from error
        if len(set(detected.values())) != 1:
            raise ValueError(
                f"import sources resolve to different adapters {detected}; import each format "
                "separately or pass adapter= explicitly"
            )
        adapter_name = next(iter(detected.values()))
    else:
        adapter_name = adapter
        if adapter_name not in active_registry.names("adapter"):
            raise ValueError(
                f"unknown adapter {adapter_name!r}; registered: "
                f"{active_registry.names('adapter')}"
            )
    if adapter_name == "curated":
        raise ValueError(
            "source is already a Kuration dataset; use it directly with kur-inspect/kur-check"
        )

    factory = active_registry.get("adapter", adapter_name)
    array_loader = getattr(factory, "import_arrays", None)
    # The lossless array hook copies stored tensors verbatim: it has no schema, so it cannot
    # honor any schema-shaping option. Any of them sends the import down the tick path (where
    # `input()` applies them) instead of dropping them.
    shaping = _given_options(
        schema=schema, fields=fields, time=time, entity=entity, hz=hz, kinds=kinds,
        adapter_options=adapter_options,
    )
    with _pin_import_sources(paths) as pinned_sources:
        if callable(array_loader) and not shaping:
            return _import_array_dataset(
                pinned_sources,
                out,
                adapter_name=adapter_name,
                factory=factory,
                source_id=source_id,
                split=split,
                shard_length=shard_length,
                container=container,
                force=force,
                destination_state=destination_state,
                registry=active_registry,
            )

        from .pipeline import input as open_input

        read_paths = [source.read_path for source in pinned_sources]
        options = dict(adapter_options or {})
        auto_pipeline_adapters = {
            "timeseries", "jsonl", "npz", "hdf5", "parquet-dataset", "zarr"
        }
        # These read named arrays: there are no table columns to nominate as the time/entity role
        # and no per-column kind inference to override. `hz=` still applies (schema sample rate).
        array_pipeline_adapters = {"npz", "hdf5", "zarr"}
        if adapter is None and adapter_name in auto_pipeline_adapters:
            if adapter_name in array_pipeline_adapters:
                columnless = _given_options(time=time, entity=entity, kinds=kinds)
                if columnless:
                    raise ValueError(
                        f"adapter {adapter_name!r} reads named arrays rather than table columns, "
                        f"so {columnless} cannot be applied. Declare the entity axis and field "
                        "kinds in schema= instead (hz= is supported)."
                    )
            source_arg: Any = read_paths[0] if len(read_paths) == 1 else read_paths
            pipeline = open_input(
                source_arg,
                schema=schema,
                fields=dict(fields) if fields is not None else None,
                time=time,
                entity=entity,
                hz=hz,
                kinds=dict(kinds) if kinds is not None else None,
                registry=active_registry,
                **options,
            )
        else:
            # `input(adapter=...)` is the no-inference escape hatch: it builds the adapter from
            # schema= plus adapter_options and has nowhere to put the inference overrides.
            unusable = _given_options(time=time, entity=entity, hz=hz, kinds=kinds)
            if unusable:
                raise ValueError(
                    f"adapter {adapter_name!r} is constructed explicitly, so {unusable} cannot "
                    "be applied — there is no schema inference on that path. Put the sample "
                    "rate and column roles in schema=, or pass adapter-specific keywords "
                    "through adapter_options=."
                )
            if schema is None:
                raise ValueError(
                    f"explicit or plugin adapter {adapter_name!r} needs schema= because its "
                    "construction parameters cannot be inferred generically"
                )
            pipeline = open_input(
                adapter=adapter_name,
                sources=read_paths,
                schema=schema,
                fields=dict(fields) if fields is not None else None,
                registry=active_registry,
                **options,
            )
        pipeline_sources = _align_pipeline_sources(pinned_sources, pipeline.sources)
        return _import_tick_dataset(
            pipeline,
            pipeline_sources,
            out,
            adapter_name=adapter_name,
            source_id=source_id,
            split=split,
            shard_length=shard_length,
            container=container,
            force=force,
            destination_state=destination_state,
        )
