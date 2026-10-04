"""Artifact-format version bumps that change *selection*, not just structure.

The 1.1 bump records two DSL semantic changes (`dispersion`/`min_pair_dist` over all declared
position dimensions; `scope: per_entity_mean` honored on categorical fields). Migration is
structural, so the compat table in `kuration.core.versioning` is the only place that can tell a
user their old config selects differently here. The 1.2 bump records bounded/materialized parity,
experiment-arm tag isolation, corrected tag floors/filtering, and group/tag identity parity.
These tests pin the version, notes, and the `kur-migrate` surface that reports them.
"""
from __future__ import annotations

import json
import sys

import pytest

from kuration import ARTIFACT_FORMAT_VERSION, JOB_SCHEMA_VERSION, migrate_config
from kuration.core.versioning import (ARTIFACT_SEMANTIC_CHANGES, document_migration_notes,
                                      check_version, migration_notes)


def _config(version="1.0"):
    return {
        "artifact_format_version": version,
        "schema": {"name": "demo", "fields": {"x": {"kind": "scalar"}}},
        "strategy": {"selection": {"method": "top-k", "weights": {"x": 1}}},
        "output": {"container": "npz"},
        "results": {"selected": 1},
    }


def test_artifact_format_version_records_the_dsl_semantic_changes():
    assert ARTIFACT_FORMAT_VERSION == "1.3"
    assert set(ARTIFACT_SEMANTIC_CHANGES) == {"1.1", "1.2", "1.3"}

    notes = migration_notes("1.0")
    assert len(notes) == 7
    assert sum(note.startswith("1.1: ") for note in notes) == 2
    assert sum(note.startswith("1.2: ") for note in notes) == 4
    assert sum(note.startswith("1.3: ") for note in notes) == 1
    joined = " ".join(notes)
    assert "dispersion" in joined and "min_pair_dist" in joined and "dimension" in joined
    assert "per_entity_mean" in joined and "categorical" in joined
    assert "bounded uniform-bin" in joined and "partition" in joined
    assert "each arm" in joined and "min_tag_frac=0" in joined
    assert "round up" in joined and "predicate-failing" in joined
    assert "Python hash/equality" in joined
    assert "shard naming" in joined and "rank among selected" in joined

    # A legacy (unversioned) artifact predates the change too; a current one has nothing to say.
    assert migration_notes(None) == notes
    assert migration_notes(ARTIFACT_FORMAT_VERSION) == []
    assert migration_notes("1.0", kind="job") == []
    with pytest.raises(ValueError, match="unknown document kind"):
        migration_notes("1.0", kind="strategy")


def test_migrate_config_bumps_to_the_new_version_and_stays_idempotent():
    migrated = migrate_config(_config("1.0"))
    assert migrated["artifact_format_version"] == "1.3"
    assert migrate_config(migrated) == migrated
    assert document_migration_notes(_config("1.0")) == migration_notes("1.0")
    assert document_migration_notes(_config(ARTIFACT_FORMAT_VERSION)) == []
    # Jobs are classified separately and did not change semantically.
    assert document_migration_notes({"job_schema_version": JOB_SCHEMA_VERSION,
                                     "command": "curate"}) == []


@pytest.mark.parametrize(
    "malformed",
    ["1", "1.2.3", "1.x", "-1.2", "+1.2", "01.2", "1.02", "", True],
)
def test_version_parser_rejects_noncanonical_or_partial_versions(malformed):
    with pytest.raises(ValueError, match="canonical '<major>.<minor>'"):
        check_version(malformed, kind="artifact")


@pytest.mark.parametrize(
    ("version", "kind"),
    [("1.4", "artifact"), ("1.999", "artifact"), ("1.1", "job"), ("2.0", "job")],
)
def test_version_gate_rejects_every_future_minor_or_major(version, kind):
    with pytest.raises(ValueError, match=rf"unsupported future {kind} format {version}"):
        check_version(version, kind=kind)
    with pytest.raises(ValueError, match=rf"unsupported future {kind} format {version}"):
        migration_notes(version, kind=kind)


def test_migration_cannot_downgrade_a_future_minor_artifact():
    future = _config("1.999")
    with pytest.raises(ValueError, match="unsupported future artifact format"):
        migrate_config(future)
    assert future["artifact_format_version"] == "1.999"


def test_written_artifacts_declare_the_current_version():
    from kuration.core.manifest import build_config
    from kuration.core.schema import SchemaSpec
    from kuration.core.strategy import load_strategy

    schema = SchemaSpec.from_dict({"name": "demo", "sample_hz": 1.0,
                                   "fields": {"x": {"kind": "scalar"}}})
    strategy = load_strategy({"define": {"m": {"source": "x", "reduce": "mean"}},
                              "selection": {"method": "top-k", "weights": {"m": 1}}}, schema)
    config = build_config(schema, strategy, results={"selected": 0}, seed=0)
    assert config["artifact_format_version"] == ARTIFACT_FORMAT_VERSION


def test_kur_migrate_reports_the_semantic_notes(tmp_path, monkeypatch, capsys):
    from kuration.core.cli.migrate import main

    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config("1.0")))
    monkeypatch.setattr(sys, "argv", ["kur-migrate", str(path)])
    main()
    captured = capsys.readouterr()
    assert json.loads(captured.out)["artifact_format_version"] == ARTIFACT_FORMAT_VERSION
    assert captured.err.count("note: 1.1:") == 2
    assert captured.err.count("note: 1.2:") == 4
    assert "dispersion" in captured.err and "per_entity_mean" in captured.err

    out = tmp_path / "migrated.json"
    path.write_text(json.dumps(_config(ARTIFACT_FORMAT_VERSION)))
    monkeypatch.setattr(sys, "argv", ["kur-migrate", str(path), "--out", str(out)])
    main()
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(out.read_text())["artifact_format_version"] == ARTIFACT_FORMAT_VERSION
