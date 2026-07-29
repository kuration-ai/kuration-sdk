"""Authoring, artifact, and dispatcher CLIs: kur-infer/init/preview/stats/export/checkpoint.

These cover the on-ramp (infer -> init -> preview -> curate) and the artifact-side commands
that were previously reachable only from Python.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import run_cli


def _make_csv(tmp_path, name="d.csv"):
    path = tmp_path / name
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["t", "id", "x", "y", "act", "team"])
        writer.writeheader()
        for step in range(12):                 # time-major, the adapter's documented order
            for entity in range(1, 4):
                writer.writerow({"t": step, "id": f"e{entity}", "x": step * 0.5,
                                 "y": entity * 1.5, "act": entity,
                                 "team": "A" if entity < 3 else "B"})
    return path


_run = run_cli          # in-process, so the executed CLI lines are visible to coverage


def _run_subprocess(module, *args, expect=0):
    """Run a CLI the way a shell does, for the contracts only a real process can show."""
    proc = subprocess.run([sys.executable, "-m", module, *args],
                          capture_output=True, text=True, timeout=120)
    if expect is not None:
        assert proc.returncode == expect, f"{module} {args}\n{proc.stdout}\n{proc.stderr}"
    return proc


def _scaffold(tmp_path, *extra):
    """Write a job spec covering the sample CSV and return (job_path, dataset_dir)."""
    source = _make_csv(tmp_path)
    job = tmp_path / "job.json"
    out = tmp_path / "curated"
    _run("kuration.core.cli.init_job", str(source), "--out", str(job),
         "--window", "4", "--dataset-out", str(out), *extra)
    return job, out


def _curate(tmp_path, *extra):
    job, out = _scaffold(tmp_path)
    proc = _run("kuration.core.cli.curate", str(job), "--quiet", *extra)
    return job, out, proc


# ---- kur-infer --------------------------------------------------------------
def test_infer_emits_a_schema_a_job_can_reference(tmp_path):
    source = _make_csv(tmp_path)
    schema_path = tmp_path / "schema.json"
    proc = _run("kuration.core.cli.infer_schema_cli", str(source),
                "--out", str(schema_path))
    assert "adapter: timeseries" in proc.stdout

    from kuration.core.schema import SchemaSpec
    schema = SchemaSpec.from_dict(json.loads(schema_path.read_text()))
    assert schema.entity_axis == "id"
    assert set(schema.fields) == {"x", "y", "act", "team"}
    assert schema.fields["team"].kind == "categorical"


def test_infer_json_carries_hints_and_an_adapter_fragment(tmp_path):
    source = _make_csv(tmp_path)
    proc = _run("kuration.core.cli.infer_schema_cli", str(source), "--json")
    result = json.loads(proc.stdout)
    assert result["hints"]["time_column"] == "t"
    assert result["hints"]["entity_column"] == "id"
    assert result["adapter"]["type"] == "timeseries"
    assert result["adapter"]["entity_column"] == "id"


def test_infer_reports_an_unknown_sample_rate_as_an_actionable_warning(tmp_path):
    """No usable time column means hours budgets are meaningless — say so, don't guess."""
    path = tmp_path / "no_time.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "x"])
        writer.writeheader()
        for entity in range(4):
            writer.writerow({"id": f"e{entity}", "x": entity})
    proc = _run("kuration.core.cli.infer_schema_cli", str(path), "--json")
    codes = {warning["code"] for warning in json.loads(proc.stdout)["warnings"]}
    assert "sample-rate-unknown" in codes


def test_infer_rejects_an_unsupported_source_in_one_line(tmp_path):
    bad = tmp_path / "data.unknown"
    bad.write_text("nothing")
    proc = _run("kuration.core.cli.infer_schema_cli", str(bad), expect=1)
    assert "Traceback" not in proc.stderr and "error:" in proc.stderr


# ---- --kinds/--fields/--adapter-options: inline JSON or a JSON file ---------
def test_kinds_option_accepts_inline_json_and_a_json_file(tmp_path):
    """`act` is numeric and infers as scalar by default; forcing it to categorical via
    --kinds proves the override took effect, in both its inline-JSON and file forms."""
    source = _make_csv(tmp_path)
    override = {"act": "categorical"}
    inline = json.dumps(override)
    kinds_file = tmp_path / "kinds.json"
    kinds_file.write_text(inline)

    for label, kinds_arg in (("inline", inline), ("file", str(kinds_file))):
        proc = _run("kuration.core.cli.infer_schema_cli", str(source),
                    "--kinds", kinds_arg, "--json")
        result = json.loads(proc.stdout)
        assert result["schema"]["fields"]["act"]["kind"] == "categorical", label

        job = tmp_path / f"job_{label}.json"
        _run("kuration.core.cli.init_job", str(source), "--out", str(job),
             "--window", "4", "--kinds", kinds_arg)
        assert json.loads(job.read_text())["schema"]["fields"]["act"]["kind"] == "categorical"

        proc = _run("kuration.core.cli.preview", "--infer", str(source),
                    "--kinds", kinds_arg, "--json")
        assert json.loads(proc.stdout)["fields"]["act"]["kind"] == "categorical", label


def test_kinds_option_rejects_a_value_that_is_neither_json_nor_an_existing_path(tmp_path):
    source = _make_csv(tmp_path)
    proc = _run("kuration.core.cli.infer_schema_cli", str(source),
                "--kinds", "not json and not a path", expect=1)
    assert "Traceback" not in proc.stderr
    assert "inline JSON or a path to a JSON file" in proc.stderr


def test_import_fields_and_kinds_and_adapter_options_accept_both_forms(tmp_path):
    """kur-import previously read --fields/--kinds/--adapter-options as PATH only; the shared
    _common.parse_mapping_option helper now accepts inline JSON here too."""
    import csv as csv_module

    source = tmp_path / "telemetry.csv"
    with source.open("w", newline="") as file:
        writer = csv_module.DictWriter(file, fieldnames=["t", "id", "signal"])
        writer.writeheader()
        writer.writerow({"t": 0, "id": "d1", "signal": 1.5})
        writer.writerow({"t": 1, "id": "d1", "signal": 2.5})

    fields = {"signal": "signal"}
    fields_file = tmp_path / "fields.json"
    fields_file.write_text(json.dumps(fields))
    kinds_file = tmp_path / "kinds.json"
    kinds_file.write_text(json.dumps({"signal": "scalar"}))

    for label, fields_arg, kinds_arg in (
        ("inline", json.dumps(fields), json.dumps({"signal": "scalar"})),
        ("file", str(fields_file), str(kinds_file)),
    ):
        dest = tmp_path / f"dataset_{label}"
        _run("kuration.core.cli.import_dataset", str(source), "--out", str(dest),
             "--time", "t", "--entity", "id", "--fields", fields_arg,
             "--kinds", kinds_arg, "--adapter-options", "{}")
        assert (dest / "dataset_manifest.json").exists(), label


# ---- kur-init ---------------------------------------------------------------
def test_init_scaffold_validates_explains_and_curates(tmp_path):
    """The scaffold's contract: what it writes is a job the rest of the CLI family accepts."""
    job, out = _scaffold(tmp_path)
    spec = json.loads(job.read_text())
    assert spec["job_schema_version"]
    assert spec["strategy"]["binning"]["axes"][0]["bins"] == 4
    # A categorical field is present, so the preset needs an explicit policy to resolve.
    assert spec["output_preset"]["categorical"] == "omit"

    assert "OK" in _run("kuration.core.cli.validate", str(job)).stdout
    _run("kuration.core.cli.explain", str(job))
    _run("kuration.core.cli.curate", str(job), "--quiet")
    manifest = json.loads((out / "dataset_manifest.json").read_text())
    assert manifest["count"] > 0


def test_init_refuses_to_clobber_without_force(tmp_path):
    job, _out = _scaffold(tmp_path)
    source = tmp_path / "d.csv"
    proc = _run("kuration.core.cli.init_job", str(source), "--out", str(job), expect=1)
    assert "--force" in proc.stderr
    _run("kuration.core.cli.init_job", str(source), "--out", str(job), "--force")


def test_init_says_which_block_to_write_when_no_axis_can_be_inferred(tmp_path):
    """A schema with nothing scalar cannot get a default binning axis; fail loudly."""
    path = tmp_path / "cats.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["t", "id", "team"])
        writer.writeheader()
        for step in range(4):
            for entity in range(2):
                writer.writerow({"t": step, "id": f"e{entity}", "team": "A"})
    proc = _run("kuration.core.cli.init_job", str(path),
                "--out", str(tmp_path / "job.json"), expect=1)
    assert "no scalar field" in proc.stderr
    assert not (tmp_path / "job.json").exists()   # nothing half-written


# ---- kur-preview ------------------------------------------------------------
def test_preview_reads_bounded_samples_in_job_and_infer_modes(tmp_path):
    job, _out = _scaffold(tmp_path)
    source = tmp_path / "d.csv"
    for args in ([str(job)], ["--infer", str(source)]):
        proc = _run("kuration.core.cli.preview", *args)
        assert "records: 12 ticks" in proc.stdout
        assert "field x (scalar)" in proc.stdout


def test_preview_json_and_html_carry_field_quality(tmp_path):
    job, _out = _scaffold(tmp_path)
    html = tmp_path / "preview.html"
    proc = _run("kuration.core.cli.preview", str(job), "--json", "--html", str(html))
    preview = json.loads(proc.stdout)
    assert preview["fields"]["x"]["missing_fraction"] == 0.0
    assert preview["records"]["entity_records"] == 36
    assert html.exists() and "<html" in html.read_text().lower()


def test_preview_rejects_inference_flags_in_job_mode(tmp_path):
    job, _out = _scaffold(tmp_path)
    proc = _run("kuration.core.cli.preview", str(job), "--time", "t", expect=1)
    assert "--infer mode only" in proc.stderr


# ---- kur-stats --------------------------------------------------------------
def test_stats_reports_artifact_distribution_and_writes_html(tmp_path):
    _job, out, _proc = _curate(tmp_path)
    html = tmp_path / "stats.html"
    proc = _run("kuration.core.cli.stats", str(out), "--quiet", "--html", str(html))
    assert "stats: output" in proc.stdout
    assert "artifact:" in proc.stdout
    assert html.exists() and "<html" in html.read_text().lower()

    report = json.loads(_run("kuration.core.cli.stats", str(out), "--quiet",
                             "--json").stdout)
    assert report["sections"]["artifact"]["shards"] > 0
    assert report["sections"]["contents"]["modalities"]


def test_stats_capabilities_are_discoverable(tmp_path):
    _job, out, _proc = _curate(tmp_path)
    proc = _run("kuration.core.cli.stats", str(out), "--quiet", "--capabilities")
    capabilities = json.loads(proc.stdout)
    assert capabilities["html"] is True
    assert "distribution" in capabilities["plots"]


# ---- kur-curate --review ----------------------------------------------------
def test_curate_review_explains_the_hours_budget(tmp_path):
    review_json = tmp_path / "review.json"
    _job, _out, proc = _curate(tmp_path, "--review", "--review-json", str(review_json))
    assert "hours selection:" in proc.stdout
    review = json.loads(review_json.read_text())
    assert review["admitted_items"] > 0
    assert review["budget"]["train"]["requested_seconds"] == pytest.approx(3600.0)
    # The tiny corpus cannot fill an hour, and the review must say so rather than imply success.
    assert any(warning["code"] == "budget-underfill" for warning in review["warnings"])


def test_curate_summary_distinguishes_span_items_from_source_tracks(tmp_path):
    """kur-init's default span_steps chunks each track into several admitted items; the old
    "N/M tracks" wording conflated item count with track count (e.g. "9/3 tracks" read as
    9-out-of-3, which is nonsensical). With a generous budget every item of all 3 entity
    tracks is admitted, so the summary must name both quantities truthfully."""
    _job, _out, proc = _curate(tmp_path)
    assert "wrote 9 shards from 9 span items from 3/3 tracks (" in proc.stdout


def test_curate_summary_keeps_plain_tracks_wording_without_spans(tmp_path):
    """Without span_steps, one admitted item is one whole track, so item count and track
    count agree and the plain "N/M tracks" phrasing (no earlier fix needed) stays truthful."""
    source = _make_csv(tmp_path)
    job_spec = {
        "schema": {"name": "j", "entity_axis": "id", "sample_hz": 1.0,
                   "fields": {"x": {"kind": "scalar"}, "y": {"kind": "scalar"},
                              "act": {"kind": "scalar"}, "team": {"kind": "categorical"}}},
        "adapter": {"type": "timeseries",
                    "fields": {"x": "x", "y": "y", "act": "act", "team": "team"},
                    "time_column": "t", "entity_column": "id"},
        "sources": [str(source)],
        "segmenter": {"window": 4, "stride": 4},
        "strategy": {"name": "s", "define": {"am": {"source": "id.act", "reduce": "mean"}},
                     "selection": {"method": "top-k", "weights": {"am": 1.0}}},
        "output": {"shard_length": 4, "container": "npz",
                   "modalities": {"obs": {"dtype": "float32", "shape": [], "source": "field:x"}}},
        "train_hours": 100, "test_hours": 0, "out": str(tmp_path / "curated_nospan"),
    }
    job = tmp_path / "job.json"
    job.write_text(json.dumps(job_spec))
    proc = _run("kuration.core.cli.curate", str(job), "--quiet")
    assert "wrote 9 shards from 3/3 tracks (" in proc.stdout
    assert "span items" not in proc.stdout


# ---- kur-export -------------------------------------------------------------
@pytest.mark.parametrize("fmt", ["npz", "npy"])
def test_export_writes_arrays_and_an_index(tmp_path, fmt):
    import numpy as np

    _job, out, _proc = _curate(tmp_path)
    dest = tmp_path / f"export_{fmt}"
    _run("kuration.core.cli.export", str(out), "--out", str(dest), "--format", fmt)
    index = json.loads((dest / "export_index.json").read_text())
    assert index["shards"] > 0 and index["format"] == fmt

    for entry in index["entries"]:
        for path in entry["files"]:
            assert (dest / path).exists()
        arrays = (dict(np.load(dest / entry["files"][0])) if fmt == "npz"
                  else {"x": np.load(dest / next(
                      p for p in entry["files"] if p.endswith("__x.npy")))})
        # Trimming is on by default, so rows match the manifest's n_steps exactly.
        assert len(arrays["x"]) == entry["n_steps"]


def test_export_filters_by_split_and_modality(tmp_path):
    import numpy as np

    _job, out, _proc = _curate(tmp_path)
    dest = tmp_path / "subset"
    _run("kuration.core.cli.export", str(out), "--out", str(dest),
         "--split", "train", "--modalities", "x")
    index = json.loads((dest / "export_index.json").read_text())
    assert index["split"] == "train"
    for entry in index["entries"]:
        assert entry["modalities"] == ["x"]
        assert list(np.load(dest / entry["files"][0])) == ["x"]


def test_export_says_so_when_a_split_matches_nothing(tmp_path):
    _job, out, _proc = _curate(tmp_path)
    proc = _run("kuration.core.cli.export", str(out), "--out", str(tmp_path / "none"),
                "--split", "validation", expect=1)
    assert "no shards matched" in proc.stderr and "Traceback" not in proc.stderr


def test_export_parquet_roundtrips_every_modality(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    _job, out, _proc = _curate(tmp_path)
    dest = tmp_path / "parquet"
    _run("kuration.core.cli.export", str(out), "--out", str(dest), "--format", "parquet")
    index = json.loads((dest / "export_index.json").read_text())
    table = pq.read_table(dest / index["entries"][0]["files"][0])
    assert table.num_rows == index["entries"][0]["n_steps"]
    assert "x" in table.schema.names


# ---- kur-checkpoint ---------------------------------------------------------
@pytest.fixture()
def checkpoint_root(tmp_path):
    pytest.importorskip("pyarrow")
    from kuration import (CheckpointIdentity, Segment, SegmentBatch, StageCheckpoint,
                          canonical_hash)
    identity = CheckpointIdentity(logical_plan_hash=canonical_hash({"pipeline": 1}),
                                  physical_plan_hash=canonical_hash({"plan": "a"}),
                                  schema={"name": "demo"}, parameters={"seed": 7})
    root = tmp_path / "ckpt"
    stage = StageCheckpoint(str(root), "features", identity)
    for name in ("a", "b"):
        stage.write_partition(f"source:{name}", SegmentBatch.from_segments(
            [Segment(name, 0, 4, (0, 3), {"score": 0.5})]))
    stage.finalize(["source:a", "source:b"])
    return root


def test_checkpoint_list_and_verify_accept_a_healthy_tree(checkpoint_root):
    listing = _run("kuration.core.cli.checkpoint", "list", str(checkpoint_root))
    assert "stage features: 2 partition(s)" in listing.stdout
    assert "finalized" in listing.stdout

    scan = json.loads(_run("kuration.core.cli.checkpoint", "verify",
                           str(checkpoint_root), "--json").stdout)
    assert scan["invalid"] == 0 and scan["verified"] is True
    assert all(partition["valid"] for partition in scan["stages"][0]["details"])


def test_checkpoint_verify_detects_a_corrupted_partition(checkpoint_root):
    """This is the whole point: a silently damaged commit must not be reported reusable."""
    data = sorted((checkpoint_root / "features" / "data").glob("*.arrow"))[0]
    with data.open("ab") as file:
        file.write(b"corrupt")

    # Metadata-only listing still sees a present file; only verify re-checksums it.
    assert json.loads(_run("kuration.core.cli.checkpoint", "list", str(checkpoint_root),
                           "--json").stdout)["invalid"] == 0
    proc = _run("kuration.core.cli.checkpoint", "verify", str(checkpoint_root), expect=2)
    assert "sha256 mismatch" in proc.stdout


def test_checkpoint_clear_requires_explicit_confirmation(checkpoint_root):
    proc = _run("kuration.core.cli.checkpoint", "clear", str(checkpoint_root), expect=1)
    assert "--yes" in proc.stderr
    assert (checkpoint_root / "features").exists()      # nothing removed

    _run("kuration.core.cli.checkpoint", "clear", str(checkpoint_root), "--yes")
    assert not (checkpoint_root / "features").exists()


def test_checkpoint_clear_attempts_only_keeps_commits(checkpoint_root):
    attempts = checkpoint_root / "features" / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    (attempts / "stale.arrow").write_bytes(b"leftover")
    _run("kuration.core.cli.checkpoint", "clear", str(checkpoint_root),
         "--attempts-only", "--yes")
    assert not (attempts / "stale.arrow").exists()
    assert sorted((checkpoint_root / "features" / "commits").glob("*.json"))


def test_checkpoint_rejects_an_unknown_stage(checkpoint_root):
    proc = _run("kuration.core.cli.checkpoint", "list", str(checkpoint_root),
                "--stage", "nope", expect=1)
    assert "not found" in proc.stderr and "features" in proc.stderr


# ---- kur dispatcher ---------------------------------------------------------
def test_dispatcher_lists_and_forwards_commands(tmp_path):
    _job, out, _proc = _curate(tmp_path)
    usage = _run("kuration.core.cli.main").stdout
    for name in ("curate", "infer", "init", "preview", "stats", "export", "checkpoint"):
        assert name in usage

    dispatched = _run("kuration.core.cli.main", "inspect", str(out)).stdout
    direct = _run("kuration.core.cli.inspect", str(out)).stdout
    assert dispatched == direct


def test_dispatcher_help_reads_as_the_dispatched_form():
    proc = _run("kuration.core.cli.main", "stats", "--help")
    assert proc.stdout.startswith("usage: kur stats")


def test_dispatcher_help_command_is_equivalent_to_the_help_flag():
    """`kur help <command>` previously discarded <command> and printed global usage."""
    via_help = _run("kuration.core.cli.main", "help", "stats")
    via_flag = _run("kuration.core.cli.main", "stats", "--help")
    assert via_help.stdout == via_flag.stdout
    assert via_help.stdout.startswith("usage: kur stats")

    # A bare `kur help` (no command) still prints the dispatcher's own usage.
    bare = _run("kuration.core.cli.main", "help")
    assert bare.stdout.startswith("usage: kur <command>")


def test_dispatcher_help_unknown_command_still_suggests():
    proc = _run("kuration.core.cli.main", "help", "curat", expect=2)
    assert "kur curate" in proc.stderr


def test_dispatcher_suggests_a_near_miss():
    proc = _run("kuration.core.cli.main", "curat", expect=2)
    assert "kur curate" in proc.stderr


def test_dispatcher_plugin_import_error_is_a_one_line_error(monkeypatch):
    """An ImportError while loading a discovered plugin's module (e.g. a plugin's own
    missing optional dependency) previously escaped as a raw traceback."""
    import kuration.core.cli.main as main_mod
    monkeypatch.setattr(
        main_mod, "_discovered",
        lambda: {"broken-plugin": "kuration.core.cli._no_such_plugin_module:main"})

    proc = _run("kuration.core.cli.main", "broken-plugin", expect=1)
    assert proc.stderr.startswith("error: command 'broken-plugin' failed to load:")
    assert "Traceback" not in proc.stderr


def test_dispatcher_plugin_import_error_reraises_with_debug(monkeypatch):
    import kuration.core.cli.main as main_mod
    monkeypatch.setattr(
        main_mod, "_discovered",
        lambda: {"broken-plugin": "kuration.core.cli._no_such_plugin_module:main"})

    with pytest.raises(ImportError):
        _run("kuration.core.cli.main", "broken-plugin", "--debug")


def test_dispatcher_discovers_installed_plugin_clis():
    """Domain CLIs must appear without the core naming the domain."""
    from kuration.core.cli.main import _CORE, commands
    table = commands()
    installed = {name for name in table if name not in _CORE}
    assert "slam" in installed, f"kur-slam not discovered; found {sorted(installed)}"


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
def test_completion_scripts_name_every_command(shell):
    from kuration.core.cli.main import commands
    script = _run("kuration.core.cli.main", "completion", shell).stdout
    for name in commands():
        assert name in script


def test_completion_requires_a_known_shell():
    proc = _run("kuration.core.cli.main", "completion", "tcsh", expect=1)
    assert "bash, zsh, fish" in proc.stderr


def test_init_scaffolds_a_span_so_the_hours_budget_can_bite(tmp_path):
    """Without span_steps the budget admits whole tracks, emitting the entire source."""
    job, out = _scaffold(tmp_path, "--train-hours", "0.002")
    spec = json.loads(job.read_text())
    assert spec["span_steps"] == spec["segmenter"]["window"]

    review = tmp_path / "review.json"
    _run("kuration.core.cli.curate", str(job), "--quiet", "--review-json", str(review))
    budget = json.loads(review.read_text())
    # Admission is per window, so most candidates are refused once the budget fills.
    assert budget["candidate_items"] > budget["admitted_items"] > 0
    emitted = budget["budget"]["train"]["emitted_seconds"]
    requested = budget["budget"]["train"]["requested_seconds"]
    assert emitted <= requested * 2, f"emitted {emitted}s against a {requested}s budget"
    assert json.loads((out / "dataset_manifest.json").read_text())["count"] > 0


# ---- installed console-script contract ---------------------------------------
# The tests above call entry points in-process. These few spawn a real interpreter, so the
# behavior a shell actually sees — exit codes, one-line stderr, --debug — stays pinned.
def test_console_scripts_report_failures_as_one_line(tmp_path):
    missing = tmp_path / "nope.json"
    proc = _run_subprocess("kuration.core.cli.preview", str(missing), expect=1)
    assert "Traceback" not in proc.stderr
    assert proc.stderr.startswith("error:") and "not found" in proc.stderr


def test_console_script_debug_flag_reraises(tmp_path):
    proc = _run_subprocess("kuration.core.cli.stats", str(tmp_path / "nope"),
                           "--debug", expect=None)
    assert proc.returncode != 0 and "Traceback" in proc.stderr


def test_console_script_runs_the_authoring_chain(tmp_path):
    source = _make_csv(tmp_path)
    job, out = tmp_path / "job.json", tmp_path / "curated"
    _run_subprocess("kuration.core.cli.init_job", str(source), "--out", str(job),
                    "--window", "4", "--dataset-out", str(out))
    _run_subprocess("kuration.core.cli.curate", str(job), "--quiet")
    assert json.loads((out / "dataset_manifest.json").read_text())["count"] > 0


def test_dispatcher_console_script_forwards_argv(tmp_path):
    proc = _run_subprocess("kuration.core.cli.main", "--list")
    assert "checkpoint" in proc.stdout and "export" in proc.stdout
