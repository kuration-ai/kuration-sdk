"""Adversarial path-overlap checks for CLI-authored report files."""
from __future__ import annotations

import importlib
import json

import pytest
from conftest import run_cli


def test_file_output_guard_uses_path_components_not_string_prefixes(tmp_path):
    from kuration.core.cli._common import guard_file_output

    artifact = tmp_path / "data"
    with pytest.raises(ValueError, match="overlaps input path.*descendant"):
        guard_file_output(
            artifact / "report.html",
            force=False,
            protected_inputs=(artifact,),
            label="report output",
        )
    with pytest.raises(ValueError, match="overlaps input path.*ancestor"):
        guard_file_output(
            artifact,
            force=False,
            protected_inputs=(artifact / "config.json",),
            label="report output",
        )

    # A lexical prefix without a component boundary is a separate tree.
    guard_file_output(
        tmp_path / "data-report.html",
        force=False,
        protected_inputs=(artifact,),
        label="report output",
    )


def test_file_output_guard_resolves_symlinks_before_comparing(tmp_path):
    from kuration.core.cli._common import guard_file_output

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(artifact, target_is_directory=True)

    with pytest.raises(ValueError, match="overlaps input path"):
        guard_file_output(
            alias / "report.html",
            force=False,
            protected_inputs=(artifact,),
            label="report output",
        )


def test_file_output_guard_is_component_aware_for_remote_paths():
    from kuration.core.cli._common import guard_file_output

    artifact = "memory://cli-overlap/artifact"
    with pytest.raises(ValueError, match="overlaps input path.*descendant"):
        guard_file_output(
            f"{artifact}/report.html",
            force=False,
            protected_inputs=(artifact,),
            label="report output",
        )

    guard_file_output(
        "memory://cli-overlap/artifact-report.html",
        force=False,
        protected_inputs=(artifact,),
        label="report output",
    )


@pytest.mark.parametrize(
    ("module", "extra"),
    [
        ("kuration.core.cli.inspect", ()),
        ("kuration.core.cli.stats", ("--quiet",)),
    ],
)
def test_artifact_report_clis_cannot_replace_config_before_reading(
        tmp_path, monkeypatch, module, extra):
    artifact = tmp_path / "dataset"
    artifact.mkdir()
    config = artifact / "config.json"
    config.write_text("last-good")

    if module.endswith("inspect"):
        from kuration.core import inspection
        monkeypatch.setattr(
            inspection,
            "inspect_dataset",
            lambda *args, **kwargs: pytest.fail("artifact read occurred before preflight"),
        )
    else:
        stats_module = importlib.import_module("kuration.core.stats")
        monkeypatch.setattr(
            stats_module,
            "stats",
            lambda *args, **kwargs: pytest.fail("artifact read occurred before preflight"),
        )

    result = run_cli(module, artifact, *extra, "--html", config, expect=1)
    assert "overlaps input path" in result.stderr
    assert config.read_text() == "last-good"


def test_check_html_cannot_replace_config_or_policy_before_reads(tmp_path, monkeypatch):
    from kuration.core import inspection

    artifact = tmp_path / "dataset"
    artifact.mkdir()
    config = artifact / "config.json"
    config.write_text("last-good-config")
    policy = tmp_path / "policy.json"
    policy.write_text('{"min_shards": 1}')
    monkeypatch.setattr(
        inspection,
        "inspect_dataset",
        lambda *args, **kwargs: pytest.fail("artifact read occurred before preflight"),
    )

    config_attack = run_cli(
        "kuration.core.cli.check",
        artifact,
        "--policy",
        policy,
        "--html",
        config,
        expect=1,
    )
    assert "overlaps input path" in config_attack.stderr
    assert config.read_text() == "last-good-config"

    policy_attack = run_cli(
        "kuration.core.cli.check",
        artifact,
        "--policy",
        policy,
        "--html",
        policy,
        expect=1,
    )
    assert "also an input path" in policy_attack.stderr
    assert policy.read_text() == '{"min_shards": 1}'


def test_diff_html_cannot_replace_either_artifacts_shard(tmp_path, monkeypatch):
    from kuration.core import inspection

    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    shard = right / "train__source__0.npz"
    shard.write_bytes(b"last-good-shard")
    monkeypatch.setattr(
        inspection,
        "compare_datasets",
        lambda *args, **kwargs: pytest.fail("artifact read occurred before preflight"),
    )

    result = run_cli(
        "kuration.core.cli.diff", left, right, "--html", shard, expect=1)
    assert "overlaps input path" in result.stderr
    assert shard.read_bytes() == b"last-good-shard"


def test_manifest_argument_spelling_still_protects_the_whole_artifact(tmp_path):
    artifact = tmp_path / "dataset"
    artifact.mkdir()
    manifest = artifact / "dataset_manifest.json"
    manifest.write_text("{}")
    config = artifact / "config.json"
    config.write_text("last-good-config")

    result = run_cli(
        "kuration.core.cli.inspect", manifest, "--html", config, expect=1)
    assert "overlaps input path" in result.stderr
    assert config.read_text() == "last-good-config"


@pytest.mark.parametrize("infer", [False, True])
def test_preview_html_cannot_replace_its_job_or_source(tmp_path, infer):
    target = tmp_path / ("source.csv" if infer else "job.json")
    original = "t,id,x\n0,e,1\n" if infer else "{}"
    target.write_text(original)
    args = ("--infer", target) if infer else (target,)

    result = run_cli(
        "kuration.core.cli.preview", *args, "--html", target, expect=1)
    assert "also an input path" in result.stderr
    assert target.read_text() == original


def test_curate_review_cannot_replace_artifact_config_even_with_force(
        tmp_path, monkeypatch):
    from kuration.core.cli import curate

    source = tmp_path / "source.csv"
    source.write_text("t,id,x\n0,e,1\n")
    artifact = tmp_path / "dataset"
    artifact.mkdir()
    config = artifact / "config.json"
    config.write_text("last-good-config")
    job = tmp_path / "job.json"
    job.write_text(json.dumps({
        "sources": [str(source)],
        "schema": {},
        "adapter": {},
        "segmenter": {},
        "strategy": {},
        "output": {},
        "train_hours": 1,
        "out": str(artifact),
    }))
    monkeypatch.setattr(
        curate,
        "resolve_job",
        lambda value: pytest.fail("job resolution occurred after a known overlap"),
    )
    monkeypatch.setattr(
        curate,
        "write_dataset",
        lambda *args, **kwargs: pytest.fail("dataset mutation occurred before preflight"),
    )

    result = run_cli(
        "kuration.core.cli.curate",
        job,
        "--force",
        "--review-json",
        config,
        expect=1,
    )
    assert "new output path" in result.stderr
    assert config.read_text() == "last-good-config"

    prospective = artifact / "prospective-review.json"
    overlap = run_cli(
        "kuration.core.cli.curate",
        job,
        "--force",
        "--review-json",
        prospective,
        expect=1,
    )
    assert "overlaps input path" in overlap.stderr
    assert not prospective.exists()


def test_curate_review_outputs_cannot_alias_each_other_before_job_read(tmp_path):
    output = tmp_path / "review"
    missing_job = tmp_path / "missing-job.json"
    result = run_cli(
        "kuration.core.cli.curate",
        missing_job,
        "--review-json",
        output,
        "--review-html",
        output,
        expect=1,
    )
    assert "also an input path" in result.stderr
    assert not output.exists()


def test_init_job_output_cannot_be_created_inside_a_source_tree(tmp_path):
    source_tree = tmp_path / "source"
    source_tree.mkdir()
    output = source_tree / "job.json"

    result = run_cli(
        "kuration.core.cli.init_job",
        source_tree,
        "--out",
        output,
        "--force",
        expect=1,
    )
    assert "overlaps input path" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("module", "arguments"),
    [
        ("kuration.core.cli.inspect", ("missing-dataset",)),
        ("kuration.core.cli.stats", ("missing-dataset", "--quiet")),
        ("kuration.core.cli.diff", ("missing-left", "missing-right")),
        (
            "kuration.core.cli.check",
            ("missing-dataset", "--policy", "missing-policy.json"),
        ),
        ("kuration.core.cli.preview", ("missing-job.json",)),
        ("kuration.core.cli.curate", ("missing-job.json", "--review-json")),
    ],
)
def test_report_outputs_are_no_clobber_before_input_reads(
        tmp_path, module, arguments):
    output = tmp_path / "existing-report.html"
    output.write_text("last-good")
    args = (*arguments, output) if module.endswith("curate") else (
        *arguments, "--html", output)

    result = run_cli(module, *args, expect=1)
    assert "new output path" in result.stderr
    assert output.read_text() == "last-good"
