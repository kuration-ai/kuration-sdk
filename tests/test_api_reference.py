"""Keep the supported Python surface synchronized with its canonical reference."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
import re

import kuration


ROOT = Path(__file__).parents[1]
REFERENCE = ROOT / "docs" / "api.md"
DOCUMENTATION_CONTRACT = ROOT / "docs" / "documentation-contract.md"


def _reference_rows(text: str) -> set[str]:
    """Return the exact object names that own API-reference table rows."""

    return set(re.findall(r"^\| `([^`]+)` \|", text, re.MULTILINE))


def test_api_reference_covers_every_top_level_export():
    text = REFERENCE.read_text()
    rows = _reference_rows(text)
    missing = [name for name in kuration.__all__ if name not in rows]
    assert not missing, f"docs/api.md is missing kuration exports: {missing}"


def test_documentation_contract_inventories_every_top_level_export():
    text = DOCUMENTATION_CONTRACT.read_text()
    inventory = text.split("### Top-level Python API", 1)[1].split("### Pipeline", 1)[0]
    missing = [
        name for name in kuration.__all__
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", inventory) is None
    ]
    assert not missing, f"documentation contract is missing kuration exports: {missing}"


def test_api_reference_covers_every_gameplay_facade_export():
    import kuration.gameplay as gameplay

    rows = _reference_rows(REFERENCE.read_text())
    missing = [
        name for name in gameplay.__all__
        if not any(row == name or row.endswith(f".{name}") for row in rows)
    ]
    assert not missing, f"docs/api.md is missing kuration.gameplay exports: {missing}"


def test_api_reference_covers_every_public_pipeline_method():
    tree = ast.parse((ROOT / "src" / "kuration" / "core" / "pipeline.py").read_text())
    pipeline = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Pipeline")
    methods = [
        node.name for node in pipeline.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]
    text = REFERENCE.read_text()
    missing = [name for name in methods if f"Pipeline.{name}" not in text]
    assert not missing, f"docs/api.md is missing Pipeline methods: {missing}"

    contract = DOCUMENTATION_CONTRACT.read_text()
    inventory = contract.split("### Pipeline", 1)[1].split(
        "### Framework integration", 1)[0]
    contract_missing = [
        name for name in methods
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", inventory) is None
    ]
    assert not contract_missing, (
        f"documentation contract is missing Pipeline methods: {contract_missing}"
    )


def test_operator_guides_cover_every_bundled_gameplay_operator():
    import kuration.gameplay  # noqa: F401

    expected = {
        "aintropy/pixel-shift": "pixel_shift",
        "aintropy/action-state-consistency": "action_state_consistency",
        "aintropy/collision": "collision",
        "aintropy/collision-directional": "collision_directional",
        "aintropy/slam-motion-diversity": "slam_motion_diversity",
        "aintropy/slam-quality": "slam_quality",
    }
    registered = {
        name for name in kuration.default_registry.names("operator")
        if name.startswith("aintropy/")
    }
    assert registered == set(expected)

    references = [
        (ROOT / "docs" / "operators.md").read_text(),
        (ROOT / "docs" / "pipeline.md").read_text(),
        REFERENCE.read_text(),
    ]
    for component, method in expected.items():
        assert all(component in text for text in references)
        assert all(method in text for text in references)


def test_operator_guides_cover_every_bundled_selection_operator():
    import kuration.selection  # noqa: F401

    expected = {
        "kuration/balanced-selector": "balanced_selector",
        "kuration/rare-event-coverage": "rare_event_coverage",
        "kuration/diverse-subset": "diverse_subset",
    }
    registered = {
        name for name in kuration.default_registry.names("operator")
        if name.startswith("kuration/")
    }
    assert registered == set(expected)

    references = [
        (ROOT / "docs" / "operators.md").read_text(),
        (ROOT / "docs" / "pipeline.md").read_text(),
        (ROOT / "docs" / "selection.md").read_text(),
        REFERENCE.read_text(),
    ]
    for component, method in expected.items():
        assert all(component in text for text in references)
        assert all(method in text for text in references)


def _plain_signature(obj) -> str:
    signature = inspect.signature(obj)
    parameters = [
        parameter.replace(annotation=inspect.Signature.empty)
        for parameter in signature.parameters.values()
        if parameter.name != "self"
    ]
    signature = signature.replace(
        parameters=parameters,
        return_annotation=inspect.Signature.empty,
    )
    return str(signature).replace("'", '"')


def _documented_signature(text: str, name: str) -> str:
    match = re.search(rf"\| `{re.escape(name)}` \| `([^`]+)`", text)
    assert match is not None, f"docs/api.md has no signature row for {name}"
    return match.group(1).split(" -> ", 1)[0]


def test_primary_api_signatures_match_the_reference():
    text = REFERENCE.read_text()
    public = {
        "curate": kuration.core.curate,
        "write_dataset": kuration.write_dataset,
        "run_experiment": kuration.run_experiment,
        "dataset_report": kuration.dataset_report,
        "input": kuration.input,
        "Pipeline": kuration.Pipeline,
        "Pipeline.sample": kuration.Pipeline.sample,
        "Pipeline.curate": kuration.Pipeline.curate,
        "Pipeline.output": kuration.Pipeline.output,
        "Pipeline.review": kuration.Pipeline.review,
        "schema_from_output": kuration.schema_from_output,
        "FeatureEngine": kuration.FeatureEngine,
        "CuratedDatasetAdapter": kuration.CuratedDatasetAdapter,
        "FrameRef": kuration.FrameRef,
        "VideoMetadata": kuration.VideoMetadata,
        "PyAVMediaProvider": kuration.PyAVMediaProvider,
        "VideoAdapter": kuration.VideoAdapter,
        "JoinedMediaAdapter": kuration.JoinedMediaAdapter,
        "get_adapter": kuration.get_adapter,
        "register_batch_transform": kuration.register_batch_transform,
        "register_transform": kuration.register_transform,
        "get_transform": kuration.get_transform,
        "apply_transforms": kuration.apply_transforms,
        "LocalExecutionBackend": kuration.LocalExecutionBackend,
        "RayExecutionBackend": kuration.RayExecutionBackend,
        "ResourceSpec": kuration.ResourceSpec,
        "get_execution_backend": kuration.get_execution_backend,
        "load_strategy": kuration.load_strategy,
        "get_scorer": kuration.get_scorer,
        "PassthroughEncoder": kuration.PassthroughEncoder,
        "get_encoder": kuration.get_encoder,
        "cli_progress": kuration.cli_progress,
        "register_analyzer": kuration.register_analyzer,
        "Arm": kuration.Arm,
        "PartitionCommit": kuration.PartitionCommit,
        "checkpoint_feature_stage": kuration.checkpoint_feature_stage,
        "external_top_k": kuration.external_top_k,
        "external_uniform_bins": kuration.external_uniform_bins,
        "SplitLeakageAnalyzer": kuration.SplitLeakageAnalyzer,
        "SplitLeakageAnalyzer.add_assignment": kuration.SplitLeakageAnalyzer.add_assignment,
        "SplitLeakageAnalyzer.add_assignments": kuration.SplitLeakageAnalyzer.add_assignments,
        "SplitLeakageAnalyzer.summary": kuration.SplitLeakageAnalyzer.summary,
    }
    drift = {
        name: {"documented": _documented_signature(text, name),
               "runtime": _plain_signature(obj)}
        for name, obj in public.items()
        if _documented_signature(text, name) != _plain_signature(obj)
    }
    assert not drift, f"docs/api.md signature drift: {drift}"


def test_api_reference_covers_optional_torch_surface():
    assert "kuration.core.torch.ShardDataset" in REFERENCE.read_text()


def test_public_objects_have_source_docstrings():
    # Typing aliases have no independent runtime ``__doc__`` slot; their contracts are
    # documented explicitly in the API reference. Every ordinary exported object must
    # explain itself.
    exemptions = {"ProgressCallback", "ProgressSpec"}
    missing = [
        name for name in kuration.__all__
        if name not in exemptions and not inspect.getdoc(getattr(kuration, name))
    ]
    assert not missing, f"public kuration objects without docstrings: {missing}"


def test_runtime_dependency_hints_use_the_published_package_name():
    offenders = []
    for path in (ROOT / "src" / "kuration").rglob("*.py"):
        if "pip install -e '.[" in path.read_text():
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"runtime install hints are contributor-only editable commands: {offenders}"
