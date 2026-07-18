"""Keep the supported Python surface synchronized with its canonical reference."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
import re

import kuration


ROOT = Path(__file__).parents[1]
REFERENCE = ROOT / "docs" / "api.md"


def _mentions_identifier(text: str, name: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text) is not None


def test_api_reference_covers_every_top_level_export():
    text = REFERENCE.read_text()
    missing = [name for name in kuration.__all__ if not _mentions_identifier(text, name)]
    assert not missing, f"docs/api.md is missing kuration exports: {missing}"


def test_api_reference_covers_every_public_pipeline_method():
    tree = ast.parse((ROOT / "src" / "kuration" / "pipeline.py").read_text())
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
    match = re.search(rf"\| `{re.escape(name)}` \| `([^`]+)` \|", text)
    assert match is not None, f"docs/api.md has no signature row for {name}"
    return match.group(1).split(" -> ", 1)[0]


def test_primary_api_signatures_match_the_reference():
    text = REFERENCE.read_text()
    public = {
        "curate": kuration.curate,
        "write_dataset": kuration.write_dataset,
        "run_experiment": kuration.run_experiment,
        "dataset_report": kuration.dataset_report,
        "input": kuration.input,
        "Pipeline.sample": kuration.Pipeline.sample,
        "Pipeline.output": kuration.Pipeline.output,
        "Pipeline.review": kuration.Pipeline.review,
    }
    drift = {
        name: {"documented": _documented_signature(text, name),
               "runtime": _plain_signature(obj)}
        for name, obj in public.items()
        if _documented_signature(text, name) != _plain_signature(obj)
    }
    assert not drift, f"docs/api.md signature drift: {drift}"


def test_api_reference_covers_optional_torch_surface():
    assert "kuration.torch.ShardDataset" in REFERENCE.read_text()


def test_public_objects_have_source_docstrings():
    # A typing alias has no independent runtime ``__doc__`` slot; its contract is documented
    # explicitly in the API reference. Every ordinary exported object must explain itself.
    exemptions = {"ProgressCallback"}
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
