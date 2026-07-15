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
    tree = ast.parse((ROOT / "kuration" / "pipeline.py").read_text())
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
