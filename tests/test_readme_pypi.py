"""The README is the PyPI project description, where relative paths do not resolve."""
from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).parents[1]


def test_readme_targets_are_absolute_so_they_resolve_on_pypi():
    readme = (ROOT / "README.md").read_text()
    targets = re.findall(r"\]\(([^)\s]+)", readme) + re.findall(r'(?:src|href)="([^"]+)"', readme)
    relative = sorted({t for t in targets if not re.match(r"(https?:|mailto:|#)", t)})
    assert not relative, (
        f"README.md is rendered by PyPI, which does not resolve relative paths; "
        f"make these absolute URLs: {relative}"
    )
