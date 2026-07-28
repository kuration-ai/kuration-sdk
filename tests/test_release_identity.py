"""Development and release artifacts must never reuse an already released identity."""
from __future__ import annotations

from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).parents[1]


def test_unreleased_work_does_not_reuse_a_changelog_release_version():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project_version = tomllib.load(handle)["project"]["version"]

    changelog = (ROOT / "CHANGELOG.md").read_text()
    unreleased = changelog.split("## Unreleased", 1)[1].split("\n## ", 1)[0]
    released_versions = set(re.findall(r"^## ([0-9][^\s]*)\s*$", changelog, re.MULTILINE))

    assert re.search(r"^- ", unreleased, re.MULTILINE), \
        "the development branch should record its changes under Changelog / Unreleased"
    assert project_version not in released_versions, (
        f"Unreleased work is still building as released version {project_version}; "
        "use the next .devN or release-candidate identity"
    )
