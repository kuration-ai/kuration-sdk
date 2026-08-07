"""Development and release artifacts must never reuse an already released identity."""
from __future__ import annotations

from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).parents[1]


def test_development_work_does_not_reuse_a_changelog_release_version():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project_version = tomllib.load(handle)["project"]["version"]

    changelog = (ROOT / "CHANGELOG.md").read_text()
    current_match = re.search(r"^## (Unreleased|[0-9][^\s]*)\s*$", changelog, re.MULTILINE)
    assert current_match is not None, "the changelog needs a current release section"
    current_heading = current_match.group(1)
    current = changelog[current_match.end():].split("\n## ", 1)[0]
    released_versions = set(re.findall(r"^## ([0-9][^\s]*)\s*$", changelog, re.MULTILINE))

    assert re.search(r"^- ", current, re.MULTILINE), \
        "the current changelog section should record the development branch's changes"
    if current_heading != "Unreleased":
        target_version = re.sub(r"\.dev[0-9]+$", "", project_version)
        assert current_heading == target_version, (
            f"current changelog section {current_heading} does not match development target "
            f"{target_version}"
        )
    assert project_version not in released_versions, (
        f"Development work is still building as released version {project_version}; "
        "use the next .devN or release-candidate identity"
    )
