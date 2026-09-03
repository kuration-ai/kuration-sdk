"""Offline checks for repository-local links in public Markdown documentation."""

from __future__ import annotations

import re
from pathlib import Path
import textwrap
from urllib.parse import unquote, urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INLINE_LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)")
REFERENCE_LINK = re.compile(r"^\s*\[[^\]\n]+\]:\s*(\S+)", re.MULTILINE)
INTERNAL_REFERENCE = re.compile(r"(?:CLAUDE\.md|(?<![A-Za-z0-9_])/home(?:/|\b))")
PYTHON_FENCE = re.compile(
    r"^[ \t]*```python[ \t]*\n(.*?)^[ \t]*```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
SNIPPET_INCLUDE = re.compile(r"^\s*--8<--\s+\"[^\"\n]+\"\s*$")


def _public_markdown_files() -> list[Path]:
    """Return version-controlled documentation surfaces, excluding the private agent guide."""

    files = [path for path in REPOSITORY_ROOT.glob("*.md") if path.name != "CLAUDE.md"]
    for directory in ("docs", "examples"):
        files.extend((REPOSITORY_ROOT / directory).rglob("*.md"))
    return sorted(set(files))


def _link_target(raw_target: str) -> str:
    """Remove optional Markdown titles and angle brackets from a link destination."""

    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")]
    return target.split(maxsplit=1)[0]


def _local_path(source: Path, target: str) -> Path | None:
    """Resolve a local Markdown destination without making any network request."""

    if not target or target.startswith("#") or target.startswith("//"):
        return None

    parsed = urlsplit(target)
    if parsed.scheme:
        return None

    link_path = unquote(parsed.path)
    if not link_path:
        return None
    if link_path.startswith("/"):
        return REPOSITORY_ROOT / link_path.lstrip("/")
    return source.parent / link_path


def test_repository_local_markdown_links_exist() -> None:
    missing: list[str] = []

    for source in _public_markdown_files():
        text = source.read_text(encoding="utf-8")
        matches = [*INLINE_LINK.finditer(text), *REFERENCE_LINK.finditer(text)]
        for match in matches:
            target = _link_target(match.group(1))
            local_path = _local_path(source, target)
            if local_path is None or local_path.exists():
                continue
            line = text.count("\n", 0, match.start()) + 1
            relative_source = source.relative_to(REPOSITORY_ROOT)
            missing.append(f"{relative_source}:{line}: {target} -> {local_path}")

    assert not missing, "Missing repository-local Markdown links:\n" + "\n".join(missing)


def test_public_markdown_has_no_internal_machine_or_agent_references() -> None:
    leaked: list[str] = []

    for source in _public_markdown_files():
        text = source.read_text(encoding="utf-8")
        for match in INTERNAL_REFERENCE.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            relative_source = source.relative_to(REPOSITORY_ROOT)
            leaked.append(f"{relative_source}:{line}: {match.group(0)}")

    assert not leaked, "Internal-only references found in public Markdown:\n" + "\n".join(leaked)


def test_cli_reference_mentions_every_public_core_flag() -> None:
    """Keep the human CLI reference synchronized with every installed core parser option."""

    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (REPOSITORY_ROOT / "src" / "kuration" / "core" / "cli").glob("*.py")
    )
    flags = set(re.findall(r"""add_argument\(\s*["'](--[a-z0-9-]+)""", source))
    reference = (REPOSITORY_ROOT / "docs" / "jobs.md").read_text(encoding="utf-8")
    missing = sorted(flag for flag in flags if flag not in reference)

    assert not missing, f"docs/jobs.md is missing CLI flags: {missing}"


def test_fenced_python_examples_are_syntactically_valid() -> None:
    failures: list[str] = []

    for source in _public_markdown_files():
        text = source.read_text(encoding="utf-8")
        for match in PYTHON_FENCE.finditer(text):
            if SNIPPET_INCLUDE.match(match.group(1)):
                continue  # pymdownx.snippets expands this at mkdocs build time, not here.
            line = text.count("\n", 0, match.start()) + 1
            try:
                compile(textwrap.dedent(match.group(1)), str(source), "exec")
            except SyntaxError as error:
                relative_source = source.relative_to(REPOSITORY_ROOT)
                failures.append(f"{relative_source}:{line}: {error.msg}")

    assert not failures, "Invalid fenced Python examples:\n" + "\n".join(failures)


def test_numbered_examples_are_cataloged_and_run_by_ci() -> None:
    catalog = (REPOSITORY_ROOT / "examples" / "README.md").read_text(encoding="utf-8")
    # CI is split across several ci*.yml files so that no single workflow run
    # has a job list large enough to time out the self-hosted runner's scaler,
    # so an example may be run from any of them.
    workflow_paths = sorted((REPOSITORY_ROOT / ".github" / "workflows").glob("ci*.yml"))
    assert workflow_paths, "no ci*.yml workflow files found"
    workflows = "\n".join(path.read_text(encoding="utf-8") for path in workflow_paths)
    missing: list[str] = []

    for example in sorted((REPOSITORY_ROOT / "examples").glob("[0-9][0-9]_*.py")):
        relative = example.relative_to(REPOSITORY_ROOT).as_posix()
        if relative not in catalog:
            missing.append(f"{relative}: examples/README.md")
        if relative not in workflows:
            missing.append(f"{relative}: .github/workflows/ci*.yml")

    assert not missing, "Numbered examples missing from catalog or CI:\n" + "\n".join(missing)
