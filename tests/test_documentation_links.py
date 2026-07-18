"""Offline checks for repository-local links in public Markdown documentation."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INLINE_LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)")
REFERENCE_LINK = re.compile(r"^\s*\[[^\]\n]+\]:\s*(\S+)", re.MULTILINE)
INTERNAL_REFERENCE = re.compile(r"(?:CLAUDE\.md|(?<![A-Za-z0-9_])/home(?:/|\b))")


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
