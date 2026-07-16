"""Dependency-free static HTML rendering for product review results."""
from __future__ import annotations

from html import escape
import json

from .io import write_text


def render_html_report(*reports, title: str = "kuration review") -> str:
    """Render one or more typed/mapping reports as a self-contained static HTML document."""
    if not reports:
        raise ValueError("render_html_report needs at least one report")
    sections = []
    for report in reports:
        data = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        name = type(report).__name__ if type(report) is not dict else "Report"
        payload = escape(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        summary = []
        for key in ("candidates", "selected", "selection_rate", "shards", "container",
                    "identical"):
            if key in data:
                summary.append(f"<dt>{escape(key.replace('_', ' ').title())}</dt>"
                               f"<dd>{escape(str(data[key]))}</dd>")
        sections.append(f"<section><h2>{escape(name)}</h2>"
                        f"<dl>{''.join(summary)}</dl><details open><summary>Structured data"
                        f"</summary><pre>{payload}</pre></details></section>")
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>{title}</title><style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ max-width: 1100px; margin: 2rem auto; padding: 0 1rem; line-height: 1.45; }}
section {{ border: 1px solid #8886; border-radius: .6rem; padding: 1rem; margin: 1rem 0; }}
dl {{ display: grid; grid-template-columns: max-content 1fr; gap: .3rem 1rem; }}
dt {{ font-weight: 650; }} dd {{ margin: 0; }}
pre {{ overflow: auto; padding: 1rem; background: #8881; border-radius: .4rem; }}
summary {{ cursor: pointer; font-weight: 600; margin: .5rem 0; }}
</style></head><body><h1>{title}</h1>{sections}</body></html>""".format(
        title=escape(title), sections="".join(sections))


def write_html_report(path, *reports, title: str = "kuration review") -> str:
    """Write a self-contained review HTML document and return ``path`` as a string."""
    path = str(path)
    write_text(path, render_html_report(*reports, title=title))
    return path
