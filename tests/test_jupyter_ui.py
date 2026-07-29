"""Opt-in browser checks for notebook-facing HTML.

The ordinary suite skips these because Playwright browsers are intentionally not SDK
dependencies. CI enables them in its dedicated notebook/UI job.
"""
from __future__ import annotations

import os

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("KURATION_BROWSER_TESTS") != "1",
    reason="set KURATION_BROWSER_TESTS=1 in the dedicated Playwright environment",
)


@pytest.fixture(scope="module")
def page():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        page = browser.new_page()
        yield page
        browser.close()


def test_progress_markup_has_visible_browser_geometry(page):
    from kuration.core.progress import _notebook_progress_html

    page.set_content(
        _notebook_progress_html(
            {
                "phase": "scoring",
                "event": "advance",
                "current": 2,
                "total": 4,
                "source": "match-01",
            }
        )
    )

    assert page.locator(".kur-progress strong").inner_text() == "scoring"
    assert page.locator(".kur-progress").get_by_text("2/4").count() == 1
    assert page.locator(".kur-progress-bar").evaluate(
        "(element) => element.style.width"
    ) == "50%"


def test_video_comparison_synchronization_runs_in_a_browser(page, tmp_path):
    from kuration.gameplay import compare_videos

    left = tmp_path / "left.mp4"
    right = tmp_path / "right.mp4"
    left.write_bytes(b"left")
    right.write_bytes(b"right")
    html = compare_videos(left, right).render(embed=True)
    page.set_content(html)

    assert page.locator(".kur-video-compare video").count() == 2
    mirrored = page.evaluate(
        """() => {
          const [left, right] = document.querySelectorAll('video');
          window.__kurationMirrored = false;
          right.play = () => {
            window.__kurationMirrored = true;
            return Promise.resolve();
          };
          left.dispatchEvent(new Event('play'));
          return window.__kurationMirrored;
        }"""
    )
    assert mirrored is True


def test_gameplay_frame_png_decodes_in_a_browser(page):
    from kuration.gameplay._stats import _png_data_url

    source = _png_data_url(
        np.full((12, 16, 3), [20, 100, 220], dtype=np.uint8),
        color_space="RGB",
    )
    page.set_content(f'<img class="evidence" src="{source}">')
    page.locator("img.evidence").wait_for(state="visible")

    assert page.locator("img.evidence").evaluate(
        "(image) => image.complete && image.naturalWidth === 16 && image.naturalHeight === 12"
    )
