"""Product-level checks for Kuration's domain package boundaries."""
from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import kuration
import kuration.core


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
PACKAGE_ROOT = SOURCE_ROOT / "kuration"


def test_source_root_contains_one_product_package() -> None:
    packages = {
        path.name
        for path in SOURCE_ROOT.iterdir()
        if path.is_dir() and not path.name.endswith(".egg-info") and path.name != "__pycache__"
    }
    assert packages == {"kuration"}


def test_kuration_implementation_is_split_into_explicit_domains() -> None:
    implementation_modules = {
        path.name for path in PACKAGE_ROOT.glob("*.py") if path.name != "__init__.py"
    }
    domains = {
        path.name
        for path in PACKAGE_ROOT.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }

    assert implementation_modules == set()
    assert domains == {"core", "gameplay"}


def test_removed_root_compatibility_packages_are_not_importable() -> None:
    assert find_spec("depth_anything_3") is None
    assert find_spec("kuration_action_state_consistency") is None
    assert find_spec("kuration_pixel_shift") is None


def test_package_level_api_is_exported_by_core() -> None:
    assert kuration.__all__ is kuration.core.__all__
    assert kuration.Pipeline.__module__ == "kuration.core.pipeline"
