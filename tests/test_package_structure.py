"""Product-level checks for Kuration's domain package boundaries."""
from __future__ import annotations

from importlib.util import find_spec
from importlib.resources import files
from pathlib import Path
import tomllib

import kuration
import kuration.core
import kuration.gameplay


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
    assert domains == {"core", "gameplay", "sample_data", "selection"}


def test_removed_root_compatibility_packages_are_not_importable() -> None:
    assert find_spec("depth_anything_3") is None
    assert find_spec("kuration_action_state_consistency") is None
    assert find_spec("kuration_pixel_shift") is None


def test_package_level_api_is_exported_by_core() -> None:
    assert kuration.__all__ is kuration.core.__all__
    assert kuration.Pipeline.__module__ == "kuration.core.pipeline"


def test_domain_resources_resolve_from_their_own_package() -> None:
    assert files("kuration.core").joinpath("schemas/job.schema.json").is_file()
    assert files("kuration").joinpath("py.typed").is_file()


def test_slam_reconstruction_and_curation_pipeline_boundaries_are_explicit() -> None:
    assert not hasattr(kuration.gameplay, "GameplayPipeline")
    assert not hasattr(kuration.gameplay, "as_gameplay")
    assert not hasattr(kuration.gameplay, "input")
    assert not issubclass(kuration.gameplay.SLAMReconstructor, kuration.Pipeline)
    assert callable(kuration.gameplay.slam_reconstruct)
    assert not isinstance(kuration.gameplay.slam_reconstruct, type)


def test_vendored_slam_runtime_is_outside_first_party_coverage() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        configuration = tomllib.load(handle)

    omitted = configuration["tool"]["coverage"]["run"]["omit"]
    assert "src/kuration/gameplay/slam/_vendor/**" in omitted
    assert "src/kuration/gameplay/slam/_runtime.py" in omitted
    assert "src/kuration/gameplay/slam/slam_pipeline.py" not in omitted
    assert all(not path.startswith("src/kuration/core/") for path in omitted)
