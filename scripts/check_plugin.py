#!/usr/bin/env python3
"""Reproduce the repository plugin CI contract in one isolated temporary environment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, cwd: Path = ROOT) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _executable(environment: Path, name: str) -> str:
    folder = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return str(environment / folder / f"{name}{suffix}")


def _metadata(plugin: Path) -> tuple[str, tuple[str, ...]]:
    config = plugin / "pyproject.toml"
    if not config.is_file():
        raise ValueError(f"plugin has no pyproject.toml: {plugin}")
    with config.open("rb") as handle:
        document = tomllib.load(handle)
    project = document.get("project", {})
    package = project.get("name")
    points = project.get("entry-points", {}).get("kuration.plugins", {})
    if not isinstance(package, str) or not package:
        raise ValueError(f"plugin project.name is missing in {config}")
    if not isinstance(points, dict) or not points \
            or any(not isinstance(name, str) or not name for name in points):
        raise ValueError(f"plugin must declare at least one kuration.plugins entry point: {config}")
    return package, tuple(sorted(points))


def check_plugin(plugin: str | Path, *, artifacts: str | Path | None = None) -> list[str]:
    """Install, test, build, reinstall, and contract-check one plugin distribution."""
    path = Path(plugin)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    try:
        path.relative_to(ROOT / "plugins")
    except ValueError as error:
        raise ValueError(f"plugin must be inside {ROOT / 'plugins'}: {path}") from error
    package, entry_points = _metadata(path)

    with tempfile.TemporaryDirectory(prefix="kuration-plugin-check-") as directory:
        work = Path(directory)
        environment = work / "venv"
        distribution_dir = work / "dist"
        _run([sys.executable, "-m", "venv", str(environment)])
        python = _executable(environment, "python")
        checker = _executable(environment, "check-wheel-contents")
        _run([python, "-m", "pip", "install", "-e", str(ROOT),
              "-e", f"{path}[test]"])
        _run([python, "-m", "pip", "check"])
        _run([python, "-m", "pytest", str(path / "tests"), "-q"])
        _run([python, "-m", "ruff", "check", str(path / "src"), str(path / "tests")])
        _run([python, "-m", "build", "--outdir", str(distribution_dir), str(path)])
        wheels = sorted(distribution_dir.glob("*.whl"))
        sdists = sorted(distribution_dir.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise RuntimeError(
                f"{package} must build exactly one wheel and one sdist; got {wheels}, {sdists}")
        _run([checker, str(wheels[0])])
        _run([python, "-m", "pip", "install", "--force-reinstall", "--no-deps",
              str(wheels[0])])
        for entry_point in entry_points:
            _run([python, "-m", "kuration.cli.plugin_check", entry_point, "--json"], cwd=work)
        _run([python, "-m", "pip", "check"])
        built = [*wheels, *sdists]
        labels = [item.name for item in built]
        if artifacts is not None:
            destination = Path(artifacts)
            if not destination.is_absolute():
                destination = ROOT / destination
            destination.mkdir(parents=True, exist_ok=True)
            labels = [str(Path(shutil.copy2(item, destination / item.name))) for item in built]
        return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plugin", help="Plugin directory below plugins/.")
    parser.add_argument("--artifacts", help="Optional directory receiving the wheel and sdist.")
    args = parser.parse_args()
    built = check_plugin(args.plugin, artifacts=args.artifacts)
    print(f"PASS {args.plugin}: " + ", ".join(built))


if __name__ == "__main__":
    main()
