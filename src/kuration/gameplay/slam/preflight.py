"""Side-effect-free readiness checks for the optional SLAM preparation path."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

from .export import EXPORT_FORMAT, EXPORT_VERSION
from .runner import _runtime_factory, _validated_frames


@dataclass(frozen=True)
class SLAMPreflightResult:
    """JSON-safe, renderable readiness report."""

    input_kind: str
    will_run_slam: bool
    checks: tuple[dict, ...]

    @property
    def ready(self) -> bool:
        return not any(check["status"] == "fail" for check in self.checks)

    def to_dict(self) -> dict:
        return {
            "ready": self.ready,
            "input_kind": self.input_kind,
            "will_run_slam": self.will_run_slam,
            "checks": [dict(check) for check in self.checks],
        }

    def render(self) -> str:
        lines = [
            f"SLAM preflight: {'PASS' if self.ready else 'FAIL'}",
            f"input: {self.input_kind}; reconstruction: "
            f"{'required' if self.will_run_slam else 'skipped'}",
        ]
        for check in self.checks:
            marker = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[check["status"]]
            lines.append(f"{marker} {check['name']}: {check['message']}")
            if check.get("hint"):
                lines.append(f"  fix: {check['hint']}")
        return "\n".join(lines)

    def raise_for_errors(self) -> None:
        if not self.ready:
            failed = "; ".join(
                check["message"] for check in self.checks if check["status"] == "fail"
            )
            raise ValueError(f"SLAM preflight failed: {failed}")


def _check(name: str, status: str, message: str, hint: str | None = None) -> dict:
    result = {"name": name, "status": status, "message": message}
    if hint:
        result["hint"] = hint
    return result


def _manifest(path: Path) -> bool:
    candidate = path / "manifest.json" if path.is_dir() else path
    if not candidate.is_file():
        return False
    try:
        document = json.loads(candidate.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(document, dict):
        return False
    return (
        document.get("format") == EXPORT_FORMAT
        and document.get("version") == EXPORT_VERSION
    )


def _classify(source) -> str:
    if isinstance(source, (str, Path)):
        return "manifest" if _manifest(Path(source)) else "invalid_path"
    if isinstance(getattr(source, "trajectories", None), Mapping) \
            and isinstance(getattr(source, "poses", None), Mapping):
        return "completed_output"
    if isinstance(source, Mapping) and source:
        if all(isinstance(value, (str, Path)) for value in source.values()):
            return "video_paths"
        return "frames"
    return "unsupported"


def _writable_parent(path: Path) -> Path | None:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate if candidate.exists() and os.access(candidate, os.W_OK) else None


def _gpu_available() -> tuple[bool, str]:
    if importlib.util.find_spec("torch") is None:
        return False, "PyTorch is not installed"
    try:
        import torch

        available = bool(torch.cuda.is_available())
        return available, "CUDA is available" if available else "PyTorch reports no CUDA device"
    except Exception as error:  # pragma: no cover - backend-specific import failures
        return False, f"GPU detection failed: {type(error).__name__}: {error}"


def preflight_slam(
    source,
    out_dir: str | Path | None = None,
    *,
    run_if_needed: bool = True,
    runner_options: Mapping[str, object] | None = None,
    decoder_options: Mapping[str, object] | None = None,
    require_gpu: bool = False,
) -> SLAMPreflightResult:
    """Inspect readiness without decoding video, constructing models, or writing output."""
    options = dict(runner_options or {})
    decoder = dict(decoder_options or {})
    kind = _classify(source)
    will_run = kind in {"video_paths", "frames"}
    checks: list[dict] = []

    if kind in {"invalid_path", "unsupported"}:
        checks.append(_check(
            "input",
            "fail",
            "input is not a valid SLAM manifest, completed output, frame mapping, or video mapping",
            "Pass manifest.json, an export directory, SLAMOperator output, or {clip: input}.",
        ))
    else:
        checks.append(_check("input", "pass", f"recognized {kind.replace('_', ' ')} input"))

    if kind == "manifest":
        try:
            from .adapter import SLAMTrajectoryAdapter

            path = Path(source)
            path = path / "manifest.json" if path.is_dir() else path
            adapter = SLAMTrajectoryAdapter(manifest=path)
            for clip in adapter.sources():
                adapter.check_source(clip)
            checks.append(_check(
                "manifest", "pass", f"validated {len(adapter.sources())} trajectory archive(s)"
            ))
        except (OSError, TypeError, ValueError) as error:
            checks.append(_check("manifest", "fail", str(error)))

    if kind == "completed_output":
        trajectories = source.trajectories
        missing = sorted(set(trajectories) - set(source.poses), key=str)
        if not trajectories:
            checks.append(_check("completed-output", "fail", "trajectory mapping is empty"))
        elif missing:
            checks.append(_check(
                "completed-output", "fail", f"pose mapping is missing clip(s): {missing}"
            ))
        else:
            checks.append(_check(
                "completed-output", "pass", f"found {len(trajectories)} reconstructed clip(s)"
            ))

    if will_run and not run_if_needed:
        checks.append(_check(
            "execution-policy", "fail", "raw input requires SLAM but run_if_needed=False"
        ))
    elif will_run:
        checks.append(_check("execution-policy", "pass", "raw input is allowed to run SLAM"))
    else:
        checks.append(_check("execution-policy", "pass", "SLAM reconstruction will be skipped"))

    if kind != "manifest":
        if out_dir is None:
            checks.append(_check("output", "fail", "an output directory is required"))
        elif Path(out_dir).exists() and not Path(out_dir).is_dir():
            checks.append(_check("output", "fail", f"output path is not a directory: {out_dir}"))
        else:
            writable = _writable_parent(Path(out_dir))
            if writable is None:
                checks.append(_check("output", "fail", f"output path is not writable: {out_dir}"))
            else:
                checks.append(_check("output", "pass", f"output parent is writable: {writable}"))

    if kind == "frames":
        try:
            frames = _validated_frames(source)
            count = sum(len(value) for value in frames.values())
            checks.append(_check(
                "frames", "pass", f"validated {count} frame(s) across {len(frames)} clip(s)"
            ))
        except (TypeError, ValueError) as error:
            checks.append(_check("frames", "fail", str(error)))

    if kind == "video_paths":
        missing = []
        for value in source.values():
            parsed = urlparse(str(value))
            if not parsed.scheme and not Path(value).is_file():
                missing.append(str(value))
        if missing and decoder.get("provider") is None:
            checks.append(_check("videos", "fail", f"video path(s) do not exist: {missing}"))
        elif missing:
            checks.append(_check(
                "videos",
                "warn",
                f"custom provider will resolve non-local video reference(s): {missing}",
            ))
        else:
            checks.append(_check("videos", "pass", f"validated {len(source)} video reference(s)"))
        if decoder.get("provider") is not None or importlib.util.find_spec("av") is not None:
            checks.append(_check("video-runtime", "pass", "a video decoder is available"))
        else:
            checks.append(_check(
                "video-runtime",
                "fail",
                "PyAV video decoding support is not installed",
                "Run pip install 'kuration-sdk[slam]' or inject a provider.",
            ))

    if will_run:
        if options.get("depth_model") is None:
            checks.append(_check(
                "depth-model",
                "fail",
                "no depth model was supplied",
                "Provide runner_options['depth_model'] from the chosen SLAM runtime.",
            ))
        else:
            checks.append(_check("depth-model", "pass", "a depth model was supplied"))
        if options.get("embedding_model") is not None and options.get("embeddings") is not None:
            checks.append(_check(
                "embeddings", "fail", "embedding_model and embeddings cannot both be supplied"
            ))
        else:
            checks.append(_check("embeddings", "pass", "embedding inputs are unambiguous"))
        try:
            factory_option = options.get("runtime_factory")
            if factory_option is not None and not (
                isinstance(factory_option, str) or callable(factory_option)
            ):
                raise TypeError("runtime_factory must be callable or an import string")
            factory = _runtime_factory(cast(Callable | str | None, factory_option))
            checks.append(_check(
                "slam-runtime", "pass", f"loaded runtime factory {factory.__module__}."
                f"{getattr(factory, '__qualname__', type(factory).__name__)}"
            ))
        except (ImportError, TypeError, ValueError) as error:
            checks.append(_check("slam-runtime", "fail", str(error)))

    if require_gpu and will_run:
        available, detail = _gpu_available()
        checks.append(_check(
            "gpu",
            "pass" if available else "fail",
            detail,
            None if available else "Use a CUDA-capable environment or omit require_gpu.",
        ))
    elif will_run:
        checks.append(_check(
            "gpu",
            "warn",
            "GPU availability was not required; large reconstructions may be slow on CPU",
        ))

    return SLAMPreflightResult(kind, will_run, tuple(checks))
