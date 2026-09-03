"""First-class convenience entry points for maintained gameplay capabilities."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.analyzers import Analyzer
    from ..core.pipeline import Pipeline


def action_state_checker(
    *,
    baselines: dict | None = None,
    anomaly_type: str = "action_state",
    calibration: dict | None = None,
) -> "Analyzer":
    """Construct the built-in action-state consistency analyzer.

    This is the direct Python spelling for
    ``get_analyzer("aintropy/action-state-consistency", ...)``. Without ``baselines`` the
    checker self-calibrates (leave-one-out consensus outlier detection); driving it then
    requires explicit ``controls=``.
    """
    from ..core.analyzers import get_analyzer

    params: dict = {"anomaly_type": anomaly_type}
    if baselines is not None:
        params["baselines"] = baselines
    if calibration is not None:
        params["calibration"] = calibration
    return get_analyzer("aintropy/action-state-consistency", **params)


def slam_reconstruct(
    source: Any,
    output: str | Path | None = None,
    *,
    depth_model: Any = None,
    embedding_model: Any = None,
    embeddings: Any = None,
    depth_model_id: str | None = None,
    embedding_model_id: str | None = None,
    runtime_factory: Any = None,
    join_video: bool = False,
    tolerance: float | None = None,
    mismatch: str = "error",
    provider: Any = None,
    run_if_needed: bool = True,
    decoder_options: Mapping[str, object] | None = None,
    export_options: Mapping[str, object] | None = None,
    resume: bool = False,
    cache_key: str | None = None,
    progress: Any = None,
    preflight: bool = True,
    require_gpu: bool = False,
    **runtime_options: Any,
) -> "Pipeline":
    """Prepare SLAM input and return a gameplay pipeline.

    ``source`` may be an existing manifest, completed reconstruction, raw frame mapping,
    or ``{clip: video}`` mapping. Raw input requires ``output`` and the ``slam`` extra.
    Existing manifests remain NumPy-only. Advanced callers can still construct
    :class:`SLAMReconstructor` directly when they need the intermediate reconstruction object.
    """
    from .slam import SLAMReconstructor
    from ..core.progress import progress_callback

    if progress is not None:
        progress = progress_callback(progress)

    decode = dict(decoder_options or {})
    if provider is not None:
        decode.setdefault("provider", provider)
    runner = SLAMReconstructor(
        depth_model,
        embedding_model=embedding_model,
        embeddings=embeddings,
        depth_model_id=depth_model_id,
        embedding_model_id=embedding_model_id,
        runtime_factory=runtime_factory,
        **runtime_options,
    )
    reconstruction = runner.reconstruct(
        source,
        output,
        run_if_needed=run_if_needed,
        decoder_options=decode,
        export_options=export_options,
        resume=resume,
        cache_key=cache_key,
        progress=progress,
        preflight=preflight,
        require_gpu=require_gpu,
    )
    return reconstruction.to_pipeline(
        join_video=join_video,
        tolerance=tolerance,
        mismatch=mismatch,
        provider=provider,
    )
