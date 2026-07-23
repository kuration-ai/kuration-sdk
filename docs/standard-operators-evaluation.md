# Standard operators evaluation

This branch evaluates shipping the repository's maintained domain capabilities inside the
`kuration-sdk` distribution instead of publishing them as separate plugin distributions. It is
an implementation experiment, not a permanent admission policy.

## Included capabilities

| capability | built-in surface | optional runtime |
|---|---|---|
| Pixel shift | `Pipeline.pixel_shift()`; lower-level `kuration.pixel_shift` and `aintropy/pixel-shift` | `[pixel-shift]` |
| Action-state consistency | `action_state_checker()`; lower-level `kuration.action_state_consistency` and `aintropy/action-state-consistency` | `[action-state]` for frame measurement |
| SLAM | `slam_pipeline()`, `Pipeline.slam_motion()`, and `Pipeline.slam_quality()`; advanced `kuration.slam` API | `[slam]`, `[slam-viz]` |

The SLAM implementation includes the three commits at the tip of the original `slam` branch:
`009901a`, `3324564`, and `b98877c`.

## What changed

- First-party implementations live under `src/kuration` and ship in the SDK wheel.
- Their descriptors register during `import kuration`, before external plugin discovery.
- Ordinary Python use needs one import and no component-name strings:
  `import kuration as kur`.
- The base dependency list remains NumPy-only. OpenCV, PyTorch, FAISS, PyAV, and related
  frameworks remain optional and are not imported during SDK startup.
- `gameplay` installs every bundled gameplay runtime for a single-command setup.
- `standard-operators` remains an equivalent compatibility alias during the evaluation.
- External plugins and the `kuration.plugins` entry-point mechanism remain supported.
- Compatibility packages preserve the former `kuration_pixel_shift` and
  `kuration_action_state_consistency` imports and tolerate old installed entry-point metadata.

## Current measurements

Measured from clean wheels built on the same machine and revision family:

| check | post-revert `dev/main` | evaluation branch |
|---|---:|---:|
| wheel size | 254,873 bytes | 421,060 bytes |
| wheel increase | — | 166,187 bytes (65.2%) |
| required dependencies | NumPy | NumPy |
| eager `torch`, `cv2`, or `faiss` import | no | no |
| full test suite | — | 823 passed, 7 skipped |
| focused maintained-capability suite | — | 150 passed, 1 skipped |

The wheel increase measures bundled Python/configuration code, including the vendored Depth
Anything 3 implementation. It does not include the much larger optional runtime wheels or model
weights.

## Installation modes

```bash
pip install kuration-sdk
pip install "kuration-sdk[gameplay]"
pip install "kuration-sdk[all]"
```

The base package is the NumPy-only core, `gameplay` installs Pixel Shift, action-state, SLAM, and
SLAM visualization, and `all` installs every runtime integration and backend. Granular installs
remain available:

```bash
pip install "kuration-sdk[pixel-shift]"
pip install "kuration-sdk[action-state]"
pip install "kuration-sdk[slam]"
```

The base installation exposes descriptors and lightweight APIs. Execution that reaches a missing
framework fails with the relevant extra's installation command. Existing
`kuration-sdk[standard-operators]` installation commands continue to work as a compatibility
alias for `gameplay`.

## Single-import facade

```python
import kuration as kur

pixel_pipeline = kur.input(telemetry, ...).join_video(videos).pixel_shift()
slam_pipeline = (
    kur.slam_pipeline("slam-export/manifest.json")
    .window(32, stride=16)
    .slam_motion()
    .slam_quality(min_score=0.6)
)
checker = kur.action_state_checker(baselines=baselines)
```

These methods are aliases over the same registered descriptors used by declarative jobs and
provenance. They improve discovery and imports without forking operator behavior.

## Evaluation questions

1. Does built-in discovery materially improve first-run usability compared with separately
   installed first-party plugins?
2. Is a 65.2% wheel increase acceptable when required dependencies and eager imports remain
   unchanged?
3. Should domain-specific APIs share the SDK's release cadence and compatibility guarantees?
4. Is vendoring a model implementation acceptable, or should the built-in SLAM adapter depend on
   a separately versioned runtime?
5. Does the compatibility layer provide a credible migration path from already-installed plugin
   distributions?
6. Which objective admission criteria distinguish future built-ins from external plugins?

## Reproduce the checks

```bash
python -m pytest -q
python -m ruff check src tests examples
python -m mypy
python -m mkdocs build --strict
python -m build
check-wheel-contents dist/*.whl
twine check dist/*
```
