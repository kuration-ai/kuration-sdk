# Standard operators evaluation

This branch evaluates shipping the repository's maintained domain capabilities inside the
`kuration-sdk` distribution instead of publishing them as separate plugin distributions. It is
an implementation experiment, not a permanent admission policy.

## Included capabilities

| capability | built-in surface | optional runtime |
|---|---|---|
| Pixel shift | `kuration.pixel_shift`; operator and transform `aintropy/pixel-shift` | `[pixel-shift]` |
| Action-state consistency | `kuration.action_state_consistency`; analyzer `aintropy/action-state-consistency` | `[action-state]` for frame measurement |
| SLAM | `kuration.slam`; trajectory adapter, motion/quality transforms and operators, native reconstruction API | `[slam]`, `[slam-viz]` |

The SLAM implementation includes the three commits at the tip of the original `slam` branch:
`009901a`, `3324564`, and `b98877c`.

## What changed

- First-party implementations live under `src/kuration` and ship in the SDK wheel.
- Their descriptors register during `import kuration`, before external plugin discovery.
- The base dependency list remains NumPy-only. OpenCV, PyTorch, FAISS, PyAV, and related
  frameworks remain optional and are not imported during SDK startup.
- `standard-operators` installs every first-party operator runtime for a single-command setup.
- External plugins and the `kuration.plugins` entry-point mechanism remain supported.
- Compatibility packages preserve the former `kuration_pixel_shift` and
  `kuration_action_state_consistency` imports and tolerate old installed entry-point metadata.

## Current measurements

Measured from clean wheels built on the same machine and revision family:

| check | post-revert `dev/main` | evaluation branch |
|---|---:|---:|
| wheel size | 254,873 bytes | 418,662 bytes |
| wheel increase | — | 163,789 bytes (64.3%) |
| required dependencies | NumPy | NumPy |
| eager `torch`, `cv2`, or `faiss` import | no | no |
| full test suite | — | 817 passed, 7 skipped |
| focused maintained-capability suite | — | 144 passed, 1 skipped |

The wheel increase measures bundled Python/configuration code, including the vendored Depth
Anything 3 implementation. It does not include the much larger optional runtime wheels or model
weights.

## Installation modes

```bash
pip install kuration-sdk
pip install "kuration-sdk[pixel-shift]"
pip install "kuration-sdk[action-state]"
pip install "kuration-sdk[slam]"
pip install "kuration-sdk[standard-operators]"
```

The base installation exposes descriptors and lightweight APIs. Execution that reaches a missing
framework fails with the relevant extra's installation command.

## Evaluation questions

1. Does built-in discovery materially improve first-run usability compared with separately
   installed first-party plugins?
2. Is a 64% wheel increase acceptable when required dependencies and eager imports remain
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
