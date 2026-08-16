# starter

A working minimal Kuration SDK pipeline you can copy, run, and extend. Runs green out of the box against a tiny synthetic dataset. Ends with one TODO for you to fill in.

[![Colab](https://img.shields.io/badge/Colab-open-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aintropy-solutions/beta-solutions/blob/main/kuration-wrapper-and-guide/notebooks/starter/starter.ipynb) [![Binder](https://img.shields.io/badge/Binder-launch-579ACA?logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/aintropy-solutions/beta-solutions/HEAD?labpath=kuration-wrapper-and-guide%2Fnotebooks%2Fstarter%2Fstarter.ipynb)

> **Launcher status.** The URLs above point at `aintropy-solutions/beta-solutions`. Colab and Binder can only clone public repositories, so the badges activate once this repo goes public. In the meantime, clone and run locally.

## What's in the notebook

1. Install `kuration-sdk`
2. Generate a tiny synthetic dataset (5 devices, 500 ticks each)
3. Run a working pipeline: `window` → `measure` → `bin` → `sample("equal")` → `collect`
4. One TODO cell for you to extend

## Try it locally

```bash
git clone https://github.com/aintropy-solutions/beta-solutions.git
cd beta-solutions/kuration-wrapper-and-guide/notebooks/starter
uv pip install kuration-sdk jupyter
uv run jupyter lab starter.ipynb
```

## Related

- [`../action-state-consistency/`](../action-state-consistency/) walkthrough of the `action_state_consistency` operator
- [`../collision-aware/`](../collision-aware/) walkthrough of the `aintropy/collision-directional` operator
- [`../coverage-based-curation/`](../coverage-based-curation/) coverage-based sampling walkthrough
- Back to the wrapper index: [`../../README.md`](../../README.md)
