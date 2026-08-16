# starter

A working minimal Kuration SDK pipeline you can copy, run, and extend. Runs green out of the box against a tiny synthetic dataset.

![](https://img.shields.io/badge/working_minimal_example-copy_and_extend-0D9488?style=for-the-badge)
![](https://img.shields.io/badge/single_install_cell-one_TODO_at_the_end-0D9488?style=for-the-badge)
![](https://img.shields.io/badge/no_external_data-runs_locally-0D9488?style=for-the-badge)

![](https://img.shields.io/badge/5_devices-500_ticks_each-0D9488)
![](https://img.shields.io/badge/pipeline-window_measure_bin_sample_collect-0D9488)
![](https://img.shields.io/badge/dependency-kuration--sdk-0D9488)

[![Colab](https://img.shields.io/badge/Colab-open-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aintropy-solutions/beta-solutions/blob/main/kuration-wrapper-and-guide/notebooks/starter/starter.ipynb) [![Binder](https://img.shields.io/badge/Binder-launch-579ACA?logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/aintropy-solutions/beta-solutions/HEAD?labpath=kuration-wrapper-and-guide%2Fnotebooks%2Fstarter%2Fstarter.ipynb)

> **Launcher status.** The URLs above point at `aintropy-solutions/beta-solutions`. Colab and Binder can only clone public repositories, so the badges activate once this repo goes public. In the meantime, clone and run the notebook locally with `pip install kuration-sdk`.

<details><summary><b>Contents</b></summary>

- [The idea in one paragraph](#the-idea-in-one-paragraph)
- [What's in the notebook](#whats-in-the-notebook)
- [Try it locally](#try-it-locally)
- [Related](#related)

</details>

## The idea in one paragraph

Before jumping into the operator-specific walkthroughs, get a feel for the pipeline shape. This notebook generates five synthetic device sessions, cuts them into windows, measures activity per window, bins the results into three buckets, and samples uniformly across bins so quiet and loud regions both show up. Read the cells, run them, then extend the last cell with your own data or a different measure. Every walkthrough in this folder builds on the same shape.

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
