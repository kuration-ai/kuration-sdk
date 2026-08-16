<p align="center">
  <img src="assets/beta-solutions-hero.svg" alt="Kuration SDK wrapper and solutions guide" width="100%">
</p>

# kuration-wrapper-and-guide

Runnable notebooks and operator reference for solutions built on the [Kuration SDK](https://github.com/aintropy-ai/kuration-sdk).

![](https://img.shields.io/badge/20.9%25-LPIPS_improvement_over_DIAMOND-0D9488?style=for-the-badge)
![](https://img.shields.io/badge/62%25-JEDi_reduction_across_600_epochs-0D9488?style=for-the-badge)
![](https://img.shields.io/badge/<18%25-training_budget_to_reach_near--baseline-0D9488?style=for-the-badge)

![](https://img.shields.io/badge/4-runnable_notebooks-0D9488)
![](https://img.shields.io/badge/1,090-test_episodes_scored-0D9488)
![](https://img.shields.io/badge/4--second_segments-16_fps-0D9488)
![](https://img.shields.io/badge/51--dim_action-20_categories-0D9488)

## What's in here

- **[`notebooks/`](./notebooks/)**. One-click Binder and Colab launchers. The first cell in each notebook installs the SDK.
- **[`solutions-guide/`](./solutions-guide/)**. The companion `kuration-ai` Python package (`kai` CLI, three shipped operations: `BalancedSelector`, `DiverseSubset`, `RareEventCoverage`), mirrored from `kuration-ai/python-sdk`.

Long-form blog posts on physical-AI data curation live on [aintropy.ai/blog](https://aintropy.ai/blog), published directly from Drive through the content pipeline.

## Run any notebook in a browser

Every notebook has a Colab and a Binder badge. Click, wait, run.

> **Launcher status.** The URLs below point at `aintropy-solutions/beta-solutions`. Colab and Binder can only clone public repositories, so the badges activate once this repo goes public. In the meantime, clone and run each notebook locally. See the notebook's own README for the exact install command (the `[gameplay]` extra is only needed by `action-state-consistency` and `collision-aware`).


| Notebook | Read | Launch |
|---|---|---|
| `starter` | [README](./notebooks/starter/) | [![Colab](https://img.shields.io/badge/Colab-open-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aintropy-solutions/beta-solutions/blob/main/kuration-wrapper-and-guide/notebooks/starter/starter.ipynb) [![Binder](https://img.shields.io/badge/Binder-launch-579ACA?logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/aintropy-solutions/beta-solutions/HEAD?labpath=kuration-wrapper-and-guide%2Fnotebooks%2Fstarter%2Fstarter.ipynb) |
| `action-state-consistency` | [README](./notebooks/action-state-consistency/) | [![Colab](https://img.shields.io/badge/Colab-open-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aintropy-solutions/beta-solutions/blob/main/kuration-wrapper-and-guide/notebooks/action-state-consistency/action-state-consistency.ipynb) [![Binder](https://img.shields.io/badge/Binder-launch-579ACA?logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/aintropy-solutions/beta-solutions/HEAD?labpath=kuration-wrapper-and-guide%2Fnotebooks%2Faction-state-consistency%2Faction-state-consistency.ipynb) |
| `collision-aware` | [README](./notebooks/collision-aware/) | [![Colab](https://img.shields.io/badge/Colab-open-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aintropy-solutions/beta-solutions/blob/main/kuration-wrapper-and-guide/notebooks/collision-aware/collision-aware.ipynb) [![Binder](https://img.shields.io/badge/Binder-launch-579ACA?logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/aintropy-solutions/beta-solutions/HEAD?labpath=kuration-wrapper-and-guide%2Fnotebooks%2Fcollision-aware%2Fcollision-aware.ipynb) |
| `coverage-based-curation` | [README](./notebooks/coverage-based-curation/) | [![Colab](https://img.shields.io/badge/Colab-open-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aintropy-solutions/beta-solutions/blob/main/kuration-wrapper-and-guide/notebooks/coverage-based-curation/coverage-based-curation.ipynb) [![Binder](https://img.shields.io/badge/Binder-launch-579ACA?logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/aintropy-solutions/beta-solutions/HEAD?labpath=kuration-wrapper-and-guide%2Fnotebooks%2Fcoverage-based-curation%2Fcoverage-based-curation.ipynb) |

## Contributing

See the [contributing guide](../CONTRIBUTING.md) at the repo root.

## License

[Apache 2.0](../LICENSE).
