# coverage-based-curation

Sample equally across feature bins so training sees the whole distribution, not just the peak.

[![7,530 training episodes](https://img.shields.io/badge/7,530-training_episodes-0D9488)](./coverage-based-curation.ipynb)
[![35 candidate strategies triaged](https://img.shields.io/badge/35-candidate_strategies_triaged-0D9488)](./coverage-based-curation.ipynb)

[![Colab](https://img.shields.io/badge/Colab-open-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aintropy-solutions/kuration-sdk/blob/main/kuration-sdk-wrapper/notebooks/coverage-based-curation/coverage-based-curation.ipynb) [![Binder](https://img.shields.io/badge/Binder-launch-579ACA?logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/aintropy-solutions/kuration-sdk/HEAD?labpath=kuration-sdk-wrapper%2Fnotebooks%2Fcoverage-based-curation%2Fcoverage-based-curation.ipynb)

> **Launcher status.** The URLs above point at `aintropy-solutions/kuration-sdk`. Colab and Binder can only clone public repositories, so the badges activate once this repo goes public. In the meantime, clone and run the notebook locally with `pip install kuration-sdk matplotlib`.

<details><summary><b>Contents</b></summary>

- [Setup](#setup)
- [The idea in one paragraph](#the-idea-in-one-paragraph)
- [Build a skewed sensor corpus](#build-a-skewed-sensor-corpus)
- [Row-uniform baseline](#row-uniform-baseline)
- [Coverage-based sampling gives every bin equal weight](#coverage-based-sampling-gives-every-bin-equal-weight)
- [Why this matters at scale](#why-this-matters-at-scale)
- [Try it locally](#try-it-locally)
- [Related](#related)

</details>

## Setup

```python
%pip install --quiet kuration-sdk matplotlib
```

See cell 2.

## The idea in one paragraph

Ordinary matches cluster tightly around a mean player movement speed in the high 20s. The rare but informative segments (players moving at physical extremes, taking impacts, colliding with geometry) sit far out in the tail. Row-uniform sampling barely touches that tail. Coverage-based curation quantile-bins the range and samples equally from each bin, giving quiet and extreme regions equal representation regardless of their volume in the raw data. It is the coverage-maximizing default: distribution coverage alone, no domain-specific hypothesis about which behavior is most informative.

## Build a skewed sensor corpus

10 devices, each with 1000 ticks. Activity is heavily concentrated around a low regime. A small tail of devices (indices 8 and 9) runs at high activity, deliberately skewed.

```python
for i in range(N_DEVICES):
    amplitude = 0.3 if i < 8 else 3.5
    ...
```

See cell 5.

## Row-uniform baseline

Enumerate every candidate window, pick 100 at random with numpy. This is what row-uniform sampling produces without any strategy at all: the high-activity tail (devices 8 and 9) is under-represented in proportion to how much time it spends in that regime.

```python
pipeline = input(sessions, time="t", hz=1).window(50, stride=25)

# .collect() always needs a resolvable selection method; "filter" with a trivially-true
# predicate gets the full, unfiltered candidate pool back.
all_candidates = (
    pipeline
    .measure("_all", "count", of="signal")
    .filter("_all", ">=", 0)
    .sample("filter", n=1)
    .collect()
)
uniform_pick_idx = rng.choice(len(all_candidates.selected), size=100, replace=False)
uniform_devices = [all_candidates.selected[i].source_id for i in uniform_pick_idx]
```

See cell 7.

## Coverage-based sampling gives every bin equal weight

Measure activity per window, quantile-bin it into 4 bins, then sample equally from each bin. The tail gets a proper share.

```python
coverage_result = (
    pipeline
    .measure("activity", "std", of="signal")
    .bin("activity", bins=4)
    .sample("equal", n=100)
    .options(seed=1)
    .collect()
)
```

See cell 9.

## Why this matters at scale

Internal evaluation on a coverage-based curation of CS2 gameplay found that training on this selection reaches near-baseline generation quality in a small fraction of the training compute needed by training on the full, uncurated corpus — without any domain-specific hypothesis about which behavior is most informative.

The CS2 strategy is a JSON spec, not fluent-API code. A `collision-aware` variant uses the same three axes but with a weighted selection method:

```json
{
  "name": "collision-aware",
  "define": {
    "impact_rate":  {"source": "impact.type", "reduce": "rate"},
    "move_speed":   {"source": "player.vel", "map": "magnitude",   "reduce": "mean", "scope": "per_player_mean"},
    "aim_velocity": {"source": "player.ang", "map": "angle_delta", "reduce": "mean", "scope": "per_player_mean"}
  },
  "binning": {
    "axes": [
      {"metric": "impact_rate",  "bins": 10},
      {"metric": "move_speed",   "bins": 10},
      {"metric": "aim_velocity", "bins": 10}
    ]
  },
  "selection": {
    "method": "weighted",
    "weights": {"impact_rate": 2.0, "move_speed": 1.0, "aim_velocity": 0.5}
  }
}
```

Ten bins per axis. Weighted selection over three axes. Impact rate carries 2x the weight of movement speed and 4x aim velocity.

The `coverage-based` strategy swaps `weighted` for `uniform-bins` selection over the same three axes. Selecting equally across all three axes at once gives the model the genuinely distinct segments it needs. Selecting by any single axis produces a similar dataset each time because those properties co-occur.

Internal triage across dozens of candidate strategies found that many collapsed to near-baseline behavior; coverage-based, rare-trajectory, and collision-aware were among the few that produced genuinely distinct datasets.

See cell 10.

## Try it locally

```bash
git clone https://github.com/aintropy-solutions/kuration-sdk.git
cd kuration-sdk/kuration-sdk-wrapper/notebooks/coverage-based-curation
uv pip install kuration-sdk matplotlib jupyter
uv run jupyter lab coverage-based-curation.ipynb
```

## Related

- Paired long-form post: [aintropy.ai/blog/virtual2real-gap-paper](https://aintropy.ai/blog/virtual2real-gap-paper)
- Back to the wrapper index: [`../../README.md`](../../README.md)
