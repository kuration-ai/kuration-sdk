# Motion-diversity coverage

**Cover a motion axis evenly, so the tail isn't crowded out by the boring middle.**

Most gameplay is the same movement: walking forward, standard turns, level ground. Sample naively and the training set inherits that natural distribution and barely touches the harder motions (tight turns, sharp climbs, high-speed segments). Quantile binning matches the distribution, which is exactly what discards rare events. Coverage has to be measured on the axis you actually care about.

`BalancedSelector` is the general-purpose form: measure a per-window metric on one axis (turning, speed, climb, whatever), bin the values evenly across the observed range, then sample the same count from every bin. That's the pattern the SDK's `slam_motion_diversity` uses on `slam_turn_degrees`, generalized to any per-window column.

**Composes:** `.measure("motion", "std", of="turn_rate") -> .bin("motion", bins=4) -> .sample("equal", n=20)` .

**Extras needed:** none. Core install works.

## Run

```
python examples/motion-coverage/motion_coverage.py
```

## Expected output

```
selected: 20 windows
features on first window: ['motion']

compiled chain:
  BalancedSelector: .measure('motion', 'std', of='turn_rate') -> .bin('motion', bins=4) -> .sample('equal', n=20)
```

## What to try next

- Increase `bins` (4 → 8 → 16) to see how coverage granularity changes. `slam_motion_diversity` in the Intro World-Model-Data-Curation blog uses `bins=8` on `slam_turn_degrees`.
- Swap `kind="std"` for `kind="mean"` or `kind="max"` to bin on a different summary of the same axis.
- Replace the synthetic `sessions` dict with real SLAM output. The SDK's `slam_pipeline` produces columns like `slam_turn_degrees` that plug straight in as `of=...`.

## Related

- [Kit gallery](../README.md)
- [`design/01_operation_abstraction.md`](../../design/01_operation_abstraction.md): the Operation contract
- [`design/03_config_model.md`](../../design/03_config_model.md): driving this from a YAML config
