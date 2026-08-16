# Balanced sensor shards

Turn a folder of device logs into windows selected across the full range of a
signal's variability, not just its natural distribution peak. Useful when your
telemetry has quiet devices and busy devices and you don't want the busy ones
to dominate the training set.

**Composes:** `.measure("activity", "std", of="signal") -> .bin("activity", bins=4) -> .sample("equal", n=20)` .

**Extras needed:** none. Core install (`pip install kuration-ai`) works.

## Run

```
python examples/balanced-sensor-shards/balanced_sensor_shards.py
```

## Expected output

```
selected: 20 windows
features on first window: ['activity']

compiled chain:
  BalancedSelector: .measure('activity', 'std', of='signal') -> .bin('activity', bins=4) -> .sample('equal', n=20)
```

## What to try next

- Increase `bins` (4 → 8 → 16) to see how coverage granularity changes.
- Swap `kind="std"` for `kind="mean"` or `kind="max"` to balance by a different metric.
- Replace the synthetic `sessions` dict with a directory of CSV device logs: Kuration's `timeseries` adapter reads a folder the same way.

## Related

- [Kit gallery](../README.md)
- [`design/01_operation_abstraction.md`](../../design/01_operation_abstraction.md): the Operation contract
- [`design/03_config_model.md`](../../design/03_config_model.md): driving this from a YAML config
