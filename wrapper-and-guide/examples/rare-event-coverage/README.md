# Rare-event coverage

Weighted-sample a telemetry stream so rare failure windows don't get dropped
when ordinary distribution matching kicks in. Useful when the rare event lives
in a much larger stream of healthy telemetry and you need it represented in
your training set.

**Composes:** `.measure("event_rate", "rate", of="fault") -> .sample("weighted", n=15, weights={"event_rate": 2.0})` .

**Extras needed:** none. Core install works.

## Run

```
python examples/rare-event-coverage/rare_event_coverage.py
```

## Expected output

```
selected: 15 windows
features on first window: ['event_rate']

compiled chain:
  RareEventCoverage: .measure('event_rate', 'rate', of='fault') -> .sample('weighted', n=15, weights={'event_rate': 2.0})
```

## What to try next

- Increase `weight` (2.0 → 5.0) to focus the sampler more tightly on rare-event windows.
- Change `of="fault"` to another boolean or 0/1 field (`latch_alarm`, `sensor_dropout`, etc.).
- Rename the measured feature via `feature="alarm_rate"` if the default `event_rate` name is not specific enough for your data.

## Related

- [Kit gallery](../README.md)
- [`design/01_operation_abstraction.md`](../../design/01_operation_abstraction.md): the Operation contract
- [`design/03_config_model.md`](../../design/03_config_model.md): driving this from a YAML config
