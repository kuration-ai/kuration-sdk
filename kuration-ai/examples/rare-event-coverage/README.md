# Rare-event coverage

**Oversample the moments a model needs to see, but good players avoid.**

A world model only learns the physics that show up consistently in its training data. But the exact interactions a world model needs to learn (a moving object meeting a solid one, a player-object contact) are systematically underrepresented in any dataset built from real play. Good players don't run into walls. DIAMOND's CS:GO world model let players double- and triple-jump against walls precisely because successive jumps didn't appear often enough in the training data for the model to learn they shouldn't be possible.

The fix isn't collecting more footage. It's finding the events that are already buried in the footage you have and oversampling them relative to their natural, very low, base rate. `RareEventCoverage` handles the oversampling half of that: measure a per-window rate of the rare event, then weight the sampler so windows containing it are picked far more often than they occur in the wild.

**Composes:** `.measure("event_rate", "rate", of="collision") -> .sample("weighted", n=15, weights={"event_rate": 2.0})` .

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
  RareEventCoverage: .measure('event_rate', 'rate', of='collision') -> .sample('weighted', n=15, weights={'event_rate': 2.0})
```

## What to try next

- Increase `weight` (2.0 → 5.0) to focus the sampler more tightly on the tail. Coverage has to be measured on the axis you actually care about, and a stronger weight buys more of that axis at the cost of other diversity.
- Change `of="collision"` to another 0/1 event column: `glitch` from the SDK quickstart, a violation-rate column derived from `action_state_consistency`, or your own flag column.
- Rename the measured feature via `feature="collision_rate"` if the default `event_rate` name is not specific enough for your data.

## How this fits the wider curation loop

Rare-event coverage is the sampling step. Upstream, the Kuration SDK ships operators that produce the flag column this operation weights on: `collision` for player-object and player-wall contact, and `action_state_consistency` for actions whose screen motion breaks the dataset's own baseline. Flag first, then oversample.

## Related

- [Kit gallery](../README.md)
- [`design/01_operation_abstraction.md`](../../design/01_operation_abstraction.md): the Operation contract
- [`design/03_config_model.md`](../../design/03_config_model.md): driving this from a YAML config
